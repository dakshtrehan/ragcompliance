"""Storage façade: owns the sink, the async writer and the read API.

``AuditStorage`` is what handlers call ``save()`` on. It decides where a
record goes (a sink built from ``RAGComplianceConfig``), whether the
write is synchronous or queued, and exposes the read side (``query``,
``get_by_id``, ``iter_records``, ``verify``, ``purge_expired``) that the
dashboard, the SOC 2 report and the evidence exporter use.

Sink selection
--------------
1. An explicit ``sink=`` argument.
2. ``config.sink_url`` (``RAGCOMPLIANCE_SINK``).
3. Supabase credentials in the config (0.1.x behaviour).
4. ``dev_mode``: an in-memory sink plus a stdout print of every record.
5. Nothing: ``save()`` logs one warning, counts the record as dropped
   and returns ``False``. In 0.1.x this path returned ``True``; a
   compliance library must not report success for a record it did not
   store (AUDIT.md B04).

Misconfiguration (credentials present, driver missing) raises
``ConfigurationError`` at construction unless ``dev_mode`` is on.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from .chain import VerificationResult, verify_records
from .config import RAGComplianceConfig
from .models import AuditRecord
from .retention import PurgeResult, RetentionPolicy
from .sinks import AuditSink, MemorySink, make_sink

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when storage is configured in a way that cannot work."""


class AuditStorage:
    """Persist audit records through a sink, optionally on a background thread."""

    # Class-level defaults so objects created via __new__ (older tests) still
    # have a safe synchronous code path.
    _queue: queue.Queue | None = None
    _stop_event: threading.Event | None = None
    _worker: threading.Thread | None = None
    _sink: AuditSink | None = None
    _dev_print: bool = False
    dropped_count: int = 0

    def __init__(self, config: RAGComplianceConfig, sink: AuditSink | None = None):
        self.config = config
        self._queue = None
        self._stop_event = None
        self._worker = None
        self._dev_print = False
        self._warned_no_sink = False
        self.dropped_count = 0
        self._counter_lock = threading.Lock()

        self._sink = sink if sink is not None else self._build_sink(config)

        if self._sink is not None and config.async_writes and not self._dev_print:
            self._start_worker()

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    def _build_sink(self, config: RAGComplianceConfig) -> AuditSink | None:
        if config.sink_url:
            try:
                return make_sink(config.sink_url, config)
            except ImportError as exc:
                if config.dev_mode:
                    logger.warning("RAGCompliance: %s; falling back to dev mode.", exc)
                else:
                    raise ConfigurationError(str(exc)) from exc
        elif config.supabase_url and config.supabase_key:
            try:
                return make_sink("supabase://", config)
            except ImportError as exc:
                if config.dev_mode:
                    logger.warning("RAGCompliance: %s; falling back to dev mode.", exc)
                else:
                    raise ConfigurationError(
                        "RAGCOMPLIANCE_SUPABASE_URL/_KEY are set but the supabase package "
                        "is not installed. Run: pip install 'ragcompliance[supabase]'. "
                        "Set RAGCOMPLIANCE_DEV_MODE=true to run without persistence."
                    ) from exc

        if config.dev_mode:
            self._dev_print = True
            return MemorySink(chain_id=config.chain_id, hmac_key=config.hmac_key_bytes)

        logger.warning(
            "RAGCompliance: no audit sink configured (set RAGCOMPLIANCE_SINK or "
            "Supabase credentials). Records will NOT be persisted. Set "
            "RAGCOMPLIANCE_DEV_MODE=true for a local in-memory sink."
        )
        return None

    # ------------------------------------------------------------------ #
    # Public write API                                                   #
    # ------------------------------------------------------------------ #

    @property
    def sink(self) -> AuditSink | None:
        return self._sink

    @property
    def _client(self) -> Any:
        """Compatibility shim: the raw Supabase client when that is the sink."""
        return getattr(self._sink, "_client", None) if self._sink is not None else None

    def save(self, record: AuditRecord) -> bool:
        """Persist an audit record.

        Returns ``True`` when the record was written (or enqueued in async
        mode) and ``False`` when it was dropped. Never raises.
        """
        if not self.config.enabled:
            return True

        if self._sink is None:
            self._count_drop()
            if not self._warned_no_sink:
                self._warned_no_sink = True
                logger.warning(
                    "RAGCompliance: dropping audit record for session=%r because no "
                    "sink is configured.",
                    record.session_id,
                )
            return False

        if self._queue is not None:
            try:
                if self.config.async_overflow == "block":
                    self._queue.put(record, timeout=self.config.async_shutdown_timeout)
                else:
                    self._queue.put_nowait(record)
                return True
            except queue.Full:
                self._count_drop()
                logger.warning(
                    "RAGCompliance: async queue full (size=%d). Dropping audit record for "
                    "session=%r. Check sink reachability or raise "
                    "RAGCOMPLIANCE_ASYNC_MAX_QUEUE. dropped_count=%d",
                    self.config.async_max_queue,
                    record.session_id,
                    self.dropped_count,
                )
                return False

        return self._save_sync(record)

    def flush(self, timeout: float | None = None) -> bool:
        """Block until the async queue is empty or ``timeout`` elapses."""
        if self._queue is None:
            if self._sink is not None:
                self._sink.flush()
            return True
        t = timeout if timeout is not None else self.config.async_shutdown_timeout
        deadline = time.monotonic() + t
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        # Wait for the in-flight record too.
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.empty()

    def close(self) -> None:
        """Drain, stop the worker and close the sink."""
        self._shutdown()
        if self._sink is not None:
            try:
                self._sink.close()
            except Exception as exc:
                logger.debug("RAGCompliance: sink.close raised (%s)", exc)

    # ------------------------------------------------------------------ #
    # Public read API                                                    #
    # ------------------------------------------------------------------ #

    def iter_records(
        self,
        *,
        workspace_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        chain_id: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        if self._sink is None:
            return iter(())
        return self._sink.iter_records(
            workspace_id=workspace_id, start=start, end=end, chain_id=chain_id, limit=limit
        )

    def get_by_id(
        self, record_id: str, workspace_id: str | None = None
    ) -> dict[str, Any] | None:
        if self._sink is None:
            logger.warning("RAGCompliance: no sink configured; cannot get_by_id.")
            return None
        try:
            return self._sink.get(record_id, workspace_id=workspace_id)
        except Exception as exc:
            logger.error("RAGCompliance: get_by_id failed: %s", exc)
            return None

    def query(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first page of records (dashboard shape)."""
        if self._sink is None:
            logger.warning("RAGCompliance: no sink configured; cannot query.")
            return []
        try:
            rows = list(self._sink.iter_records(workspace_id=workspace_id))
        except Exception as exc:
            logger.error("RAGCompliance: query failed: %s", exc)
            return []
        if session_id:
            rows = [r for r in rows if r.get("session_id") == session_id]
        rows.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        return rows[:limit]

    def count(self, *, workspace_id: str | None = None) -> int:
        return self._sink.count(workspace_id=workspace_id) if self._sink is not None else 0

    def verify(
        self,
        *,
        workspace_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        chain_id: str | None = None,
        trust_anchors: list[str] | None = None,
    ) -> VerificationResult:
        """Verify every stored record in scope against the hash chain."""
        records = self.iter_records(
            workspace_id=workspace_id, start=start, end=end, chain_id=chain_id
        )
        return verify_records(
            records,
            hmac_key=self.config.hmac_key_bytes,
            trust_anchors=trust_anchors or (),
        )

    def purge_expired(
        self,
        policy: RetentionPolicy | None = None,
        *,
        workspace_id: str | None = None,
        chain_id: str | None = None,
        now: datetime | None = None,
    ) -> PurgeResult | None:
        """Delete records older than the retention cutoff for one workspace.

        Returns ``None`` when retention is disabled. Raises
        ``PurgeNotSupported`` when the sink cannot delete.
        """
        policy = policy or RetentionPolicy(days=self.config.retention_days)
        cutoff = policy.cutoff(now)
        if cutoff is None or self._sink is None:
            return None
        self.flush()
        return self._sink.purge_before(
            cutoff,
            workspace_id=workspace_id or self.config.workspace_id,
            chain_id=chain_id or self.config.chain_id,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "sink": self._sink.describe() if self._sink is not None else None,
            "async_writes": self._queue is not None,
            "dropped_count": self.dropped_count,
            "retention_days": self.config.retention_days,
            "dev_mode": self.config.dev_mode,
        }

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _count_drop(self) -> None:
        with self._counter_lock:
            self.dropped_count += 1

    def _save_sync(self, record: AuditRecord) -> bool:
        assert self._sink is not None
        try:
            sealed = self._sink.append(record.to_dict())
            record.apply_seal(sealed)
        except Exception as exc:
            self._count_drop()
            logger.error("RAGCompliance: failed to save audit record: %s", exc)
            return False
        if self._dev_print:
            print(
                "[RAGCompliance DEV] Audit record:\n"
                + json.dumps(sealed, indent=2, default=str, ensure_ascii=False)
            )
        return True

    def _start_worker(self) -> None:
        self._queue = queue.Queue(maxsize=self.config.async_max_queue)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._drain_loop, name="ragcompliance-writer", daemon=True
        )
        self._worker.start()
        atexit.register(self._shutdown)
        logger.debug(
            "RAGCompliance: async writer started (max_queue=%d).", self.config.async_max_queue
        )

    def _drain_loop(self) -> None:
        assert self._queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._save_sync(record)
            except Exception as exc:
                logger.error("RAGCompliance: writer worker exception: %s", exc)
            finally:
                try:
                    self._queue.task_done()
                except ValueError:  # pragma: no cover - defensive
                    pass

    def _shutdown(self) -> None:
        if self._queue is None or self._stop_event is None:
            return
        drained = self.flush(timeout=self.config.async_shutdown_timeout)
        if not drained:
            logger.warning(
                "RAGCompliance: shutdown timeout with %d audit record(s) still pending. "
                "Records may be lost.",
                self._queue.qsize(),
            )
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._queue = None
