import atexit
import json
import logging
import queue
import threading
import time
from typing import Any

from .config import RAGComplianceConfig
from .models import AuditRecord

logger = logging.getLogger(__name__)


class AuditStorage:
    """
    Persists audit records to Supabase.

    Two write paths:
      * Synchronous — save() runs the INSERT inline and returns once Supabase
        acknowledges. Used when async_writes=False, or when the Supabase
        client isn't available (dev mode / missing creds).
      * Asynchronous (default) — save() enqueues the record for a single
        daemon worker thread and returns immediately. The chain's hot path
        never blocks on Supabase. On process shutdown, an atexit hook drains
        any pending records within async_shutdown_timeout.
    """

    # Class-level defaults so that tests which construct AuditStorage via
    # __new__ (bypassing __init__) still see a safe sync code path instead
    # of AttributeError on self._queue.
    _queue: queue.Queue | None = None
    _stop_event: threading.Event | None = None
    _worker: threading.Thread | None = None

    def __init__(self, config: RAGComplianceConfig):
        self.config = config
        self._client = None

        # Async worker state — initialized below only if we actually have a
        # live Supabase client AND config.async_writes is on.
        self._queue = None
        self._stop_event = None
        self._worker = None

        if config.supabase_url and config.supabase_key:
            try:
                from supabase import create_client
                self._client = create_client(config.supabase_url, config.supabase_key)
                logger.info("RAGCompliance: Supabase client initialized.")
            except ImportError:
                logger.warning(
                    "RAGCompliance: supabase package not installed. "
                    "Run: pip install supabase. Falling back to dev mode."
                )
        else:
            if not config.dev_mode:
                logger.warning(
                    "RAGCompliance: No Supabase credentials found. "
                    "Set RAGCOMPLIANCE_DEV_MODE=true to suppress this warning."
                )

        if self._client is not None and config.async_writes:
            self._start_worker()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def save(self, record: AuditRecord) -> bool:
        """
        Persist an audit record. Returns True on success or on successful
        enqueue in async mode. Never raises — storage errors must not break
        the host chain.
        """
        if not self.config.enabled:
            return True

        if self._client is None:
            if self.config.dev_mode:
                print(
                    "[RAGCompliance DEV] Audit record:\n"
                    + json.dumps(record.to_dict(), indent=2, default=str)
                )
            return True

        if self._queue is not None:
            try:
                self._queue.put_nowait(record)
                return True
            except queue.Full:
                logger.warning(
                    "RAGCompliance: async queue full (size=%d). Dropping "
                    "audit record for session=%r. Check Supabase reachability "
                    "or raise RAGCOMPLIANCE_ASYNC_MAX_QUEUE.",
                    self.config.async_max_queue,
                    record.session_id,
                )
                return False

        return self._save_sync(record)

    def flush(self, timeout: float | None = None) -> bool:
        """
        Block until the async queue is empty, or timeout seconds elapse.
        Useful in tests and in app shutdown hooks. Returns True if the
        queue drained cleanly, False if the timeout was hit.

        No-op when async writes are disabled.
        """
        if self._queue is None:
            return True
        t = timeout if timeout is not None else self.config.async_shutdown_timeout
        deadline = time.monotonic() + t
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.empty()

    def query(
        self,
        workspace_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._client is None:
            logger.warning("RAGCompliance: No Supabase client — cannot query.")
            return []

        try:
            q = self._client.table(self.config.table_name).select("*")
            if workspace_id:
                q = q.eq("workspace_id", workspace_id)
            if session_id:
                q = q.eq("session_id", session_id)
            result = q.order("timestamp", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"RAGCompliance: Query failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _save_sync(self, record: AuditRecord) -> bool:
        try:
            self._client.table(self.config.table_name).insert(record.to_dict()).execute()
            return True
        except Exception as e:
            logger.error(f"RAGCompliance: Failed to save audit record: {e}")
            return False

    def _start_worker(self) -> None:
        self._queue = queue.Queue(maxsize=self.config.async_max_queue)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._drain_loop,
            name="ragcompliance-writer",
            daemon=True,
        )
        self._worker.start()
        # Best-effort shutdown drain. Won't fire on os._exit() or segfaults,
        # but catches normal process exit (including Ctrl-C via sys.exit).
        atexit.register(self._shutdown)
        logger.debug(
            "RAGCompliance: async writer started (max_queue=%d).",
            self.config.async_max_queue,
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
            except Exception as e:
                # _save_sync already swallows, but belt-and-suspenders: a dead
                # worker thread would silently stop draining the queue.
                logger.error(f"RAGCompliance: writer worker exception: {e}")
            finally:
                # Always mark done so flush()/_shutdown() can progress even on
                # repeated failures.
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    def _shutdown(self) -> None:
        """Drain pending writes, then signal the worker to stop. Called on
        process exit via atexit. Intentionally bounded by
        async_shutdown_timeout so a stuck Supabase connection can't hang
        interpreter shutdown forever."""
        if self._queue is None or self._stop_event is None:
            return
        drained = self.flush(timeout=self.config.async_shutdown_timeout)
        if not drained:
            pending = self._queue.qsize()
            logger.warning(
                "RAGCompliance: shutdown timeout with %d audit record(s) "
                "still pending. Records may be lost.",
                pending,
            )
        self._stop_event.set()
        if self._worker is not None and self._worker.is_alive():
            # Give the worker a final short grace window to exit the loop.
            self._worker.join(timeout=1.0)
