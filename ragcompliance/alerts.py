"""
Opt-in anomaly alerts to Slack (or any webhook that accepts the Slack
incoming-webhook JSON shape — Discord, Teams with a shim, a custom
receiver).

Four detection rules, all configurable via env:

  * retrieval_returned_zero_chunks — retriever returned no documents, so
    the LLM was running blind. Common failure mode when a vectorstore
    index drifts or a query embedding is malformed.
  * low_similarity — best matching chunk was below
    RAGCOMPLIANCE_SLACK_MIN_SIMILARITY (default 0.3). Signals retrieval
    matched but poorly, which usually produces hallucinations.
  * chain_errored — LangChain or LlamaIndex surfaced an exception before
    the chain completed.
  * chain_slow — end-to-end chain latency exceeded
    RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS (default 10000ms). Signals either
    a slow LLM, a slow vectorstore, or a stuck retrieval.

Alerts are posted on a single daemon worker thread with a bounded queue
so a Slack outage can't back-pressure the chain, and a full queue
drops alerts rather than leaking memory.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    name: str
    detail: str = ""


@dataclass
class SlackAlerter:
    webhook_url: str
    dashboard_url: str = ""
    min_similarity: float = 0.3
    slow_chain_ms: int = 10_000
    max_queue: int = 100
    # Filled in __post_init__ so we don't start threads from dataclass default.
    _queue: queue.Queue = field(init=False, repr=False)
    _stop: threading.Event = field(init=False, repr=False)
    _worker: threading.Thread = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._drain,
            name="ragcompliance-alerts",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def evaluate(self, record: Any, error: BaseException | None = None) -> list[AlertRule]:
        """Run all detection rules against an audit record. Returns the list
        of rules that fired (empty list means the chain looked healthy)."""
        fired: list[AlertRule] = []

        if error is not None:
            fired.append(AlertRule("chain_errored", str(error)[:200]))

        chunks = getattr(record, "retrieved_chunks", []) or []
        if not chunks:
            fired.append(AlertRule("retrieval_returned_zero_chunks", "no documents retrieved"))
        else:
            scores = [
                c.similarity_score
                for c in chunks
                if getattr(c, "similarity_score", None) is not None
            ]
            if scores and max(scores) < self.min_similarity:
                fired.append(
                    AlertRule(
                        "low_similarity",
                        f"max similarity {max(scores):.2f} < threshold {self.min_similarity}",
                    )
                )

        latency_ms = getattr(record, "latency_ms", 0) or 0
        if latency_ms > self.slow_chain_ms:
            fired.append(
                AlertRule("chain_slow", f"{latency_ms}ms > {self.slow_chain_ms}ms threshold")
            )

        return fired

    def maybe_alert(self, record: Any, error: BaseException | None = None) -> bool:
        """Evaluate rules against this record and, if any fired, enqueue a
        Slack payload. Returns True if an alert was enqueued. Never raises."""
        try:
            fired = self.evaluate(record, error)
        except Exception as e:
            logger.debug(f"RAGCompliance: alert evaluate() errored ({e})")
            return False
        if not fired:
            return False
        payload = self._build_payload(record, fired)
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning(
                "RAGCompliance: Slack alert queue full (size=%d). Dropping alert "
                "for session=%r.",
                self.max_queue,
                getattr(record, "session_id", "unknown"),
            )
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until the alert queue drains or timeout elapses. Returns
        True if the queue drained cleanly, False on timeout. Useful in tests."""
        import time
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.empty()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_payload(self, record: Any, fired: list[AlertRule]) -> dict[str, Any]:
        query = (getattr(record, "query", "") or "")[:240]
        if len(getattr(record, "query", "") or "") > 240:
            query += "..."

        lines = [
            ":rotating_light: *RAGCompliance alert*",
            f"workspace: `{getattr(record, 'workspace_id', '?')}` | "
            f"session: `{getattr(record, 'session_id', '?')}`",
            f"query: `{query}`",
            "",
        ]
        for rule in fired:
            lines.append(f"• *{rule.name}*: {rule.detail}")

        if self.dashboard_url and getattr(record, "id", None):
            lines.append("")
            lines.append(
                f"<{self.dashboard_url.rstrip('/')}/logs/detail/{record.id}|"
                f"View in dashboard>"
            )

        return {"text": "\n".join(lines)}

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as _:
                    pass
            except Exception as e:
                # Slack outage or bad webhook URL — log and move on. We do NOT
                # retry: alerts are best-effort and an accumulating retry
                # queue would mask the underlying issue.
                logger.warning(f"RAGCompliance: Slack alert post failed: {e}")
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    @classmethod
    def from_env(cls, config: Any | None = None) -> "SlackAlerter | None":
        """Construct from environment variables if RAGCOMPLIANCE_SLACK_WEBHOOK_URL
        is set, else return None so the handler can skip alert evaluation
        entirely with no overhead."""
        import os

        webhook = os.getenv("RAGCOMPLIANCE_SLACK_WEBHOOK_URL", "").strip()
        if not webhook:
            return None
        return cls(
            webhook_url=webhook,
            dashboard_url=os.getenv("RAGCOMPLIANCE_SLACK_DASHBOARD_URL", "").strip(),
            min_similarity=float(os.getenv("RAGCOMPLIANCE_SLACK_MIN_SIMILARITY", "0.3")),
            slow_chain_ms=int(os.getenv("RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS", "10000")),
            max_queue=int(os.getenv("RAGCOMPLIANCE_SLACK_MAX_QUEUE", "100")),
        )
