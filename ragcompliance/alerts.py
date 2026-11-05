"""Opt-in anomaly alerts to Slack (or any webhook that accepts the Slack
incoming-webhook JSON shape).

Four detection rules, all configurable via env:

* ``retrieval_returned_zero_chunks``: the retriever returned no documents.
* ``low_similarity``: best matching chunk was below
  ``RAGCOMPLIANCE_SLACK_MIN_SIMILARITY`` (default 0.3).
* ``chain_errored``: the framework surfaced an exception.
* ``chain_slow``: end-to-end latency exceeded
  ``RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS`` (default 10000).

Data egress: by default the payload carries ids, the rule names and the
record id, never the query text. Set ``RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true``
to include the (redacted, if redaction is on) query, truncated to 240
characters (DECISIONS.md D08).

Alerts are posted on a single daemon worker thread with a bounded queue
so a Slack outage cannot back-pressure the chain.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
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
    include_query: bool = False
    _queue: queue.Queue = field(init=False, repr=False)
    _stop: threading.Event = field(init=False, repr=False)
    _worker: threading.Thread = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=self.max_queue)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._drain, name="ragcompliance-alerts", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def evaluate(self, record: Any, error: BaseException | None = None) -> list[AlertRule]:
        """Run all detection rules. Returns the rules that fired."""
        fired: list[AlertRule] = []

        if error is not None:
            fired.append(AlertRule("chain_errored", f"{error.__class__.__name__}: {str(error)[:200]}"))

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
        """Evaluate and enqueue a payload if any rule fired. Never raises."""
        try:
            fired = self.evaluate(record, error)
        except Exception as exc:
            logger.debug("RAGCompliance: alert evaluate() errored (%s)", exc)
            return False
        if not fired:
            return False
        payload = self._build_payload(record, fired)
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            logger.warning(
                "RAGCompliance: Slack alert queue full (size=%d). Dropping alert for session=%r.",
                self.max_queue,
                getattr(record, "session_id", "unknown"),
            )
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        return self._queue.empty()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _build_payload(self, record: Any, fired: list[AlertRule]) -> dict[str, Any]:
        lines = [
            ":rotating_light: *RAGCompliance alert*",
            (
                f"workspace: `{getattr(record, 'workspace_id', '?')}` | "
                f"session: `{getattr(record, 'session_id', '?')}` | "
                f"record: `{getattr(record, 'id', '?')}`"
            ),
        ]
        actor_id = (getattr(record, "actor", None) or {}).get("id")
        if actor_id:
            lines.append(f"actor: `{actor_id}`")
        if self.include_query:
            query = (getattr(record, "query", "") or "")[:240]
            if len(getattr(record, "query", "") or "") > 240:
                query += "..."
            lines.append(f"query: `{query}`")
        lines.append("")
        for rule in fired:
            lines.append(f"- *{rule.name}*: {rule.detail}")

        if self.dashboard_url and getattr(record, "id", None):
            lines.append("")
            lines.append(
                f"<{self.dashboard_url.rstrip('/')}/logs/detail/{record.id}|View in dashboard>"
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
                with urllib.request.urlopen(req, timeout=5.0):
                    pass
            except Exception as exc:
                logger.warning("RAGCompliance: Slack alert post failed: %s", exc)
            finally:
                try:
                    self._queue.task_done()
                except ValueError:  # pragma: no cover - defensive
                    pass

    @classmethod
    def from_env(cls, config: Any | None = None) -> SlackAlerter | None:
        """Construct from env if ``RAGCOMPLIANCE_SLACK_WEBHOOK_URL`` is set."""
        webhook = os.getenv("RAGCOMPLIANCE_SLACK_WEBHOOK_URL", "").strip()
        if not webhook:
            return None
        return cls(
            webhook_url=webhook,
            dashboard_url=os.getenv("RAGCOMPLIANCE_SLACK_DASHBOARD_URL", "").strip(),
            min_similarity=float(os.getenv("RAGCOMPLIANCE_SLACK_MIN_SIMILARITY", "0.3")),
            slow_chain_ms=int(os.getenv("RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS", "10000")),
            max_queue=int(os.getenv("RAGCOMPLIANCE_SLACK_MAX_QUEUE", "100")),
            include_query=os.getenv("RAGCOMPLIANCE_SLACK_INCLUDE_QUERY", "false").lower()
            in ("1", "true", "yes", "on"),
        )
