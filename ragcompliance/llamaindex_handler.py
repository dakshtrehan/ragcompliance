"""
RAGCompliance LlamaIndex callback handler.

Captures retrieval + LLM events from a LlamaIndex pipeline and writes the same
audit record shape as the LangChain handler. Depends only on `llama_index.core`.

Usage:

    from llama_index.core.callbacks import CallbackManager
    from ragcompliance import RAGComplianceConfig
    from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler

    handler = LlamaIndexRAGComplianceHandler(
        config=RAGComplianceConfig.from_env(),
        session_id="user-abc",
    )
    callback_manager = CallbackManager([handler])
    # pass callback_manager into Settings or your LLM / query engine
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .alerts import SlackAlerter
from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler as _LIBase
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    _LLAMA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep path
    _LIBase = object  # type: ignore
    CBEventType = None  # type: ignore
    EventPayload = None  # type: ignore
    _LLAMA_AVAILABLE = False


# Sentinel used when LlamaIndex passes trace_id=None (default single-trace
# usage). Keyed on a stable value so concurrent None-trace calls from
# different threads still collide (and the caller is then expected to use
# distinct handler instances, which is the legacy behavior).
_DEFAULT_TRACE = "__ragcompliance_default_trace__"


@dataclass
class _TraceState:
    """Per-trace state for one LlamaIndex query lifecycle.

    One of these lives in ``_traces`` keyed by ``trace_id`` while a trace
    is active. This replaces the previous single instance-level buffer and
    makes concurrent query engines on a shared handler correct.
    """

    query: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    llm_answer: str = ""
    model_name: str = ""
    start_time: float = 0.0


class LlamaIndexRAGComplianceHandler(_LIBase):  # type: ignore[misc]
    """
    LlamaIndex callback that mirrors the LangChain audit capture:
    query -> retrieved chunks -> LLM answer -> SHA-256 chain signature.

    Thread safety
    -------------
    This handler is safe to share across concurrent LlamaIndex query
    engines. Per-trace state lives in a lock-guarded ``dict`` keyed by the
    ``trace_id`` LlamaIndex passes into ``start_trace``/``end_trace``, so
    events from different traces cannot interleave. The configured
    ``AuditStorage``, ``BillingManager``, and ``SlackAlerter`` objects are
    also safe to share.
    """

    def __init__(
        self,
        config: RAGComplianceConfig | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        billing: Any | None = None,
        alerter: SlackAlerter | None = None,
    ):
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "llama-index-core is not installed. "
                "Install with: pip install ragcompliance[llamaindex]"
            )
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

        self.config = config or RAGComplianceConfig.from_env()
        self.storage = AuditStorage(self.config)
        self.session_id = session_id or str(uuid.uuid4())
        self.extra = extra or {}

        if billing is None:
            try:
                from .billing import BillingManager
                self.billing = BillingManager.from_env()
            except Exception as e:
                logger.debug(f"RAGCompliance: billing disabled ({e})")
                self.billing = None
        else:
            self.billing = billing

        # Optional Slack alerter — opt-in via RAGCOMPLIANCE_SLACK_WEBHOOK_URL.
        if alerter is None:
            try:
                self.alerter = SlackAlerter.from_env()
            except Exception as e:
                logger.debug(f"RAGCompliance: alerter disabled ({e})")
                self.alerter = None
        else:
            self.alerter = alerter

        # Per-trace state, keyed by trace_id. LlamaIndex calls
        # start_trace/end_trace to bracket a query lifecycle; between them
        # on_event_* fires without a trace_id. We use thread-local state
        # to remember the active trace for the current thread, so a
        # shared handler across threads stays correct.
        self._traces: dict[Any, _TraceState] = {}
        self._active = threading.local()
        self._lock = threading.Lock()

        try:
            self._max_pending_traces = max(
                1, int(os.getenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "10000"))
            )
        except ValueError:
            self._max_pending_traces = 10000
        self._pending_cap_warned = False

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _trace_key(self, trace_id: Any) -> Any:
        return trace_id if trace_id is not None else _DEFAULT_TRACE

    def _enforce_pending_cap_locked(self) -> None:
        if len(self._traces) <= self._max_pending_traces:
            return
        if not self._pending_cap_warned:
            logger.warning(
                "RAGCompliance: pending trace state exceeded "
                "RAGCOMPLIANCE_MAX_PENDING_RUNS=%d; evicting oldest traces. "
                "This usually means end_trace was not delivered for some "
                "queries (crashed workers, misconfigured callbacks).",
                self._max_pending_traces,
            )
            self._pending_cap_warned = True
        while len(self._traces) > self._max_pending_traces:
            victim_id, _ = next(iter(self._traces.items()))
            self._traces.pop(victim_id, None)

    def _current_trace_id(self) -> Any:
        """The trace_id LlamaIndex most recently called start_trace with
        on this thread. Falls back to _DEFAULT_TRACE so direct calls to
        on_event_* without a surrounding trace still collect state."""
        return getattr(self._active, "trace_id", _DEFAULT_TRACE)

    def _current_state_locked(self) -> _TraceState | None:
        """Return the _TraceState for the current thread's active trace,
        or None if no trace is active. Caller holds the lock."""
        return self._traces.get(self._current_trace_id())

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                      #
    # ------------------------------------------------------------------ #

    # LlamaIndex calls these four methods on the handler.
    def start_trace(self, trace_id: str | None = None) -> None:
        key = self._trace_key(trace_id)
        # Remember the active trace for the CURRENT THREAD so on_event_*
        # can route to the right state even in concurrent usage.
        self._active.trace_id = key
        with self._lock:
            self._traces[key] = _TraceState(start_time=time.time())
            self._enforce_pending_cap_locked()

        # Billing quota check runs outside the lock.
        if self.billing is not None:
            try:
                within = self.billing.check_query_quota(self.config.workspace_id)
            except Exception:
                within = True
            if not within:
                msg = (
                    f"RAGCompliance: workspace {self.config.workspace_id!r} "
                    "has exceeded its plan query quota for the current period."
                )
                if self.config.enforce_quota:
                    raise RuntimeError(msg)
                logger.warning(msg)

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        key = self._trace_key(trace_id)
        with self._lock:
            state = self._traces.pop(key, None)
        # Clear the thread-local pointer regardless — a subsequent
        # start_trace will reset it.
        if getattr(self._active, "trace_id", None) == key:
            self._active.trace_id = None
        if state is None:
            return

        latency_ms = int((time.time() - state.start_time) * 1000)
        signature = self._sign_chain(state)
        record = AuditRecord(
            session_id=self.session_id,
            workspace_id=self.config.workspace_id,
            query=state.query,
            retrieved_chunks=state.chunks,
            llm_answer=state.llm_answer,
            model_name=state.model_name,
            chain_signature=signature,
            latency_ms=latency_ms,
            extra=self.extra,
        )

        # Defensive save: custom storage backends might raise. A compliance
        # library must never take down the host chain because its own audit
        # write failed.
        try:
            self.storage.save(record)
        except Exception as e:
            logger.error(
                "RAGCompliance: storage.save raised (%s); dropping audit "
                "record for session=%r. This should not normally happen — "
                "the built-in Supabase storage catches its own errors. If "
                "you wrote a custom storage backend, it must not raise.",
                e, self.session_id,
            )

        if self.billing is not None:
            try:
                self.billing.increment_usage(self.config.workspace_id)
            except Exception:
                pass
        if self.alerter is not None:
            try:
                self.alerter.maybe_alert(record)
            except Exception as e:
                logger.debug(f"RAGCompliance: alerter.maybe_alert errored ({e})")

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        payload = payload or {}
        # LlamaIndex doesn't pass trace_id into event callbacks — it
        # brackets them with start_trace/end_trace on a per-thread basis.
        # _current_state_locked() reads thread-local state set by
        # start_trace to route the event to the right trace.
        if CBEventType and event_type == CBEventType.QUERY:
            q = None
            if EventPayload is not None:
                q = payload.get(EventPayload.QUERY_STR)
            if not q:
                q = payload.get("query_str") or payload.get("query")
            if q:
                with self._lock:
                    state = self._current_state_locked()
                    if state is not None:
                        state.query = str(q)
        return event_id or str(uuid.uuid4())

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        payload = payload or {}
        if CBEventType is None:
            return

        if event_type == CBEventType.RETRIEVE:
            # Payload key is EventPayload.NODES, which is a StrEnum equal to "nodes".
            nodes = (
                (payload.get(EventPayload.NODES) if EventPayload is not None else None)
                or payload.get("nodes")
                or []
            )
            new_chunks: list[RetrievedChunk] = []
            for i, node_with_score in enumerate(nodes):
                # LlamaIndex yields NodeWithScore; attribute access is safest.
                node = getattr(node_with_score, "node", node_with_score)
                score = getattr(node_with_score, "score", None)
                meta = getattr(node, "metadata", {}) or {}
                content = (
                    getattr(node, "text", None)
                    or getattr(node, "get_content", lambda: "")()
                )
                new_chunks.append(
                    RetrievedChunk(
                        content=str(content),
                        source_url=str(
                            meta.get("source")
                            or meta.get("file_path")
                            or meta.get("url")
                            or "unknown"
                        ),
                        chunk_id=str(
                            meta.get("chunk_id")
                            or getattr(node, "node_id", None)
                            or f"chunk-{i}"
                        ),
                        similarity_score=float(score) if score is not None else None,
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if k not in ("source", "file_path", "url", "chunk_id")
                        },
                    )
                )
            with self._lock:
                state = self._current_state_locked()
                if state is not None:
                    state.chunks.extend(new_chunks)

        elif event_type == CBEventType.LLM:
            # LLM end payload holds a CompletionResponse under EventPayload.COMPLETION
            # (with a .text attribute) OR a ChatResponse under EventPayload.MESSAGES.
            # Older code assumed a "response" key — that key doesn't exist in
            # 0.10+. Prefer COMPLETION, then MESSAGES, then legacy "response".
            completion = None
            if EventPayload is not None:
                completion = payload.get(EventPayload.COMPLETION) or payload.get(
                    EventPayload.MESSAGES
                )
            if completion is None:
                completion = payload.get("completion") or payload.get("response")
            answer = ""
            if completion is not None:
                text = (
                    getattr(completion, "text", None)
                    or getattr(completion, "message", None)
                    or str(completion)
                )
                answer = text.content if hasattr(text, "content") else str(text)

            model_name = ""
            serialized = (
                payload.get(EventPayload.SERIALIZED) if EventPayload is not None else None
            )
            if not isinstance(serialized, dict):
                serialized = payload.get("serialized")
            if isinstance(serialized, dict):
                model = serialized.get("model") or serialized.get("model_name")
                if model:
                    model_name = str(model)

            with self._lock:
                state = self._current_state_locked()
                if state is not None:
                    # Only overwrite answer if non-empty — protects multi-LLM-call
                    # flows where a refine pass might emit an empty continuation.
                    if answer:
                        state.llm_answer = answer
                    if model_name:
                        state.model_name = model_name

        elif event_type == CBEventType.SYNTHESIZE:
            # The synthesize event's end payload carries the FINAL Response
            # object with the assembled answer in .response. This is the
            # canonical capture point for the answer — more reliable than
            # catching the last LLM call, especially in refine/compact modes.
            response = None
            if EventPayload is not None:
                response = payload.get(EventPayload.RESPONSE)
            if response is None:
                response = payload.get("response")
            if response is not None:
                answer = (
                    getattr(response, "response", None)
                    or getattr(response, "text", None)
                    or str(response)
                )
                if answer:
                    with self._lock:
                        state = self._current_state_locked()
                        if state is not None:
                            state.llm_answer = str(answer)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _sign_chain(self, state: _TraceState) -> str:
        payload = {
            "query": state.query,
            "chunks": [
                {"content": c.content, "source_url": c.source_url, "chunk_id": c.chunk_id}
                for c in state.chunks
            ],
            "answer": state.llm_answer,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
