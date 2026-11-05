"""LlamaIndex callback handler.

Captures retrieval and LLM events from a LlamaIndex pipeline and writes
the same audit record shape as the LangChain handler, through the same
``Auditor`` core (redaction, attribution, lineage, signing, storage).

Usage::

    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager
    from ragcompliance import RAGComplianceConfig
    from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler

    handler = LlamaIndexRAGComplianceHandler(config=RAGComplianceConfig.from_env(),
                                             session_id="user-abc")
    Settings.callback_manager = CallbackManager([handler])

    with handler.auditor.bind(actor={"id": "alice@example.com", "type": "user"}):
        response = query_engine.query("What does section 4.2 say?")

Thread safety: per trace state lives in a lock guarded dict keyed by the
``trace_id`` LlamaIndex passes to ``start_trace`` / ``end_trace``; the
active trace for the current thread is remembered in a thread local so
event callbacks (which carry no trace id) route correctly. Several
concurrent ``aquery`` calls on one thread can interleave; use one
handler per concurrent async query in that case (AUDIT.md B22).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .auditor import Auditor, Trace
from .config import RAGComplianceConfig
from .models import RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)

try:
    import llama_index.core as _li_core
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler as _LIBase
    from llama_index.core.callbacks.schema import CBEventType, EventPayload

    _LLAMA_AVAILABLE = True
    _LLAMA_VERSION = getattr(_li_core, "__version__", "")
except ImportError:  # pragma: no cover - optional dep path
    _LIBase = object  # type: ignore[assignment,misc]
    CBEventType = None  # type: ignore[assignment,misc]
    EventPayload = None  # type: ignore[assignment,misc]
    _LLAMA_AVAILABLE = False
    _LLAMA_VERSION = ""


_DEFAULT_TRACE = "__ragcompliance_default_trace__"


@dataclass
class _TraceState(Trace):
    """Per trace capture state (a ``Trace`` with no extra fields)."""


def _payload_get(payload: Mapping[str, Any], enum_key: Any, *fallbacks: str) -> Any:
    if enum_key is not None and enum_key in payload:
        return payload[enum_key]
    for key in fallbacks:
        if key in payload:
            return payload[key]
    return None


class LlamaIndexRAGComplianceHandler(_LIBase):  # type: ignore[misc]
    """LlamaIndex callback that mirrors the LangChain audit capture."""

    def __init__(
        self,
        config: RAGComplianceConfig | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        billing: Any | None = None,
        alerter: Any | None = None,
        redactor: Any | None = None,
        storage: AuditStorage | None = None,
        actor: dict[str, Any] | None = None,
        auditor: Auditor | None = None,
    ):
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "llama-index-core is not installed. "
                "Install with: pip install 'ragcompliance[llamaindex]'"
            )
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.auditor = auditor or Auditor(
            config,
            session_id=session_id,
            extra=extra,
            actor=actor,
            billing=billing,
            alerter=alerter,
            redactor=redactor,
            storage=storage,
            framework="llamaindex",
            framework_version=_LLAMA_VERSION,
        )
        self._traces: dict[Any, _TraceState] = {}
        self._active = threading.local()
        self._lock = threading.Lock()
        raw = os.getenv("RAGCOMPLIANCE_MAX_PENDING_RUNS")
        try:
            self._max_pending_traces = max(1, int(raw)) if raw else int(
                self.auditor.config.max_pending_runs
            )
        except ValueError:
            self._max_pending_traces = 10000
        self._pending_cap_warned = False

    # ------------------------------------------------------------------ #
    # Compatibility properties                                           #
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> RAGComplianceConfig:
        return self.auditor.config

    @property
    def storage(self) -> AuditStorage:
        return self.auditor.storage

    @storage.setter
    def storage(self, value: AuditStorage) -> None:
        self.auditor.storage = value

    @property
    def session_id(self) -> str:
        return self.auditor.session_id

    @property
    def extra(self) -> dict[str, Any]:
        return self.auditor.extra

    @property
    def redactor(self) -> Any:
        return self.auditor.redactor

    @property
    def billing(self) -> Any:
        return self.auditor.billing

    @billing.setter
    def billing(self, value: Any) -> None:
        self.auditor.billing = value

    @property
    def alerter(self) -> Any:
        return self.auditor.alerter

    @alerter.setter
    def alerter(self, value: Any) -> None:
        self.auditor.alerter = value

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _trace_key(self, trace_id: Any) -> Any:
        return trace_id if trace_id is not None else _DEFAULT_TRACE

    def _enforce_pending_cap_locked(self) -> None:
        if len(self._traces) <= self._max_pending_traces:
            return
        if not self._pending_cap_warned:
            logger.warning(
                "RAGCompliance: pending trace state exceeded RAGCOMPLIANCE_MAX_PENDING_RUNS=%d; "
                "evicting oldest traces (end_trace was not delivered for some queries).",
                self._max_pending_traces,
            )
            self._pending_cap_warned = True
        while len(self._traces) > self._max_pending_traces:
            victim_id, _ = next(iter(self._traces.items()))
            self._traces.pop(victim_id, None)

    def _current_trace_id(self) -> Any:
        return getattr(self._active, "trace_id", None) or _DEFAULT_TRACE

    def _current_state_locked(self) -> _TraceState | None:
        state = self._traces.get(self._current_trace_id())
        if state is None and len(self._traces) == 1:
            # Events for a trace started on another thread (e.g. a worker
            # pool) route to the only live trace rather than being dropped.
            state = next(iter(self._traces.values()))
        return state

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def start_trace(self, trace_id: str | None = None) -> None:
        key = self._trace_key(trace_id)
        self._active.trace_id = key
        base = self.auditor.new_trace()
        state = _TraceState(actor=base.actor, extra=base.extra, chunks=base.chunks)
        state.start_time = time.time()
        with self._lock:
            self._traces[key] = state
            self._enforce_pending_cap_locked()
        self.auditor.check_quota()

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        key = self._trace_key(trace_id)
        with self._lock:
            state = self._traces.pop(key, None)
        if getattr(self._active, "trace_id", None) == key:
            self._active.trace_id = None
        if state is None:
            return
        try:
            error = state.extra.pop("_exception", None)
            if error is not None:
                self.auditor.fail(state, error)
            else:
                self.auditor.finalize(state)
        except Exception:
            logger.exception("RAGCompliance: finalize failed; audit record lost")

    # ------------------------------------------------------------------ #
    # Events                                                             #
    # ------------------------------------------------------------------ #

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        payload = payload or {}
        if CBEventType is None:  # pragma: no cover
            return event_id or str(uuid.uuid4())
        try:
            if event_type == CBEventType.QUERY:
                q = _payload_get(payload, EventPayload.QUERY_STR, "query_str", "query")
                if q:
                    with self._lock:
                        state = self._current_state_locked()
                        if state is not None:
                            state.query = self.auditor.coerce_query(str(q), state)
            elif event_type == CBEventType.LLM:
                self._note_llm_start(payload)
            elif event_type == CBEventType.RETRIEVE:
                with self._lock:
                    state = self._current_state_locked()
                    if state is not None:
                        state.note_retriever("llamaindex.retrieve")
        except Exception:
            logger.exception("RAGCompliance: on_event_start failed; audit capture degraded")
        return event_id or str(uuid.uuid4())

    def on_event_end(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        payload = payload or {}
        if CBEventType is None:  # pragma: no cover
            return
        try:
            if event_type == CBEventType.RETRIEVE:
                self._on_retrieve_end(payload)
            elif event_type == CBEventType.LLM:
                self._on_llm_end(payload)
            elif event_type == CBEventType.SYNTHESIZE:
                self._on_synthesize_end(payload)
            elif event_type == CBEventType.EXCEPTION:
                exc = _payload_get(payload, EventPayload.EXCEPTION, "exception")
                if exc is not None:
                    with self._lock:
                        state = self._current_state_locked()
                        if state is not None:
                            state.extra["_exception"] = exc
        except Exception:
            logger.exception("RAGCompliance: on_event_end failed; audit capture degraded")

    # ------------------------------------------------------------------ #
    # Event handlers                                                     #
    # ------------------------------------------------------------------ #

    def _note_llm_start(self, payload: Mapping[str, Any]) -> None:
        prompt = _payload_get(payload, EventPayload.PROMPT, "formatted_prompt")
        messages = _payload_get(payload, EventPayload.MESSAGES, "messages")
        serialized = _payload_get(payload, EventPayload.SERIALIZED, "serialized")
        text = ""
        if isinstance(prompt, str):
            text = prompt
        elif messages:
            parts = []
            for m in messages:
                role = getattr(m, "role", None)
                role = getattr(role, "value", role) or "message"
                parts.append(f"{role}: {getattr(m, 'content', m)}")
            text = "\n".join(parts)
        with self._lock:
            state = self._current_state_locked()
            if state is None:
                return
            if text:
                state.set_prompt(text)
            if isinstance(serialized, Mapping):
                state.set_model_params(serialized)

    def _on_retrieve_end(self, payload: Mapping[str, Any]) -> None:
        nodes = _payload_get(payload, EventPayload.NODES, "nodes") or []
        new_chunks: list[RetrievedChunk] = []
        for i, node_with_score in enumerate(nodes):
            node = getattr(node_with_score, "node", node_with_score)
            score = getattr(node_with_score, "score", None)
            meta = getattr(node, "metadata", {}) or {}
            content = getattr(node, "text", None)
            if not content:
                getter = getattr(node, "get_content", None)
                content = getter() if callable(getter) else str(node)
            new_chunks.append(
                RetrievedChunk(
                    content=str(content),
                    source_url=str(
                        meta.get("source") or meta.get("file_path") or meta.get("url") or "unknown"
                    ),
                    chunk_id=str(
                        meta.get("chunk_id") or getattr(node, "node_id", None) or f"chunk-{i}"
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

    def _on_llm_end(self, payload: Mapping[str, Any]) -> None:
        completion = _payload_get(payload, EventPayload.COMPLETION, "completion")
        if completion is None:
            completion = _payload_get(payload, EventPayload.RESPONSE, "response")
        if completion is None:
            completion = _payload_get(payload, EventPayload.MESSAGES, "messages")
        answer = ""
        if completion is not None:
            text = (
                getattr(completion, "text", None)
                or getattr(completion, "message", None)
                or completion
            )
            answer = text.content if hasattr(text, "content") else str(text)
            if isinstance(text, list):
                answer = ""
        model_name = ""
        serialized = _payload_get(payload, EventPayload.SERIALIZED, "serialized")
        if isinstance(serialized, Mapping):
            model = serialized.get("model") or serialized.get("model_name")
            if model:
                model_name = str(model)
        raw = getattr(completion, "raw", None)
        if not model_name and raw is not None:
            model_name = str(getattr(raw, "model", "") or (raw.get("model", "") if isinstance(raw, Mapping) else ""))
        with self._lock:
            state = self._current_state_locked()
            if state is None:
                return
            if answer:
                state.llm_answer = answer
            if model_name:
                state.model_name = model_name
            if isinstance(serialized, Mapping):
                state.set_model_params(serialized)

    def _on_synthesize_end(self, payload: Mapping[str, Any]) -> None:
        response = _payload_get(payload, EventPayload.RESPONSE, "response")
        if response is None:
            return
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
    # Compatibility                                                      #
    # ------------------------------------------------------------------ #

    def _sign_chain(self, state: Trace) -> str:
        from .chain import content_signature

        return content_signature(state.query, state.chunks, state.llm_answer)
