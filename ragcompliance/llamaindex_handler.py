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
import time
import uuid
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


class LlamaIndexRAGComplianceHandler(_LIBase):  # type: ignore[misc]
    """
    LlamaIndex callback that mirrors the LangChain audit capture:
    query -> retrieved chunks -> LLM answer -> SHA-256 chain signature.
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

        self._query: str = ""
        self._chunks: list[RetrievedChunk] = []
        self._llm_answer: str = ""
        self._model_name: str = ""
        self._start_time: float = time.time()

    # LlamaIndex calls these four methods on the handler.
    def start_trace(self, trace_id: str | None = None) -> None:
        self._reset()
        self._start_time = time.time()
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
        latency_ms = int((time.time() - self._start_time) * 1000)
        signature = self._sign_chain()
        record = AuditRecord(
            session_id=self.session_id,
            workspace_id=self.config.workspace_id,
            query=self._query,
            retrieved_chunks=self._chunks,
            llm_answer=self._llm_answer,
            model_name=self._model_name,
            chain_signature=signature,
            latency_ms=latency_ms,
            extra=self.extra,
        )
        self.storage.save(record)
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
        self._reset()

    def on_event_start(
        self,
        event_type: Any,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        payload = payload or {}
        # LlamaIndex emits a QUERY event at the start of a query engine call.
        # Payload keys are EventPayload StrEnum values; string keys still match
        # because StrEnum compares equal to its value, but prefer the enum when
        # available for clarity and forward-compat.
        if CBEventType and event_type == CBEventType.QUERY:
            q = None
            if EventPayload is not None:
                q = payload.get(EventPayload.QUERY_STR)
            if not q:
                q = payload.get("query_str") or payload.get("query")
            if q:
                self._query = str(q)
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
            for i, node_with_score in enumerate(nodes):
                # LlamaIndex yields NodeWithScore; attribute access is safest.
                node = getattr(node_with_score, "node", node_with_score)
                score = getattr(node_with_score, "score", None)
                meta = getattr(node, "metadata", {}) or {}
                content = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
                self._chunks.append(
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
            if completion is not None:
                text = (
                    getattr(completion, "text", None)
                    or getattr(completion, "message", None)
                    or str(completion)
                )
                answer = text.content if hasattr(text, "content") else str(text)
                # Only overwrite if non-empty — protects multi-LLM-call flows
                # where a refine pass might emit an empty continuation.
                if answer:
                    self._llm_answer = answer
            # Try to pull model name from the serialized payload.
            serialized = payload.get(EventPayload.SERIALIZED) if EventPayload is not None else None
            if not isinstance(serialized, dict):
                serialized = payload.get("serialized")
            if isinstance(serialized, dict):
                model = serialized.get("model") or serialized.get("model_name")
                if model:
                    self._model_name = str(model)

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
                    self._llm_answer = str(answer)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _sign_chain(self) -> str:
        payload = {
            "query": self._query,
            "chunks": [
                {"content": c.content, "source_url": c.source_url, "chunk_id": c.chunk_id}
                for c in self._chunks
            ],
            "answer": self._llm_answer,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _reset(self) -> None:
        self._query = ""
        self._chunks = []
        self._llm_answer = ""
        self._model_name = ""
