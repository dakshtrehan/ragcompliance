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

from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler as _LIBase
    from llama_index.core.callbacks.schema import CBEventType
    _LLAMA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep path
    _LIBase = object  # type: ignore
    CBEventType = None  # type: ignore
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
        if CBEventType and event_type == CBEventType.QUERY:
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
            nodes = payload.get("nodes") or []
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
            response = payload.get("response")
            if response is not None:
                text = (
                    getattr(response, "message", None)
                    or getattr(response, "text", None)
                    or str(response)
                )
                self._llm_answer = (
                    text.content if hasattr(text, "content") else str(text)
                )
            model = payload.get("serialized", {}).get("model") if isinstance(payload.get("serialized"), dict) else None
            if model:
                self._model_name = str(model)

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
