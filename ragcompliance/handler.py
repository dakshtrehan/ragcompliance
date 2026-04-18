"""
RAGCompliance — LangChain callback handler for RAG audit trails.

Usage:
    from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

    config = RAGComplianceConfig.from_env()
    handler = RAGComplianceHandler(config=config, session_id="user-123")

    # Pass to any LangChain chain
    chain.invoke({"query": "..."}, config={"callbacks": [handler]})
"""

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


class RAGComplianceHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the full RAG chain —
    query, retrieved chunks with scores and sources, and LLM answer —
    signs everything with SHA-256, and persists to Supabase.
    """

    def __init__(
        self,
        config: RAGComplianceConfig | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        billing: Any | None = None,
    ):
        super().__init__()
        self.config = config or RAGComplianceConfig.from_env()
        self.storage = AuditStorage(self.config)
        self.session_id = session_id or str(uuid.uuid4())
        self.extra = extra or {}

        # Optional billing manager for quota + usage metering. Lazy import so
        # `stripe` only becomes required when billing features are used.
        if billing is None:
            try:
                from .billing import BillingManager
                self.billing = BillingManager.from_env()
            except Exception as e:  # stripe missing, bad env, etc.
                logger.debug(f"RAGCompliance: billing disabled ({e})")
                self.billing = None
        else:
            self.billing = billing

        # State accumulated across callback events in one chain run
        self._query: str = ""
        self._chunks: list[RetrievedChunk] = []
        self._llm_answer: str = ""
        self._model_name: str = ""
        self._start_time: float = time.time()

    # ------------------------------------------------------------------ #
    # Query capture                                                        #
    # ------------------------------------------------------------------ #

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self._start_time = time.time()
        # Capture whatever looks like the user query
        for key in ("query", "question", "input", "human_input"):
            if key in inputs:
                self._query = str(inputs[key])
                break

        # Soft quota check — log when over, hard-block only when enforce_quota=True.
        if self.billing is not None:
            try:
                within = self.billing.check_query_quota(self.config.workspace_id)
            except Exception as e:
                logger.debug(f"RAGCompliance: quota check errored, allowing ({e})")
                within = True
            if not within:
                msg = (
                    f"RAGCompliance: workspace {self.config.workspace_id!r} "
                    "has exceeded its plan query quota for the current period."
                )
                if self.config.enforce_quota:
                    raise RuntimeError(msg)
                logger.warning(msg)

    # ------------------------------------------------------------------ #
    # Retriever capture                                                    #
    # ------------------------------------------------------------------ #

    def on_retriever_end(
        self, documents: list[Document], **kwargs: Any
    ) -> None:
        for i, doc in enumerate(documents):
            meta = doc.metadata or {}
            self._chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source_url=str(meta.get("source", meta.get("url", "unknown"))),
                    chunk_id=str(meta.get("chunk_id", meta.get("id", f"chunk-{i}"))),
                    similarity_score=meta.get("score", meta.get("similarity_score")),
                    metadata={k: v for k, v in meta.items() if k not in ("source", "url", "chunk_id", "id", "score", "similarity_score")},
                )
            )

    # ------------------------------------------------------------------ #
    # LLM answer capture                                                   #
    # ------------------------------------------------------------------ #

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            self._llm_answer = response.generations[0][0].text
        except (IndexError, AttributeError):
            self._llm_answer = str(response)

        if response.llm_output:
            self._model_name = response.llm_output.get("model_name", "")

    # ------------------------------------------------------------------ #
    # Chain end — build, sign, and persist the audit record               #
    # ------------------------------------------------------------------ #

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        # If we didn't catch the answer from on_llm_end, try chain outputs
        if not self._llm_answer:
            for key in ("answer", "result", "output", "response"):
                if key in outputs:
                    self._llm_answer = str(outputs[key])
                    break

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

        # Best-effort usage metering; never let a metering failure break the chain.
        if self.billing is not None:
            try:
                self.billing.increment_usage(self.config.workspace_id)
            except Exception as e:
                logger.debug(f"RAGCompliance: increment_usage errored ({e})")

        self._reset()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _sign_chain(self) -> str:
        """SHA-256 of the full chain: query + chunks + answer."""
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
        self._start_time = time.time()

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error(f"RAGCompliance: Chain error — {error}")
        self._reset()
