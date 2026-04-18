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

from .alerts import SlackAlerter
from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


class RAGComplianceHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the full RAG chain —
    query, retrieved chunks with scores and sources, and LLM answer —
    signs everything with SHA-256, and persists to Supabase.

    Thread safety
    -------------
    This handler accumulates per-run state on the instance
    (``_root_run_id``, ``_query``, ``_chunks``, ``_llm_answer``,
    ``_model_name``, ``_start_time``) and is therefore NOT safe to share
    across concurrent chain invocations. Create one handler per chain
    invocation, per thread, or per async task. Sharing a single instance
    across concurrent ``chain.invoke`` / ``chain.ainvoke`` calls will
    interleave events from different runs and produce corrupted audit
    records.

    The configured ``AuditStorage``, ``BillingManager``, and
    ``SlackAlerter`` objects are safe to share across handler instances.
    """

    # Propagate exceptions raised inside our callback methods instead of
    # swallowing them. Required so the quota-exceeded RuntimeError raised in
    # on_chain_start actually blocks the chain rather than being logged and
    # ignored by LangChain's default observability behavior.
    raise_error = True

    def __init__(
        self,
        config: RAGComplianceConfig | None = None,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        billing: Any | None = None,
        alerter: SlackAlerter | None = None,
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

        # Optional Slack alerter. Opt-in via RAGCOMPLIANCE_SLACK_WEBHOOK_URL.
        # When not configured this stays None so evaluate() overhead is skipped.
        if alerter is None:
            try:
                self.alerter = SlackAlerter.from_env()
            except Exception as e:
                logger.debug(f"RAGCompliance: alerter disabled ({e})")
                self.alerter = None
        else:
            self.alerter = alerter

        # State accumulated across callback events in one chain run.
        # LCEL pipelines fire on_chain_start/on_chain_end for EVERY sub-runnable,
        # so we latch onto the outermost chain via run_id/parent_run_id and ignore
        # intermediate events.
        self._root_run_id: Any = None
        self._query: str = ""
        self._chunks: list[RetrievedChunk] = []
        self._llm_answer: str = ""
        self._model_name: str = ""
        self._start_time: float = time.time()

    # ------------------------------------------------------------------ #
    # Query capture                                                        #
    # ------------------------------------------------------------------ #

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: Any, **kwargs: Any
    ) -> None:
        # LCEL fires this for every sub-runnable. Only latch onto the outermost
        # one (parent_run_id is None) and ignore the rest — otherwise we'd
        # overwrite query/answer state mid-chain and save a half-built record.
        parent_run_id = kwargs.get("parent_run_id")
        if self._root_run_id is not None or parent_run_id is not None:
            return

        self._root_run_id = kwargs.get("run_id")
        self._start_time = time.time()
        self._query = self._coerce_query(inputs)

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

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        # Only finalize when the OUTERMOST chain ends. Every inner runnable
        # also fires on_chain_end; ignoring those keeps us from saving an
        # empty record before the LLM has even run.
        run_id = kwargs.get("run_id")
        if self._root_run_id is None or run_id != self._root_run_id:
            return

        # If we didn't catch the answer from on_llm_end, try chain outputs.
        if not self._llm_answer:
            self._llm_answer = self._coerce_answer(outputs)

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

        # Best-effort anomaly alerting. Runs after save() so any alert can
        # link back to a persisted row, and after increment_usage so metering
        # isn't gated on alert evaluation succeeding.
        if self.alerter is not None:
            try:
                self.alerter.maybe_alert(record)
            except Exception as e:
                logger.debug(f"RAGCompliance: alerter.maybe_alert errored ({e})")

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
        self._root_run_id = None
        self._query = ""
        self._chunks = []
        self._llm_answer = ""
        self._model_name = ""
        self._start_time = time.time()

    def _coerce_query(self, inputs: Any) -> str:
        """LCEL hands us dicts, strings, lists of messages, or BaseMessages.
        Normalize to a single query string."""
        if isinstance(inputs, str):
            return inputs
        if isinstance(inputs, dict):
            for key in ("query", "question", "input", "human_input", "prompt"):
                if key in inputs and inputs[key] is not None:
                    return str(inputs[key])
            # Fallback: stringify the whole dict so we at least have something.
            return json.dumps(inputs, default=str)[:2000]
        # BaseMessage, list of messages, or anything else — stringify.
        content = getattr(inputs, "content", None)
        if isinstance(content, str):
            return content
        return str(inputs)[:2000]

    def _coerce_answer(self, outputs: Any) -> str:
        """Same idea as _coerce_query but for the chain's final output."""
        if isinstance(outputs, str):
            return outputs
        if isinstance(outputs, dict):
            for key in ("answer", "result", "output", "response", "text"):
                if key in outputs and outputs[key] is not None:
                    return str(outputs[key])
            return json.dumps(outputs, default=str)[:4000]
        content = getattr(outputs, "content", None)
        if isinstance(content, str):
            return content
        return str(outputs)[:4000]

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        # Only clear state if the OUTERMOST chain errored, so retries on
        # intermediate runnables don't wipe accumulated audit state.
        run_id = kwargs.get("run_id")
        if self._root_run_id is not None and run_id == self._root_run_id:
            logger.error(f"RAGCompliance: Chain error — {error}")
            # Fire a chain_errored alert with whatever state we captured
            # before the failure. We don't persist a partial record; the
            # alerter is passed a lightweight synthetic record so it has
            # enough metadata to link back.
            if self.alerter is not None:
                try:
                    partial = AuditRecord(
                        session_id=self.session_id,
                        workspace_id=self.config.workspace_id,
                        query=self._query,
                        retrieved_chunks=self._chunks,
                        llm_answer=self._llm_answer,
                        model_name=self._model_name,
                        chain_signature="",
                        latency_ms=int((time.time() - self._start_time) * 1000),
                        extra=self.extra,
                    )
                    self.alerter.maybe_alert(partial, error=error)
                except Exception as e:
                    logger.debug(f"RAGCompliance: error-path alert failed ({e})")
            self._reset()
