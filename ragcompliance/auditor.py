"""Framework neutral audit core.

``Auditor`` owns everything that is not specific to LangChain, LlamaIndex
or an LLM SDK: configuration, storage, redaction, access attribution,
lineage, signing, billing hooks and alerting. Every integration is a thin
adapter that opens a ``Trace``, feeds it the query, retrieved chunks,
prompt and answer, and calls ``finalize``.

Minimal direct use::

    from ragcompliance import Auditor, RetrievedChunk

    auditor = Auditor(session_id="user-42", actor={"id": "alice@example.com"})
    with auditor.trace("What does clause 4.2 say?") as t:
        t.add_chunks([RetrievedChunk("Clause 4.2 ...", "contracts/msa.pdf", "p3-c1", 0.91)])
        answer = my_llm(...)
        t.set_answer(answer, model_name="my-model")
    # the record is sealed and stored when the block exits
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .chain import content_signature
from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


class QuotaExceededError(RuntimeError):
    """Raised by ``Auditor.check_quota`` when ``enforce_quota`` is on."""


MODEL_PARAM_KEYS = (
    "model",
    "model_name",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "top_p",
    "top_k",
    "stop",
    "seed",
    "n",
    "frequency_penalty",
    "presence_penalty",
    "response_format",
    "_type",
)
"""Whitelist of invocation parameters recorded in lineage. Everything
else (API keys, headers, raw prompts) is deliberately excluded."""


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    out: dict[str, Any] = {}
    for key in MODEL_PARAM_KEYS:
        if key in params and params[key] is not None:
            value = params[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            else:
                out[key] = json.dumps(value, default=str)[:200]
    return out


@dataclass
class Trace:
    """Mutable capture state for one RAG invocation."""

    query: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    llm_answer: str = ""
    model_name: str = ""
    start_time: float = field(default_factory=time.time)
    started_at: datetime = field(default_factory=_utcnow)
    actor: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    retrievers: list[str] = field(default_factory=list)
    truncation: dict[str, Any] = field(default_factory=dict)

    # -- builders ---------------------------------------------------- #

    def add_chunk(self, chunk: RetrievedChunk) -> None:
        self.chunks.append(chunk)

    def add_chunks(self, chunks: Iterable[Any]) -> None:
        for c in chunks:
            self.chunks.append(coerce_chunk(c, index=len(self.chunks)))

    def set_answer(
        self,
        text: str,
        *,
        model_name: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.llm_answer = text or ""
        if model_name:
            self.model_name = model_name
        if params:
            self.set_model_params(params)

    def set_prompt(self, text: str) -> None:
        if text:
            self.prompt_text = text

    def set_model_params(self, params: Mapping[str, Any] | None) -> None:
        cleaned = _clean_params(params)
        if cleaned:
            self.model_params.update(cleaned)
            if not self.model_name:
                self.model_name = str(cleaned.get("model_name") or cleaned.get("model") or "")

    def note_retriever(self, name: str) -> None:
        if name and name not in self.retrievers:
            self.retrievers.append(name)


def coerce_chunk(obj: Any, *, index: int = 0) -> RetrievedChunk:
    """Accept ``RetrievedChunk``, dict, or plain string context."""
    if isinstance(obj, RetrievedChunk):
        return obj
    if isinstance(obj, Mapping):
        meta = dict(obj.get("metadata") or {})
        return RetrievedChunk(
            content=str(obj.get("content") or obj.get("text") or obj.get("page_content") or ""),
            source_url=str(obj.get("source_url") or obj.get("source") or obj.get("url") or "unknown"),
            chunk_id=str(obj.get("chunk_id") or obj.get("id") or f"chunk-{index}"),
            similarity_score=(
                float(obj["similarity_score"])
                if obj.get("similarity_score") is not None
                else (float(obj["score"]) if obj.get("score") is not None else None)
            ),
            metadata=meta,
        )
    page_content = getattr(obj, "page_content", None)
    if isinstance(page_content, str):  # LangChain Document without importing langchain
        meta = dict(getattr(obj, "metadata", None) or {})
        return RetrievedChunk(
            content=page_content,
            source_url=str(meta.get("source", meta.get("url", "unknown"))),
            chunk_id=str(meta.get("chunk_id", meta.get("id", f"chunk-{index}"))),
            similarity_score=meta.get("score", meta.get("similarity_score")),
            metadata={
                k: v
                for k, v in meta.items()
                if k not in ("source", "url", "chunk_id", "id", "score", "similarity_score")
            },
        )
    return RetrievedChunk(content=str(obj), source_url="unknown", chunk_id=f"chunk-{index}")


class Auditor:
    """Framework neutral audit core shared by every integration."""

    def __init__(
        self,
        config: RAGComplianceConfig | None = None,
        *,
        session_id: str | None = None,
        extra: dict[str, Any] | None = None,
        actor: dict[str, Any] | None = None,
        billing: Any | None = None,
        alerter: Any | None = None,
        redactor: Any | None = None,
        storage: AuditStorage | None = None,
        framework: str = "",
        framework_version: str = "",
    ):
        import uuid

        self.config = config or RAGComplianceConfig.from_env()
        self.storage = storage if storage is not None else AuditStorage(self.config)
        self.session_id = session_id or str(uuid.uuid4())
        self.extra: dict[str, Any] = dict(extra or {})
        self.actor: dict[str, Any] = dict(actor or {})
        self.framework = framework
        self.framework_version = framework_version
        self._local = threading.local()

        if redactor is not None:
            self.redactor = redactor
        elif self.config.redact_pii:
            from .redaction import Redactor

            self.redactor = Redactor(
                patterns=self.config.redaction_patterns,
                replacement=self.config.redaction_replacement,
            )
        else:
            self.redactor = None

        self.billing = billing if billing is not None else self._default_billing()
        self.alerter = alerter if alerter is not None else self._default_alerter()

    # ------------------------------------------------------------------ #
    # Defaults                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_billing() -> Any | None:
        # Billing is a reference implementation and must stay out of the
        # hot path unless the operator configured Stripe (AUDIT.md B14).
        if not os.getenv("STRIPE_SECRET_KEY"):
            return None
        try:
            from .billing import BillingManager

            return BillingManager.from_env()
        except Exception as exc:
            logger.debug("RAGCompliance: billing disabled (%s)", exc)
            return None

    @staticmethod
    def _default_alerter() -> Any | None:
        try:
            from .alerts import SlackAlerter

            return SlackAlerter.from_env()
        except Exception as exc:
            logger.debug("RAGCompliance: alerter disabled (%s)", exc)
            return None

    # ------------------------------------------------------------------ #
    # Thread-local binding (used by SDK wrappers and LlamaIndex)         #
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def bind(
        self,
        *,
        actor: dict[str, Any] | None = None,
        chunks: Iterable[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Attach an actor, retrieved chunks or extra fields to every trace
        opened on this thread inside the block."""
        prev = getattr(self._local, "bound", None)
        bound = dict(prev or {})
        if actor:
            bound["actor"] = {**bound.get("actor", {}), **actor}
        if chunks is not None:
            bound["chunks"] = list(chunks)
        if extra:
            bound["extra"] = {**bound.get("extra", {}), **extra}
        self._local.bound = bound
        try:
            yield
        finally:
            self._local.bound = prev

    def _bound(self) -> dict[str, Any]:
        return dict(getattr(self._local, "bound", None) or {})

    # ------------------------------------------------------------------ #
    # Trace lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def new_trace(
        self,
        query: Any = "",
        *,
        actor: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        chunks: Iterable[Any] | None = None,
    ) -> Trace:
        bound = self._bound()
        trace = Trace()
        if query:
            trace.query = self.coerce_query(query, trace)
        trace.actor = {**bound.get("actor", {}), **dict(actor or {})}
        trace.extra = {**bound.get("extra", {}), **dict(extra or {})}
        if bound.get("chunks"):
            trace.add_chunks(bound["chunks"])
        if chunks:
            trace.add_chunks(chunks)
        return trace

    def check_quota(self) -> None:
        """Raise ``RuntimeError`` when the workspace is over quota and
        ``enforce_quota`` is on; otherwise log a warning."""
        if self.billing is None:
            return
        try:
            within = self.billing.check_query_quota(self.config.workspace_id)
        except Exception as exc:
            logger.debug("RAGCompliance: quota check errored, allowing (%s)", exc)
            within = True
        if not within:
            msg = (
                f"RAGCompliance: workspace {self.config.workspace_id!r} has exceeded its "
                "plan query quota for the current period."
            )
            if self.config.enforce_quota:
                raise QuotaExceededError(msg)
            logger.warning(msg)

    @contextlib.contextmanager
    def trace(
        self,
        query: Any = "",
        *,
        actor: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        chunks: Iterable[Any] | None = None,
    ) -> Iterator[Trace]:
        """Open a trace, run the block, finalize (or record the failure)."""
        self.check_quota()
        t = self.new_trace(query, actor=actor, extra=extra, chunks=chunks)
        try:
            yield t
        except BaseException as exc:
            self.fail(t, exc)
            raise
        self.finalize(t)

    # ------------------------------------------------------------------ #
    # Finalisation                                                       #
    # ------------------------------------------------------------------ #

    def finalize(self, trace: Trace, *, outputs: Any = None) -> AuditRecord | None:
        """Build, redact, sign and store the record for a completed trace."""
        if not trace.llm_answer and outputs is not None:
            trace.llm_answer = self.coerce_answer(outputs, trace)
        record = self._build_record(trace, status="ok")
        self._persist(record)
        self._meter()
        self._alert(record)
        return record

    def fail(self, trace: Trace, error: BaseException) -> AuditRecord | None:
        """Handle a failed invocation: alert (redacted) and optionally persist."""
        record = self._build_record(trace, status="error")
        record.extra["error"] = {
            "type": error.__class__.__name__,
            "message": str(error)[:500],
        }
        if self.config.record_errors:
            self._persist(record)
        self._alert(record, error=error)
        return record if self.config.record_errors else None

    def _build_record(self, trace: Trace, *, status: str) -> AuditRecord:
        ended_at = _utcnow()
        latency_ms = int((time.time() - trace.start_time) * 1000)
        extra = dict(self.extra)
        extra.update(trace.extra)
        actor = dict(self.config.default_actor)
        actor.update(self.actor)
        actor.update(trace.actor)

        lineage: dict[str, Any] = {
            "status": status,
            "framework": self.framework or "direct",
            "framework_version": self.framework_version,
            "tool_version": __version__,
            "python_version": platform.python_version(),
            "started_at": trace.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
        }
        if trace.retrievers:
            lineage["retrievers"] = list(trace.retrievers)
        if trace.model_params:
            lineage["model_params"] = dict(trace.model_params)
        if trace.prompt_text:
            lineage["prompt_sha256"] = _sha256(trace.prompt_text)
            if self.config.capture_prompts:
                lineage["prompt_text"] = trace.prompt_text
        if trace.truncation:
            lineage["truncation"] = dict(trace.truncation)
        lineage.update(trace.lineage)

        record = AuditRecord(
            session_id=self.session_id,
            workspace_id=self.config.workspace_id,
            query=trace.query,
            retrieved_chunks=trace.chunks,
            llm_answer=trace.llm_answer,
            model_name=trace.model_name,
            timestamp=ended_at,
            latency_ms=latency_ms,
            extra=extra,
            actor=actor,
            lineage=lineage,
        )

        # Redaction runs before hashing so every hash is reproducible from
        # the stored record alone.
        if self.redactor is not None:
            findings = self._apply_redaction(record)
            if findings:
                record.extra["redaction_findings"] = findings

        record.lineage["query_sha256"] = _sha256(record.query)
        record.lineage["answer_sha256"] = _sha256(record.llm_answer)
        record.lineage["chunks"] = [
            {
                "chunk_id": c.chunk_id,
                "source_url": c.source_url,
                "content_sha256": c.content_sha256(),
                "similarity_score": c.similarity_score,
            }
            for c in record.retrieved_chunks
        ]
        record.chain_signature = content_signature(
            record.query, record.retrieved_chunks, record.llm_answer
        )
        return record

    def _apply_redaction(self, record: AuditRecord) -> dict[str, int]:
        redactor = self.redactor
        try:
            if hasattr(redactor, "redact_record"):
                return dict(redactor.redact_record(record) or {})
            # Duck-typed redactor exposing only redact(text) -> RedactionResult.
            findings: dict[str, int] = {}

            def _apply(text: str) -> str:
                res = redactor.redact(text)
                new_text = getattr(res, "text", res)
                for k, v in (getattr(res, "findings", None) or {}).items():
                    findings[k] = findings.get(k, 0) + v
                return str(new_text)

            record.query = _apply(record.query)
            for c in record.retrieved_chunks:
                c.content = _apply(c.content)
                c.source_url = _apply(c.source_url)
            record.llm_answer = _apply(record.llm_answer)
            return findings
        except Exception as exc:
            logger.error("RAGCompliance: redactor raised (%s); record stored unredacted", exc)
            return {"redactor_error": 1}

    def _persist(self, record: AuditRecord) -> None:
        try:
            self.storage.save(record)
        except Exception as exc:
            logger.error(
                "RAGCompliance: storage.save raised (%s); dropping audit record for "
                "session=%r. Custom storage backends must not raise.",
                exc,
                self.session_id,
            )

    def _meter(self) -> None:
        if self.billing is None:
            return
        try:
            self.billing.increment_usage(self.config.workspace_id)
        except Exception as exc:
            logger.debug("RAGCompliance: increment_usage errored (%s)", exc)

    def _alert(self, record: AuditRecord, error: BaseException | None = None) -> None:
        if self.alerter is None:
            return
        try:
            self.alerter.maybe_alert(record, error=error)
        except Exception as exc:
            logger.debug("RAGCompliance: alerter.maybe_alert errored (%s)", exc)

    # ------------------------------------------------------------------ #
    # Coercion                                                           #
    # ------------------------------------------------------------------ #

    def coerce_query(self, inputs: Any, trace: Trace | None = None) -> str:
        return self._truncate(_coerce_text(inputs, ("query", "question", "input", "human_input", "prompt")), "query", trace)

    def coerce_answer(self, outputs: Any, trace: Trace | None = None) -> str:
        return self._truncate(_coerce_text(outputs, ("answer", "result", "output", "response", "text")), "answer", trace)

    def _truncate(self, text: str, which: str, trace: Trace | None) -> str:
        limit = self.config.max_query_chars if which == "query" else self.config.max_answer_chars
        if limit and len(text) > limit:
            if trace is not None:
                trace.truncation[which] = {"original_chars": len(text), "stored_chars": limit}
            return text[:limit]
        return text

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def flush(self, timeout: float | None = None) -> bool:
        return self.storage.flush(timeout)

    def close(self) -> None:
        self.storage.close()


def _coerce_text(obj: Any, keys: tuple[str, ...]) -> str:
    """Normalise LCEL style dicts, strings, messages and misc objects to text."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj and obj[key] is not None:
                value = obj[key]
                return value if isinstance(value, str) else _coerce_text(value, keys)
        return json.dumps(obj, default=str)
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # chat message content blocks
        parts = [
            p.get("text", "") if isinstance(p, Mapping) else getattr(p, "text", "")
            for p in content
        ]
        return "\n".join(str(p) for p in parts if p)
    if isinstance(obj, (list, tuple)):
        return "\n".join(_coerce_text(item, keys) for item in obj)
    return str(obj)
