"""
RAGCompliance — LangChain callback handler for RAG audit trails.

Usage:
    from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

    config = RAGComplianceConfig.from_env()
    handler = RAGComplianceHandler(config=config, session_id="user-123")

    # Pass to any LangChain chain — including chain.batch() and concurrent
    # chain.invoke() / chain.ainvoke() calls on the same handler instance.
    chain.invoke({"query": "..."}, config={"callbacks": [handler]})
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

from .alerts import SlackAlerter
from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

logger = logging.getLogger(__name__)


@dataclass
class _RunState:
    """Per-invocation state for one root chain run.

    A single ``RAGComplianceHandler`` instance holds one of these per
    concurrently-running root chain, keyed by root ``run_id``. This is what
    makes ``chain.batch([q1, q2, q3])`` and concurrent ``chain.invoke(...)``
    calls on a shared handler correct: events are routed to the right
    per-run state instead of clobbering a single instance-level buffer.
    """

    query: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    llm_answer: str = ""
    model_name: str = ""
    start_time: float = 0.0
    # Transitive set of descendant run_ids (every inner LCEL sub-runnable
    # whose on_chain_start fired with this run as an ancestor). Used to
    # clean up ``_run_parents`` when the root finishes.
    descendants: set = field(default_factory=set)


class RAGComplianceHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures the full RAG chain —
    query, retrieved chunks with scores and sources, and LLM answer —
    signs everything with SHA-256, and persists to Supabase.

    Thread safety
    -------------
    This handler is safe to share across concurrent chain invocations and
    across ``chain.batch(...)`` calls. Per-invocation state is kept in a
    lock-guarded ``dict`` keyed by the root ``run_id`` LangChain assigns to
    each top-level call, so events from different invocations cannot
    interleave. The built-in ``AuditStorage``, ``BillingManager``, and
    ``SlackAlerter`` objects are also safe to share.

    The one pragmatic guard is ``RAGCOMPLIANCE_MAX_PENDING_RUNS`` (default
    10000): if on_chain_end is never delivered for some runs (crashed host
    process, misconfigured callbacks), the oldest pending run state is
    evicted with a warning so the handler can't leak unbounded memory.
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

        # Per-run state. _runs holds one _RunState per root (top-level) run,
        # keyed by root run_id. _run_parents maps inner runnable run_id →
        # its parent run_id so we can walk up to the root in O(depth).
        # _lock guards both dicts.
        self._runs: dict[Any, _RunState] = {}
        self._run_parents: dict[Any, Any] = {}
        self._lock = threading.Lock()

        try:
            self._max_pending_runs = max(
                1, int(os.getenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "10000"))
            )
        except ValueError:
            self._max_pending_runs = 10000
        self._pending_cap_warned = False

    # ------------------------------------------------------------------ #
    # Root resolution helpers                                              #
    # ------------------------------------------------------------------ #

    def _resolve_root(self, run_id: Any) -> Any | None:
        """Walk ``_run_parents`` upward from ``run_id`` to find a tracked
        root. Returns the root run_id or None if the run isn't trackable
        (e.g. callback fired for an unknown inner runnable). Caller must
        hold ``self._lock``."""
        if run_id is None:
            return None
        if run_id in self._runs:
            return run_id
        current = run_id
        seen = set()
        while current is not None:
            if current in self._runs:
                return current
            if current in seen:  # cycle guard — should never happen
                return None
            seen.add(current)
            parent = self._run_parents.get(current)
            if parent is None:
                return None
            current = parent
        return None

    def _register_descendant(
        self, run_id: Any, parent_run_id: Any
    ) -> None:
        """Track a child ``run_id`` under its parent so later ``*_end``
        events can resolve back to the correct root audit state.

        Called from every ``on_*_start`` callback that fires for an inner
        runnable (chain, retriever, LLM, chat model, tool, ...). Without
        this, a child's ``run_id`` never lands in ``_run_parents`` and
        ``_resolve_root`` walks off the end, silently dropping whatever
        that child produced from the audit record.

        No-op when either id is missing or when the parent isn't
        traceable to a tracked root (e.g. callback fired for a runnable
        whose root chain never started, or whose root has already
        finished). Caller must NOT hold ``self._lock``.
        """
        if run_id is None or parent_run_id is None:
            return
        with self._lock:
            self._run_parents[run_id] = parent_run_id
            root_id = self._resolve_root(parent_run_id)
            if root_id is not None:
                root_state = self._runs.get(root_id)
                if root_state is not None:
                    root_state.descendants.add(run_id)

    def _enforce_pending_cap_locked(self) -> None:
        """Evict oldest root runs if we're over the soft cap. Caller holds
        the lock."""
        if len(self._runs) <= self._max_pending_runs:
            return
        if not self._pending_cap_warned:
            logger.warning(
                "RAGCompliance: pending run state exceeded "
                "RAGCOMPLIANCE_MAX_PENDING_RUNS=%d; evicting oldest runs. "
                "This usually means on_chain_end is not being delivered for "
                "some chains (crashed workers, misconfigured callbacks, or "
                "handlers shared across processes). Raise the env var if "
                "this is expected.",
                self._max_pending_runs,
            )
            self._pending_cap_warned = True
        while len(self._runs) > self._max_pending_runs:
            victim_id, victim_state = next(iter(self._runs.items()))
            self._runs.pop(victim_id, None)
            for child_id in victim_state.descendants:
                self._run_parents.pop(child_id, None)

    # ------------------------------------------------------------------ #
    # Query capture                                                        #
    # ------------------------------------------------------------------ #

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: Any, **kwargs: Any
    ) -> None:
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")

        if parent_run_id is not None:
            # Inner LCEL sub-runnable — record the parent mapping so later
            # retriever/llm events can resolve to the correct root, and add
            # this run_id to the root's descendants set for O(1) cleanup.
            self._register_descendant(run_id, parent_run_id)
            return

        # Root chain starting. Create fresh per-run state.
        # run_id is None only when on_chain_start is invoked directly (e.g.
        # from tests). Synthesize one so the rest of the callback surface
        # still routes correctly.
        if run_id is None:
            run_id = uuid.uuid4()
            # Stash on kwargs so the caller can read it back if desired.
            kwargs["run_id"] = run_id

        state = _RunState(
            query=self._coerce_query(inputs),
            start_time=time.time(),
        )
        with self._lock:
            self._runs[run_id] = state
            self._enforce_pending_cap_locked()

        # Billing quota check runs outside the lock — we don't want to hold
        # it while calling into a potentially slow billing backend.
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
    # Parent registration for non-chain runnables                          #
    # ------------------------------------------------------------------ #
    # LangChain fires on_retriever_start, on_llm_start, on_chat_model_start,
    # and on_tool_start with their own fresh run_id and the parent chain's
    # run_id as parent_run_id. Without these overrides the child run_id is
    # never recorded in _run_parents, so the matching on_*_end event can't
    # walk up to the root audit state and the event's payload (chunks, llm
    # answer, tool output) is silently dropped.
    #
    # This became a visible correctness bug on langchain-core >=1.3.0 where
    # retrievers stopped double-firing on_chain_start — the handler was
    # accidentally relying on that old side channel in 0.1.4.

    def on_retriever_start(
        self, serialized: dict[str, Any], query: str, **kwargs: Any
    ) -> None:
        self._register_descendant(
            kwargs.get("run_id"), kwargs.get("parent_run_id")
        )

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        self._register_descendant(
            kwargs.get("run_id"), kwargs.get("parent_run_id")
        )

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        """LangChain's chat-model path; same routing need as on_llm_start."""
        self._register_descendant(
            kwargs.get("run_id"), kwargs.get("parent_run_id")
        )

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self._register_descendant(
            kwargs.get("run_id"), kwargs.get("parent_run_id")
        )

    # ------------------------------------------------------------------ #
    # Retriever capture                                                    #
    # ------------------------------------------------------------------ #

    def on_retriever_end(
        self, documents: list[Document], **kwargs: Any
    ) -> None:
        run_id = kwargs.get("run_id")
        with self._lock:
            root_id = self._resolve_root(run_id)
            if root_id is None:
                return
            state = self._runs.get(root_id)
            if state is None:
                return
            for i, doc in enumerate(documents):
                meta = doc.metadata or {}
                state.chunks.append(
                    RetrievedChunk(
                        content=doc.page_content,
                        source_url=str(meta.get("source", meta.get("url", "unknown"))),
                        chunk_id=str(meta.get("chunk_id", meta.get("id", f"chunk-{i}"))),
                        similarity_score=meta.get("score", meta.get("similarity_score")),
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if k not in (
                                "source", "url", "chunk_id", "id", "score",
                                "similarity_score",
                            )
                        },
                    )
                )

    # ------------------------------------------------------------------ #
    # LLM answer capture                                                   #
    # ------------------------------------------------------------------ #

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        with self._lock:
            root_id = self._resolve_root(run_id)
            if root_id is None:
                return
            state = self._runs.get(root_id)
            if state is None:
                return
            try:
                state.llm_answer = response.generations[0][0].text
            except (IndexError, AttributeError):
                state.llm_answer = str(response)
            if response.llm_output:
                state.model_name = (
                    response.llm_output.get("model_name", "") or state.model_name
                )

    # ------------------------------------------------------------------ #
    # Chain end — build, sign, and persist the audit record               #
    # ------------------------------------------------------------------ #

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")

        # Pop the root's state atomically. If run_id isn't a tracked root,
        # this is an inner runnable — do nothing and let the root finalize.
        with self._lock:
            if run_id not in self._runs:
                return
            state = self._runs.pop(run_id)
            for child_id in state.descendants:
                self._run_parents.pop(child_id, None)

        # Finalize outside the lock to avoid holding it during storage I/O.
        if not state.llm_answer:
            state.llm_answer = self._coerce_answer(outputs)

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

        # Defensive save: the built-in Supabase storage catches its own
        # errors, but a user-supplied custom storage backend might raise.
        # A compliance library must never take down the host chain because
        # its own audit write failed.
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

        # Best-effort usage metering; never let a metering failure break
        # the chain.
        if self.billing is not None:
            try:
                self.billing.increment_usage(self.config.workspace_id)
            except Exception as e:
                logger.debug(f"RAGCompliance: increment_usage errored ({e})")

        # Best-effort anomaly alerting. Runs after save() so any alert can
        # link back to a persisted row, and after increment_usage so
        # metering isn't gated on alert evaluation succeeding.
        if self.alerter is not None:
            try:
                self.alerter.maybe_alert(record)
            except Exception as e:
                logger.debug(f"RAGCompliance: alerter.maybe_alert errored ({e})")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _sign_chain(self, state: _RunState) -> str:
        """SHA-256 of the full chain: query + chunks + answer."""
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
        run_id = kwargs.get("run_id")

        with self._lock:
            if run_id not in self._runs:
                # Inner runnable error. LangChain will propagate the error
                # up to the root, which will also fire on_chain_error — we
                # finalize there.
                return
            state = self._runs.pop(run_id)
            for child_id in state.descendants:
                self._run_parents.pop(child_id, None)

        logger.error(f"RAGCompliance: Chain error — {error}")
        # Fire a chain_errored alert with whatever state we captured before
        # the failure. We don't persist a partial record; the alerter gets
        # a lightweight synthetic record with enough metadata to link back.
        if self.alerter is not None:
            try:
                partial = AuditRecord(
                    session_id=self.session_id,
                    workspace_id=self.config.workspace_id,
                    query=state.query,
                    retrieved_chunks=state.chunks,
                    llm_answer=state.llm_answer,
                    model_name=state.model_name,
                    chain_signature="",
                    latency_ms=int((time.time() - state.start_time) * 1000),
                    extra=self.extra,
                )
                self.alerter.maybe_alert(partial, error=error)
            except Exception as e:
                logger.debug(f"RAGCompliance: error-path alert failed ({e})")
