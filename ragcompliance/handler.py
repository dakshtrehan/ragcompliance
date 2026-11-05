"""LangChain callback handler.

Usage::

    from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

    handler = RAGComplianceHandler(config=RAGComplianceConfig.from_env(), session_id="user-123")
    chain.invoke({"query": "..."}, config={"callbacks": [handler]})

Per invocation attribution goes through LangChain metadata::

    chain.invoke(
        {"query": "..."},
        config={
            "callbacks": [handler],
            "metadata": {"ragcompliance_actor": {"id": "alice@example.com", "type": "user"}},
        },
    )

The handler is a thin adapter: it routes LangChain callback events to a
per run ``Trace`` and hands the finished trace to ``Auditor`` (see
``auditor.py``) which redacts, signs, stores, meters and alerts.

Thread safety: per invocation state is kept in a lock guarded dict keyed
by the root ``run_id`` LangChain assigns to each top level call, so a
single handler can be shared across ``chain.batch(...)`` and concurrent
``invoke`` / ``ainvoke`` calls.
"""

from __future__ import annotations

import functools
import logging
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .auditor import Auditor, QuotaExceededError, Trace, coerce_chunk
from .config import RAGComplianceConfig
from .storage import AuditStorage

logger = logging.getLogger(__name__)

try:
    import langchain_core
    from langchain_core.callbacks import BaseCallbackHandler as _LCBase

    _LANGCHAIN_AVAILABLE = True
    _LANGCHAIN_VERSION = getattr(langchain_core, "__version__", "")
except ImportError:  # pragma: no cover - exercised only without the extra
    _LCBase = object  # type: ignore[assignment,misc]
    _LANGCHAIN_AVAILABLE = False
    _LANGCHAIN_VERSION = ""


@dataclass
class _RunState(Trace):
    """Per root run capture state plus the transitive set of descendant
    run ids, used to clean ``_run_parents`` when the root finishes."""

    descendants: set = field(default_factory=set)


def _guarded(method: Callable[..., Any]) -> Callable[..., Any]:
    """Never let a bug in the audit handler take down the user's chain.

    ``raise_error = True`` is required so the deliberate quota error reaches
    the caller, but it also means any exception escaping a callback kills
    the chain. Everything except ``QuotaExceededError`` is logged instead.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except QuotaExceededError:
            raise
        except Exception:
            logger.exception("RAGCompliance: %s failed; audit capture degraded", method.__name__)
            return None

    return wrapper


class RAGComplianceHandler(_LCBase):  # type: ignore[misc]
    """LangChain callback handler that captures the full RAG chain and
    hands it to the ``Auditor``."""

    raise_error = True

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
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is not installed. Install with: pip install 'ragcompliance[langchain]'"
            )
        super().__init__()
        self.auditor = auditor or Auditor(
            config,
            session_id=session_id,
            extra=extra,
            actor=actor,
            billing=billing,
            alerter=alerter,
            redactor=redactor,
            storage=storage,
            framework="langchain",
            framework_version=_LANGCHAIN_VERSION,
        )
        self._runs: dict[Any, _RunState] = {}
        self._run_parents: dict[Any, Any] = {}
        import threading

        self._lock = threading.Lock()
        self._max_pending_runs = self._resolve_max_pending()
        self._pending_cap_warned = False

    def _resolve_max_pending(self) -> int:
        raw = os.getenv("RAGCOMPLIANCE_MAX_PENDING_RUNS")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                return 10000
        return max(1, int(self.auditor.config.max_pending_runs))

    # ------------------------------------------------------------------ #
    # Compatibility properties (0.1.x public attributes)                 #
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

    @session_id.setter
    def session_id(self, value: str) -> None:
        self.auditor.session_id = value

    @property
    def extra(self) -> dict[str, Any]:
        return self.auditor.extra

    @property
    def redactor(self) -> Any:
        return self.auditor.redactor

    @redactor.setter
    def redactor(self, value: Any) -> None:
        self.auditor.redactor = value

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

    @property
    def actor(self) -> dict[str, Any]:
        return self.auditor.actor

    # ------------------------------------------------------------------ #
    # Root resolution helpers                                            #
    # ------------------------------------------------------------------ #

    def _resolve_root(self, run_id: Any) -> Any | None:
        """Walk ``_run_parents`` upward to a tracked root. A missing
        ``run_id`` resolves to the only pending root when exactly one is in
        flight (direct calls without LangChain, see AUDIT.md B01). Caller
        holds ``self._lock``."""
        if run_id is None:
            return next(iter(self._runs)) if len(self._runs) == 1 else None
        if run_id in self._runs:
            return run_id
        current = run_id
        seen: set[Any] = set()
        while current is not None:
            if current in self._runs:
                return current
            if current in seen:
                return None
            seen.add(current)
            current = self._run_parents.get(current)
        return None

    def _register_descendant(self, run_id: Any, parent_run_id: Any) -> Any | None:
        """Record ``run_id`` under ``parent_run_id`` and return the root it
        resolves to (or None)."""
        if run_id is None or parent_run_id is None:
            return None
        with self._lock:
            self._run_parents[run_id] = parent_run_id
            root_id = self._resolve_root(parent_run_id)
            if root_id is not None:
                state = self._runs.get(root_id)
                if state is not None:
                    state.descendants.add(run_id)
            return root_id

    def _state_for(self, run_id: Any) -> _RunState | None:
        """Caller holds the lock."""
        root_id = self._resolve_root(run_id)
        return self._runs.get(root_id) if root_id is not None else None

    def _enforce_pending_cap_locked(self) -> None:
        if len(self._runs) <= self._max_pending_runs:
            return
        if not self._pending_cap_warned:
            logger.warning(
                "RAGCompliance: pending run state exceeded RAGCOMPLIANCE_MAX_PENDING_RUNS=%d; "
                "evicting oldest runs. This usually means on_chain_end is not being delivered "
                "for some chains. Raise the limit if this is expected.",
                self._max_pending_runs,
            )
            self._pending_cap_warned = True
        while len(self._runs) > self._max_pending_runs:
            victim_id, victim_state = next(iter(self._runs.items()))
            self._runs.pop(victim_id, None)
            for child_id in victim_state.descendants:
                self._run_parents.pop(child_id, None)

    def _pop_root(self, run_id: Any) -> _RunState | None:
        with self._lock:
            if run_id is None:
                run_id = self._resolve_root(None)
            if run_id not in self._runs:
                return None
            state = self._runs.pop(run_id)
            for child_id in state.descendants:
                self._run_parents.pop(child_id, None)
            return state

    # ------------------------------------------------------------------ #
    # Chain start / end                                                  #
    # ------------------------------------------------------------------ #

    @_guarded
    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")

        if parent_run_id is not None:
            self._register_descendant(run_id, parent_run_id)
            return

        if run_id is None:
            run_id = uuid.uuid4()
            kwargs["run_id"] = run_id

        metadata = kwargs.get("metadata") or {}
        actor, extra = _actor_and_extra_from_metadata(metadata)
        base = self.auditor.new_trace(actor=actor, extra=extra)
        state = _RunState(
            actor=base.actor, extra=base.extra, chunks=base.chunks,
        )
        state.query = self.auditor.coerce_query(inputs, state)
        tags = kwargs.get("tags")
        if tags:
            state.lineage["tags"] = [str(t) for t in tags]
        if isinstance(serialized, Mapping) and serialized.get("name"):
            state.lineage["chain"] = str(serialized["name"])
        elif kwargs.get("name"):
            state.lineage["chain"] = str(kwargs["name"])

        with self._lock:
            self._runs[run_id] = state
            self._enforce_pending_cap_locked()

        # Quota check outside the lock; raises QuotaExceededError when enforced.
        self.auditor.check_quota()

    @_guarded
    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        state = self._pop_root(kwargs.get("run_id"))
        if state is None:
            return
        if not state.llm_answer:
            state.llm_answer = self.auditor.coerce_answer(outputs, state)
        self.auditor.finalize(state)

    @_guarded
    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        state = self._pop_root(kwargs.get("run_id"))
        if state is None:
            return
        logger.error("RAGCompliance: chain error: %s", error)
        self.auditor.fail(state, error)

    # ------------------------------------------------------------------ #
    # Inner runnable starts                                              #
    # ------------------------------------------------------------------ #

    @_guarded
    def on_retriever_start(self, serialized: Any, query: str, **kwargs: Any) -> None:
        root = self._register_descendant(kwargs.get("run_id"), kwargs.get("parent_run_id"))
        name = _name_of(serialized, kwargs)
        if root is not None and name:
            with self._lock:
                state = self._runs.get(root)
                if state is not None:
                    state.note_retriever(name)

    @_guarded
    def on_llm_start(self, serialized: Any, prompts: list[str], **kwargs: Any) -> None:
        root = self._register_descendant(kwargs.get("run_id"), kwargs.get("parent_run_id"))
        self._note_model_call(root, kwargs, "\n".join(str(p) for p in (prompts or [])))

    @_guarded
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        root = self._register_descendant(kwargs.get("run_id"), kwargs.get("parent_run_id"))
        self._note_model_call(root, kwargs, _render_messages(messages))

    @_guarded
    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        self._register_descendant(kwargs.get("run_id"), kwargs.get("parent_run_id"))

    def _note_model_call(self, root: Any, kwargs: Mapping[str, Any], prompt: str) -> None:
        if root is None:
            return
        params = kwargs.get("invocation_params")
        with self._lock:
            state = self._runs.get(root)
            if state is None:
                return
            if isinstance(params, Mapping):
                state.set_model_params(params)
            if prompt:
                state.set_prompt(prompt)

    # ------------------------------------------------------------------ #
    # Retriever and LLM ends                                             #
    # ------------------------------------------------------------------ #

    @_guarded
    def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
        with self._lock:
            state = self._state_for(kwargs.get("run_id"))
            if state is None:
                return
            for doc in documents or []:
                state.chunks.append(coerce_chunk(doc, index=len(state.chunks)))

    @_guarded
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        answer, model_name = _answer_and_model(response)
        with self._lock:
            state = self._state_for(kwargs.get("run_id"))
            if state is None:
                return
            if answer:
                state.llm_answer = answer
            if model_name:
                state.model_name = model_name
            usage = _usage_from(response)
            if usage:
                state.lineage["usage"] = usage

    # ------------------------------------------------------------------ #
    # Helpers kept for compatibility with 0.1.x tests                    #
    # ------------------------------------------------------------------ #

    def _sign_chain(self, state: Trace) -> str:
        from .chain import content_signature

        return content_signature(state.query, state.chunks, state.llm_answer)

    def _coerce_query(self, inputs: Any) -> str:
        return self.auditor.coerce_query(inputs)

    def _coerce_answer(self, outputs: Any) -> str:
        return self.auditor.coerce_answer(outputs)

    def flush(self, timeout: float | None = None) -> bool:
        return self.auditor.flush(timeout)


# ---------------------------------------------------------------------- #
# Module helpers                                                         #
# ---------------------------------------------------------------------- #


def _actor_and_extra_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor: dict[str, Any] = {}
    raw_actor = metadata.get("ragcompliance_actor")
    if isinstance(raw_actor, Mapping):
        actor.update({str(k): v for k, v in raw_actor.items()})
    for key, target in (("actor_id", "id"), ("user_id", "id"), ("actor_type", "type")):
        if key in metadata and metadata[key] is not None and target not in actor:
            actor[target] = metadata[key]
    extra_raw = metadata.get("ragcompliance_extra")
    extra = dict(extra_raw) if isinstance(extra_raw, Mapping) else {}
    return actor, extra


def _name_of(serialized: Any, kwargs: Mapping[str, Any]) -> str:
    if isinstance(serialized, Mapping):
        name = serialized.get("name")
        if name:
            return str(name)
        ident = serialized.get("id")
        if isinstance(ident, list) and ident:
            return str(ident[-1])
    name = kwargs.get("name")
    return str(name) if name else ""


def _render_messages(messages: Any) -> str:
    lines: list[str] = []
    batches = messages if isinstance(messages, list) else [messages]
    for batch in batches:
        items = batch if isinstance(batch, list) else [batch]
        for m in items:
            role = getattr(m, "type", None) or getattr(m, "role", None) or "message"
            content = getattr(m, "content", m)
            if isinstance(content, list):
                content = " ".join(
                    str(p.get("text", "")) if isinstance(p, Mapping) else str(p) for p in content
                )
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _answer_and_model(response: Any) -> tuple[str, str]:
    answer = ""
    model_name = ""
    try:
        gen = response.generations[0][0]
    except (IndexError, AttributeError, TypeError):
        gen = None
    if gen is not None:
        text = getattr(gen, "text", None)
        if isinstance(text, str):
            answer = text
        message = getattr(gen, "message", None)
        meta = getattr(message, "response_metadata", None) or {}
        if isinstance(meta, Mapping):
            model_name = str(meta.get("model_name") or meta.get("model") or "")
        if not answer and message is not None:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                answer = content
    llm_output = getattr(response, "llm_output", None) or {}
    if isinstance(llm_output, Mapping):
        model_name = str(llm_output.get("model_name") or llm_output.get("model") or model_name)
    return answer, model_name


def _usage_from(response: Any) -> dict[str, Any]:
    llm_output = getattr(response, "llm_output", None) or {}
    usage: Any = None
    if isinstance(llm_output, Mapping):
        usage = llm_output.get("token_usage") or llm_output.get("usage")
    if not usage:
        try:
            message = response.generations[0][0].message
            usage = getattr(message, "usage_metadata", None)
        except (IndexError, AttributeError, TypeError):
            usage = None
    if isinstance(usage, Mapping):
        return {
            k: v
            for k, v in usage.items()
            if isinstance(v, (int, float)) and k in (
                "prompt_tokens", "completion_tokens", "total_tokens",
                "input_tokens", "output_tokens",
            )
        }
    return {}
