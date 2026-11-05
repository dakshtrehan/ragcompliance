"""Shared machinery for the plain SDK wrappers (OpenAI, Anthropic).

A wrapper intercepts ``create`` style calls, opens a ``Trace`` with the
query taken from the last user message, forwards the call unchanged
(minus the audit-only keyword arguments), records the answer and model,
and finalises the trace. Streaming responses are wrapped so the record is
written when the stream is exhausted. Async clients work because the
wrapper detects an awaitable result.

Audit-only keyword arguments accepted by every wrapped ``create``:

``retrieved_chunks``
    The context the application retrieved (``RetrievedChunk`` objects,
    dicts, LangChain documents or plain strings). The SDK never sees it.
``actor``
    Per call access attribution dict.
``audit_extra``
    Extra fields merged into the record's ``extra``.

The same values can be bound for a block of code with
``auditor.bind(actor=..., chunks=...)``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..auditor import Auditor, Trace

logger = logging.getLogger(__name__)

AUDIT_KWARGS = ("retrieved_chunks", "actor", "audit_extra")


def pop_audit_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: kwargs.pop(k) for k in AUDIT_KWARGS if k in kwargs}


def text_of(content: Any) -> str:
    """Flatten message content (string or list of parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        if content.get("type") in (None, "text", "input_text", "output_text"):
            return str(content.get("text", "") or "")
        return ""
    if isinstance(content, Iterable):
        return "\n".join(t for t in (text_of(p) for p in content) if t)
    text = getattr(content, "text", None)
    return str(text) if isinstance(text, str) else ""


def last_user_message(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, Iterable):
        return ""
    last = ""
    for m in messages:
        role = m.get("role") if isinstance(m, Mapping) else getattr(m, "role", None)
        if role == "user":
            content = m.get("content") if isinstance(m, Mapping) else getattr(m, "content", None)
            last = text_of(content) or last
    return last


def render_messages(messages: Any, system: Any = None) -> str:
    lines: list[str] = []
    if system:
        lines.append(f"system: {text_of(system)}")
    if isinstance(messages, str):
        lines.append(f"user: {messages}")
    elif isinstance(messages, Iterable):
        for m in messages:
            role = m.get("role") if isinstance(m, Mapping) else getattr(m, "role", "message")
            content = m.get("content") if isinstance(m, Mapping) else getattr(m, "content", None)
            lines.append(f"{role}: {text_of(content)}")
    return "\n".join(lines)


def unsupported_params(fn: Callable[..., Any], candidates: Iterable[str]) -> list[str]:
    """Names in ``candidates`` that ``fn`` does not accept. If the signature
    has ``**kwargs`` nothing is reported, because we cannot tell."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return []
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return [c for c in candidates if c not in params]


class AuditedStream:
    """Wrap a sync or async stream, accumulate text deltas, finalise at end."""

    def __init__(
        self,
        stream: Any,
        *,
        auditor: Auditor,
        trace: Trace,
        delta_fn: Callable[[Any], str],
        model_fn: Callable[[Any], str] | None = None,
    ):
        self._stream = stream
        self._auditor = auditor
        self._trace = trace
        self._delta_fn = delta_fn
        self._model_fn = model_fn
        self._parts: list[str] = []
        self._done = False
        self._model_seen = False

    # -- passthrough ------------------------------------------------- #

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> AuditedStream:
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        exit_ = getattr(self._stream, "__exit__", None)
        if exit_ is not None:
            exit_(exc_type, exc, tb)
        if exc is not None:
            self._fail(exc)
        else:
            self._finish()

    async def __aenter__(self) -> AuditedStream:
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        exit_ = getattr(self._stream, "__aexit__", None)
        if exit_ is not None:
            await exit_(exc_type, exc, tb)
        if exc is not None:
            self._fail(exc)
        else:
            self._finish()

    # -- iteration --------------------------------------------------- #

    def __iter__(self) -> Any:
        try:
            for event in self._stream:
                self._note(event)
                yield event
        except BaseException as exc:
            self._fail(exc)
            raise
        self._finish()

    async def __aiter__(self) -> Any:
        try:
            async for event in self._stream:
                self._note(event)
                yield event
        except BaseException as exc:
            self._fail(exc)
            raise
        self._finish()

    # -- internals --------------------------------------------------- #

    def _note(self, event: Any) -> None:
        try:
            delta = self._delta_fn(event)
            if delta:
                self._parts.append(delta)
            if self._model_fn is not None and not self._model_seen:
                model = self._model_fn(event)
                if model:
                    # The served model (from the response) wins over the
                    # requested one recorded from the call parameters.
                    self._trace.model_name = model
                    self._model_seen = True
        except Exception as exc:
            logger.debug("RAGCompliance: stream delta extraction failed (%s)", exc)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._trace.llm_answer = "".join(self._parts)
        self._trace.lineage["streamed"] = True
        self._auditor.finalize(self._trace)

    def _fail(self, exc: BaseException) -> None:
        if self._done:
            return
        self._done = True
        self._trace.llm_answer = "".join(self._parts)
        self._trace.lineage["streamed"] = True
        self._auditor.fail(self._trace, exc)


class SDKWrapperBase:
    """Common constructor and call pipeline for the SDK wrappers."""

    framework = "sdk"

    def __init__(
        self,
        client: Any,
        auditor: Auditor | None = None,
        *,
        framework_version: str = "",
        **auditor_kwargs: Any,
    ):
        self._client = client
        self.auditor = auditor or Auditor(
            auditor_kwargs.pop("config", None),
            framework=self.framework,
            framework_version=framework_version,
            **auditor_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _run(
        self,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
        *,
        query: str,
        prompt: str,
        answer_fn: Callable[[Any], tuple[str, str]],
        delta_fn: Callable[[Any], str],
        stream_model_fn: Callable[[Any], str] | None,
        usage_fn: Callable[[Any], dict[str, Any]] | None = None,
        dropped: Mapping[str, Any] | None = None,
    ) -> Any:
        audit = pop_audit_kwargs(kwargs)
        trace = self.auditor.new_trace(
            query,
            actor=audit.get("actor"),
            extra=audit.get("audit_extra"),
            chunks=audit.get("retrieved_chunks"),
        )
        self.auditor.check_quota()
        trace.set_model_params(kwargs)
        trace.set_prompt(prompt)
        if dropped:
            trace.lineage["model_params_dropped"] = dict(dropped)

        try:
            result = fn(**kwargs)
        except BaseException as exc:
            self.auditor.fail(trace, exc)
            raise

        if inspect.isawaitable(result):
            return self._finish_async(
                result, trace, kwargs, answer_fn, delta_fn, stream_model_fn, usage_fn
            )
        return self._finish(result, trace, kwargs, answer_fn, delta_fn, stream_model_fn, usage_fn)

    def _finish(
        self,
        result: Any,
        trace: Trace,
        kwargs: Mapping[str, Any],
        answer_fn: Callable[[Any], tuple[str, str]],
        delta_fn: Callable[[Any], str],
        stream_model_fn: Callable[[Any], str] | None,
        usage_fn: Callable[[Any], dict[str, Any]] | None,
    ) -> Any:
        if kwargs.get("stream"):
            return AuditedStream(
                result,
                auditor=self.auditor,
                trace=trace,
                delta_fn=delta_fn,
                model_fn=stream_model_fn,
            )
        try:
            answer, model = answer_fn(result)
            trace.set_answer(answer, model_name=model)
            if usage_fn is not None:
                usage = usage_fn(result)
                if usage:
                    trace.lineage["usage"] = usage
        except Exception as exc:
            logger.debug("RAGCompliance: answer extraction failed (%s)", exc)
        self.auditor.finalize(trace)
        return result

    async def _finish_async(
        self,
        awaitable: Any,
        trace: Trace,
        kwargs: Mapping[str, Any],
        answer_fn: Callable[[Any], tuple[str, str]],
        delta_fn: Callable[[Any], str],
        stream_model_fn: Callable[[Any], str] | None,
        usage_fn: Callable[[Any], dict[str, Any]] | None,
    ) -> Any:
        try:
            result = await awaitable
        except BaseException as exc:
            self.auditor.fail(trace, exc)
            raise
        return self._finish(result, trace, kwargs, answer_fn, delta_fn, stream_model_fn, usage_fn)
