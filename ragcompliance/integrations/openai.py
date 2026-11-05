"""OpenAI SDK wrapper (extra ``openai``).

::

    from openai import OpenAI
    from ragcompliance.integrations.openai import AuditedOpenAI

    client = AuditedOpenAI(OpenAI(), session_id="user-1")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
        retrieved_chunks=my_chunks,                 # audit only, stripped before the call
        actor={"id": "alice@example.com"},          # audit only
    )

Wrapped: ``chat.completions.create`` and ``responses.create``, sync and
async clients, streaming and non streaming. Everything else on the
client is passed through untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._sdk import SDKWrapperBase, last_user_message, render_messages, text_of

__all__ = ["AuditedOpenAI", "audit_openai"]


def _openai_version() -> str:
    try:
        import openai

        return str(getattr(openai, "__version__", ""))
    except ImportError:
        return ""


# ---- chat.completions ------------------------------------------------- #


def _chat_answer(result: Any) -> tuple[str, str]:
    choices = getattr(result, "choices", None) or []
    text = ""
    if choices:
        message = getattr(choices[0], "message", None)
        text = text_of(getattr(message, "content", None))
    return text, str(getattr(result, "model", "") or "")


def _chat_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return text_of(getattr(delta, "content", None))


def _model_attr(obj: Any) -> str:
    return str(getattr(obj, "model", "") or "")


def _usage(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


# ---- responses -------------------------------------------------------- #


def _responses_answer(result: Any) -> tuple[str, str]:
    text = getattr(result, "output_text", None)
    if not isinstance(text, str) or not text:
        parts: list[str] = []
        for item in getattr(result, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") in ("output_text", "text"):
                    parts.append(str(getattr(part, "text", "")))
        text = "".join(parts)
    return text, _model_attr(result)


def _responses_delta(event: Any) -> str:
    if getattr(event, "type", "") == "response.output_text.delta":
        return str(getattr(event, "delta", "") or "")
    return ""


def _responses_stream_model(event: Any) -> str:
    response = getattr(event, "response", None)
    return _model_attr(response) if response is not None else ""


def _responses_query(kwargs: Mapping[str, Any]) -> str:
    inp = kwargs.get("input")
    if isinstance(inp, str):
        return inp
    return last_user_message(inp)


class _ChatCompletions:
    def __init__(self, wrapper: AuditedOpenAI):
        self._w = wrapper

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") or []
        return self._w._run(
            self._w._client.chat.completions.create,
            kwargs,
            query=last_user_message(messages),
            prompt=render_messages(messages),
            answer_fn=_chat_answer,
            delta_fn=_chat_delta,
            stream_model_fn=_model_attr,
            usage_fn=_usage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._w._client.chat.completions, name)


class _Chat:
    def __init__(self, wrapper: AuditedOpenAI):
        self._w = wrapper
        self.completions = _ChatCompletions(wrapper)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._w._client.chat, name)


class _Responses:
    def __init__(self, wrapper: AuditedOpenAI):
        self._w = wrapper

    def create(self, **kwargs: Any) -> Any:
        prompt_parts = []
        if kwargs.get("instructions"):
            prompt_parts.append(f"system: {kwargs['instructions']}")
        prompt_parts.append(render_messages(kwargs.get("input")))
        return self._w._run(
            self._w._client.responses.create,
            kwargs,
            query=_responses_query(kwargs),
            prompt="\n".join(p for p in prompt_parts if p),
            answer_fn=_responses_answer,
            delta_fn=_responses_delta,
            stream_model_fn=_responses_stream_model,
            usage_fn=_usage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._w._client.responses, name)


class AuditedOpenAI(SDKWrapperBase):
    """Audit wrapper around ``openai.OpenAI`` / ``openai.AsyncOpenAI``."""

    framework = "openai"

    def __init__(self, client: Any, auditor: Any | None = None, **auditor_kwargs: Any):
        super().__init__(
            client, auditor, framework_version=_openai_version(), **auditor_kwargs
        )
        self.chat = _Chat(self)
        self.responses = _Responses(self)


def audit_openai(client: Any, **kwargs: Any) -> AuditedOpenAI:
    """``AuditedOpenAI(client, **kwargs)`` for people who prefer a function."""
    return AuditedOpenAI(client, **kwargs)
