"""Anthropic SDK wrapper (extra ``anthropic``).

::

    from anthropic import Anthropic
    from ragcompliance.integrations.anthropic import AuditedAnthropic

    client = AuditedAnthropic(Anthropic(), session_id="user-1")
    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        system="Answer from the context only.",
        messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
        retrieved_chunks=my_chunks,            # audit only
        actor={"id": "alice@example.com"},     # audit only
    )

API version note
----------------
Anthropic SDK 2.x defaults to API version ``2026-10-21``, which rejects
``temperature`` (and the installed SDK's ``messages.create`` no longer
accepts ``temperature``, ``top_p`` or ``top_k`` at all). The wrapper
inspects the installed ``create`` signature and drops any of those
sampling parameters it does not accept, logs a warning once per process,
and records what was dropped in ``lineage.model_params_dropped`` so the
audit record still shows what the application asked for. See
DECISIONS.md D12. If a future SDK accepts the parameter again it is
passed through unchanged.

Wrapped: ``messages.create`` (sync and async, streaming via
``stream=True``). ``messages.stream(...)`` is passed through unwrapped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ._sdk import SDKWrapperBase, last_user_message, render_messages, unsupported_params

__all__ = ["AuditedAnthropic", "audit_anthropic", "SAMPLING_PARAMS"]

logger = logging.getLogger(__name__)

SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
_warned: set[str] = set()


def _anthropic_version() -> str:
    try:
        import anthropic

        return str(getattr(anthropic, "__version__", ""))
    except ImportError:
        return ""


def _message_answer(result: Any) -> tuple[str, str]:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        if isinstance(block, Mapping):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        elif getattr(block, "type", "") == "text":
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts), str(getattr(result, "model", "") or "")


def _message_delta(event: Any) -> str:
    if getattr(event, "type", "") == "content_block_delta":
        delta = getattr(event, "delta", None)
        if getattr(delta, "type", "") == "text_delta":
            return str(getattr(delta, "text", "") or "")
    return ""


def _stream_model(event: Any) -> str:
    if getattr(event, "type", "") == "message_start":
        message = getattr(event, "message", None)
        return str(getattr(message, "model", "") or "")
    return ""


def _usage(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


class _Messages:
    def __init__(self, wrapper: AuditedAnthropic):
        self._w = wrapper

    def create(self, **kwargs: Any) -> Any:
        fn = self._w._client.messages.create
        dropped = self._w._drop_unsupported(fn, kwargs)
        messages = kwargs.get("messages") or []
        return self._w._run(
            fn,
            kwargs,
            query=last_user_message(messages),
            prompt=render_messages(messages, system=kwargs.get("system")),
            answer_fn=_message_answer,
            delta_fn=_message_delta,
            stream_model_fn=_stream_model,
            usage_fn=_usage,
            dropped=dropped,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._w._client.messages, name)


class AuditedAnthropic(SDKWrapperBase):
    """Audit wrapper around ``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic``."""

    framework = "anthropic"

    def __init__(
        self,
        client: Any,
        auditor: Any | None = None,
        *,
        drop_unsupported_params: bool = True,
        **auditor_kwargs: Any,
    ):
        super().__init__(
            client, auditor, framework_version=_anthropic_version(), **auditor_kwargs
        )
        self.drop_unsupported_params = drop_unsupported_params
        self.messages = _Messages(self)

    def _drop_unsupported(self, fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.drop_unsupported_params:
            return {}
        present = [p for p in SAMPLING_PARAMS if p in kwargs]
        if not present:
            return {}
        rejected = unsupported_params(fn, present)
        dropped = {p: kwargs.pop(p) for p in rejected}
        for name in dropped:
            if name not in _warned:
                _warned.add(name)
                logger.warning(
                    "RAGCompliance: dropping %r from Anthropic messages.create; the installed "
                    "SDK (API version 2026-10-21 or later) rejects it. The requested value is "
                    "recorded in the audit record under lineage.model_params_dropped.",
                    name,
                )
        return dropped


def audit_anthropic(client: Any, **kwargs: Any) -> AuditedAnthropic:
    """``AuditedAnthropic(client, **kwargs)`` for people who prefer a function."""
    return AuditedAnthropic(client, **kwargs)
