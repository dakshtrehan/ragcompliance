"""LlamaIndex integration (extra ``llamaindex``).

::

    from ragcompliance.integrations.llamaindex import LlamaIndexRAGComplianceHandler, install

    handler = install(session_id="user-1")   # registers on llama_index.core.Settings
    response = query_engine.query("...")
"""

from __future__ import annotations

from typing import Any

from ..llamaindex_handler import LlamaIndexRAGComplianceHandler

__all__ = ["LlamaIndexRAGComplianceHandler", "install"]


def install(**handler_kwargs: Any) -> LlamaIndexRAGComplianceHandler:
    """Create a handler and register it on the global LlamaIndex
    ``Settings.callback_manager`` (keeping any handlers already there)."""
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager

    handler = LlamaIndexRAGComplianceHandler(**handler_kwargs)
    existing = getattr(Settings, "callback_manager", None)
    handlers = list(getattr(existing, "handlers", []) or [])
    handlers.append(handler)
    Settings.callback_manager = CallbackManager(handlers)
    return handler
