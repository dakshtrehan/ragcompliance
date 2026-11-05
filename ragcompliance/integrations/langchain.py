"""LangChain integration (extra ``langchain``).

``RAGComplianceHandler`` is the callback handler; ``audited_chain`` binds
it to a runnable so callers no longer need to pass ``callbacks`` on every
invoke::

    from ragcompliance.integrations.langchain import RAGComplianceHandler, audited_chain

    handler = RAGComplianceHandler(session_id="user-1")
    chain = audited_chain(my_chain, handler)
    chain.invoke({"query": "..."})
"""

from __future__ import annotations

from typing import Any

from ..handler import RAGComplianceHandler

__all__ = ["RAGComplianceHandler", "audited_chain"]


def audited_chain(runnable: Any, handler: RAGComplianceHandler, **config: Any) -> Any:
    """Return ``runnable.with_config(callbacks=[handler], **config)``.

    ``config`` may carry ``metadata`` with a ``ragcompliance_actor`` entry
    to attribute every invocation of the returned chain.
    """
    callbacks = list(config.pop("callbacks", []) or [])
    callbacks.append(handler)
    return runnable.with_config(callbacks=callbacks, **config)
