"""First class integrations, each behind an optional extra.

``ragcompliance.integrations.langchain``
    ``RAGComplianceHandler`` plus ``audited_chain`` helper (extra ``langchain``).
``ragcompliance.integrations.llamaindex``
    ``LlamaIndexRAGComplianceHandler`` plus ``install`` helper (extra ``llamaindex``).
``ragcompliance.integrations.openai``
    ``AuditedOpenAI`` wrapper around an OpenAI client (extra ``openai``).
``ragcompliance.integrations.anthropic``
    ``AuditedAnthropic`` wrapper around an Anthropic client (extra ``anthropic``).

The OpenAI and Anthropic wrappers only need the SDK at call time; they
import nothing from it at module import, so they can be tested with a
fake client too.
"""

from __future__ import annotations

__all__ = ["anthropic", "langchain", "llamaindex", "openai"]
