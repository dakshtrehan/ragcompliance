"""RAGCompliance: tamper evident audit trails for RAG pipelines.

Framework neutral core::

    from ragcompliance import Auditor, RetrievedChunk

Integrations (each needs its optional extra)::

    from ragcompliance import RAGComplianceHandler            # langchain
    from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler
    from ragcompliance.integrations.openai import AuditedOpenAI
    from ragcompliance.integrations.anthropic import AuditedAnthropic
"""

from __future__ import annotations

from typing import Any

from ._version import __version__
from .alerts import SlackAlerter
from .auditor import Auditor, QuotaExceededError, Trace
from .chain import (
    GENESIS_HASH,
    VerificationResult,
    content_signature,
    verify_records,
)
from .config import RAGComplianceConfig
from .models import AuditRecord, RetrievedChunk
from .redaction import (
    BUILTIN_PATTERNS,
    DEFAULT_PATTERN_ORDER,
    Pattern,
    RedactionResult,
    Redactor,
    pseudonymize,
    redact,
)
from .retention import PurgeNotSupported, PurgeResult, RetentionPolicy
from .sinks import AuditSink, MemorySink, make_sink
from .storage import AuditStorage, ConfigurationError

__all__ = [
    "BUILTIN_PATTERNS",
    "DEFAULT_PATTERN_ORDER",
    "GENESIS_HASH",
    "PLANS",
    "AuditRecord",
    "AuditSink",
    "AuditStorage",
    "Auditor",
    "BillingManager",
    "BillingReadiness",
    "ConfigurationError",
    "MemorySink",
    "Pattern",
    "PurgeNotSupported",
    "PurgeResult",
    "QuotaExceededError",
    "RAGComplianceConfig",
    "RAGComplianceHandler",
    "RedactionResult",
    "Redactor",
    "RetentionPolicy",
    "RetrievedChunk",
    "SlackAlerter",
    "Trace",
    "VerificationResult",
    "WorkspaceSubscription",
    "__version__",
    "content_signature",
    "make_sink",
    "pseudonymize",
    "redact",
    "verify_records",
]

_LAZY = {
    "RAGComplianceHandler": ("ragcompliance.handler", "RAGComplianceHandler"),
    "BillingManager": ("ragcompliance.billing", "BillingManager"),
    "BillingReadiness": ("ragcompliance.billing", "BillingReadiness"),
    "PLANS": ("ragcompliance.billing", "PLANS"),
    "WorkspaceSubscription": ("ragcompliance.billing", "WorkspaceSubscription"),
}


def __getattr__(name: str) -> Any:
    # Optional-dependency symbols are imported on first access so that
    # ``import ragcompliance`` works without langchain-core or stripe.
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'ragcompliance' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])
