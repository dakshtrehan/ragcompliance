"""
RAGCompliance — Audit trail middleware for RAG in regulated industries.
"""

from .alerts import SlackAlerter
from .billing import BillingManager, BillingReadiness, PLANS, WorkspaceSubscription
from .config import RAGComplianceConfig
from .handler import RAGComplianceHandler
from .models import AuditRecord, RetrievedChunk
from .redaction import (
    BUILTIN_PATTERNS,
    DEFAULT_PATTERN_ORDER,
    Pattern,
    Redactor,
    RedactionResult,
    redact,
)
from .storage import AuditStorage

__version__ = "0.1.8"
__all__ = [
    "AuditRecord",
    "AuditStorage",
    "BillingManager",
    "BillingReadiness",
    "BUILTIN_PATTERNS",
    "DEFAULT_PATTERN_ORDER",
    "PLANS",
    "Pattern",
    "RAGComplianceConfig",
    "RAGComplianceHandler",
    "RedactionResult",
    "Redactor",
    "RetrievedChunk",
    "SlackAlerter",
    "WorkspaceSubscription",
    "redact",
]
