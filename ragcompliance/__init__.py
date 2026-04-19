"""
RAGCompliance — Audit trail middleware for RAG in regulated industries.
"""

from .alerts import SlackAlerter
from .billing import BillingManager, BillingReadiness, PLANS, WorkspaceSubscription
from .config import RAGComplianceConfig
from .handler import RAGComplianceHandler
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

__version__ = "0.1.5"
__all__ = [
    "AuditRecord",
    "AuditStorage",
    "BillingManager",
    "BillingReadiness",
    "PLANS",
    "RAGComplianceConfig",
    "RAGComplianceHandler",
    "RetrievedChunk",
    "SlackAlerter",
    "WorkspaceSubscription",
]
