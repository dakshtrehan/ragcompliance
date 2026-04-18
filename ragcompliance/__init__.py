"""
RAGCompliance — Audit trail middleware for RAG in regulated industries.
"""

from .billing import PLANS, BillingManager, WorkspaceSubscription
from .config import RAGComplianceConfig
from .handler import RAGComplianceHandler
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

__version__ = "0.1.0"
__all__ = [
    "AuditRecord",
    "AuditStorage",
    "BillingManager",
    "PLANS",
    "RAGComplianceConfig",
    "RAGComplianceHandler",
    "RetrievedChunk",
    "WorkspaceSubscription",
]
