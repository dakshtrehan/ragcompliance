"""
RAGCompliance — Audit trail middleware for RAG in regulated industries.
"""

from .config import RAGComplianceConfig
from .handler import RAGComplianceHandler
from .models import AuditRecord, RetrievedChunk
from .storage import AuditStorage

__version__ = "0.1.0"
__all__ = [
    "RAGComplianceConfig",
    "RAGComplianceHandler",
    "AuditRecord",
    "RetrievedChunk",
    "AuditStorage",
]
