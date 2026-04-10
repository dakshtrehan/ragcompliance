from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class RetrievedChunk:
    content: str
    source_url: str
    chunk_id: str
    similarity_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    workspace_id: str = ""
    query: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    llm_answer: str = ""
    model_name: str = ""
    chain_signature: str = ""  # SHA-256 of the full chain
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "query": self.query,
            "retrieved_chunks": [
                {
                    "content": c.content,
                    "source_url": c.source_url,
                    "chunk_id": c.chunk_id,
                    "similarity_score": c.similarity_score,
                    "metadata": c.metadata,
                }
                for c in self.retrieved_chunks
            ],
            "llm_answer": self.llm_answer,
            "model_name": self.model_name,
            "chain_signature": self.chain_signature,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "extra": self.extra,
        }
