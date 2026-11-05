"""Audit record model.

``AuditRecord`` is the unit the whole library moves around: handlers
build one per RAG invocation, sinks seal and persist it, the verifier
and the evidence exporter read it back.

Schema versions
---------------
``schema_version`` 1 is the 0.1.x shape (id, session_id, workspace_id,
query, retrieved_chunks, llm_answer, model_name, chain_signature,
timestamp, latency_ms, extra). Version 2 adds, all with defaults so that
existing call sites keep working:

``kind``
    ``"rag_invocation"`` for normal records, ``"retention_checkpoint"``
    for the marker written after a purge.
``actor``
    Who issued the query: ``{"id", "type", "auth_method", "ip", ...}``.
``lineage``
    How the answer was produced: chunk hashes, prompt hash, model
    parameters, framework and library versions, timing, truncation.
``tool_version``
    The ragcompliance version that wrote the record.
``chain_id``, ``sequence``, ``prev_hash``, ``record_hash``, ``record_hmac``
    Assigned by the sink when the record is sealed into a hash chain.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._version import __version__

SCHEMA_VERSION = 2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return _utcnow()
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # PostgREST can emit a variable number of fractional digits.
        head, _, tail = text.partition(".")
        frac = ""
        rest = tail
        while rest and rest[0].isdigit():
            frac += rest[0]
            rest = rest[1:]
        dt = datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}{rest}")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class RetrievedChunk:
    content: str
    source_url: str
    chunk_id: str
    similarity_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def content_sha256(self) -> str:
        return hashlib.sha256((self.content or "").encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "source_url": self.source_url,
            "chunk_id": self.chunk_id,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievedChunk:
        score = data.get("similarity_score")
        return cls(
            content=str(data.get("content", "") or ""),
            source_url=str(data.get("source_url", "") or ""),
            chunk_id=str(data.get("chunk_id", "") or ""),
            similarity_score=float(score) if score is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AuditRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    workspace_id: str = ""
    query: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    llm_answer: str = ""
    model_name: str = ""
    chain_signature: str = ""  # SHA-256 over (query, chunks, answer), see chain.py
    timestamp: datetime = field(default_factory=_utcnow)
    latency_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    # ---- schema version 2 ------------------------------------------- #
    kind: str = "rag_invocation"
    actor: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    tool_version: str = __version__
    chain_id: str = ""
    sequence: int | None = None
    prev_hash: str = ""
    record_hash: str = ""
    record_hmac: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "query": self.query,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "llm_answer": self.llm_answer,
            "model_name": self.model_name,
            "chain_signature": self.chain_signature,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "extra": self.extra,
            "kind": self.kind,
            "actor": self.actor,
            "lineage": self.lineage,
            "schema_version": self.schema_version,
            "tool_version": self.tool_version,
            "chain_id": self.chain_id,
            "sequence": self.sequence,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
            "record_hmac": self.record_hmac,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        """Rebuild a record from ``to_dict()`` output or a storage row.

        Unknown keys are ignored. Missing version 2 keys take their
        defaults, which is how 0.1.x rows are read.
        """
        chunks_raw = data.get("retrieved_chunks") or []
        chunks = [
            c if isinstance(c, RetrievedChunk) else RetrievedChunk.from_dict(c)
            for c in chunks_raw
        ]
        seq = data.get("sequence")
        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            session_id=str(data.get("session_id", "") or ""),
            workspace_id=str(data.get("workspace_id", "") or ""),
            query=str(data.get("query", "") or ""),
            retrieved_chunks=chunks,
            llm_answer=str(data.get("llm_answer", "") or ""),
            model_name=str(data.get("model_name", "") or ""),
            chain_signature=str(data.get("chain_signature", "") or ""),
            timestamp=_parse_timestamp(data.get("timestamp")),
            latency_ms=int(data.get("latency_ms") or 0),
            extra=dict(data.get("extra") or {}),
            kind=str(data.get("kind") or "rag_invocation"),
            actor=dict(data.get("actor") or {}),
            lineage=dict(data.get("lineage") or {}),
            schema_version=int(data.get("schema_version") or 1),
            tool_version=str(data.get("tool_version", "") or ""),
            chain_id=str(data.get("chain_id", "") or ""),
            sequence=int(seq) if seq is not None else None,
            prev_hash=str(data.get("prev_hash", "") or ""),
            record_hash=str(data.get("record_hash", "") or ""),
            record_hmac=str(data.get("record_hmac", "") or ""),
        )

    def apply_seal(self, sealed: dict[str, Any]) -> None:
        """Copy chain fields from a sealed dict back onto this record."""
        self.chain_id = str(sealed.get("chain_id", ""))
        seq = sealed.get("sequence")
        self.sequence = int(seq) if seq is not None else None
        self.prev_hash = str(sealed.get("prev_hash", ""))
        self.record_hash = str(sealed.get("record_hash", ""))
        self.record_hmac = str(sealed.get("record_hmac", ""))
