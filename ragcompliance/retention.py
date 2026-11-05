"""Retention policy and purge bookkeeping.

A retention policy says how long audit records are kept. Enforcement is
explicit: ``AuditStorage.purge_expired()`` or ``ragcompliance purge``
deletes records older than the cutoff on sinks that support deletion.

Deleting records from a hash chain breaks the chain at the deletion
point, so every purge appends a ``retention_checkpoint`` record that
names the ``record_hash`` and ``sequence`` of the last purged record,
how many records were removed, and the cutoff. The verifier treats the
first surviving record's ``prev_hash`` as anchored when a checkpoint in
the same chain names it (see DECISIONS.md D04).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ._version import __version__
from .chain import RECORD_KIND_RETENTION_CHECKPOINT
from .models import AuditRecord


class PurgeNotSupported(RuntimeError):
    """The sink cannot delete records (for example an Object Lock bucket)."""


@dataclass
class RetentionPolicy:
    """How long records are retained. ``days=None`` means forever."""

    days: int | None = None

    @classmethod
    def from_env(cls) -> RetentionPolicy:
        raw = os.getenv("RAGCOMPLIANCE_RETENTION_DAYS", "").strip()
        if not raw:
            return cls()
        try:
            return cls(days=int(raw))
        except ValueError:
            return cls()

    @property
    def enabled(self) -> bool:
        return self.days is not None and self.days >= 0

    def cutoff(self, now: datetime | None = None) -> datetime | None:
        if not self.enabled:
            return None
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now - timedelta(days=int(self.days or 0))

    def to_dict(self) -> dict[str, Any]:
        return {"days": self.days, "enabled": self.enabled}


@dataclass
class PurgeResult:
    workspace_id: str
    cutoff: datetime
    purged: int = 0
    last_purged_sequence: int | None = None
    last_purged_hash: str = ""
    checkpoint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "cutoff": self.cutoff.isoformat(),
            "purged": self.purged,
            "last_purged_sequence": self.last_purged_sequence,
            "last_purged_hash": self.last_purged_hash,
            "checkpoint_id": (self.checkpoint or {}).get("id"),
        }


def build_retention_checkpoint(
    *,
    workspace_id: str,
    cutoff: datetime,
    purged: int,
    last_purged_sequence: int | None,
    last_purged_hash: str,
    actor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The record appended to a chain after a purge.

    It is a normal ``AuditRecord`` so every sink can store it without
    special casing. ``query`` and ``llm_answer`` are empty, and
    ``chain_signature`` is left empty because the content signature is
    only meaningful for invocations.
    """
    record = AuditRecord(
        session_id="retention",
        workspace_id=workspace_id,
        kind=RECORD_KIND_RETENTION_CHECKPOINT,
        actor=dict(actor or {"id": "ragcompliance-retention", "type": "system"}),
        tool_version=__version__,
        extra={
            "purged": int(purged),
            "cutoff": cutoff.isoformat(),
            "last_purged_sequence": last_purged_sequence,
            "last_purged_hash": last_purged_hash,
        },
        lineage={"event": "retention_purge"},
    )
    return record.to_dict()
