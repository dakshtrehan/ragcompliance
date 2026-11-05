"""Sink protocol.

A sink persists sealed audit records and can read them back in chain
order. The base class implements the sealing template: take the chain
tail under the sink's own lock, seal the record with the next sequence
number, write it. Subclasses supply the lock, the tail lookup and the
write.

Chain identity is ``(workspace_id, chain_id)``. ``chain_id`` comes from
the record when set, otherwise from the sink's default, which in turn
comes from ``RAGComplianceConfig.chain_id``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from ..chain import GENESIS_HASH, seal
from ..models import _parse_timestamp
from ..retention import PurgeNotSupported, PurgeResult, build_retention_checkpoint


def normalise_ts(value: Any) -> str:
    """ISO-8601 in UTC with microseconds, so string comparison equals
    chronological comparison for every row a sink stores."""
    dt = _parse_timestamp(value).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _in_window(record: Mapping[str, Any], start: datetime | None, end: datetime | None) -> bool:
    if start is None and end is None:
        return True
    ts = _parse_timestamp(record.get("timestamp"))
    if start is not None and ts < start:
        return False
    return not (end is not None and ts > end)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class AuditSink(ABC):
    """Append-only, chain-sealing record store."""

    name: str = "base"
    supports_purge: bool = False

    def __init__(self, *, chain_id: str = "default", hmac_key: bytes | str | None = None):
        self.chain_id = chain_id or "default"
        self.hmac_key = hmac_key or None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Seal ``record`` onto its chain and persist it. Returns the sealed
        dict. Raises on write failure; the storage façade decides whether
        that is fatal."""
        rec = dict(record)
        workspace_id = str(rec.get("workspace_id", "") or "")
        chain_id = str(rec.get("chain_id") or self.chain_id)
        with self._locked(workspace_id, chain_id):
            return self._append_locked(rec, workspace_id, chain_id)

    def iter_records(
        self,
        *,
        workspace_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        chain_id: str | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield records, oldest first, in ``(workspace_id, chain_id,
        sequence)`` order where the sink can guarantee it."""
        start, end = _as_utc(start), _as_utc(end)
        yielded = 0
        for rec in self._iter_all(workspace_id=workspace_id, chain_id=chain_id):
            if not _in_window(rec, start, end):
                continue
            yield rec
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def count(self, *, workspace_id: str | None = None, chain_id: str | None = None) -> int:
        return sum(1 for _ in self.iter_records(workspace_id=workspace_id, chain_id=chain_id))

    def get(self, record_id: str, *, workspace_id: str | None = None) -> dict[str, Any] | None:
        for rec in self._iter_all(workspace_id=workspace_id, chain_id=None):
            if str(rec.get("id")) == record_id:
                return rec
        return None

    def tail(self, workspace_id: str, chain_id: str | None = None) -> tuple[int, str] | None:
        chain_id = chain_id or self.chain_id
        with self._locked(workspace_id, chain_id):
            return self._tail_locked(workspace_id, chain_id)

    def purge_before(
        self,
        cutoff: datetime,
        *,
        workspace_id: str,
        chain_id: str | None = None,
    ) -> PurgeResult:
        """Delete records older than ``cutoff`` for one workspace and chain,
        then append a retention checkpoint. Sinks that cannot delete raise
        ``PurgeNotSupported``."""
        if not self.supports_purge:
            raise PurgeNotSupported(f"{self.name} sink does not support purging")
        cutoff = _as_utc(cutoff) or datetime.now(timezone.utc)
        chain_id = chain_id or self.chain_id
        with self._locked(workspace_id, chain_id):
            purged, last_seq, last_hash = self._purge_locked(cutoff, workspace_id, chain_id)
            result = PurgeResult(
                workspace_id=workspace_id,
                cutoff=cutoff,
                purged=purged,
                last_purged_sequence=last_seq,
                last_purged_hash=last_hash,
            )
            if purged:
                checkpoint = build_retention_checkpoint(
                    workspace_id=workspace_id,
                    cutoff=cutoff,
                    purged=purged,
                    last_purged_sequence=last_seq,
                    last_purged_hash=last_hash,
                )
                checkpoint["chain_id"] = chain_id
                if self._tail_locked(workspace_id, chain_id) is None and last_seq is not None:
                    # Everything was purged: continue the chain from the last
                    # purged record so the checkpoint itself anchors the gap.
                    sealed = seal(
                        checkpoint,
                        prev_hash=last_hash,
                        sequence=last_seq + 1,
                        chain_id=chain_id,
                        hmac_key=self.hmac_key,
                    )
                    self._write_locked(sealed)
                    result.checkpoint = sealed
                else:
                    result.checkpoint = self._append_locked(checkpoint, workspace_id, chain_id)
            return result

    def flush(self) -> None:  # pragma: no cover - trivial default
        return None

    def close(self) -> None:  # pragma: no cover - trivial default
        return None

    def describe(self) -> dict[str, Any]:
        """Non-secret description for evidence packs and selftest."""
        return {"type": self.name, "chain_id": self.chain_id, "hmac": bool(self.hmac_key)}

    # ------------------------------------------------------------------ #
    # Template internals                                                 #
    # ------------------------------------------------------------------ #

    def _append_locked(
        self, rec: dict[str, Any], workspace_id: str, chain_id: str
    ) -> dict[str, Any]:
        tail = self._tail_locked(workspace_id, chain_id)
        if tail is None:
            sequence, prev_hash = 1, GENESIS_HASH
        else:
            sequence, prev_hash = tail[0] + 1, tail[1]
        sealed = seal(
            rec,
            prev_hash=prev_hash,
            sequence=sequence,
            chain_id=chain_id,
            hmac_key=self.hmac_key,
        )
        self._write_locked(sealed)
        return sealed

    @contextmanager
    def _locked(self, workspace_id: str, chain_id: str) -> Iterator[None]:
        """Critical section around tail lookup and write. Subclasses extend
        this with file locks or database transactions."""
        with self._lock:
            yield

    # ------------------------------------------------------------------ #
    # Subclass hooks                                                     #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        """``(sequence, record_hash)`` of the newest record in the chain."""

    @abstractmethod
    def _write_locked(self, sealed: dict[str, Any]) -> None:
        """Persist one sealed record."""

    @abstractmethod
    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        """Every stored record matching the filters, oldest first."""

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        """Delete and return ``(count, last_purged_sequence, last_purged_hash)``."""
        raise PurgeNotSupported(f"{self.name} sink does not support purging")


def select_purge_victims(
    records: list[dict[str, Any]], cutoff: datetime
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``records`` (one chain, sequence order) into ``(purged, kept)``.

    Only a prefix of the chain can be purged without leaving holes, so
    the split stops at the first record that is newer than the cutoff.
    """
    purged: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    cutoff_str = normalise_ts(cutoff)
    boundary_hit = False
    for rec in records:
        if not boundary_hit and normalise_ts(rec.get("timestamp")) < cutoff_str:
            purged.append(rec)
        else:
            boundary_hit = True
            kept.append(rec)
    return purged, kept
