"""In-memory sink for tests and short lived scripts."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from .base import AuditSink, select_purge_victims


class MemorySink(AuditSink):
    name = "memory"
    supports_purge = True

    def __init__(self, *, chain_id: str = "default", hmac_key: bytes | str | None = None):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.records: list[dict[str, Any]] = []

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        for rec in reversed(self.records):
            if rec.get("workspace_id") == workspace_id and rec.get("chain_id") == chain_id:
                return int(rec["sequence"]), str(rec["record_hash"])
        return None

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        self.records.append(dict(sealed))

    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        with self._lock:
            snapshot = list(self.records)
        snapshot.sort(
            key=lambda r: (
                str(r.get("workspace_id", "")),
                str(r.get("chain_id", "")),
                int(r.get("sequence") or 0),
            )
        )
        for rec in snapshot:
            if workspace_id is not None and rec.get("workspace_id") != workspace_id:
                continue
            if chain_id is not None and rec.get("chain_id") != chain_id:
                continue
            yield dict(rec)

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        chain = [
            r
            for r in self.records
            if r.get("workspace_id") == workspace_id and r.get("chain_id") == chain_id
        ]
        chain.sort(key=lambda r: int(r.get("sequence") or 0))
        purged, _kept = select_purge_victims(chain, cutoff)
        if not purged:
            return 0, None, ""
        purged_ids = {r["id"] for r in purged}
        self.records = [r for r in self.records if r.get("id") not in purged_ids]
        last = purged[-1]
        return len(purged), int(last["sequence"]), str(last["record_hash"])
