"""Supabase sink (the 0.1.x storage backend, now behind the sink API).

Two table layouts are supported (DECISIONS.md D07):

Layout 1 (default, no migration needed)
    The 0.1.x ``rag_audit_logs`` columns. The version 2 fields (kind,
    actor, lineage, schema_version, tool_version and the chain fields)
    are nested under ``extra["_ragcompliance"]`` so PostgREST accepts
    the row. Reads unpack them again.

Layout 2 (after ``supabase_migration_v1.sql``)
    Dedicated columns and a unique index on
    ``(workspace_id, chain_id, sequence)``. A duplicate key error means
    another writer appended first; the sink re-reads the tail and
    retries, which keeps the chain linear across processes.

With layout 1 the tail is read once per chain from the newest row and
then tracked in memory, so chain continuity across several processes
that share a ``chain_id`` is not guaranteed. Migrate to layout 2 or give
each writer its own ``chain_id``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from ..models import AuditRecord
from ..retention import PurgeNotSupported
from .base import AuditSink

logger = logging.getLogger(__name__)

LEGACY_COLUMNS = (
    "id",
    "session_id",
    "workspace_id",
    "query",
    "retrieved_chunks",
    "llm_answer",
    "model_name",
    "chain_signature",
    "timestamp",
    "latency_ms",
    "extra",
)
NESTED_KEY = "_ragcompliance"
V2_COLUMNS = (
    "kind",
    "actor",
    "lineage",
    "schema_version",
    "tool_version",
    "chain_id",
    "sequence",
    "prev_hash",
    "record_hash",
    "record_hmac",
)
PAGE_SIZE = 1000


def pack_row(sealed: dict[str, Any], schema_version: int) -> dict[str, Any]:
    """Shape a sealed record for the configured table layout."""
    if schema_version >= 2:
        return {k: sealed.get(k) for k in LEGACY_COLUMNS + V2_COLUMNS}
    row = {k: sealed.get(k) for k in LEGACY_COLUMNS}
    extra = dict(sealed.get("extra") or {})
    extra[NESTED_KEY] = {k: sealed.get(k) for k in V2_COLUMNS}
    row["extra"] = extra
    return row


def unpack_row(row: dict[str, Any]) -> dict[str, Any]:
    """Inverse of ``pack_row`` for either layout."""
    rec = dict(row)
    extra = dict(rec.get("extra") or {})
    nested = extra.pop(NESTED_KEY, None)
    rec["extra"] = extra
    if isinstance(nested, dict):
        for k in V2_COLUMNS:
            if rec.get(k) in (None, "") and k in nested:
                rec[k] = nested[k]
    # Normalise through the model so defaults and types are consistent.
    return AuditRecord.from_dict(rec).to_dict()


class SupabaseSink(AuditSink):
    name = "supabase"
    supports_purge = True

    def __init__(
        self,
        url: str = "",
        key: str = "",
        *,
        table: str = "rag_audit_logs",
        schema_version: int = 1,
        chain_id: str = "default",
        hmac_key: bytes | str | None = None,
        client: Any | None = None,
    ):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.table = table
        self.schema_version = int(schema_version)
        if client is None:
            if not (url and key):
                raise ValueError("SupabaseSink needs url and key (or a client)")
            try:
                from supabase import create_client
            except ImportError as exc:
                raise ImportError(
                    "supabase package not installed. Install with: "
                    "pip install 'ragcompliance[supabase]'"
                ) from exc
            client = create_client(url, key)
        self._client = client
        self._tails: dict[tuple[str, str], tuple[int, str]] = {}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"table": self.table, "schema_version": self.schema_version})
        return d

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        key = (workspace_id, chain_id)
        if key in self._tails:
            return self._tails[key]
        tail = self._fetch_tail(workspace_id, chain_id)
        if tail is not None:
            self._tails[key] = tail
        return tail

    def _fetch_tail(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        q = self._client.table(self.table).select("*").eq("workspace_id", workspace_id)
        if self.schema_version >= 2:
            q = q.eq("chain_id", chain_id).order("sequence", desc=True)
        else:
            q = q.order("timestamp", desc=True)
        rows = q.limit(1).execute().data or []
        if not rows:
            return None
        rec = unpack_row(rows[0])
        if rec.get("sequence") is None or not rec.get("record_hash"):
            return None
        if self.schema_version < 2 and rec.get("chain_id") not in ("", chain_id):
            return None
        return int(rec["sequence"]), str(rec["record_hash"])

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        self._client.table(self.table).insert(pack_row(sealed, self.schema_version)).execute()
        key = (str(sealed.get("workspace_id", "")), str(sealed["chain_id"]))
        self._tails[key] = (int(sealed["sequence"]), str(sealed["record_hash"]))

    def _append_locked(
        self, rec: dict[str, Any], workspace_id: str, chain_id: str
    ) -> dict[str, Any]:
        attempts = 3 if self.schema_version >= 2 else 1
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                return super()._append_locked(rec, workspace_id, chain_id)
            except Exception as exc:
                last_exc = exc
                if "23505" in str(exc) or "duplicate" in str(exc).lower():
                    self._tails.pop((workspace_id, chain_id), None)
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            q = self._client.table(self.table).select("*")
            if workspace_id is not None:
                q = q.eq("workspace_id", workspace_id)
            if chain_id is not None and self.schema_version >= 2:
                q = q.eq("chain_id", chain_id)
            if self.schema_version >= 2:
                q = q.order("workspace_id").order("chain_id").order("sequence")
            else:
                q = q.order("timestamp")
            rows = q.range(offset, offset + PAGE_SIZE - 1).execute().data or []
            for row in rows:
                rec = unpack_row(row)
                if chain_id is not None and rec.get("chain_id") != chain_id:
                    continue
                yield rec
            if len(rows) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    def get(self, record_id: str, *, workspace_id: str | None = None) -> dict[str, Any] | None:
        q = self._client.table(self.table).select("*").eq("id", record_id)
        if workspace_id:
            q = q.eq("workspace_id", workspace_id)
        rows = q.limit(1).execute().data or []
        return unpack_row(rows[0]) if rows else None

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        if self.schema_version < 2:
            raise PurgeNotSupported(
                "purging the Supabase sink requires the layout 2 migration "
                "(supabase_migration_v1.sql) so the chain prefix can be selected safely"
            )
        chain = list(self._iter_all(workspace_id=workspace_id, chain_id=chain_id))
        from .base import select_purge_victims

        purged, _kept = select_purge_victims(chain, cutoff)
        if not purged:
            return 0, None, ""
        last = purged[-1]
        (
            self._client.table(self.table)
            .delete()
            .eq("workspace_id", workspace_id)
            .eq("chain_id", chain_id)
            .lte("sequence", int(last["sequence"]))
            .execute()
        )
        self._tails.pop((workspace_id, chain_id), None)
        return len(purged), int(last["sequence"]), str(last["record_hash"])
