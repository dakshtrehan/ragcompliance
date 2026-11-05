"""SQLite sink with database level append-only enforcement.

Every record is stored as a row with indexed columns for the fields the
verifier and the dashboard filter on, plus the full canonical JSON in
``payload``. Two triggers abort any ``UPDATE`` or ``DELETE``; the purge
path drops the delete trigger inside the same ``BEGIN IMMEDIATE``
transaction that performs the deletion and recreates it before commit,
so there is never a committed state without the guard.

``BEGIN IMMEDIATE`` takes the database write lock for the whole
tail-read-then-insert sequence, which keeps the chain linear across
processes as well as threads.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from ..chain import canonical_json
from .base import AuditSink, normalise_ts

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str) -> str:
    """Table names are interpolated into DDL, so they must be plain
    identifiers. Everything else goes through bound parameters."""
    if not _IDENT.match(name or ""):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


class SQLiteSink(AuditSink):
    name = "sqlite"
    supports_purge = True

    def __init__(
        self,
        path: str,
        *,
        table: str = "rag_audit_records",
        chain_id: str = "default",
        hmac_key: bytes | str | None = None,
    ):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.path = str(path)
        self.table = validate_identifier(table)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"path": self.path, "table": self.table, "append_only_triggers": True})
        return d

    # ------------------------------------------------------------------ #
    # Schema                                                             #
    # ------------------------------------------------------------------ #

    def schema_sql(self) -> list[str]:
        t = self.table
        return [
            f"""CREATE TABLE IF NOT EXISTS {t} (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                workspace_id TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                ts_utc TEXT NOT NULL,
                kind TEXT NOT NULL,
                session_id TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE (workspace_id, chain_id, sequence)
            )""",
            f"CREATE INDEX IF NOT EXISTS {t}_ws_ts ON {t} (workspace_id, ts_utc)",
            f"CREATE INDEX IF NOT EXISTS {t}_session ON {t} (session_id)",
            self._no_update_trigger_sql(),
            self._no_delete_trigger_sql(),
        ]

    def _no_update_trigger_sql(self) -> str:
        return (
            f"CREATE TRIGGER IF NOT EXISTS {self.table}_no_update BEFORE UPDATE ON "
            f"{self.table} BEGIN SELECT RAISE(ABORT, 'ragcompliance audit log is append-only'); END"
        )

    def _no_delete_trigger_sql(self) -> str:
        return (
            f"CREATE TRIGGER IF NOT EXISTS {self.table}_no_delete BEFORE DELETE ON "
            f"{self.table} BEGIN SELECT RAISE(ABORT, 'ragcompliance audit log is append-only'); END"
        )

    def _ensure_schema(self) -> None:
        with self._lock:
            for stmt in self.schema_sql():
                self._conn.execute(stmt)

    # ------------------------------------------------------------------ #
    # Locking                                                            #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _locked(self, workspace_id: str, chain_id: str) -> Iterator[None]:
        with self._lock:
            if self._conn.in_transaction:
                # Re-entrant call (purge appending its checkpoint): already
                # inside the outer transaction.
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        row = self._conn.execute(
            f"SELECT sequence, record_hash FROM {self.table} "
            "WHERE workspace_id = ? AND chain_id = ? ORDER BY sequence DESC LIMIT 1",
            (workspace_id, chain_id),
        ).fetchone()
        if row is None:
            return None
        return int(row[0]), str(row[1])

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        self._conn.execute(
            f"INSERT INTO {self.table} (record_id, workspace_id, chain_id, sequence, ts_utc, "
            "kind, session_id, prev_hash, record_hash, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(sealed["id"]),
                str(sealed.get("workspace_id", "")),
                str(sealed["chain_id"]),
                int(sealed["sequence"]),
                normalise_ts(sealed.get("timestamp")),
                str(sealed.get("kind", "rag_invocation")),
                str(sealed.get("session_id", "")),
                str(sealed["prev_hash"]),
                str(sealed["record_hash"]),
                canonical_json(sealed),
            ),
        )

    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if chain_id is not None:
            clauses.append("chain_id = ?")
            params.append(chain_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT payload FROM {self.table}{where} "
            "ORDER BY workspace_id, chain_id, sequence"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        for (payload,) in rows:
            yield json.loads(payload)

    def get(self, record_id: str, *, workspace_id: str | None = None) -> dict[str, Any] | None:
        sql = f"SELECT payload FROM {self.table} WHERE record_id = ?"
        params: list[Any] = [record_id]
        if workspace_id is not None:
            sql += " AND workspace_id = ?"
            params.append(workspace_id)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return json.loads(row[0]) if row else None

    def count(self, *, workspace_id: str | None = None, chain_id: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if chain_id is not None:
            clauses.append("chain_id = ?")
            params.append(chain_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            row = self._conn.execute(f"SELECT COUNT(*) FROM {self.table}{where}", params).fetchone()
        return int(row[0]) if row else 0

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        cutoff_str = normalise_ts(cutoff)
        # Only a prefix of the chain may go, so stop at the first survivor.
        first_survivor = self._conn.execute(
            f"SELECT sequence FROM {self.table} WHERE workspace_id = ? AND chain_id = ? "
            "AND ts_utc >= ? ORDER BY sequence ASC LIMIT 1",
            (workspace_id, chain_id, cutoff_str),
        ).fetchone()
        if first_survivor is None:
            victims_sql = (
                f"SELECT sequence, record_hash FROM {self.table} WHERE workspace_id = ? "
                "AND chain_id = ? ORDER BY sequence DESC LIMIT 1"
            )
            victims_params: tuple[Any, ...] = (workspace_id, chain_id)
            delete_sql = f"DELETE FROM {self.table} WHERE workspace_id = ? AND chain_id = ?"
            delete_params: tuple[Any, ...] = (workspace_id, chain_id)
        else:
            boundary = int(first_survivor[0])
            victims_sql = (
                f"SELECT sequence, record_hash FROM {self.table} WHERE workspace_id = ? "
                "AND chain_id = ? AND sequence < ? ORDER BY sequence DESC LIMIT 1"
            )
            victims_params = (workspace_id, chain_id, boundary)
            delete_sql = (
                f"DELETE FROM {self.table} WHERE workspace_id = ? AND chain_id = ? "
                "AND sequence < ?"
            )
            delete_params = (workspace_id, chain_id, boundary)

        last = self._conn.execute(victims_sql, victims_params).fetchone()
        if last is None:
            return 0, None, ""

        self._conn.execute(f"DROP TRIGGER IF EXISTS {self.table}_no_delete")
        try:
            cur = self._conn.execute(delete_sql, delete_params)
            purged = int(cur.rowcount)
        finally:
            self._conn.execute(self._no_delete_trigger_sql())
        return purged, int(last[0]), str(last[1])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
