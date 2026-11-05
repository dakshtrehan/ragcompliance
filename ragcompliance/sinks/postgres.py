"""PostgreSQL sink (optional extra ``postgres``, uses ``psycopg`` 3).

Same table shape as the SQLite sink. Append-only enforcement is a
``BEFORE UPDATE OR DELETE`` trigger whose function raises unless the
session setting ``ragcompliance.allow_purge`` is ``on``; the purge path
sets it with ``SET LOCAL`` so it applies to that transaction only and
needs no DDL privileges.

Writers are serialised per chain with ``pg_advisory_xact_lock`` on
``hashtext(workspace_id || ':' || chain_id)``, so any number of
processes can append to the same chain without forking it.

A ``connection_factory`` can be injected, which is how the test suite
exercises the sink without a database server.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from ..chain import canonical_json
from .base import AuditSink, normalise_ts
from .sqlite import validate_identifier


class PostgresSink(AuditSink):
    name = "postgres"
    supports_purge = True

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "rag_audit_records",
        chain_id: str = "default",
        hmac_key: bytes | str | None = None,
        create_schema: bool = True,
        connection_factory: Callable[[str], Any] | None = None,
    ):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.dsn = dsn
        self.table = validate_identifier(table)
        if connection_factory is None:
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ImportError(
                    "psycopg is not installed. Install with: pip install 'ragcompliance[postgres]'"
                ) from exc
            connection_factory = psycopg.connect
        self._conn = connection_factory(dsn)
        if create_schema:
            self.ensure_schema()

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"table": self.table, "append_only_trigger": True, "dsn": _redact_dsn(self.dsn)})
        return d

    # ------------------------------------------------------------------ #
    # Schema                                                             #
    # ------------------------------------------------------------------ #

    def schema_sql(self) -> list[str]:
        t = self.table
        return [
            f"""CREATE TABLE IF NOT EXISTS {t} (
                record_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                ts_utc TIMESTAMPTZ NOT NULL,
                kind TEXT NOT NULL,
                session_id TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                payload JSONB NOT NULL,
                UNIQUE (workspace_id, chain_id, sequence)
            )""",
            f"CREATE INDEX IF NOT EXISTS {t}_ws_ts ON {t} (workspace_id, ts_utc)",
            f"CREATE INDEX IF NOT EXISTS {t}_session ON {t} (session_id)",
            f"""CREATE OR REPLACE FUNCTION {t}_append_only() RETURNS trigger AS $$
            BEGIN
                IF current_setting('ragcompliance.allow_purge', true) = 'on' THEN
                    RETURN OLD;
                END IF;
                RAISE EXCEPTION 'ragcompliance audit log is append-only';
            END;
            $$ LANGUAGE plpgsql""",
            f"DROP TRIGGER IF EXISTS {t}_append_only ON {t}",
            f"CREATE TRIGGER {t}_append_only BEFORE UPDATE OR DELETE ON {t} "
            f"FOR EACH ROW EXECUTE FUNCTION {t}_append_only()",
        ]

    def ensure_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                for stmt in self.schema_sql():
                    cur.execute(stmt)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    # ------------------------------------------------------------------ #
    # Locking                                                            #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _locked(self, workspace_id: str, chain_id: str) -> Iterator[None]:
        with self._lock:
            if getattr(self, "_in_txn", False):
                yield
                return
            self._in_txn = True
            cur = self._conn.cursor()
            try:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"{workspace_id}:{chain_id}",),
                )
                self._cur = cur
                yield
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
            finally:
                self._cur = None
                self._in_txn = False
                cur.close()

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        cur = self._cur
        cur.execute(
            f"SELECT sequence, record_hash FROM {self.table} "
            "WHERE workspace_id = %s AND chain_id = %s ORDER BY sequence DESC LIMIT 1",
            (workspace_id, chain_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return int(row[0]), str(row[1])

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        self._cur.execute(
            f"INSERT INTO {self.table} (record_id, workspace_id, chain_id, sequence, ts_utc, "
            "kind, session_id, prev_hash, record_hash, payload) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
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
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        if chain_id is not None:
            clauses.append("chain_id = %s")
            params.append(chain_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT payload FROM {self.table}{where} "
            "ORDER BY workspace_id, chain_id, sequence"
        )
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
                self._conn.commit()
            finally:
                cur.close()
        for (payload,) in rows:
            yield payload if isinstance(payload, dict) else json.loads(payload)

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        cur = self._cur
        cutoff_str = normalise_ts(cutoff)
        cur.execute(
            f"SELECT sequence FROM {self.table} WHERE workspace_id = %s AND chain_id = %s "
            "AND ts_utc >= %s ORDER BY sequence ASC LIMIT 1",
            (workspace_id, chain_id, cutoff_str),
        )
        survivor = cur.fetchone()
        if survivor is None:
            cur.execute(
                f"SELECT sequence, record_hash FROM {self.table} WHERE workspace_id = %s "
                "AND chain_id = %s ORDER BY sequence DESC LIMIT 1",
                (workspace_id, chain_id),
            )
            last = cur.fetchone()
            delete_sql = f"DELETE FROM {self.table} WHERE workspace_id = %s AND chain_id = %s"
            delete_params: tuple[Any, ...] = (workspace_id, chain_id)
        else:
            boundary = int(survivor[0])
            cur.execute(
                f"SELECT sequence, record_hash FROM {self.table} WHERE workspace_id = %s "
                "AND chain_id = %s AND sequence < %s ORDER BY sequence DESC LIMIT 1",
                (workspace_id, chain_id, boundary),
            )
            last = cur.fetchone()
            delete_sql = (
                f"DELETE FROM {self.table} WHERE workspace_id = %s AND chain_id = %s "
                "AND sequence < %s"
            )
            delete_params = (workspace_id, chain_id, boundary)
        if last is None:
            return 0, None, ""
        cur.execute("SET LOCAL ragcompliance.allow_purge = 'on'")
        cur.execute(delete_sql, delete_params)
        purged = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0)
        return purged, int(last[0]), str(last[1])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _redact_dsn(dsn: str) -> str:
    """Hide the password in a DSN for describe()."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
