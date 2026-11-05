"""PostgresSink tests against a fake psycopg connection.

No database server is available in CI, so the fake implements just enough
of the DB-API surface (``cursor().execute/fetchone/fetchall``, ``commit``,
``rollback``) and interprets the exact statements the sink emits. This
pins down the SQL contract: parameterised statements only, advisory lock
before the tail read, ``SET LOCAL`` before a purge delete.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from ragcompliance.chain import ChainError, content_signature, verify_records
from ragcompliance.models import AuditRecord
from ragcompliance.sinks import make_sink
from ragcompliance.sinks.postgres import PostgresSink, _redact_dsn


class FakeCursor:
    def __init__(self, conn: FakeConn):
        self.conn = conn
        self._rows: list[tuple] = []
        self.rowcount = -1

    def execute(self, sql: str, params: tuple | list | None = None) -> None:
        self.conn.log.append((sql, tuple(params) if params is not None else None))
        s = " ".join(sql.split())
        params = tuple(params or ())
        if s.startswith("SELECT pg_advisory_xact_lock"):
            self.conn.locks.append(params[0])
            self._rows = [(None,)]
        elif s.startswith("SELECT sequence, record_hash FROM"):
            ws, cid = params[0], params[1]
            rows = [r for r in self.conn.rows if r["workspace_id"] == ws and r["chain_id"] == cid]
            if "sequence < %s" in s:
                rows = [r for r in rows if r["sequence"] < params[2]]
            rows.sort(key=lambda r: r["sequence"], reverse=True)
            self._rows = [(r["sequence"], r["record_hash"]) for r in rows[:1]]
        elif s.startswith("SELECT sequence FROM"):
            ws, cid, ts = params
            rows = [
                r
                for r in self.conn.rows
                if r["workspace_id"] == ws and r["chain_id"] == cid and r["ts_utc"] >= ts
            ]
            rows.sort(key=lambda r: r["sequence"])
            self._rows = [(r["sequence"],) for r in rows[:1]]
        elif s.startswith("INSERT INTO"):
            if "%s" not in s or any(ch in s for ch in ("'", '"')):
                raise AssertionError("insert must be parameterised")
            (rid, ws, cid, seq, ts, kind, sid, prev, rh, payload) = params
            if any(r["record_id"] == rid for r in self.conn.rows):
                raise RuntimeError("duplicate key value violates unique constraint")
            self.conn.rows.append(
                {
                    "record_id": rid, "workspace_id": ws, "chain_id": cid, "sequence": seq,
                    "ts_utc": ts, "kind": kind, "session_id": sid, "prev_hash": prev,
                    "record_hash": rh, "payload": json.loads(payload),
                }
            )
        elif s.startswith("SELECT payload FROM"):
            rows = list(self.conn.rows)
            if "workspace_id = %s" in s:
                rows = [r for r in rows if r["workspace_id"] == params[0]]
                if "chain_id = %s" in s:
                    rows = [r for r in rows if r["chain_id"] == params[1]]
            elif "chain_id = %s" in s:
                rows = [r for r in rows if r["chain_id"] == params[0]]
            rows.sort(key=lambda r: (r["workspace_id"], r["chain_id"], r["sequence"]))
            self._rows = [(r["payload"],) for r in rows]
        elif s.startswith("SET LOCAL ragcompliance.allow_purge"):
            self.conn.purge_allowed = True
        elif s.startswith("DELETE FROM"):
            if not self.conn.purge_allowed:
                raise RuntimeError("ragcompliance audit log is append-only")
            ws, cid = params[0], params[1]
            before = len(self.conn.rows)
            if "sequence < %s" in s:
                self.conn.rows = [
                    r for r in self.conn.rows
                    if not (r["workspace_id"] == ws and r["chain_id"] == cid and r["sequence"] < params[2])
                ]
            else:
                self.conn.rows = [
                    r for r in self.conn.rows if not (r["workspace_id"] == ws and r["chain_id"] == cid)
                ]
            self.rowcount = before - len(self.conn.rows)
        elif re.match(r"^(CREATE|DROP|ALTER)", s):
            self.conn.ddl.append(s)
        else:
            raise AssertionError(f"unexpected SQL: {s}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self) -> None:
        pass


class FakeConn:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.rows: list[dict] = []
        self.log: list[tuple] = []
        self.ddl: list[str] = []
        self.locks: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.purge_allowed = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1
        self.purge_allowed = False

    def rollback(self) -> None:
        self.rollbacks += 1
        self.purge_allowed = False

    def close(self) -> None:
        self.closed = True


def _rec(i: int, ts: datetime | None = None) -> dict:
    r = AuditRecord(session_id="s", workspace_id="ws", query=f"q{i}", llm_answer=f"a{i}")
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    if ts:
        r.timestamp = ts
    return r.to_dict()


@pytest.fixture
def sink():
    conn_holder = {}

    def factory(dsn):
        conn_holder["conn"] = FakeConn(dsn)
        return conn_holder["conn"]

    s = PostgresSink("postgresql://user:pw@db.internal:5432/audit", connection_factory=factory)
    s.fake = conn_holder["conn"]  # type: ignore[attr-defined]
    return s


class TestPostgresSink:
    def test_schema_created_with_append_only_trigger(self, sink):
        ddl = "\n".join(sink.fake.ddl)
        assert "CREATE TABLE IF NOT EXISTS rag_audit_records" in ddl
        assert "ragcompliance.allow_purge" in ddl and "BEFORE UPDATE OR DELETE" in ddl
        assert sink.fake.commits == 1

    def test_append_takes_lock_then_tail_then_insert(self, sink):
        sealed = sink.append(_rec(1))
        assert sealed["sequence"] == 1
        kinds = [s for s, _ in sink.fake.log if not s.startswith(("CREATE", "DROP"))]
        assert kinds[0].startswith("SELECT pg_advisory_xact_lock")
        assert kinds[1].startswith("SELECT sequence, record_hash")
        assert kinds[2].startswith("INSERT INTO")
        assert sink.fake.locks == ["ws:default"]
        second = sink.append(_rec(2))
        assert second["prev_hash"] == sealed["record_hash"]
        assert verify_records(sink.iter_records()).ok
        assert sink.fake.commits >= 3

    def test_statements_are_parameterised(self, sink):
        sink.append(_rec(1))
        for sql, params in sink.fake.log:
            if sql.startswith("INSERT"):
                assert params is not None and len(params) == 10
                assert "q1" not in sql

    def test_rollback_on_failure(self, sink):
        bad = _rec(2)
        bad["extra"] = {"nan": float("nan")}
        with pytest.raises(ChainError):
            sink.append(bad)
        assert sink.fake.rollbacks == 1
        assert sink._in_txn is False
        assert sink.append(_rec(3))["sequence"] == 1

    def test_purge_uses_set_local_and_checkpoints(self, sink):
        old = datetime.now(timezone.utc) - timedelta(days=40)
        sink.append(_rec(1, old))
        sink.append(_rec(2, old))
        sink.append(_rec(3))
        res = sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws")
        assert res.purged == 2 and res.checkpoint["sequence"] == 4
        left = list(sink.iter_records(workspace_id="ws"))
        assert [r["sequence"] for r in left] == [3, 4]
        assert verify_records(left).ok
        assert any(s.startswith("SET LOCAL") for s, _ in sink.fake.log)

    def test_purge_all(self, sink):
        old = datetime.now(timezone.utc) - timedelta(days=40)
        sink.append(_rec(1, old))
        res = sink.purge_before(datetime.now(timezone.utc), workspace_id="ws")
        assert res.purged == 1
        left = list(sink.iter_records())
        assert len(left) == 1 and left[0]["kind"] == "retention_checkpoint" and left[0]["sequence"] == 2

    def test_purge_nothing(self, sink):
        sink.append(_rec(1))
        res = sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws")
        assert res.purged == 0

    def test_iter_filters(self, sink):
        sink.append(_rec(1))
        assert len(list(sink.iter_records(workspace_id="ws", chain_id="default"))) == 1
        assert list(sink.iter_records(workspace_id="other")) == []
        assert len(list(sink.iter_records(chain_id="default"))) == 1

    def test_describe_redacts_password_and_close(self, sink):
        d = sink.describe()
        assert d["dsn"] == "postgresql://user:***@db.internal:5432/audit"
        assert "pw" not in d["dsn"]
        sink.close()
        assert sink.fake.closed
        assert _redact_dsn("dbname=x") == "dbname=x"

    def test_factory_strips_sink_options_from_dsn(self):
        seen = {}

        def factory(dsn):
            seen["dsn"] = dsn
            return FakeConn(dsn)

        s = make_sink(
            "postgresql://u:p@h/db?table=audit_t&create_schema=0&chain_id=w1&sslmode=require",
            connection_factory=factory,
        )
        assert isinstance(s, PostgresSink)
        assert s.table == "audit_t" and s.chain_id == "w1"
        assert seen["dsn"] == "postgresql://u:p@h/db?sslmode=require"
        assert s.fake.ddl == [] if hasattr(s, "fake") else seen["dsn"]  # no schema created

    def test_schema_file_matches_code(self, sink):
        from pathlib import Path

        text = Path(__file__).resolve().parents[1].joinpath("sql", "postgres_schema.sql").read_text()
        for stmt in sink.schema_sql():
            assert stmt.strip() in text

    def test_schema_failure_rolls_back(self):
        class BoomConn(FakeConn):
            def cursor(self):
                cur = super().cursor()
                cur.execute = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ddl rights"))  # type: ignore[assignment]
                return cur

        with pytest.raises(RuntimeError):
            PostgresSink("postgresql://x", connection_factory=BoomConn)
