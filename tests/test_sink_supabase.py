"""SupabaseSink tests against a fake supabase-py client (both table layouts)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ragcompliance.chain import content_signature, verify_records
from ragcompliance.config import RAGComplianceConfig
from ragcompliance.models import AuditRecord
from ragcompliance.retention import PurgeNotSupported
from ragcompliance.sinks import make_sink
from ragcompliance.sinks.supabase import (
    LEGACY_COLUMNS,
    NESTED_KEY,
    SupabaseSink,
    pack_row,
    unpack_row,
)


class FakeQuery:
    """Enough of the PostgREST builder chain for the sink."""

    def __init__(self, table: FakeTable):
        self.t = table
        self.filters: list = []
        self.order_by: list = []
        self._range = None
        self._limit = None
        self._delete = False

    def select(self, *a, **k):
        return self

    def insert(self, row):
        self.t.pending_insert = row
        return self

    def delete(self):
        self._delete = True
        return self

    def eq(self, col, val):
        self.filters.append(("eq", col, val))
        return self

    def lte(self, col, val):
        self.filters.append(("lte", col, val))
        return self

    def order(self, col, desc=False):
        self.order_by.append((col, desc))
        return self

    def limit(self, n):
        self._limit = n
        return self

    def range(self, a, b):
        self._range = (a, b)
        return self

    def execute(self):
        if self.t.pending_insert is not None:
            row = self.t.pending_insert
            self.t.pending_insert = None
            if self.t.fail_insert:
                exc = self.t.fail_insert
                self.t.fail_insert = None
                raise exc
            if self.t.unique_chain and any(
                r.get("workspace_id") == row.get("workspace_id")
                and r.get("chain_id") == row.get("chain_id")
                and r.get("sequence") == row.get("sequence")
                for r in self.t.rows
            ):
                raise RuntimeError("23505 duplicate key value")
            self.t.rows.append(dict(row))
            return SimpleNamespace(data=[row])
        rows = list(self.t.rows)
        for op, col, val in self.filters:
            if op == "eq":
                rows = [r for r in rows if r.get(col) == val]
            elif op == "lte":
                rows = [r for r in rows if r.get(col) is not None and r.get(col) <= val]
        if self._delete:
            keep = [r for r in self.t.rows if r not in rows]
            self.t.rows = keep
            return SimpleNamespace(data=rows)
        for col, desc in reversed(self.order_by):
            rows.sort(key=lambda r: (r.get(col) is None, r.get(col)), reverse=desc)
        if self._range:
            a, b = self._range
            rows = rows[a : b + 1]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeTable:
    def __init__(self, unique_chain: bool):
        self.rows: list[dict] = []
        self.pending_insert = None
        self.fail_insert: Exception | None = None
        self.unique_chain = unique_chain


class FakeClient:
    def __init__(self, unique_chain: bool = False):
        self.tables: dict[str, FakeTable] = {}
        self.unique_chain = unique_chain

    def table(self, name):
        t = self.tables.setdefault(name, FakeTable(self.unique_chain))
        return FakeQuery(t)


def _rec(i: int, ts: datetime | None = None) -> dict:
    r = AuditRecord(session_id="s", workspace_id="ws", query=f"q{i}", llm_answer=f"a{i}", actor={"id": "u"})
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    if ts:
        r.timestamp = ts
    return r.to_dict()


class TestPackUnpack:
    def test_layout1_nests_v2_fields(self):
        sealed = {**_rec(1), "chain_id": "c", "sequence": 3, "prev_hash": "p", "record_hash": "h", "record_hmac": ""}
        row = pack_row(sealed, 1)
        assert set(row) == set(LEGACY_COLUMNS)
        assert row["extra"][NESTED_KEY]["sequence"] == 3 and row["extra"][NESTED_KEY]["actor"] == {"id": "u"}
        back = unpack_row(row)
        assert back["sequence"] == 3 and back["actor"] == {"id": "u"} and NESTED_KEY not in back["extra"]

    def test_layout2_uses_columns(self):
        sealed = {**_rec(1), "chain_id": "c", "sequence": 3, "prev_hash": "p", "record_hash": "h", "record_hmac": ""}
        row = pack_row(sealed, 2)
        assert row["sequence"] == 3 and NESTED_KEY not in row["extra"]
        assert unpack_row(row)["chain_id"] == "c"

    def test_unpack_legacy_row_without_nested(self):
        row = {k: v for k, v in _rec(1).items() if k in LEGACY_COLUMNS}
        rec = unpack_row(row)
        assert rec["schema_version"] == 1 and rec["sequence"] is None and rec["kind"] == "rag_invocation"


@pytest.mark.parametrize("layout", [1, 2])
class TestSupabaseSink:
    def test_append_and_iter(self, layout):
        client = FakeClient(unique_chain=layout == 2)
        sink = SupabaseSink(client=client, schema_version=layout)
        s1 = sink.append(_rec(1))
        s2 = sink.append(_rec(2))
        assert s2["prev_hash"] == s1["record_hash"]
        recs = list(sink.iter_records(workspace_id="ws"))
        assert [r["sequence"] for r in recs] == [1, 2]
        assert verify_records(recs).ok
        assert sink.get(s1["id"], workspace_id="ws")["query"] == "q1"
        assert sink.get("nope") is None
        assert list(sink.iter_records(chain_id="other")) == []

    def test_tail_read_from_store(self, layout):
        client = FakeClient(unique_chain=layout == 2)
        a = SupabaseSink(client=client, schema_version=layout)
        a.append(_rec(1))
        b = SupabaseSink(client=client, schema_version=layout)
        assert b.append(_rec(2))["sequence"] == 2

    def test_describe(self, layout):
        sink = SupabaseSink(client=FakeClient(), schema_version=layout)
        assert sink.describe()["schema_version"] == layout


class TestSupabaseSpecifics:
    def test_layout1_tail_ignores_other_chain(self):
        client = FakeClient()
        a = SupabaseSink(client=client, schema_version=1, chain_id="w1")
        a.append(_rec(1))
        b = SupabaseSink(client=client, schema_version=1, chain_id="w2")
        assert b.append(_rec(2))["sequence"] == 1

    def test_layout1_tail_ignores_unchained_rows(self):
        client = FakeClient()
        client.table("rag_audit_logs").t.rows.append({k: v for k, v in _rec(0).items() if k in LEGACY_COLUMNS})
        sink = SupabaseSink(client=client, schema_version=1)
        assert sink.append(_rec(1))["sequence"] == 1

    def test_layout2_retries_on_duplicate_sequence(self):
        client = FakeClient(unique_chain=True)
        a = SupabaseSink(client=client, schema_version=2)
        b = SupabaseSink(client=client, schema_version=2)
        a.append(_rec(1))
        b.append(_rec(2))  # b reads tail 1 -> writes 2
        a.append(_rec(3))  # a has stale cached tail 1 -> duplicate -> retry -> 3
        recs = list(a.iter_records(workspace_id="ws"))
        assert [r["sequence"] for r in recs] == [1, 2, 3]
        assert verify_records(recs).ok

    def test_non_duplicate_error_propagates(self):
        client = FakeClient(unique_chain=True)
        sink = SupabaseSink(client=client, schema_version=2)
        client.table("rag_audit_logs").t.fail_insert = RuntimeError("network down")
        with pytest.raises(RuntimeError, match="network down"):
            sink.append(_rec(1))

    def test_pagination(self):
        client = FakeClient()
        sink = SupabaseSink(client=client, schema_version=2)
        from ragcompliance.sinks import supabase as mod

        old = mod.PAGE_SIZE
        mod.PAGE_SIZE = 2
        try:
            for i in range(5):
                sink.append(_rec(i))
            assert len(list(sink.iter_records())) == 5
        finally:
            mod.PAGE_SIZE = old

    def test_purge_layout2(self):
        client = FakeClient(unique_chain=True)
        sink = SupabaseSink(client=client, schema_version=2)
        old = datetime.now(timezone.utc) - timedelta(days=40)
        sink.append(_rec(1, old))
        sink.append(_rec(2, old))
        sink.append(_rec(3))
        res = sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws")
        assert res.purged == 2
        left = list(sink.iter_records(workspace_id="ws"))
        assert [r["sequence"] for r in left] == [3, 4] and verify_records(left).ok
        assert sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws").purged == 0

    def test_purge_layout1_not_supported(self):
        sink = SupabaseSink(client=FakeClient(), schema_version=1)
        with pytest.raises(PurgeNotSupported):
            sink.purge_before(datetime.now(timezone.utc), workspace_id="ws")

    def test_requires_credentials_or_client(self):
        with pytest.raises(ValueError):
            SupabaseSink()

    def test_factory_from_config(self):
        cfg = RAGComplianceConfig(
            supabase_url="https://x.supabase.co", supabase_key="k", supabase_schema_version=2, table_name="t"
        )
        sink = make_sink("supabase://", cfg, client=FakeClient())
        assert isinstance(sink, SupabaseSink) and sink.schema_version == 2 and sink.table == "t"

    def test_missing_dependency_message(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "supabase":
                raise ImportError("no supabase")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="ragcompliance\\[supabase\\]"):
            SupabaseSink("https://x", "k")
