"""Tests for the local sinks (memory, JSONL, SQLite), the URL factory and
the purge selection helper."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from ragcompliance.chain import GENESIS_HASH, ChainError, content_signature, verify_records
from ragcompliance.models import AuditRecord
from ragcompliance.retention import PurgeNotSupported
from ragcompliance.sinks import JSONLSink, MemorySink, SQLiteSink, make_sink
from ragcompliance.sinks.base import AuditSink, normalise_ts, select_purge_victims
from ragcompliance.sinks.sqlite import validate_identifier


def _rec(i: int, ws: str = "ws", ts: datetime | None = None, chain_id: str | None = None) -> dict:
    r = AuditRecord(session_id="s", workspace_id=ws, query=f"q{i}", llm_answer=f"a{i}")
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    if ts is not None:
        r.timestamp = ts
    d = r.to_dict()
    if chain_id:
        d["chain_id"] = chain_id
    return d


def _old(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


SINK_FACTORIES = {
    "memory": lambda tmp: MemorySink(),
    "jsonl": lambda tmp: JSONLSink(tmp / "audit.jsonl"),
    "sqlite": lambda tmp: SQLiteSink(str(tmp / "audit.db")),
}


@pytest.fixture(params=sorted(SINK_FACTORIES))
def sink(request, tmp_path):
    s = SINK_FACTORIES[request.param](tmp_path)
    yield s
    s.close()


class TestCommonBehaviour:
    def test_append_seals_in_order(self, sink: AuditSink):
        s1 = sink.append(_rec(1))
        s2 = sink.append(_rec(2))
        assert s1["sequence"] == 1 and s1["prev_hash"] == GENESIS_HASH
        assert s2["sequence"] == 2 and s2["prev_hash"] == s1["record_hash"]
        assert sink.tail("ws") == (2, s2["record_hash"])
        assert verify_records(sink.iter_records()).ok

    def test_chains_are_per_workspace_and_chain_id(self, sink: AuditSink):
        sink.append(_rec(1, ws="a"))
        sink.append(_rec(2, ws="b"))
        sink.append(_rec(3, ws="a", chain_id="other"))
        a = sink.append(_rec(4, ws="a"))
        assert a["sequence"] == 2
        assert sink.tail("b") == (1, next(sink.iter_records(workspace_id="b"))["record_hash"])
        assert sink.count(workspace_id="a") == 3
        assert sink.count(workspace_id="a", chain_id="other") == 1
        assert [r["workspace_id"] for r in sink.iter_records()] == ["a", "a", "a", "b"]

    def test_iter_filters_by_window_and_limit(self, sink: AuditSink):
        sink.append(_rec(1, ts=_old(10)))
        sink.append(_rec(2, ts=_old(5)))
        sink.append(_rec(3))
        got = list(sink.iter_records(start=_old(7), end=_old(1)))
        assert [r["query"] for r in got] == ["q2"]
        assert len(list(sink.iter_records(limit=2))) == 2
        naive = datetime.now() - timedelta(days=7)
        assert len(list(sink.iter_records(start=naive))) == 2

    def test_get_by_id(self, sink: AuditSink):
        sealed = sink.append(_rec(1))
        assert sink.get(sealed["id"])["query"] == "q1"
        assert sink.get(sealed["id"], workspace_id="ws")["id"] == sealed["id"]
        assert sink.get(sealed["id"], workspace_id="nope") is None
        assert sink.get("missing") is None

    def test_purge_prefix_and_checkpoint(self, sink: AuditSink):
        sink.append(_rec(1, ts=_old(40)))
        sink.append(_rec(2, ts=_old(35)))
        sink.append(_rec(3, ts=_old(1)))
        result = sink.purge_before(_old(30), workspace_id="ws")
        assert result.purged == 2 and result.last_purged_sequence == 2
        assert result.checkpoint is not None and result.checkpoint["sequence"] == 4
        left = list(sink.iter_records(workspace_id="ws"))
        assert [r["sequence"] for r in left] == [3, 4]
        assert left[1]["kind"] == "retention_checkpoint"
        assert left[1]["extra"]["last_purged_hash"] == result.last_purged_hash
        res = verify_records(left)
        assert res.ok and res.chains[0].anchored_by_checkpoint

    def test_purge_only_prefix_even_if_newer_record_is_old(self, sink: AuditSink):
        # Record 2 is newer than the cutoff, record 3 is older (clock skew):
        # only the prefix before the first survivor may go.
        sink.append(_rec(1, ts=_old(40)))
        sink.append(_rec(2, ts=_old(1)))
        sink.append(_rec(3, ts=_old(40)))
        result = sink.purge_before(_old(30), workspace_id="ws")
        assert result.purged == 1
        assert sink.count(workspace_id="ws") == 3  # 2 survivors + checkpoint

    def test_purge_everything_continues_sequence(self, sink: AuditSink):
        sink.append(_rec(1, ts=_old(40)))
        sink.append(_rec(2, ts=_old(39)))
        result = sink.purge_before(_old(30), workspace_id="ws")
        assert result.purged == 2
        left = list(sink.iter_records(workspace_id="ws"))
        assert len(left) == 1 and left[0]["sequence"] == 3
        assert left[0]["prev_hash"] == result.last_purged_hash
        assert verify_records(left).ok
        nxt = sink.append(_rec(9))
        assert nxt["sequence"] == 4

    def test_purge_nothing(self, sink: AuditSink):
        sink.append(_rec(1))
        result = sink.purge_before(_old(30), workspace_id="ws")
        assert result.purged == 0 and result.checkpoint is None
        assert sink.count() == 1

    def test_describe(self, sink: AuditSink):
        d = sink.describe()
        assert d["type"] == sink.name and d["chain_id"] == "default" and d["hmac"] is False

    def test_concurrent_appends_stay_linear(self, sink: AuditSink):
        def worker(n: int) -> None:
            for i in range(10):
                sink.append(_rec(n * 100 + i))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        res = verify_records(sink.iter_records())
        assert res.ok and res.records_checked == 50
        assert res.chains[0].last_sequence == 50


class TestJSONL:
    def test_file_is_one_canonical_json_per_line(self, tmp_path):
        s = JSONLSink(tmp_path / "a.jsonl", fsync=True)
        s.append(_rec(1))
        s.append(_rec(2))
        lines = (tmp_path / "a.jsonl").read_text().splitlines()
        assert len(lines) == 2 and json.loads(lines[1])["sequence"] == 2
        assert s.describe()["fsync"] is True

    def test_second_instance_continues_chain(self, tmp_path):
        path = tmp_path / "a.jsonl"
        a = JSONLSink(path)
        a.append(_rec(1))
        b = JSONLSink(path)  # another process, same file
        sealed = b.append(_rec(2))
        assert sealed["sequence"] == 2
        # the first instance notices the external append on its next write
        third = a.append(_rec(3))
        assert third["sequence"] == 3 and third["prev_hash"] == sealed["record_hash"]
        assert verify_records(a.iter_records()).ok

    def test_rewritten_file_is_rescanned(self, tmp_path):
        path = tmp_path / "a.jsonl"
        a = JSONLSink(path)
        b = JSONLSink(path)
        a.append(_rec(1, ts=_old(40)))
        a.append(_rec(2, ts=_old(39)))
        b.append(_rec(3))
        a.purge_before(_old(30), workspace_id="ws")
        # b's cached tail is stale and the file shrank; the next append must rescan.
        nxt = b.append(_rec(4))
        assert nxt["sequence"] == 5
        assert verify_records(b.iter_records()).ok

    def test_corrupt_lines_are_skipped(self, tmp_path):
        path = tmp_path / "a.jsonl"
        s = JSONLSink(path)
        s.append(_rec(1))
        with open(path, "a") as fh:
            fh.write("not json\n\n")
        s2 = JSONLSink(path)
        assert s2.append(_rec(2))["sequence"] == 2
        assert len(list(s2.iter_records())) == 2

    def test_missing_file_recreated(self, tmp_path):
        path = tmp_path / "a.jsonl"
        s = JSONLSink(path)
        path.unlink()
        assert s.append(_rec(1))["sequence"] == 1
        assert s._read_all() and len(s._read_all()) == 1
        path.unlink()
        assert s._read_all() == []


class TestSQLite:
    def test_append_only_triggers(self, tmp_path):
        s = SQLiteSink(str(tmp_path / "a.db"))
        s.append(_rec(1))
        conn = sqlite3.connect(str(tmp_path / "a.db"))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE rag_audit_records SET payload = 'x'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM rag_audit_records")
        conn.close()
        # The purge path restores the trigger afterwards.
        s.append(_rec(2, ts=_old(40)))
        s.purge_before(_old(30), workspace_id="ws")  # record 1 is new, so nothing purged
        conn = sqlite3.connect(str(tmp_path / "a.db"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM rag_audit_records")
        conn.close()

    def test_rollback_on_write_failure(self, tmp_path):
        s = SQLiteSink(str(tmp_path / "a.db"))
        s.append(_rec(1))
        bad = _rec(2)
        bad["extra"] = {"nan": float("nan")}
        with pytest.raises(ChainError):
            s.append(bad)
        assert not s._conn.in_transaction
        assert s.append(_rec(3))["sequence"] == 2

    def test_custom_table_and_identifier_validation(self, tmp_path):
        s = SQLiteSink(str(tmp_path / "a.db"), table="audit_x")
        assert s.append(_rec(1))["sequence"] == 1
        assert s.describe()["table"] == "audit_x"
        with pytest.raises(ValueError):
            SQLiteSink(str(tmp_path / "b.db"), table="bad name; drop")
        assert validate_identifier("ok_1") == "ok_1"

    def test_count_and_get_with_filters(self, tmp_path):
        s = SQLiteSink(":memory:")
        s.append(_rec(1, ws="a"))
        s.append(_rec(2, ws="b"))
        assert s.count() == 2 and s.count(workspace_id="a") == 1 and s.count(chain_id="default") == 2
        assert s.get("nope") is None


class TestFactory:
    def test_memory(self):
        assert isinstance(make_sink("memory://"), MemorySink)

    def test_jsonl_paths(self, tmp_path):
        s = make_sink(f"jsonl://{tmp_path}/x.jsonl?fsync=1")
        assert isinstance(s, JSONLSink) and s.fsync is True
        assert str(s.path) == f"{tmp_path}/x.jsonl"
        s2 = make_sink("jsonl://relative/x.jsonl")
        assert str(s2.path) == "relative/x.jsonl"
        import shutil

        shutil.rmtree("relative", ignore_errors=True)

    def test_sqlite(self, tmp_path):
        s = make_sink(f"sqlite:///{tmp_path}/x.db?table=t1&chain_id=writer-7")
        assert isinstance(s, SQLiteSink) and s.table == "t1" and s.chain_id == "writer-7"
        s.close()

    def test_config_supplies_chain_and_hmac(self, tmp_path):
        from ragcompliance.config import RAGComplianceConfig

        cfg = RAGComplianceConfig(chain_id="c9", chain_hmac_key="secret")
        s = make_sink("memory://", cfg)
        assert s.chain_id == "c9" and s.hmac_key == b"secret"
        s2 = make_sink("memory://", cfg, chain_id="override")
        assert s2.chain_id == "override"

    def test_supabase_requires_config(self):
        with pytest.raises(ValueError):
            make_sink("supabase://")

    def test_unknown_scheme(self):
        with pytest.raises(ValueError, match="unknown sink URL"):
            make_sink("ftp://nope")

    def test_lazy_attribute_access(self):
        import ragcompliance.sinks as sinks

        assert sinks.JSONLSink is JSONLSink
        assert sinks.PostgresSink.__name__ == "PostgresSink"
        assert sinks.S3Sink.__name__ == "S3Sink"
        assert sinks.SupabaseSink.__name__ == "SupabaseSink"
        with pytest.raises(AttributeError):
            sinks.Nope  # noqa: B018


class TestHelpers:
    def test_normalise_ts(self):
        assert normalise_ts("2026-01-01T00:00:00Z") == "2026-01-01T00:00:00.000000+00:00"
        assert normalise_ts("2026-01-01T01:00:00+01:00") == "2026-01-01T00:00:00.000000+00:00"

    def test_select_purge_victims_stops_at_first_survivor(self):
        recs = [_rec(1, ts=_old(40)), _rec(2, ts=_old(1)), _rec(3, ts=_old(40))]
        purged, kept = select_purge_victims(recs, _old(30))
        assert [r["query"] for r in purged] == ["q1"]
        assert [r["query"] for r in kept] == ["q2", "q3"]

    def test_base_purge_not_supported(self):
        class NoPurge(MemorySink):
            supports_purge = False

        with pytest.raises(PurgeNotSupported):
            NoPurge().purge_before(_old(1), workspace_id="ws")
