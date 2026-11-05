"""AuditStorage façade: sink selection, async queue, dropped counter, reads."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pytest

from ragcompliance import AuditRecord, AuditStorage, RAGComplianceConfig, RetentionPolicy
from ragcompliance.chain import content_signature
from ragcompliance.sinks import MemorySink
from ragcompliance.storage import ConfigurationError


def _record(i: int = 1, ws: str = "ws") -> AuditRecord:
    r = AuditRecord(session_id="s", workspace_id=ws, query=f"q{i}", llm_answer=f"a{i}")
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    return r


class TestSinkSelection:
    def test_explicit_sink_wins(self):
        sink = MemorySink()
        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", async_writes=False), sink=sink)
        assert st.sink is sink

    def test_sink_url(self, tmp_path):
        st = AuditStorage(RAGComplianceConfig(sink_url=f"sqlite:///{tmp_path}/a.db", async_writes=False))
        assert st.sink.name == "sqlite"
        st.close()

    def test_dev_mode_uses_memory_and_prints(self, capsys):
        st = AuditStorage(RAGComplianceConfig(dev_mode=True))
        assert st.sink.name == "memory" and st._queue is None  # dev mode is synchronous
        assert st.save(_record()) is True
        out = capsys.readouterr().out
        assert "[RAGCompliance DEV] Audit record:" in out and '"record_hash"' in out
        assert st.count() == 1 and st.verify().ok

    def test_no_sink_warns_and_returns_false(self, caplog):
        st = AuditStorage(RAGComplianceConfig())
        assert st.sink is None
        with caplog.at_level(logging.WARNING, logger="ragcompliance.storage"):
            assert st.save(_record()) is False
            assert st.save(_record()) is False
        assert st.dropped_count == 2
        assert sum("dropping audit record" in r.getMessage() for r in caplog.records) == 1
        assert st.query() == [] and list(st.iter_records()) == [] and st.count() == 0
        assert st.verify().records_checked == 0 and st.purge_expired(RetentionPolicy(days=1)) is None
        assert st.describe()["sink"] is None

    def test_missing_driver_raises_unless_dev_mode(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "supabase":
                raise ImportError("no supabase")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cfg = RAGComplianceConfig(supabase_url="https://x.supabase.co", supabase_key="k")
        with pytest.raises(ConfigurationError, match="ragcompliance\\[supabase\\]"):
            AuditStorage(cfg)
        cfg.dev_mode = True
        st = AuditStorage(cfg)
        assert st.sink.name == "memory"

    def test_missing_driver_for_sink_url(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "boto3":
                raise ImportError("no boto3")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ConfigurationError):
            AuditStorage(RAGComplianceConfig(sink_url="s3://b/p"))
        st = AuditStorage(RAGComplianceConfig(sink_url="s3://b/p", dev_mode=True))
        assert st.sink.name == "memory"

    def test_disabled(self):
        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", enabled=False, async_writes=False))
        assert st.save(_record()) is True and st.count() == 0

    def test_client_shim(self):
        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", async_writes=False))
        assert st._client is None


class TestAsyncQueue:
    def test_async_save_and_flush(self):
        st = AuditStorage(RAGComplianceConfig(sink_url="memory://"))
        assert st._queue is not None
        rec = _record()
        assert st.save(rec) is True
        assert st.flush(timeout=5) is True
        assert st.count() == 1 and rec.sequence == 1  # sealed by the worker
        st.close()
        assert st._queue is None

    def test_overflow_drop(self):
        class Slow(MemorySink):
            def _write_locked(self, sealed):
                time.sleep(0.2)
                super()._write_locked(sealed)

        cfg = RAGComplianceConfig(sink_url="memory://", async_max_queue=1)
        st = AuditStorage(cfg, sink=Slow())
        results = [st.save(_record(i)) for i in range(4)]
        assert False in results and st.dropped_count >= 1
        st.flush(timeout=5)
        st.close()

    def test_overflow_block(self):
        class Slow(MemorySink):
            def _write_locked(self, sealed):
                time.sleep(0.05)
                super()._write_locked(sealed)

        cfg = RAGComplianceConfig(sink_url="memory://", async_max_queue=1, async_overflow="block")
        st = AuditStorage(cfg, sink=Slow())
        assert all(st.save(_record(i)) for i in range(4))
        st.flush(timeout=5)
        assert st.count() == 4 and st.dropped_count == 0
        st.close()

    def test_sink_failure_counts_drop(self, caplog):
        class Broken(MemorySink):
            def _write_locked(self, sealed):
                raise RuntimeError("disk full")

        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", async_writes=False), sink=Broken())
        with caplog.at_level(logging.ERROR, logger="ragcompliance.storage"):
            assert st.save(_record()) is False
        assert st.dropped_count == 1 and "disk full" in caplog.text

    def test_worker_survives_exception(self):
        class Flaky(MemorySink):
            calls = 0

            def _write_locked(self, sealed):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("transient")
                super()._write_locked(sealed)

        st = AuditStorage(RAGComplianceConfig(sink_url="memory://"), sink=Flaky())
        st.save(_record(1))
        st.save(_record(2))
        st.flush(timeout=5)
        assert st.count() == 1 and st.dropped_count == 1
        st.close()

    def test_shutdown_timeout_warns(self, caplog):
        class Stuck(MemorySink):
            def _write_locked(self, sealed):
                time.sleep(0.5)
                super()._write_locked(sealed)

        cfg = RAGComplianceConfig(sink_url="memory://", async_shutdown_timeout=0.05, async_max_queue=100)
        st = AuditStorage(cfg, sink=Stuck())
        for i in range(3):
            st.save(_record(i))
        with caplog.at_level(logging.WARNING, logger="ragcompliance.storage"):
            st._shutdown()
        assert "shutdown timeout" in caplog.text

    def test_close_closes_sink_even_if_it_raises(self):
        class Grumpy(MemorySink):
            def close(self):
                raise RuntimeError("nope")

        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", async_writes=False), sink=Grumpy())
        st.close()  # must not raise


class TestReads:
    @pytest.fixture
    def st(self):
        st = AuditStorage(RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="ws"))
        for i in range(3):
            r = _record(i)
            r.session_id = "s1" if i < 2 else "s2"
            st.save(r)
        st.save(_record(9, ws="other"))
        return st

    def test_query_newest_first_and_filters(self, st):
        rows = st.query(workspace_id="ws")
        assert [r["query"] for r in rows] == ["q2", "q1", "q0"]
        assert [r["query"] for r in st.query(workspace_id="ws", session_id="s2")] == ["q2"]
        assert len(st.query(limit=2)) == 2

    def test_get_by_id(self, st):
        rec = st.query(workspace_id="ws")[0]
        assert st.get_by_id(rec["id"])["id"] == rec["id"]
        assert st.get_by_id(rec["id"], workspace_id="other") is None

    def test_read_errors_are_contained(self, st, caplog):
        def boom(*a, **k):
            raise RuntimeError("down")

        st.sink.iter_records = boom
        st.sink.get = boom
        with caplog.at_level(logging.ERROR, logger="ragcompliance.storage"):
            assert st.query() == [] and st.get_by_id("x") is None

    def test_verify_and_describe(self, st):
        res = st.verify(workspace_id="ws")
        assert res.ok and res.records_checked == 3
        d = st.describe()
        assert d["sink"]["type"] == "memory" and d["dropped_count"] == 0 and d["async_writes"] is False

    def test_purge_expired(self, st):
        assert st.purge_expired(RetentionPolicy(), workspace_id="ws") is None
        res = st.purge_expired(
            RetentionPolicy(days=0), workspace_id="ws", now=datetime.now(timezone.utc) + timedelta(seconds=5)
        )
        assert res.purged == 3 and st.count(workspace_id="ws") == 1
        assert st.count(workspace_id="other") == 1
        assert st.verify(workspace_id="ws").ok
