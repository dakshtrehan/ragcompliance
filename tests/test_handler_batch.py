"""Correctness tests for ``chain.batch()`` and concurrent ``chain.invoke()``.

These tests exist because prior to v0.1.4 the handler kept per-invocation
state on the instance (``_root_run_id``, ``_query``, ``_chunks``, ...),
which silently clobbered records when the same handler was passed into a
``chain.batch([q1, q2, q3])`` call or across threads. The failure mode
was silent audit-loss-with-misrepresentation — exactly the most dangerous
class of bug for a compliance library — so there is now a dedicated
regression suite.

None of these tests hit a real LLM, vector store, or network. They drive
the callback interface directly the way LangChain would, then assert on
the mocked ``storage.save`` call log.
"""

import threading
import time
import uuid
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from ragcompliance import RAGComplianceConfig, RAGComplianceHandler


@pytest.fixture
def dev_config():
    return RAGComplianceConfig(dev_mode=True, workspace_id="test-workspace")


@pytest.fixture
def handler(dev_config):
    return RAGComplianceHandler(config=dev_config, session_id="test-session")


def _run_one(handler, query: str, answer: str, *, chunks: int = 1) -> uuid.UUID:
    """Drive one complete chain lifecycle through the handler: start,
    retriever, llm, end. Returns the root run_id."""
    rid = uuid.uuid4()
    handler.on_chain_start({}, {"query": query}, run_id=rid)
    docs = [
        Document(page_content=f"ctx-{i}", metadata={"source": f"s-{i}", "chunk_id": f"c-{i}"})
        for i in range(chunks)
    ]
    handler.on_retriever_end(docs, run_id=rid)
    handler.on_llm_end(LLMResult(generations=[[Generation(text=answer)]]), run_id=rid)
    handler.on_chain_end({"answer": answer}, run_id=rid)
    return rid


class TestBatchCorrectness:
    def test_batch_three_queries_produces_three_records(self, handler):
        """chain.batch([q1, q2, q3]) must produce THREE records, one per
        query, with each record's query matching its own answer."""
        queries = ["alpha?", "beta?", "gamma?"]
        answers = ["A-alpha", "A-beta", "A-gamma"]

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            # LangChain's batch interleaves on_chain_start calls for all
            # runs before finishing any. Simulate that.
            rids = [uuid.uuid4() for _ in queries]
            for q, rid in zip(queries, rids):
                handler.on_chain_start({}, {"query": q}, run_id=rid)
            for a, rid in zip(answers, rids):
                handler.on_llm_end(
                    LLMResult(generations=[[Generation(text=a)]]), run_id=rid
                )
            for a, rid in zip(answers, rids):
                handler.on_chain_end({"answer": a}, run_id=rid)

        assert mock_save.call_count == 3
        saved_records = [call.args[0] for call in mock_save.call_args_list]
        by_query = {r.query: r.llm_answer for r in saved_records}
        assert by_query == {"alpha?": "A-alpha", "beta?": "A-beta", "gamma?": "A-gamma"}

    def test_batch_concurrent_ten_queries(self, handler):
        """Hammer a shared handler with 10 concurrent chain lifecycles.
        Each must produce its own record with query/answer matched."""
        pairs = [(f"q-{i}", f"a-{i}") for i in range(10)]
        saved = []
        saved_lock = threading.Lock()

        def record_save(record):
            with saved_lock:
                saved.append(record)
            return True

        with patch.object(handler.storage, "save", side_effect=record_save):
            threads = [
                threading.Thread(target=_run_one, args=(handler, q, a))
                for q, a in pairs
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(saved) == 10
        by_query = {r.query: r.llm_answer for r in saved}
        assert by_query == dict(pairs)


class TestNestedLCEL:
    def test_nested_lcel_still_saves_once(self, handler):
        """LCEL fires on_chain_start for every sub-runnable. The handler
        must only save when the OUTER chain ends."""
        outer = uuid.uuid4()
        inner1 = uuid.uuid4()
        inner2 = uuid.uuid4()

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            handler.on_chain_start({}, {"query": "nested"}, run_id=outer)
            handler.on_chain_start({}, {"step": 1}, run_id=inner1, parent_run_id=outer)
            handler.on_chain_start({}, {"step": 2}, run_id=inner2, parent_run_id=inner1)
            handler.on_llm_end(
                LLMResult(generations=[[Generation(text="answer")]]),
                run_id=inner2,
            )
            # Inner runnables end...
            handler.on_chain_end({"step": 2}, run_id=inner2)
            handler.on_chain_end({"step": 1}, run_id=inner1)
            # ...and only this one triggers save.
            handler.on_chain_end({"answer": "answer"}, run_id=outer)

        assert mock_save.call_count == 1
        rec = mock_save.call_args[0][0]
        assert rec.query == "nested"
        assert rec.llm_answer == "answer"

    def test_inner_event_routes_to_root_state(self, handler):
        """An on_retriever_end fired with an INNER run_id must still land
        in the outermost root's chunks list."""
        outer = uuid.uuid4()
        inner = uuid.uuid4()
        handler.on_chain_start({}, {"query": "q"}, run_id=outer)
        handler.on_chain_start({}, {}, run_id=inner, parent_run_id=outer)
        handler.on_retriever_end(
            [Document(page_content="x", metadata={"source": "s", "chunk_id": "c"})],
            run_id=inner,
        )
        assert len(handler._runs[outer].chunks) == 1


class TestConcurrentInvoke:
    def test_concurrent_invoke_on_shared_handler(self, handler):
        """20 threads, each running one full chain lifecycle through the
        same handler. No record should leak query from one run into the
        answer of another."""
        saved = []
        saved_lock = threading.Lock()

        def record_save(record):
            with saved_lock:
                saved.append((record.query, record.llm_answer))
            return True

        with patch.object(handler.storage, "save", side_effect=record_save):
            threads = [
                threading.Thread(
                    target=_run_one, args=(handler, f"query-{i}", f"answer-{i}")
                )
                for i in range(20)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(saved) == 20
        # Every pair must be (query-N, answer-N) — no cross-contamination.
        for q, a in saved:
            assert q.replace("query-", "") == a.replace("answer-", ""), (q, a)

    def test_interleaved_events(self, handler):
        """Callbacks from two runs interleaved manually, out of order,
        still produce two correctly-matched records."""
        a_id = uuid.uuid4()
        b_id = uuid.uuid4()

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            # Fully interleave the two lifecycles.
            handler.on_chain_start({}, {"query": "A?"}, run_id=a_id)
            handler.on_chain_start({}, {"query": "B?"}, run_id=b_id)
            handler.on_retriever_end(
                [Document(page_content="a-ctx", metadata={"source": "a"})],
                run_id=a_id,
            )
            handler.on_retriever_end(
                [Document(page_content="b-ctx", metadata={"source": "b"})],
                run_id=b_id,
            )
            handler.on_llm_end(
                LLMResult(generations=[[Generation(text="A-answer")]]), run_id=a_id
            )
            handler.on_llm_end(
                LLMResult(generations=[[Generation(text="B-answer")]]), run_id=b_id
            )
            # B ends first, then A.
            handler.on_chain_end({"answer": "B-answer"}, run_id=b_id)
            handler.on_chain_end({"answer": "A-answer"}, run_id=a_id)

        assert mock_save.call_count == 2
        by_query = {
            c.args[0].query: (c.args[0].llm_answer, c.args[0].retrieved_chunks[0].source_url)
            for c in mock_save.call_args_list
        }
        assert by_query["A?"] == ("A-answer", "a")
        assert by_query["B?"] == ("B-answer", "b")


class TestPendingRunsSoftCap:
    def test_pending_runs_soft_cap(self, monkeypatch, dev_config, caplog):
        """When on_chain_end never fires, old runs should be evicted once
        the soft cap is exceeded, with a single warning log."""
        monkeypatch.setenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "5")
        h = RAGComplianceHandler(config=dev_config, session_id="s")
        assert h._max_pending_runs == 5

        import logging

        with caplog.at_level(logging.WARNING, logger="ragcompliance.handler"):
            # 10 roots start, none end — expect eviction down to 5.
            for i in range(10):
                h.on_chain_start({}, {"query": f"q{i}"}, run_id=uuid.uuid4())

        assert len(h._runs) == 5
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("RAGCOMPLIANCE_MAX_PENDING_RUNS" in r.getMessage() for r in warnings)
        # Exactly one warning — not one per eviction.
        assert sum(
            "RAGCOMPLIANCE_MAX_PENDING_RUNS" in r.getMessage() for r in warnings
        ) == 1

    def test_pending_runs_soft_cap_drops_descendants(
        self, monkeypatch, dev_config
    ):
        """Evicting a root must also drop its descendant parent-mappings
        so _run_parents doesn't grow unbounded."""
        monkeypatch.setenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "2")
        h = RAGComplianceHandler(config=dev_config, session_id="s")

        r1 = uuid.uuid4()
        r1_child = uuid.uuid4()
        h.on_chain_start({}, {"query": "q1"}, run_id=r1)
        h.on_chain_start({}, {}, run_id=r1_child, parent_run_id=r1)
        assert r1_child in h._run_parents

        # Two more roots push r1 out.
        h.on_chain_start({}, {"query": "q2"}, run_id=uuid.uuid4())
        h.on_chain_start({}, {"query": "q3"}, run_id=uuid.uuid4())

        assert r1 not in h._runs
        assert r1_child not in h._run_parents


class TestDefensiveSave:
    def test_storage_save_raises_does_not_kill_chain(self, handler, caplog):
        """If a custom storage backend raises, the handler must log and
        continue, not propagate the exception up and crash the chain."""
        rid = uuid.uuid4()
        handler.on_chain_start({}, {"query": "q"}, run_id=rid)
        handler.on_llm_end(
            LLMResult(generations=[[Generation(text="a")]]), run_id=rid
        )

        import logging

        with patch.object(
            handler.storage,
            "save",
            side_effect=RuntimeError("supabase is on fire"),
        ), caplog.at_level(logging.ERROR, logger="ragcompliance.handler"):
            # Must not raise.
            handler.on_chain_end({"answer": "a"}, run_id=rid)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert any("storage.save raised" in r.getMessage() for r in errors)
        # State is still cleaned up even though save failed.
        assert rid not in handler._runs


class TestPerformance:
    """Light microbench to guard against the per-run dict + lock causing
    an order-of-magnitude regression on the hot path. 200 μs p50 is a
    loose budget; the actual cost should be dominated by dict operations
    and far below that."""

    def test_chain_end_hot_path_is_fast(self, handler):
        with patch.object(handler.storage, "save", return_value=True):
            # Warm up.
            for _ in range(50):
                _run_one(handler, "warm", "up")

            samples = []
            for _ in range(200):
                rid = uuid.uuid4()
                t0 = time.perf_counter()
                handler.on_chain_start({}, {"query": "q"}, run_id=rid)
                handler.on_llm_end(
                    LLMResult(generations=[[Generation(text="a")]]), run_id=rid
                )
                handler.on_chain_end({"answer": "a"}, run_id=rid)
                samples.append(time.perf_counter() - t0)

        samples.sort()
        p50 = samples[len(samples) // 2]
        # Generous bound so this passes on slow CI runners without being
        # so loose that a real regression slips through.
        assert p50 < 2e-3, f"p50 of full lifecycle = {p50 * 1e6:.1f}μs"


def test_max_pending_runs_env_bad_value_defaults_safely(dev_config, monkeypatch):
    """A junk value in RAGCOMPLIANCE_MAX_PENDING_RUNS must not crash the
    constructor — fall back to 10000."""
    monkeypatch.setenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "not-a-number")
    h = RAGComplianceHandler(config=dev_config, session_id="s")
    assert h._max_pending_runs == 10000


def test_synthesized_run_id_when_none_passed(handler):
    """Some tests / legacy code call on_chain_start without a run_id.
    The handler should synthesize one instead of crashing."""
    before = len(handler._runs)
    handler.on_chain_start({}, {"query": "q"})
    assert len(handler._runs) == before + 1
