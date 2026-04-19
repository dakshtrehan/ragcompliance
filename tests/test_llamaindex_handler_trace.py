"""Concurrency regression tests for the LlamaIndex handler.

Prior to v0.1.4 the handler kept per-query state on the instance
(``_query``, ``_chunks``, ``_llm_answer``), which made it unsafe to share
across concurrent query engines. This suite verifies the per-trace,
thread-local state design fixes that."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.schema import NodeWithScore, TextNode

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler


def _make_handler() -> LlamaIndexRAGComplianceHandler:
    cfg = RAGComplianceConfig(
        workspace_id="w-llama",
        dev_mode=True,
    )
    return LlamaIndexRAGComplianceHandler(
        config=cfg, session_id="s-concurrent", billing=None
    )


def _drive_one(h, trace_id: str, query: str, answer: str) -> None:
    h.start_trace(trace_id)
    h.on_event_start(CBEventType.QUERY, payload={"query_str": query})
    h.on_event_end(
        CBEventType.RETRIEVE,
        payload={
            "nodes": [
                NodeWithScore(
                    node=TextNode(
                        text=f"{trace_id}-ctx",
                        metadata={"source": f"{trace_id}-src"},
                    ),
                    score=0.5,
                )
            ]
        },
    )

    class _Resp:
        response = answer

    h.on_event_end(CBEventType.SYNTHESIZE, payload={"response": _Resp()})
    h.end_trace(trace_id, {})


def test_concurrent_traces_produce_distinct_records():
    """Ten threads, each running a distinct LlamaIndex trace through the
    same handler. Every saved record must have a matched (query, answer)
    pair — no cross-contamination."""
    h = _make_handler()
    saved = []
    saved_lock = threading.Lock()

    def record_save(record):
        with saved_lock:
            saved.append((record.query, record.llm_answer))
        return True

    h.storage.save = record_save  # type: ignore

    pairs = [(f"trace-{i}", f"q-{i}", f"a-{i}") for i in range(10)]
    threads = [
        threading.Thread(target=_drive_one, args=(h, tid, q, a))
        for tid, q, a in pairs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(saved) == 10
    by_q = dict(saved)
    for _, q, a in pairs:
        assert by_q[q] == a, f"query {q!r} mismatched answer {by_q[q]!r}"


def test_end_trace_cleans_up_pending_state():
    """After end_trace, the trace_id is no longer in ``_traces``."""
    h = _make_handler()
    h.storage.save = lambda rec: True  # type: ignore

    h.start_trace("gone")
    assert "gone" in h._traces
    h.end_trace("gone", {})
    assert "gone" not in h._traces


def test_storage_save_raises_does_not_kill_trace():
    """A custom storage that raises must be caught — the handler logs and
    moves on."""

    h = _make_handler()

    def bad_save(rec):
        raise RuntimeError("storage on fire")

    h.storage.save = bad_save  # type: ignore

    h.start_trace("t")
    h.on_event_start(CBEventType.QUERY, payload={"query_str": "Q?"})

    class _Resp:
        response = "A"

    h.on_event_end(CBEventType.SYNTHESIZE, payload={"response": _Resp()})

    # Must not raise.
    h.end_trace("t", {})
    assert "t" not in h._traces
