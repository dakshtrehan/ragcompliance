"""Retriever-capture regression tests for RAGComplianceHandler.

These tests exist because of the v0.1.4 -> v0.1.5 bug where
``retrieved_chunks`` were silently dropped from the audit record on
langchain-core >=1.3.0. The handler only overrode ``on_chain_start``, so
``on_retriever_start`` fired with a ``parent_run_id`` that was never
recorded in ``_run_parents`` — and the matching ``on_retriever_end``
could not walk up to the tracked root audit state.

Every test here simulates a real LangChain event sequence (start /
retriever_start / retriever_end / llm_start / llm_end / end) and asserts
that the full audit record survives, not just the query and answer.
"""

from __future__ import annotations

import threading
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.handler import RAGComplianceHandler


def _make_handler():
    """Fresh handler with in-memory save() so tests can assert on records."""
    cfg = RAGComplianceConfig(workspace_id="w-retrieval", dev_mode=True)
    h = RAGComplianceHandler(config=cfg, session_id="s-retrieval", billing=None)
    saved: list = []
    h.storage.save = lambda rec: saved.append(rec) or True  # type: ignore
    return h, saved


def _doc(content: str, source: str, chunk_id: str, score: float = 0.5) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "chunk_id": chunk_id, "score": score},
    )


def _drive_lcel_chain(
    handler: RAGComplianceHandler,
    query: str,
    docs: list[Document],
    answer: str,
    *,
    chain_run_id: uuid.UUID | None = None,
    retriever_run_id: uuid.UUID | None = None,
    llm_run_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Simulate the LCEL callback sequence LangChain fires for a real chain:
    start root chain, retriever_start/end with its own run_id parented to
    the root, llm_start/end similarly, then chain_end. Returns the root
    run_id so callers can inspect state if they need to."""
    chain_run_id = chain_run_id or uuid.uuid4()
    retriever_run_id = retriever_run_id or uuid.uuid4()
    llm_run_id = llm_run_id or uuid.uuid4()

    handler.on_chain_start({}, {"query": query}, run_id=chain_run_id, parent_run_id=None)

    # Retriever fires as an inner runnable with its own run_id. The 0.1.5
    # fix records parent_run_id=chain_run_id via on_retriever_start.
    handler.on_retriever_start(
        {}, query, run_id=retriever_run_id, parent_run_id=chain_run_id
    )
    handler.on_retriever_end(docs, run_id=retriever_run_id)

    # LLM fires with its own run_id too.
    handler.on_llm_start(
        {}, [query], run_id=llm_run_id, parent_run_id=chain_run_id
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[Generation(text=answer)]],
            llm_output={"model_name": "test-model"},
        ),
        run_id=llm_run_id,
    )

    handler.on_chain_end({"answer": answer}, run_id=chain_run_id)
    return chain_run_id


def test_retriever_chunks_captured_via_lcel():
    """Baseline regression: single invoke through a realistic LCEL event
    sequence captures all chunks, not an empty list."""
    h, saved = _make_handler()
    docs = [
        _doc("context one", "doc-1", "c1", 0.9),
        _doc("context two", "doc-2", "c2", 0.8),
        _doc("context three", "doc-3", "c3", 0.7),
    ]
    _drive_lcel_chain(h, "what is x?", docs, "x is y", )

    assert len(saved) == 1
    rec = saved[0]
    assert len(rec.retrieved_chunks) == 3, (
        "Expected 3 chunks in audit record; got "
        f"{len(rec.retrieved_chunks)}. This is the v0.1.5 regression."
    )
    assert [c.source_url for c in rec.retrieved_chunks] == ["doc-1", "doc-2", "doc-3"]
    assert [c.chunk_id for c in rec.retrieved_chunks] == ["c1", "c2", "c3"]
    assert all(c.similarity_score is not None for c in rec.retrieved_chunks)
    assert rec.query == "what is x?"
    assert rec.llm_answer == "x is y"


def test_retriever_chunks_captured_via_batch():
    """chain.batch([q1, q2, q3]) — each of the three saved records must
    have its own chunks paired correctly with its own query and answer."""
    h, saved = _make_handler()
    pairs = [
        ("q1", [_doc("c1-a", "d1a", "c1a"), _doc("c1-b", "d1b", "c1b")], "a1"),
        ("q2", [_doc("c2-a", "d2a", "c2a")], "a2"),
        ("q3", [_doc("c3-a", "d3a", "c3a"), _doc("c3-b", "d3b", "c3b"),
                _doc("c3-c", "d3c", "c3c")], "a3"),
    ]

    # Interleave as LangChain's batch() would: all chain_starts, then
    # retriever/llm events, then all chain_ends.
    chain_ids = [uuid.uuid4() for _ in pairs]
    retriever_ids = [uuid.uuid4() for _ in pairs]
    llm_ids = [uuid.uuid4() for _ in pairs]

    for cid, (q, _, _) in zip(chain_ids, pairs):
        h.on_chain_start({}, {"query": q}, run_id=cid, parent_run_id=None)

    for cid, rid, (q, docs, _) in zip(chain_ids, retriever_ids, pairs):
        h.on_retriever_start({}, q, run_id=rid, parent_run_id=cid)
        h.on_retriever_end(docs, run_id=rid)

    for cid, lid, (q, _, ans) in zip(chain_ids, llm_ids, pairs):
        h.on_llm_start({}, [q], run_id=lid, parent_run_id=cid)
        h.on_llm_end(
            LLMResult(
                generations=[[Generation(text=ans)]],
                llm_output={"model_name": "m"},
            ),
            run_id=lid,
        )

    for cid, (_, _, ans) in zip(chain_ids, pairs):
        h.on_chain_end({"answer": ans}, run_id=cid)

    assert len(saved) == 3
    by_query = {rec.query: rec for rec in saved}
    for q, docs, ans in pairs:
        rec = by_query[q]
        assert rec.llm_answer == ans, f"q={q!r} paired with wrong answer"
        assert len(rec.retrieved_chunks) == len(docs), (
            f"q={q!r} expected {len(docs)} chunks, got {len(rec.retrieved_chunks)}"
        )
        assert [c.source_url for c in rec.retrieved_chunks] == [
            d.metadata["source"] for d in docs
        ]


def test_retriever_chunks_survive_deep_nesting():
    """LCEL often wraps a retriever inside a dict/map/transform pipeline:
    ``{"ctx": retriever | map_fn | transform_fn, "q": passthrough}``.
    Each wrapping runnable fires its own on_chain_start under the root,
    and the retriever fires on_retriever_start parented to one of those
    inner chains, not the root itself. The audit record must still get
    the chunks."""
    h, saved = _make_handler()
    root = uuid.uuid4()
    outer_wrap = uuid.uuid4()      # dict-builder runnable
    inner_pipe = uuid.uuid4()      # retriever | map_fn | transform_fn
    retriever = uuid.uuid4()
    map_fn = uuid.uuid4()
    transform_fn = uuid.uuid4()
    llm = uuid.uuid4()

    h.on_chain_start({}, {"query": "deep"}, run_id=root, parent_run_id=None)
    h.on_chain_start({}, {}, run_id=outer_wrap, parent_run_id=root)
    h.on_chain_start({}, {}, run_id=inner_pipe, parent_run_id=outer_wrap)

    # Retriever parented to inner_pipe, not to root.
    h.on_retriever_start({}, "deep", run_id=retriever, parent_run_id=inner_pipe)
    h.on_retriever_end(
        [_doc("nested-chunk", "nested-src", "nc1", 0.95)],
        run_id=retriever,
    )

    h.on_chain_start({}, {}, run_id=map_fn, parent_run_id=inner_pipe)
    h.on_chain_end({}, run_id=map_fn)
    h.on_chain_start({}, {}, run_id=transform_fn, parent_run_id=inner_pipe)
    h.on_chain_end({}, run_id=transform_fn)
    h.on_chain_end({}, run_id=inner_pipe)
    h.on_chain_end({}, run_id=outer_wrap)

    h.on_llm_start({}, ["deep"], run_id=llm, parent_run_id=root)
    h.on_llm_end(
        LLMResult(
            generations=[[Generation(text="deep answer")]],
            llm_output={"model_name": "m"},
        ),
        run_id=llm,
    )
    h.on_chain_end({"answer": "deep answer"}, run_id=root)

    assert len(saved) == 1
    rec = saved[0]
    assert len(rec.retrieved_chunks) == 1
    assert rec.retrieved_chunks[0].source_url == "nested-src"
    assert rec.retrieved_chunks[0].chunk_id == "nc1"


def test_llm_start_parent_registered():
    """Regression guard: the llm_answer path happens to have an
    on_chain_end fallback that coerces outputs. But if someone later
    removes that fallback, we want to catch the silent drop immediately
    via on_llm_start parent registration. This test asserts the llm-end
    path — specifically model_name — works even with no chain-outputs
    fallback."""
    h, saved = _make_handler()
    root = uuid.uuid4()
    llm = uuid.uuid4()

    h.on_chain_start({}, {"query": "q"}, run_id=root, parent_run_id=None)
    h.on_llm_start({}, ["q"], run_id=llm, parent_run_id=root)
    h.on_llm_end(
        LLMResult(
            generations=[[Generation(text="captured-via-llm-end")]],
            llm_output={"model_name": "specific-model-name"},
        ),
        run_id=llm,
    )
    # No outputs dict — on_chain_end falls back to state.llm_answer, which
    # is only populated if on_llm_end's parent resolution worked.
    h.on_chain_end({}, run_id=root)

    assert len(saved) == 1
    rec = saved[0]
    assert rec.llm_answer == "captured-via-llm-end"
    assert rec.model_name == "specific-model-name"


def test_concurrent_invokes_all_have_chunks():
    """10 threads, shared handler, each runs its own chain. Every saved
    record must have a matching query, answer, AND its own chunks —
    regression guard combining the v0.1.4 batch fix with the v0.1.5
    retriever fix."""
    h, saved = _make_handler()
    lock = threading.Lock()
    thread_safe_saved: list = []
    h.storage.save = lambda rec: (  # type: ignore
        lock.acquire(),
        thread_safe_saved.append(rec),
        lock.release(),
        True,
    )[-1]

    def run_one(i: int) -> None:
        docs = [
            _doc(f"c{i}-a", f"src-{i}-a", f"cid-{i}-a", 0.9),
            _doc(f"c{i}-b", f"src-{i}-b", f"cid-{i}-b", 0.8),
        ]
        _drive_lcel_chain(h, f"q-{i}", docs, f"a-{i}")

    threads = [threading.Thread(target=run_one, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(thread_safe_saved) == 10
    for rec in thread_safe_saved:
        i = rec.query.split("-", 1)[1]
        assert rec.llm_answer == f"a-{i}", (
            f"Query/answer mismatch for {rec.query!r}: got answer {rec.llm_answer!r}"
        )
        assert len(rec.retrieved_chunks) == 2, (
            f"Thread {i} lost chunks: got {len(rec.retrieved_chunks)} of 2"
        )
        assert rec.retrieved_chunks[0].source_url == f"src-{i}-a"
        assert rec.retrieved_chunks[1].source_url == f"src-{i}-b"
