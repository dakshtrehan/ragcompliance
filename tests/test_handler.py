"""Tests for the RAGCompliance callback handler."""

import hashlib
import json
import uuid
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult

from ragcompliance import RAGComplianceConfig, RAGComplianceHandler
from ragcompliance.handler import _RunState


@pytest.fixture
def dev_config():
    return RAGComplianceConfig(dev_mode=True, workspace_id="test-workspace")


@pytest.fixture
def handler(dev_config):
    return RAGComplianceHandler(config=dev_config, session_id="test-session")


def _start_root(handler, inputs, run_id=None):
    """Start a root chain run and return its run_id."""
    run_id = run_id or uuid.uuid4()
    handler.on_chain_start({}, inputs, run_id=run_id)
    return run_id


class TestQueryCapture:
    def test_captures_query_key(self, handler):
        rid = _start_root(handler, {"query": "What is RAG?"})
        assert handler._runs[rid].query == "What is RAG?"

    def test_captures_question_key(self, handler):
        rid = _start_root(handler, {"question": "How does this work?"})
        assert handler._runs[rid].query == "How does this work?"

    def test_captures_input_key(self, handler):
        rid = _start_root(handler, {"input": "Explain compliance."})
        assert handler._runs[rid].query == "Explain compliance."

    def test_ignores_unknown_keys(self, handler):
        rid = _start_root(handler, {"irrelevant": "nothing"})
        # Falls back to stringifying the whole dict so we at least have
        # something when the input shape is unexpected.
        assert "irrelevant" in handler._runs[rid].query


class TestRetrieverCapture:
    def test_captures_chunks_with_metadata(self, handler):
        rid = _start_root(handler, {"query": "q"})
        docs = [
            Document(
                page_content="RAG stands for Retrieval Augmented Generation.",
                metadata={"source": "https://example.com/rag", "chunk_id": "abc123", "score": 0.92},
            ),
            Document(
                page_content="Compliance is critical in fintech.",
                metadata={"source": "https://example.com/compliance", "chunk_id": "def456", "score": 0.87},
            ),
        ]
        handler.on_retriever_end(docs, run_id=rid)

        chunks = handler._runs[rid].chunks
        assert len(chunks) == 2
        assert chunks[0].source_url == "https://example.com/rag"
        assert chunks[0].chunk_id == "abc123"
        assert chunks[0].similarity_score == 0.92
        assert chunks[1].similarity_score == 0.87

    def test_handles_missing_metadata_gracefully(self, handler):
        rid = _start_root(handler, {"query": "q"})
        docs = [Document(page_content="Plain content", metadata={})]
        handler.on_retriever_end(docs, run_id=rid)

        chunks = handler._runs[rid].chunks
        assert len(chunks) == 1
        assert chunks[0].source_url == "unknown"
        assert chunks[0].chunk_id == "chunk-0"
        assert chunks[0].similarity_score is None


class TestLLMCapture:
    def test_captures_llm_answer(self, handler):
        rid = _start_root(handler, {"query": "q"})
        generation = Generation(text="RAG improves LLM accuracy using external context.")
        result = LLMResult(generations=[[generation]])
        handler.on_llm_end(result, run_id=rid)
        assert (
            handler._runs[rid].llm_answer
            == "RAG improves LLM accuracy using external context."
        )

    def test_captures_model_name(self, handler):
        rid = _start_root(handler, {"query": "q"})
        generation = Generation(text="Answer.")
        result = LLMResult(generations=[[generation]], llm_output={"model_name": "gpt-4"})
        handler.on_llm_end(result, run_id=rid)
        assert handler._runs[rid].model_name == "gpt-4"


class TestChainSignature:
    def test_signature_is_sha256(self, handler):
        state = _RunState(query="test query", llm_answer="test answer")
        sig = handler._sign_chain(state)

        payload = {"query": "test query", "chunks": [], "answer": "test answer"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert sig == expected

    def test_signature_changes_with_different_content(self, handler):
        state_a = _RunState(query="query A", llm_answer="answer A")
        state_b = _RunState(query="query B", llm_answer="answer B")
        assert handler._sign_chain(state_a) != handler._sign_chain(state_b)


class TestChainEnd:
    """The handler finalizes only when the OUTERMOST chain ends. Inner
    LCEL sub-runnables fire on_chain_end too — those are ignored because
    their run_id isn't in ``_runs`` (only root run_ids live there)."""

    def test_saves_record_on_chain_end(self, handler):
        rid = _start_root(handler, {"query": "What is auditability?"})
        # Simulate the LLM firing mid-chain.
        gen = Generation(text="It means being able to trace decisions.")
        handler.on_llm_end(LLMResult(generations=[[gen]]), run_id=rid)

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            handler.on_chain_end(
                {"answer": "It means being able to trace decisions."},
                run_id=rid,
            )
            mock_save.assert_called_once()
            record = mock_save.call_args[0][0]
            assert record.query == "What is auditability?"
            assert record.session_id == "test-session"
            assert record.workspace_id == "test-workspace"
            assert len(record.chain_signature) == 64  # SHA-256 hex

    def test_ignores_inner_chain_end(self, handler):
        """Inner LCEL sub-runnables fire on_chain_end too; the handler must
        ignore those or it'd save an empty record before the LLM ran."""
        outer = _start_root(handler, {"query": "q"})
        inner = uuid.uuid4()
        # Inner sub-runnable fires on_chain_start with parent_run_id set.
        handler.on_chain_start({}, {"sub": "input"}, run_id=inner, parent_run_id=outer)

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            handler.on_chain_end({"intermediate": "stuff"}, run_id=inner)
            mock_save.assert_not_called()
        # Outer run is still tracked.
        assert outer in handler._runs

    def test_state_cleaned_up_after_chain_end(self, handler):
        rid = _start_root(handler, {"query": "some query"})
        gen = Generation(text="some answer")
        handler.on_llm_end(LLMResult(generations=[[gen]]), run_id=rid)

        with patch.object(handler.storage, "save", return_value=True):
            handler.on_chain_end({}, run_id=rid)

        assert rid not in handler._runs

    def test_state_cleaned_up_on_error(self, handler):
        rid = _start_root(handler, {"query": "failing query"})
        handler.on_chain_error(Exception("LLM timeout"), run_id=rid)
        assert rid not in handler._runs
