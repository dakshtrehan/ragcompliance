"""Tests for the RAGCompliance callback handler."""

import hashlib
import json
from unittest.mock import MagicMock, patch

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


class TestQueryCapture:
    def test_captures_query_key(self, handler):
        handler.on_chain_start({}, {"query": "What is RAG?"})
        assert handler._query == "What is RAG?"

    def test_captures_question_key(self, handler):
        handler.on_chain_start({}, {"question": "How does this work?"})
        assert handler._query == "How does this work?"

    def test_captures_input_key(self, handler):
        handler.on_chain_start({}, {"input": "Explain compliance."})
        assert handler._query == "Explain compliance."

    def test_ignores_unknown_keys(self, handler):
        handler.on_chain_start({}, {"irrelevant": "nothing"})
        assert handler._query == ""


class TestRetrieverCapture:
    def test_captures_chunks_with_metadata(self, handler):
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
        handler.on_retriever_end(docs)

        assert len(handler._chunks) == 2
        assert handler._chunks[0].source_url == "https://example.com/rag"
        assert handler._chunks[0].chunk_id == "abc123"
        assert handler._chunks[0].similarity_score == 0.92
        assert handler._chunks[1].similarity_score == 0.87

    def test_handles_missing_metadata_gracefully(self, handler):
        docs = [Document(page_content="Plain content", metadata={})]
        handler.on_retriever_end(docs)

        assert len(handler._chunks) == 1
        assert handler._chunks[0].source_url == "unknown"
        assert handler._chunks[0].chunk_id == "chunk-0"
        assert handler._chunks[0].similarity_score is None


class TestLLMCapture:
    def test_captures_llm_answer(self, handler):
        generation = Generation(text="RAG improves LLM accuracy using external context.")
        result = LLMResult(generations=[[generation]])
        handler.on_llm_end(result)
        assert handler._llm_answer == "RAG improves LLM accuracy using external context."

    def test_captures_model_name(self, handler):
        generation = Generation(text="Answer.")
        result = LLMResult(generations=[[generation]], llm_output={"model_name": "gpt-4"})
        handler.on_llm_end(result)
        assert handler._model_name == "gpt-4"


class TestChainSignature:
    def test_signature_is_sha256(self, handler):
        handler._query = "test query"
        handler._chunks = []
        handler._llm_answer = "test answer"

        sig = handler._sign_chain()

        payload = {"query": "test query", "chunks": [], "answer": "test answer"}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert sig == expected

    def test_signature_changes_with_different_content(self, handler):
        handler._query = "query A"
        handler._llm_answer = "answer A"
        sig_a = handler._sign_chain()

        handler._query = "query B"
        handler._llm_answer = "answer B"
        sig_b = handler._sign_chain()

        assert sig_a != sig_b


class TestChainEnd:
    def test_saves_record_on_chain_end(self, handler):
        handler._query = "What is auditability?"
        handler._llm_answer = "It means being able to trace decisions."

        with patch.object(handler.storage, "save", return_value=True) as mock_save:
            handler.on_chain_end({"answer": "It means being able to trace decisions."})
            mock_save.assert_called_once()
            record = mock_save.call_args[0][0]
            assert record.query == "What is auditability?"
            assert record.session_id == "test-session"
            assert record.workspace_id == "test-workspace"
            assert len(record.chain_signature) == 64  # SHA-256 hex

    def test_resets_state_after_chain_end(self, handler):
        handler._query = "some query"
        handler._llm_answer = "some answer"

        with patch.object(handler.storage, "save", return_value=True):
            handler.on_chain_end({})

        assert handler._query == ""
        assert handler._chunks == []
        assert handler._llm_answer == ""

    def test_resets_on_error(self, handler):
        handler._query = "failing query"
        handler.on_chain_error(Exception("LLM timeout"))
        assert handler._query == ""
