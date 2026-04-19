"""Tests for the LlamaIndex callback handler."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.callbacks.schema import CBEventType
from llama_index.core.schema import NodeWithScore, TextNode

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler


def _make_handler(enforce_quota: bool = False) -> LlamaIndexRAGComplianceHandler:
    cfg = RAGComplianceConfig(
        workspace_id="w-llama",
        dev_mode=True,
        enforce_quota=enforce_quota,
    )
    h = LlamaIndexRAGComplianceHandler(config=cfg, session_id="s-1", billing=None)
    return h


class TestLlamaIndexHandler:
    def test_captures_query_on_event_start(self):
        h = _make_handler()
        h.start_trace("t1")
        h.on_event_start(
            CBEventType.QUERY,
            payload={"query_str": "What does section 4.2 say?"},
            event_id="e1",
        )
        assert h._traces["t1"].query == "What does section 4.2 say?"

    def test_captures_retrieved_nodes_with_scores(self):
        h = _make_handler()
        h.start_trace("t1")
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text="Section 4.2 defines indemnification.",
                    metadata={"source": "contract-v3.pdf", "chunk_id": "c42"},
                ),
                score=0.91,
            ),
            NodeWithScore(
                node=TextNode(
                    text="Indemnification survives termination.",
                    metadata={"source": "contract-v3.pdf", "chunk_id": "c43"},
                ),
                score=0.84,
            ),
        ]
        h.on_event_end(CBEventType.RETRIEVE, payload={"nodes": nodes}, event_id="e2")

        chunks = h._traces["t1"].chunks
        assert len(chunks) == 2
        assert chunks[0].source_url == "contract-v3.pdf"
        assert chunks[0].chunk_id == "c42"
        assert chunks[0].similarity_score == pytest.approx(0.91)

    def test_signs_chain_and_saves_on_end_trace(self):
        h = _make_handler()
        h.start_trace("t1")
        h.on_event_start(CBEventType.QUERY, payload={"query_str": "Q?"})
        h.on_event_end(
            CBEventType.RETRIEVE,
            payload={
                "nodes": [
                    NodeWithScore(
                        node=TextNode(text="A", metadata={"source": "s"}),
                        score=0.5,
                    ),
                ]
            },
        )

        class _FakeLLMResp:
            text = "final answer"

        h.on_event_end(CBEventType.LLM, payload={"response": _FakeLLMResp()})

        saved = []
        h.storage.save = lambda record: saved.append(record) or True  # type: ignore

        h.end_trace("t1", {})

        assert len(saved) == 1
        rec = saved[0]
        assert rec.query == "Q?"
        assert rec.llm_answer == "final answer"
        assert len(rec.chain_signature) == 64  # sha256 hex
        # Per-trace state is evicted on end_trace
        assert "t1" not in h._traces

    def test_enforce_quota_raises_when_over_limit(self):
        h = _make_handler(enforce_quota=True)

        class _OverLimitBilling:
            def check_query_quota(self, workspace_id):
                return False

            def increment_usage(self, workspace_id):
                return None

        h.billing = _OverLimitBilling()
        with pytest.raises(RuntimeError, match="exceeded"):
            h.start_trace("t1")

    def test_soft_quota_warns_but_does_not_raise(self):
        h = _make_handler(enforce_quota=False)

        class _OverLimitBilling:
            def check_query_quota(self, workspace_id):
                return False

            def increment_usage(self, workspace_id):
                return None

        h.billing = _OverLimitBilling()
        # Must not raise
        h.start_trace("t1")
