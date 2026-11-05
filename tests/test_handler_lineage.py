"""LangChain handler: attribution, lineage, model name capture, guards and
the run_id=None path from the shipped example (AUDIT.md B01)."""

from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from ragcompliance import QuotaExceededError, RAGComplianceConfig, RAGComplianceHandler
from ragcompliance.chain import verify_records
from ragcompliance.handler import _RunState


def _handler(**cfg) -> RAGComplianceHandler:
    config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="ws", **cfg)
    return RAGComplianceHandler(config=config, session_id="s", billing=None, alerter=None)


def _records(h: RAGComplianceHandler) -> list[dict]:
    return list(h.storage.iter_records())


class TestAttributionAndLineage:
    def test_actor_from_metadata(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start(
            {"name": "RunnableSequence"}, {"query": "q"}, run_id=rid,
            metadata={"ragcompliance_actor": {"id": "alice", "type": "user"}, "actor_type": "ignored",
                      "ragcompliance_extra": {"ticket": "T-1"}},
            tags=["prod"],
        )
        h.on_chain_end({"answer": "a"}, run_id=rid)
        r = _records(h)[0]
        assert r["actor"] == {"id": "alice", "type": "user"}
        assert r["extra"]["ticket"] == "T-1" and r["lineage"]["tags"] == ["prod"]
        assert r["lineage"]["chain"] == "RunnableSequence" and r["lineage"]["framework"] == "langchain"
        assert r["lineage"]["framework_version"]

    def test_actor_from_plain_keys_and_name_kwarg(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start(None, "q", run_id=rid, metadata={"user_id": "u-9", "actor_type": "user"}, name="my_chain")
        h.on_chain_end("a", run_id=rid)
        r = _records(h)[0]
        assert r["actor"] == {"id": "u-9", "type": "user"} and r["lineage"]["chain"] == "my_chain"

    def test_model_name_from_model_key_and_response_metadata(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        llm = uuid.uuid4()
        h.on_chat_model_start(
            {"id": ["langchain", "ChatAnthropic"]},
            [[SystemMessage(content="sys"), HumanMessage(content=[{"type": "text", "text": "hi"}])]],
            run_id=llm, parent_run_id=rid,
            invocation_params={"model": "claude-x", "temperature": 0.3, "api_key": "secret"},
        )
        gen = ChatGeneration(message=AIMessage(content="answer", response_metadata={"model": "claude-x-20260101"},
                                               usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}))
        h.on_llm_end(LLMResult(generations=[[gen]], llm_output={"model": "claude-x-20260101", "usage": {"input_tokens": 3}}), run_id=llm)
        h.on_chain_end({}, run_id=rid)
        r = _records(h)[0]
        assert r["model_name"] == "claude-x-20260101" and r["llm_answer"] == "answer"
        assert r["lineage"]["model_params"] == {"model": "claude-x", "temperature": 0.3}
        assert "prompt_sha256" in r["lineage"] and r["lineage"]["usage"] == {"input_tokens": 3}

    def test_model_name_from_invocation_params_when_llm_output_missing(self):
        h = _handler()
        rid, llm = uuid.uuid4(), uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_llm_start({}, ["prompt"], run_id=llm, parent_run_id=rid, invocation_params={"model_name": "gpt-x"})
        h.on_llm_end(LLMResult(generations=[[Generation(text="a")]]), run_id=llm)
        h.on_chain_end({}, run_id=rid)
        assert _records(h)[0]["model_name"] == "gpt-x"

    def test_empty_generations_do_not_store_repr(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_llm_end(LLMResult(generations=[]), run_id=rid)
        h.on_chain_end({"answer": "real"}, run_id=rid)
        assert _records(h)[0]["llm_answer"] == "real"

    def test_usage_from_message(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        gen = ChatGeneration(message=AIMessage(content="a", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}))
        h.on_llm_end(LLMResult(generations=[[gen]]), run_id=rid)
        h.on_chain_end({}, run_id=rid)
        assert _records(h)[0]["lineage"]["usage"]["total_tokens"] == 3

    def test_retriever_name_recorded(self):
        h = _handler()
        rid, ret = uuid.uuid4(), uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_retriever_start({"id": ["langchain", "FAISS", "VectorStoreRetriever"]}, "q", run_id=ret, parent_run_id=rid)
        h.on_retriever_end([Document(page_content="c", metadata={"source": "s"})], run_id=ret)
        h.on_chain_end({"answer": "a"}, run_id=rid)
        r = _records(h)[0]
        assert r["lineage"]["retrievers"] == ["VectorStoreRetriever"]
        assert r["lineage"]["chunks"][0]["source_url"] == "s"

    def test_redaction_applies_to_error_alert(self):
        alerts = []

        class Alerter:
            def maybe_alert(self, record, error=None):
                alerts.append(record)

        h = _handler(redact_pii=True)
        h.alerter = Alerter()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "ssn 123-45-6789"}, run_id=rid)
        h.on_chain_error(RuntimeError("boom"), run_id=rid)
        assert alerts and "[REDACTED:SSN]" in alerts[0].query and alerts[0].extra["error"]["type"] == "RuntimeError"
        assert rid not in h._runs

    def test_chunks_and_signature_verify(self):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_retriever_end([Document(page_content="c1"), Document(page_content="c2")], run_id=rid)
        h.on_chain_end({"answer": "a"}, run_id=rid)
        recs = _records(h)
        assert [c["chunk_id"] for c in recs[0]["retrieved_chunks"]] == ["chunk-0", "chunk-1"]
        assert verify_records(recs).ok


class TestDirectCallsWithoutRunIds:
    def test_single_pending_root_resolves_none_run_id(self):
        """The shipped 0.1.8 example drove the handler without run ids and
        silently wrote nothing. Now the lone pending root is used."""
        h = _handler()
        h.on_chain_start({}, {"query": "q"})
        h.on_retriever_end([Document(page_content="c", metadata={"source": "s"})])
        h.on_llm_end(LLMResult(generations=[[Generation(text="a")]], llm_output={"model_name": "m"}))
        h.on_chain_end({"answer": "a"})
        recs = _records(h)
        assert len(recs) == 1 and recs[0]["retrieved_chunks"][0]["source_url"] == "s"
        assert recs[0]["model_name"] == "m" and h._runs == {}

    def test_ambiguous_none_run_id_is_dropped(self):
        h = _handler()
        h.on_chain_start({}, {"query": "a"}, run_id=uuid.uuid4())
        h.on_chain_start({}, {"query": "b"}, run_id=uuid.uuid4())
        h.on_retriever_end([Document(page_content="c")])  # cannot tell which root
        h.on_chain_end({"answer": "x"})  # ignored
        assert len(h._runs) == 2 and all(not s.chunks for s in h._runs.values())


class TestGuards:
    def test_exceptions_inside_callbacks_are_logged_not_raised(self, caplog):
        h = _handler()
        rid = uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)

        class Weird:
            @property
            def generations(self):
                raise ValueError("odd response")

        with caplog.at_level(logging.ERROR, logger="ragcompliance.handler"):
            h.on_llm_end(Weird(), run_id=rid)  # must not raise
        h.on_chain_end({"answer": "a"}, run_id=rid)
        assert len(_records(h)) == 1

    def test_quota_error_still_propagates(self):
        class Over:
            def check_query_quota(self, ws):
                return False

            def increment_usage(self, ws):
                return None

        h = _handler(enforce_quota=True)
        h.billing = Over()
        with pytest.raises(QuotaExceededError):
            h.on_chain_start({}, {"query": "q"}, run_id=uuid.uuid4())
        assert h.raise_error is True

    def test_compat_properties(self):
        h = _handler()
        assert h.config.workspace_id == "ws" and h.session_id == "s" and h.extra == {} and h.actor == {}
        h.session_id = "t"
        assert h.auditor.session_id == "t"
        h.redactor = None
        assert h.redactor is None
        h.alerter = None
        assert h.alerter is None
        from ragcompliance import AuditStorage

        new_storage = AuditStorage(h.config)
        h.storage = new_storage
        assert h.storage is new_storage
        assert h.flush() is True
        state = _RunState(query="q", llm_answer="a")
        assert len(h._sign_chain(state)) == 64
        assert h._coerce_query({"query": "x"}) == "x" and h._coerce_answer({"output": "y"}) == "y"

    def test_max_pending_from_config(self):
        h = _handler(max_pending_runs=3)
        assert h._max_pending_runs == 3

    def test_import_error_without_langchain(self, monkeypatch):
        import ragcompliance.handler as mod

        monkeypatch.setattr(mod, "_LANGCHAIN_AVAILABLE", False)
        with pytest.raises(ImportError, match="ragcompliance\\[langchain\\]"):
            RAGComplianceHandler(config=RAGComplianceConfig(dev_mode=True))

    def test_tool_start_registers_descendant(self):
        h = _handler()
        rid, tool, inner = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_tool_start({}, "x", run_id=tool, parent_run_id=rid)
        h.on_retriever_end([Document(page_content="via tool")], run_id=tool)
        # descendants whose root is gone are ignored
        h.on_chain_end({"answer": "a"}, run_id=rid)
        h.on_retriever_start({}, "q", run_id=inner, parent_run_id=tool)
        assert _records(h)[0]["retrieved_chunks"][0]["content"] == "via tool"
        assert inner in h._run_parents  # recorded, but no root to attach to


class TestIntegrationHelpers:
    def test_audited_chain_binds_callbacks(self):
        from langchain_core.runnables import RunnableLambda

        from ragcompliance.integrations.langchain import audited_chain

        h = _handler()
        chain = audited_chain(RunnableLambda(lambda x: {"answer": x["query"].upper()}), h,
                              metadata={"ragcompliance_actor": {"id": "bob"}})
        out = chain.invoke({"query": "hello"})
        assert out == {"answer": "HELLO"}
        recs = _records(h)
        assert len(recs) == 1 and recs[0]["query"] == "hello" and recs[0]["llm_answer"] == "HELLO"
        assert recs[0]["actor"] == {"id": "bob"}
