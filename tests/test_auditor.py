"""Tests for the framework neutral Auditor core."""

from __future__ import annotations

import logging
import threading

import pytest

from ragcompliance import (
    Auditor,
    AuditStorage,
    QuotaExceededError,
    RAGComplianceConfig,
    Redactor,
    RetrievedChunk,
)
from ragcompliance.auditor import Trace, _clean_params, coerce_chunk
from ragcompliance.chain import verify_records
from ragcompliance.redaction import RedactionResult


def _auditor(**cfg) -> Auditor:
    config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="ws", **cfg)
    return Auditor(config, session_id="sess", billing=None, alerter=None)


def _records(a: Auditor) -> list[dict]:
    return list(a.storage.iter_records())


class TestTraceLifecycle:
    def test_context_manager_writes_signed_record(self):
        a = _auditor()
        with a.trace("what is x?", actor={"id": "alice"}) as t:
            t.add_chunk(RetrievedChunk("x is y", "doc.pdf", "c1", 0.9))
            t.set_answer("x is y", model_name="m", params={"temperature": 0.1, "api_key": "nope"})
        recs = _records(a)
        assert len(recs) == 1
        r = recs[0]
        assert r["query"] == "what is x?" and r["llm_answer"] == "x is y" and r["model_name"] == "m"
        assert r["actor"] == {"id": "alice"} and r["session_id"] == "sess" and r["workspace_id"] == "ws"
        assert r["lineage"]["framework"] == "direct" and r["lineage"]["status"] == "ok"
        assert r["lineage"]["model_params"] == {"temperature": 0.1}
        assert r["lineage"]["chunks"][0]["content_sha256"] == RetrievedChunk("x is y", "", "").content_sha256()
        assert len(r["chain_signature"]) == 64 and r["sequence"] == 1
        assert verify_records(recs).ok

    def test_default_actor_merge_order(self):
        a = _auditor(actor_id="svc", actor_type="service")
        a.actor = {"tenant": "t1"}
        with a.trace("q", actor={"id": "alice"}) as t:
            t.set_answer("a")
        assert _records(a)[0]["actor"] == {"id": "alice", "type": "service", "tenant": "t1"}

    def test_bind_thread_local(self):
        a = _auditor()
        with a.bind(actor={"id": "bound"}, chunks=["ctx"], extra={"ticket": 1}):
            with a.bind(extra={"more": 2}):
                with a.trace("q") as t:
                    t.set_answer("a")
        r = _records(a)[0]
        assert r["actor"] == {"id": "bound"} and r["extra"] == {"ticket": 1, "more": 2}
        assert r["retrieved_chunks"][0]["content"] == "ctx"
        # Binding is per thread.
        seen = {}

        def other():
            seen["bound"] = a._bound()

        with a.bind(actor={"id": "x"}):
            th = threading.Thread(target=other)
            th.start()
            th.join()
        assert seen["bound"] == {}

    def test_outputs_coerced_when_no_answer(self):
        a = _auditor()
        t = a.new_trace({"question": "q?"})
        a.finalize(t, outputs={"result": "the answer"})
        r = _records(a)[0]
        assert r["query"] == "q?" and r["llm_answer"] == "the answer"

    def test_truncation_is_recorded(self):
        a = _auditor(max_query_chars=5, max_answer_chars=4)
        with a.trace("0123456789") as t:
            t.llm_answer = ""
        a2 = _auditor(max_query_chars=5, max_answer_chars=4)
        t = a2.new_trace("0123456789")
        a2.finalize(t, outputs="abcdefgh")
        r = _records(a2)[0]
        assert r["query"] == "01234" and r["llm_answer"] == "abcd"
        assert r["lineage"]["truncation"] == {
            "query": {"original_chars": 10, "stored_chars": 5},
            "answer": {"original_chars": 8, "stored_chars": 4},
        }

    def test_prompt_hash_and_optional_text(self):
        a = _auditor()
        with a.trace("q") as t:
            t.set_prompt("system: be nice\nuser: q")
            t.set_answer("a")
        r = _records(a)[0]
        assert "prompt_sha256" in r["lineage"] and "prompt_text" not in r["lineage"]
        b = _auditor(capture_prompts=True)
        with b.trace("q") as t:
            t.set_prompt("the prompt")
            t.set_answer("a")
        assert _records(b)[0]["lineage"]["prompt_text"] == "the prompt"

    def test_retrievers_and_extra_lineage(self):
        a = _auditor()
        a.extra = {"app": "demo"}
        with a.trace("q", extra={"ticket": "T-1"}) as t:
            t.note_retriever("faiss")
            t.note_retriever("faiss")
            t.lineage["custom"] = True
            t.set_answer("a")
        r = _records(a)[0]
        assert r["lineage"]["retrievers"] == ["faiss"] and r["lineage"]["custom"] is True
        assert r["extra"] == {"app": "demo", "ticket": "T-1"}


class TestFailurePath:
    def test_exception_alerts_but_does_not_persist_by_default(self):
        alerts = []

        class Alerter:
            def maybe_alert(self, record, error=None):
                alerts.append((record, error))

        a = _auditor()
        a.alerter = Alerter()
        with pytest.raises(ValueError):
            with a.trace("q") as t:
                t.set_answer("partial")
                raise ValueError("boom")
        assert _records(a) == []
        record, error = alerts[0]
        assert isinstance(error, ValueError) and record.extra["error"]["type"] == "ValueError"

    def test_record_errors_persists_error_record(self):
        a = _auditor(record_errors=True, redact_pii=True)
        with pytest.raises(RuntimeError):
            with a.trace("ssn 123-45-6789") as t:
                t.set_answer("partial")
                raise RuntimeError("llm down")
        r = _records(a)[0]
        assert r["lineage"]["status"] == "error" and r["extra"]["error"]["message"] == "llm down"
        assert "[REDACTED:SSN]" in r["query"]

    def test_alerter_and_billing_errors_are_swallowed(self, caplog):
        class Bad:
            def maybe_alert(self, *a, **k):
                raise RuntimeError("slack down")

            def increment_usage(self, ws):
                raise RuntimeError("stripe down")

            def check_query_quota(self, ws):
                raise RuntimeError("stripe down")

        a = _auditor()
        a.alerter = Bad()
        a.billing = Bad()
        with a.trace("q") as t:
            t.set_answer("a")
        assert len(_records(a)) == 1

    def test_storage_raise_is_contained(self, caplog):
        a = _auditor()

        def boom(record):
            raise RuntimeError("custom storage exploded")

        a.storage.save = boom
        with caplog.at_level(logging.ERROR, logger="ragcompliance.auditor"):
            with a.trace("q") as t:
                t.set_answer("a")
        assert "storage.save raised" in caplog.text


class TestQuota:
    class Over:
        def __init__(self):
            self.usage = 0

        def check_query_quota(self, ws):
            return False

        def increment_usage(self, ws):
            self.usage += 1

    def test_enforced(self):
        a = _auditor(enforce_quota=True)
        a.billing = self.Over()
        with pytest.raises(QuotaExceededError):
            with a.trace("q"):
                pass
        assert isinstance(QuotaExceededError("x"), RuntimeError)

    def test_soft_warns_and_meters(self, caplog):
        a = _auditor()
        a.billing = self.Over()
        with caplog.at_level(logging.WARNING, logger="ragcompliance.auditor"):
            with a.trace("q") as t:
                t.set_answer("a")
        assert "exceeded" in caplog.text and a.billing.usage == 1

    def test_default_billing_only_with_stripe_key(self, monkeypatch):
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        assert Auditor._default_billing() is None
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
        billing = Auditor._default_billing()
        assert billing is None or billing.__class__.__name__ == "BillingManager"

    def test_default_alerter_from_env(self, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_SLACK_WEBHOOK_URL", raising=False)
        assert Auditor._default_alerter() is None


class TestRedaction:
    def test_config_flag_builds_redactor_and_hashes_redacted_payload(self):
        a = _auditor(redact_pii=True)
        with a.trace("mail bob@x.com") as t:
            t.add_chunk(RetrievedChunk("card 4111-1111-1111-1111", "http://10.0.0.1/f", "c1", metadata={"owner": "a@b.co"}))
            t.set_answer("call 415-555-1234")
        r = _records(a)[0]
        assert "bob@x.com" not in r["query"] and "4111" not in r["retrieved_chunks"][0]["content"]
        assert r["retrieved_chunks"][0]["metadata"]["owner"] == "[REDACTED:EMAIL]"
        assert r["extra"]["redaction_findings"] == {"email": 2, "credit_card": 1, "ipv4": 1, "phone_us": 1}
        assert verify_records([r]).ok  # hashes computed after redaction

    def test_duck_typed_redactor(self):
        class Upper:
            def redact(self, text):
                return RedactionResult(text=text.upper(), findings={"upper": 1})

        config = RAGComplianceConfig(sink_url="memory://", async_writes=False)
        a = Auditor(config, redactor=Upper(), billing=None, alerter=None)
        with a.trace("hello") as t:
            t.add_chunk(RetrievedChunk("c", "u", "i"))
            t.set_answer("bye")
        r = _records(a)[0]
        assert r["query"] == "HELLO" and r["llm_answer"] == "BYE" and r["retrieved_chunks"][0]["content"] == "C"
        assert r["extra"]["redaction_findings"]["upper"] == 4

    def test_redactor_raising_does_not_lose_record(self, caplog):
        class Broken:
            def redact_record(self, record):
                raise RuntimeError("regex exploded")

        config = RAGComplianceConfig(sink_url="memory://", async_writes=False)
        a = Auditor(config, redactor=Broken(), billing=None, alerter=None)
        with caplog.at_level(logging.ERROR, logger="ragcompliance.auditor"):
            with a.trace("q") as t:
                t.set_answer("a")
        r = _records(a)[0]
        assert r["extra"]["redaction_findings"] == {"redactor_error": 1}

    def test_explicit_redactor_instance(self):
        config = RAGComplianceConfig(sink_url="memory://", async_writes=False)
        a = Auditor(config, redactor=Redactor(patterns=["email"]), billing=None, alerter=None)
        assert a.redactor.pattern_names == ["email"]


class TestHelpers:
    def test_coerce_chunk_variants(self):
        assert coerce_chunk("plain", index=3).chunk_id == "chunk-3"
        c = coerce_chunk({"text": "t", "source": "s", "id": "i", "score": "0.5", "metadata": {"k": 1}})
        assert (c.content, c.source_url, c.chunk_id, c.similarity_score, c.metadata) == ("t", "s", "i", 0.5, {"k": 1})
        c2 = coerce_chunk({"content": "x", "similarity_score": 0.7})
        assert c2.source_url == "unknown" and c2.similarity_score == 0.7

        class Doc:
            page_content = "pc"
            metadata = {"source": "s", "chunk_id": "c", "score": 0.3, "extra": 1}

        c3 = coerce_chunk(Doc())
        assert c3.content == "pc" and c3.metadata == {"extra": 1} and c3.similarity_score == 0.3
        assert coerce_chunk(RetrievedChunk("a", "b", "c")).chunk_id == "c"

    def test_clean_params(self):
        assert _clean_params(None) == {}
        out = _clean_params({"model": "m", "stop": ["a"], "temperature": None, "api_key": "k"})
        assert out == {"model": "m", "stop": '["a"]'}

    def test_coerce_text_shapes(self):
        a = _auditor()
        assert a.coerce_query({"input": {"query": "nested"}}) == "nested"
        assert a.coerce_query({"other": 1}) == '{"other": 1}'
        assert a.coerce_answer(None) == ""

        class Msg:
            content = [{"type": "text", "text": "part1"}, {"type": "image"}, "part2"]

        assert a.coerce_answer(Msg()) == "part1"
        assert a.coerce_answer([Msg(), "tail"]) == "part1\ntail"
        assert a.coerce_answer(42) == "42"

    def test_trace_set_answer_without_model(self):
        t = Trace()
        t.set_model_params({"model_name": "mn"})
        assert t.model_name == "mn"
        t.set_answer("x", params={"model": "ignored because model_name set"})
        assert t.model_name == "mn"

    def test_flush_and_close(self):
        a = _auditor()
        assert a.flush() is True
        a.close()

    def test_storage_can_be_shared(self):
        config = RAGComplianceConfig(sink_url="memory://", async_writes=False)
        storage = AuditStorage(config)
        a1 = Auditor(config, storage=storage, session_id="a", billing=None, alerter=None)
        a2 = Auditor(config, storage=storage, session_id="b", billing=None, alerter=None)
        with a1.trace("q1") as t:
            t.set_answer("x")
        with a2.trace("q2") as t:
            t.set_answer("y")
        recs = list(storage.iter_records())
        assert [r["sequence"] for r in recs] == [1, 2] and verify_records(recs).ok
