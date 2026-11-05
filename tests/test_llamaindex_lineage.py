"""LlamaIndex handler: redaction, attribution, prompt capture, exceptions."""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, TextNode

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.integrations.llamaindex import LlamaIndexRAGComplianceHandler, install


def _handler(**cfg) -> LlamaIndexRAGComplianceHandler:
    config = RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="w", **cfg)
    return LlamaIndexRAGComplianceHandler(config=config, session_id="s", billing=None, alerter=None)


def _records(h) -> list[dict]:
    return list(h.storage.iter_records())


class _Msg:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def test_redaction_and_actor_via_bind():
    h = _handler(redact_pii=True)
    with h.auditor.bind(actor={"id": "alice", "type": "user"}):
        h.start_trace("t1")
        h.on_event_start(CBEventType.QUERY, payload={EventPayload.QUERY_STR: "ssn 123-45-6789"})
        h.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.MESSAGES: [_Msg("system", "sys"), _Msg("user", "ssn 123-45-6789")],
                     EventPayload.SERIALIZED: {"model": "gpt-x", "temperature": 0.5}},
        )
        h.on_event_end(
            CBEventType.RETRIEVE,
            payload={EventPayload.NODES: [NodeWithScore(node=TextNode(text="mail a@b.com", metadata={"file_path": "/f.pdf"}), score=0.7)]},
        )

        class Resp:
            text = "answer for a@b.com"

        h.on_event_end(CBEventType.LLM, payload={EventPayload.COMPLETION: Resp(), EventPayload.SERIALIZED: {"model": "gpt-x"}})
        h.end_trace("t1", {})
    r = _records(h)[0]
    assert "[REDACTED:SSN]" in r["query"] and "[REDACTED:EMAIL]" in r["llm_answer"]
    assert r["retrieved_chunks"][0]["source_url"] == "/f.pdf" and "[REDACTED:EMAIL]" in r["retrieved_chunks"][0]["content"]
    assert r["actor"] == {"id": "alice", "type": "user"} and r["model_name"] == "gpt-x"
    assert r["lineage"]["model_params"] == {"model": "gpt-x", "temperature": 0.5}
    assert r["lineage"]["retrievers"] == ["llamaindex.retrieve"] and "prompt_sha256" in r["lineage"]
    assert r["lineage"]["framework"] == "llamaindex" and r["extra"]["redaction_findings"]["ssn"] == 1


def test_prompt_string_and_synthesize_answer():
    h = _handler(capture_prompts=True)
    h.start_trace("t")
    h.on_event_start(CBEventType.LLM, payload={EventPayload.PROMPT: "formatted prompt"})

    class R:
        response = "final"

    h.on_event_end(CBEventType.SYNTHESIZE, payload={EventPayload.RESPONSE: R()})
    h.end_trace("t", {})
    r = _records(h)[0]
    assert r["llm_answer"] == "final" and r["lineage"]["prompt_text"] == "formatted prompt"


def test_exception_event_records_failure():
    h = _handler(record_errors=True)
    h.start_trace("t")
    h.on_event_start(CBEventType.QUERY, payload={"query_str": "q"})
    h.on_event_end(CBEventType.EXCEPTION, payload={EventPayload.EXCEPTION: ValueError("bad")})
    h.end_trace("t", {})
    r = _records(h)[0]
    assert r["lineage"]["status"] == "error" and r["extra"]["error"]["type"] == "ValueError"


def test_events_without_trace_are_ignored_and_single_live_trace_is_used():
    h = _handler()
    h.on_event_start(CBEventType.QUERY, payload={"query_str": "orphan"})
    h.on_event_end(CBEventType.RETRIEVE, payload={"nodes": []})
    assert _records(h) == []
    h.start_trace("t")
    # event routed from another thread-local context: still the only trace
    h._active.trace_id = None
    h.on_event_start(CBEventType.QUERY, payload={"query_str": "routed"})
    h.end_trace("t", {})
    assert _records(h)[0]["query"] == "routed"


def test_llm_end_with_raw_model_and_messages_fallback():
    h = _handler()
    h.start_trace("t")

    class Raw:
        model = "raw-model"

    class Completion:
        text = "t"
        raw = Raw()

    h.on_event_end(CBEventType.LLM, payload={"completion": Completion()})
    h.end_trace("t", {})
    assert _records(h)[0]["model_name"] == "raw-model"


def test_pending_cap_and_compat_properties(monkeypatch):
    monkeypatch.setenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "2")
    h = _handler()
    assert h._max_pending_traces == 2
    for i in range(4):
        h.start_trace(f"t{i}")
    assert len(h._traces) == 2
    assert h.config.workspace_id == "w" and h.session_id == "s" and h.extra == {} and h.redactor is None
    h.alerter = None
    h.billing = None
    from ragcompliance import AuditStorage

    st = AuditStorage(h.config)
    h.storage = st
    assert h.storage is st
    assert len(h._sign_chain(h._traces["t2"])) == 64
    monkeypatch.setenv("RAGCOMPLIANCE_MAX_PENDING_RUNS", "junk")
    assert _handler()._max_pending_traces == 10000


def test_finalize_failure_is_contained(caplog):
    h = _handler()
    h.start_trace("t")

    def boom(*a, **k):
        raise RuntimeError("explode")

    h.auditor.finalize = boom
    h.end_trace("t", {})  # must not raise
    assert "t" not in h._traces


def test_install_registers_on_settings():
    from llama_index.core import Settings

    before = Settings.callback_manager
    try:
        h = install(config=RAGComplianceConfig(sink_url="memory://", async_writes=False), billing=None, alerter=None)
        assert h in Settings.callback_manager.handlers
    finally:
        Settings.callback_manager = before


def test_import_error_without_llamaindex(monkeypatch):
    import ragcompliance.llamaindex_handler as mod

    monkeypatch.setattr(mod, "_LLAMA_AVAILABLE", False)
    with pytest.raises(ImportError, match="ragcompliance\\[llamaindex\\]"):
        LlamaIndexRAGComplianceHandler(config=RAGComplianceConfig(dev_mode=True))


def test_event_handler_exceptions_are_logged(caplog):
    h = _handler()
    h.start_trace("t")

    class Bad:
        @property
        def node(self):
            raise RuntimeError("bad node")

    h.on_event_end(CBEventType.RETRIEVE, payload={"nodes": [Bad()]})
    h.on_event_start(CBEventType.LLM, payload={EventPayload.MESSAGES: 5})
    h.end_trace("t", {})
    assert len(_records(h)) == 1
