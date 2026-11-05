"""Anthropic wrapper tests: real SDK through an httpx mock transport, plus
fake clients for signature handling, streaming and async."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

try:  # anthropic 2.x ships against the httpx2 package
    import httpx2 as httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]

from ragcompliance import RAGComplianceConfig, RetrievedChunk
from ragcompliance.chain import verify_records
from ragcompliance.integrations import anthropic as mod
from ragcompliance.integrations._sdk import unsupported_params
from ragcompliance.integrations.anthropic import AuditedAnthropic, audit_anthropic

anthropic = pytest.importorskip("anthropic")


def _cfg(**kw) -> RAGComplianceConfig:
    return RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="ws", **kw)


MESSAGE_JSON = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Clause 4.2 is about indemnity."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 7},
}


def _mock_client(seen: list[dict]) -> anthropic.Anthropic:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content or b"{}"),
                     "version": request.headers.get("anthropic-version")})
        return httpx.Response(200, json=MESSAGE_JSON)

    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestRealSDK:
    def test_message_is_audited(self):
        seen: list[dict] = []
        client = AuditedAnthropic(_mock_client(seen), config=_cfg(), session_id="s", billing=None, alerter=None)
        msg = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=256,
            system="Answer from context only.",
            messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
            retrieved_chunks=[RetrievedChunk("Clause 4.2 ...", "msa.pdf", "p3", 0.9)],
            actor={"id": "alice"},
        )
        assert msg.content[0].text == "Clause 4.2 is about indemnity."
        assert "retrieved_chunks" not in seen[0]["body"] and "actor" not in seen[0]["body"]
        recs = list(client.auditor.storage.iter_records())
        r = recs[0]
        assert r["query"] == "What does clause 4.2 say?" and r["llm_answer"] == "Clause 4.2 is about indemnity."
        assert r["model_name"] == "claude-sonnet-5-5" and r["lineage"]["framework"] == "anthropic"
        assert r["lineage"]["usage"] == {"input_tokens": 12, "output_tokens": 7}
        assert r["lineage"]["model_params"] == {"model": "claude-sonnet-5-5", "max_tokens": 256}
        assert verify_records(recs).ok

    def test_temperature_is_dropped_for_api_2026_10_21(self, caplog):
        """Anthropic SDK 2.x defaults to API version 2026-10-21 which rejects
        temperature; the installed SDK no longer accepts the parameter."""
        seen: list[dict] = []
        sdk_accepts = "temperature" in inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters
        client = AuditedAnthropic(_mock_client(seen), config=_cfg(), billing=None, alerter=None)
        mod._warned.clear()
        client.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "q"}], temperature=0.7, top_p=0.9)
        body = seen[0]["body"]
        r = list(client.auditor.storage.iter_records())[0]
        if sdk_accepts:
            assert body.get("temperature") == 0.7
            assert "model_params_dropped" not in r["lineage"]
        else:
            assert "temperature" not in body and "top_p" not in body
            assert r["lineage"]["model_params_dropped"] == {"temperature": 0.7, "top_p": 0.9}
            assert "dropping 'temperature'" in caplog.text
            # The warning is emitted once per parameter per process.
            caplog.clear()
            client.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "q"}], temperature=0.7)
            assert "dropping 'temperature'" not in caplog.text
        if seen[0]["version"]:
            assert seen[0]["version"] >= "2023-06-01"

    def test_passthrough(self):
        client = audit_anthropic(_mock_client([]), config=_cfg(), billing=None, alerter=None)
        assert client.api_key == "test" and client.messages.with_raw_response is not None


class _FakeMessages:
    def __init__(self, result=None, stream=None, accept_all=False):
        self.calls = []
        self._result = result
        self._stream = stream
        if accept_all:
            def create(**kwargs):  # type: ignore[no-redef]
                self.calls.append(kwargs)
                return stream if kwargs.get("stream") else result
        else:
            def create(*, model, max_tokens, messages, system=None, stream=False, metadata=None):  # type: ignore[no-redef]
                self.calls.append({"model": model, "max_tokens": max_tokens, "messages": messages,
                                   "system": system, "stream": stream})
                return self._stream if stream else self._result
        self.create = create


class TestSignatureHandling:
    def test_unsupported_params_detection(self):
        def strict(a, *, b=1):
            pass

        def loose(a, **kw):
            pass

        assert unsupported_params(strict, ["b", "c"]) == ["c"]
        assert unsupported_params(loose, ["b", "c"]) == []
        assert unsupported_params(len, ["x"]) == [] or isinstance(unsupported_params(len, ["x"]), list)

    def test_kwargs_signature_passes_temperature_through(self):
        fake = SimpleNamespace(messages=_FakeMessages(result=SimpleNamespace(content=[], model="m"), accept_all=True))
        client = AuditedAnthropic(fake, config=_cfg(), billing=None, alerter=None)
        client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}], temperature=0.3)
        assert fake.messages.calls[0]["temperature"] == 0.3

    def test_strict_signature_drops_and_can_be_disabled(self):
        result = SimpleNamespace(content=[{"type": "text", "text": "dict block"}], model="m", usage=SimpleNamespace(input_tokens=1, output_tokens=2, cache_read_input_tokens=0))
        fake = SimpleNamespace(messages=_FakeMessages(result=result))
        client = AuditedAnthropic(fake, config=_cfg(), billing=None, alerter=None)
        client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}], temperature=0.3, top_k=5)
        assert fake.messages.calls[0]["model"] == "m"
        r = list(client.auditor.storage.iter_records())[0]
        assert r["lineage"]["model_params_dropped"] == {"temperature": 0.3, "top_k": 5}
        assert r["llm_answer"] == "dict block" and r["lineage"]["usage"]["cache_read_input_tokens"] == 0

        strict = AuditedAnthropic(fake, config=_cfg(), drop_unsupported_params=False, billing=None, alerter=None)
        with pytest.raises(TypeError):
            strict.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}], temperature=0.3)

    def test_no_sampling_params_means_nothing_dropped(self):
        fake = SimpleNamespace(messages=_FakeMessages(result=SimpleNamespace(content=[], model="m")))
        client = AuditedAnthropic(fake, config=_cfg(), billing=None, alerter=None)
        client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}])
        assert "model_params_dropped" not in list(client.auditor.storage.iter_records())[0]["lineage"]


def _event(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


class TestStreaming:
    def test_stream_text_deltas_and_model(self):
        events = [
            _event("message_start", message=SimpleNamespace(model="claude-stream")),
            _event("content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hel")),
            _event("content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json="{")),
            _event("content_block_delta", delta=SimpleNamespace(type="text_delta", text="lo")),
            _event("message_stop"),
        ]
        fake = SimpleNamespace(messages=_FakeMessages(stream=iter(events)))
        client = AuditedAnthropic(fake, config=_cfg(), billing=None, alerter=None)
        out = list(client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}], stream=True))
        assert len(out) == 5
        r = list(client.auditor.storage.iter_records())[0]
        assert r["llm_answer"] == "Hello" and r["model_name"] == "claude-stream"

    def test_async_create(self):
        class AsyncMessages:
            async def create(self, **kwargs):
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="async")], model="m", usage=None)

        client = AuditedAnthropic(SimpleNamespace(messages=AsyncMessages()), config=_cfg(), billing=None, alerter=None)
        asyncio.run(client.messages.create(model="m", max_tokens=1, messages=[{"role": "user", "content": "q"}]))
        assert list(client.auditor.storage.iter_records())[0]["llm_answer"] == "async"

    def test_version_helper_without_sdk(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "anthropic":
                raise ImportError
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert mod._anthropic_version() == ""


class TestSDKHelpers:
    def test_text_helpers(self):
        from ragcompliance.integrations._sdk import last_user_message, render_messages, text_of

        assert text_of(None) == "" and text_of("x") == "x"
        assert text_of([{"type": "text", "text": "a"}, {"type": "image", "source": {}}, "b"]) == "a\nb"
        assert text_of({"type": "tool_use"}) == ""
        assert text_of(SimpleNamespace(text="obj")) == "obj" and text_of(SimpleNamespace()) == ""
        msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "r"},
                SimpleNamespace(role="user", content="second")]
        assert last_user_message(msgs) == "second"
        assert last_user_message("plain") == "plain" and last_user_message(5) == ""
        assert render_messages(msgs, system="sys") == "system: sys\nuser: first\nassistant: r\nuser: second"
        assert render_messages("hi") == "user: hi"
