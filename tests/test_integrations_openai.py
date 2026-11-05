"""OpenAI wrapper tests: real SDK with an httpx mock transport (no network),
plus fake clients for streaming and async paths."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from ragcompliance import RAGComplianceConfig, RetrievedChunk
from ragcompliance.chain import verify_records
from ragcompliance.integrations.openai import AuditedOpenAI, audit_openai

openai = pytest.importorskip("openai")


def _cfg(**kw) -> RAGComplianceConfig:
    return RAGComplianceConfig(sink_url="memory://", async_writes=False, workspace_id="ws", **kw)


CHAT_JSON = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini-2026",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Clause 4.2 covers indemnity."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

RESPONSES_JSON = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "gpt-5-mini",
    "status": "completed",
    "output": [
        {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Indemnity.", "annotations": []}]}
    ],
    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
              "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
}


def _mock_client(seen: list[dict]) -> openai.OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        seen.append({"path": request.url.path, "body": body})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=CHAT_JSON)
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json=RESPONSES_JSON)
        return httpx.Response(404, json={"error": "nope"})

    return openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestRealSDK:
    def test_chat_completion_is_audited_and_audit_kwargs_are_stripped(self):
        seen: list[dict] = []
        client = AuditedOpenAI(_mock_client(seen), config=_cfg(), session_id="s", billing=None, alerter=None)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Use the context."},
                      {"role": "user", "content": [{"type": "text", "text": "What does clause 4.2 say?"}]}],
            temperature=0.2,
            retrieved_chunks=[RetrievedChunk("Clause 4.2 ...", "msa.pdf", "p3", 0.91), {"content": "ctx2", "source": "x"}],
            actor={"id": "alice@example.com"},
            audit_extra={"ticket": "T-9"},
        )
        assert resp.choices[0].message.content == "Clause 4.2 covers indemnity."
        assert set(seen[0]["body"]) == {"model", "messages", "temperature"}  # audit kwargs never sent
        recs = list(client.auditor.storage.iter_records())
        r = recs[0]
        assert r["query"] == "What does clause 4.2 say?" and r["llm_answer"] == "Clause 4.2 covers indemnity."
        assert r["model_name"] == "gpt-4o-mini-2026" and r["actor"] == {"id": "alice@example.com"}
        assert [c["chunk_id"] for c in r["retrieved_chunks"]] == ["p3", "chunk-1"]
        assert r["lineage"]["model_params"] == {"model": "gpt-4o-mini", "temperature": 0.2}
        assert r["lineage"]["usage"]["total_tokens"] == 15 and r["lineage"]["framework"] == "openai"
        assert r["extra"]["ticket"] == "T-9" and "prompt_sha256" in r["lineage"]
        assert verify_records(recs).ok

    def test_responses_api(self):
        seen: list[dict] = []
        client = audit_openai(_mock_client(seen), config=_cfg(), billing=None, alerter=None)
        resp = client.responses.create(model="gpt-5-mini", instructions="Be brief.",
                                       input=[{"role": "user", "content": "Summarise 4.2"}], retrieved_chunks=["c"])
        assert resp.output_text == "Indemnity."
        r = list(client.auditor.storage.iter_records())[0]
        assert r["query"] == "Summarise 4.2" and r["llm_answer"] == "Indemnity." and r["model_name"] == "gpt-5-mini"
        assert r["retrieved_chunks"][0]["content"] == "c"

    def test_responses_string_input(self):
        seen: list[dict] = []
        client = AuditedOpenAI(_mock_client(seen), config=_cfg(), billing=None, alerter=None)
        client.responses.create(model="gpt-5-mini", input="plain question")
        assert list(client.auditor.storage.iter_records())[0]["query"] == "plain question"

    def test_api_error_records_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        raw = openai.OpenAI(api_key="t", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        client = AuditedOpenAI(raw, config=_cfg(record_errors=True), billing=None, alerter=None)
        with pytest.raises(openai.APIStatusError):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        r = list(client.auditor.storage.iter_records())[0]
        assert r["lineage"]["status"] == "error" and r["query"] == "q"

    def test_passthrough_attributes(self):
        client = AuditedOpenAI(_mock_client([]), config=_cfg(), billing=None, alerter=None)
        assert client.api_key == "test"
        assert hasattr(client.chat, "completions") and client.chat.completions.with_raw_response is not None
        assert client.responses.with_raw_response is not None


class _FakeChunk:
    def __init__(self, text, model="fake-model"):
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]
        self.model = model


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True


class _FakeClient:
    def __init__(self, stream=None, result=None, error=None):
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if error:
                    raise error
                if kwargs.get("stream"):
                    return stream
                return result

        self.chat = SimpleNamespace(completions=Completions())


class TestStreaming:
    def test_stream_accumulates_and_finalizes(self):
        stream = _FakeStream([_FakeChunk("Hel"), _FakeChunk("lo"), SimpleNamespace(choices=[], model="x")])
        client = AuditedOpenAI(_FakeClient(stream=stream), config=_cfg(), billing=None, alerter=None)
        out = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True)
        assert list(out) == stream._chunks[0:3] or len(list(out)) == 0  # consumed once
        r = list(client.auditor.storage.iter_records())[0]
        assert r["llm_answer"] == "Hello" and r["model_name"] == "fake-model" and r["lineage"]["streamed"] is True

    def test_stream_context_manager_and_passthrough(self):
        stream = _FakeStream([_FakeChunk("a")])
        client = AuditedOpenAI(_FakeClient(stream=stream), config=_cfg(), billing=None, alerter=None)
        with client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True) as s:
            assert s.closed is False
            for _ in s:
                pass
        assert stream.closed is True
        assert len(list(client.auditor.storage.iter_records())) == 1  # finalized once

    def test_stream_error_records_failure(self):
        class Boom(_FakeStream):
            def __iter__(self):
                yield _FakeChunk("partial")
                raise RuntimeError("cut")

        client = AuditedOpenAI(_FakeClient(stream=Boom([])), config=_cfg(record_errors=True), billing=None, alerter=None)
        s = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True)
        with pytest.raises(RuntimeError):
            list(s)
        r = list(client.auditor.storage.iter_records())[0]
        assert r["llm_answer"] == "partial" and r["lineage"]["status"] == "error"

    def test_stream_exit_with_exception(self):
        client = AuditedOpenAI(_FakeClient(stream=_FakeStream([])), config=_cfg(record_errors=True), billing=None, alerter=None)
        with pytest.raises(ValueError):
            with client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True):
                raise ValueError("inside")
        assert list(client.auditor.storage.iter_records())[0]["lineage"]["status"] == "error"

    def test_delta_extraction_errors_are_ignored(self):
        class Odd:
            @property
            def choices(self):
                raise RuntimeError("weird")

        client = AuditedOpenAI(_FakeClient(stream=_FakeStream([Odd()])), config=_cfg(), billing=None, alerter=None)
        list(client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True))
        assert list(client.auditor.storage.iter_records())[0]["llm_answer"] == ""


class TestAsync:
    def test_async_client(self):
        result = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="async answer"))], model="am", usage=None)

        class AsyncCompletions:
            async def create(self, **kwargs):
                return result

        fake = SimpleNamespace(chat=SimpleNamespace(completions=AsyncCompletions()))
        client = AuditedOpenAI(fake, config=_cfg(), billing=None, alerter=None)
        coro = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        out = asyncio.run(coro)
        assert out is result
        r = list(client.auditor.storage.iter_records())[0]
        assert r["llm_answer"] == "async answer" and r["model_name"] == "am"

    def test_async_error(self):
        class AsyncCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("async boom")

        fake = SimpleNamespace(chat=SimpleNamespace(completions=AsyncCompletions()))
        client = AuditedOpenAI(fake, config=_cfg(record_errors=True), billing=None, alerter=None)
        with pytest.raises(RuntimeError):
            asyncio.run(client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}]))
        assert list(client.auditor.storage.iter_records())[0]["lineage"]["status"] == "error"

    def test_async_stream(self):
        class AStream:
            def __init__(self):
                self.entered = False

            async def __aiter__(self):
                yield _FakeChunk("a")
                yield _FakeChunk("b")

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, *a):
                return None

        class AsyncCompletions:
            async def create(self, **kwargs):
                return AStream()

        fake = SimpleNamespace(chat=SimpleNamespace(completions=AsyncCompletions()))
        client = AuditedOpenAI(fake, config=_cfg(), billing=None, alerter=None)

        async def run():
            s = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}], stream=True)
            async with s as stream:
                async for _ in stream:
                    pass
            # async context manager with an exception
            s2 = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q2"}], stream=True)
            try:
                async with s2:
                    raise ValueError("x")
            except ValueError:
                pass

        asyncio.run(run())
        recs = list(client.auditor.storage.iter_records())
        assert recs[0]["llm_answer"] == "ab" and len(recs) == 1  # error record disabled by default


class TestExtraction:
    def test_answer_extraction_failure_is_logged_not_raised(self):
        class Odd:
            @property
            def choices(self):
                raise RuntimeError("weird")

        client = AuditedOpenAI(_FakeClient(result=Odd()), config=_cfg(), billing=None, alerter=None)
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        assert list(client.auditor.storage.iter_records())[0]["llm_answer"] == ""

    def test_sync_error_path(self):
        client = AuditedOpenAI(_FakeClient(error=RuntimeError("down")), config=_cfg(record_errors=True), billing=None, alerter=None)
        with pytest.raises(RuntimeError):
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        assert list(client.auditor.storage.iter_records())[0]["lineage"]["status"] == "error"

    def test_version_helper_without_sdk(self, monkeypatch):
        import builtins

        from ragcompliance.integrations import openai as mod

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "openai":
                raise ImportError
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert mod._openai_version() == ""

    def test_responses_output_fallback_and_stream_events(self):
        from ragcompliance.integrations.openai import (
            _responses_answer,
            _responses_delta,
            _responses_stream_model,
        )

        result = SimpleNamespace(output_text="", model="m", output=[
            SimpleNamespace(content=[SimpleNamespace(type="output_text", text="a"), SimpleNamespace(type="refusal", text="no")])
        ])
        assert _responses_answer(result) == ("a", "m")
        assert _responses_delta(SimpleNamespace(type="response.output_text.delta", delta="d")) == "d"
        assert _responses_delta(SimpleNamespace(type="other")) == ""
        assert _responses_stream_model(SimpleNamespace(response=SimpleNamespace(model="mm"))) == "mm"
        assert _responses_stream_model(SimpleNamespace()) == ""
