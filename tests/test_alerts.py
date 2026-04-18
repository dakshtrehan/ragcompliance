"""Tests for the Slack anomaly alerter."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragcompliance import AuditRecord, RetrievedChunk, SlackAlerter


# ------------------------------------------------------------------ #
# Local webhook catcher                                                #
# ------------------------------------------------------------------ #


class _Catcher(BaseHTTPRequestHandler):
    posts: list[dict] = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {"_raw": body.decode("utf-8", errors="replace")}
        _Catcher.posts.append(payload)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args, **kwargs):  # silence stderr noise
        pass


@pytest.fixture
def catcher():
    _Catcher.posts = []
    server = HTTPServer(("127.0.0.1", 0), _Catcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}", _Catcher
    server.shutdown()


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #


def make_record(**overrides) -> AuditRecord:
    defaults = {
        "session_id": "sess-1",
        "workspace_id": "ws-test",
        "query": "why does the sky look blue",
        "retrieved_chunks": [
            RetrievedChunk(
                content="Rayleigh scattering.",
                source_url="sci://sky",
                chunk_id="c1",
                similarity_score=0.9,
            )
        ],
        "llm_answer": "Rayleigh scattering causes blue sky.",
        "model_name": "gpt-4o",
        "chain_signature": "a" * 64,
        "latency_ms": 500,
    }
    defaults.update(overrides)
    return AuditRecord(**defaults)


# ------------------------------------------------------------------ #
# Evaluation rules                                                      #
# ------------------------------------------------------------------ #


class TestEvaluate:
    def test_healthy_record_triggers_nothing(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url)
        fired = alerter.evaluate(make_record())
        assert fired == []

    def test_zero_chunks_fires(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url)
        fired = alerter.evaluate(make_record(retrieved_chunks=[]))
        names = {r.name for r in fired}
        assert "retrieval_returned_zero_chunks" in names

    def test_low_similarity_fires(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url, min_similarity=0.5)
        weak_chunks = [
            RetrievedChunk(content="c", source_url="u", chunk_id="x", similarity_score=0.2),
            RetrievedChunk(content="d", source_url="u", chunk_id="y", similarity_score=0.3),
        ]
        fired = alerter.evaluate(make_record(retrieved_chunks=weak_chunks))
        assert any(r.name == "low_similarity" for r in fired)

    def test_chain_slow_fires(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url, slow_chain_ms=1000)
        fired = alerter.evaluate(make_record(latency_ms=5000))
        assert any(r.name == "chain_slow" for r in fired)

    def test_error_fires(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url)
        fired = alerter.evaluate(make_record(), error=RuntimeError("timeout"))
        assert any(r.name == "chain_errored" for r in fired)

    def test_missing_similarity_scores_do_not_trigger(self, catcher):
        url, _ = catcher
        alerter = SlackAlerter(webhook_url=url, min_similarity=0.5)
        untyped_chunks = [
            RetrievedChunk(content="c", source_url="u", chunk_id="x", similarity_score=None)
        ]
        fired = alerter.evaluate(make_record(retrieved_chunks=untyped_chunks))
        # low_similarity rule should SKIP when no scores are present at all —
        # we can't tell if retrieval was bad or just unscored.
        assert not any(r.name == "low_similarity" for r in fired)


# ------------------------------------------------------------------ #
# Actual HTTP posting                                                  #
# ------------------------------------------------------------------ #


class TestWebhookPost:
    def test_posts_to_webhook_on_fire(self, catcher):
        url, store = catcher
        alerter = SlackAlerter(webhook_url=url)
        enqueued = alerter.maybe_alert(make_record(retrieved_chunks=[]))
        assert enqueued, "alert should have been enqueued"
        alerter.flush(timeout=2.0)
        # Give the worker a moment to flush.
        for _ in range(50):
            if store.posts:
                break
            time.sleep(0.05)
        assert store.posts, "expected at least one webhook POST"
        text = store.posts[0]["text"]
        assert "RAGCompliance alert" in text
        assert "ws-test" in text
        assert "retrieval_returned_zero_chunks" in text

    def test_does_not_post_for_healthy_record(self, catcher):
        url, store = catcher
        alerter = SlackAlerter(webhook_url=url)
        enqueued = alerter.maybe_alert(make_record())
        assert not enqueued
        alerter.flush(timeout=0.3)
        assert store.posts == []

    def test_dashboard_url_appears_in_payload(self, catcher):
        url, store = catcher
        alerter = SlackAlerter(
            webhook_url=url,
            dashboard_url="https://compliance.example.com/",
        )
        rec = make_record(retrieved_chunks=[])
        alerter.maybe_alert(rec)
        alerter.flush(timeout=2.0)
        for _ in range(50):
            if store.posts:
                break
            time.sleep(0.05)
        assert store.posts
        assert f"compliance.example.com/logs/detail/{rec.id}" in store.posts[0]["text"]
