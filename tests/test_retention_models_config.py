"""Tests for retention policy, the record model and config parsing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ragcompliance.config import RAGComplianceConfig
from ragcompliance.models import SCHEMA_VERSION, AuditRecord, RetrievedChunk, _parse_timestamp
from ragcompliance.retention import PurgeResult, RetentionPolicy, build_retention_checkpoint


class TestRetentionPolicy:
    def test_disabled_by_default(self):
        p = RetentionPolicy()
        assert not p.enabled and p.cutoff() is None and p.to_dict() == {"days": None, "enabled": False}

    def test_cutoff(self):
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        assert RetentionPolicy(days=30).cutoff(now) == now - timedelta(days=30)
        naive = datetime(2026, 6, 1)
        assert RetentionPolicy(days=0).cutoff(naive).tzinfo is not None

    def test_from_env(self, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_RETENTION_DAYS", raising=False)
        assert RetentionPolicy.from_env().days is None
        monkeypatch.setenv("RAGCOMPLIANCE_RETENTION_DAYS", "90")
        assert RetentionPolicy.from_env().days == 90
        monkeypatch.setenv("RAGCOMPLIANCE_RETENTION_DAYS", "junk")
        assert RetentionPolicy.from_env().days is None

    def test_checkpoint_shape(self):
        cp = build_retention_checkpoint(
            workspace_id="ws",
            cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc),
            purged=3,
            last_purged_sequence=3,
            last_purged_hash="h" * 64,
        )
        assert cp["kind"] == "retention_checkpoint" and cp["workspace_id"] == "ws"
        assert cp["extra"]["purged"] == 3 and cp["extra"]["last_purged_hash"] == "h" * 64
        assert cp["actor"]["type"] == "system" and cp["chain_signature"] == ""

    def test_purge_result_dict(self):
        pr = PurgeResult("ws", datetime(2026, 1, 1, tzinfo=timezone.utc), purged=1, checkpoint={"id": "x"})
        assert pr.to_dict()["checkpoint_id"] == "x"


class TestModels:
    def test_round_trip(self):
        r = AuditRecord(
            session_id="s", workspace_id="w", query="q",
            retrieved_chunks=[RetrievedChunk("c", "u", "i", 0.5, {"k": "v"})],
            llm_answer="a", model_name="m", latency_ms=5, extra={"e": 1},
            actor={"id": "u"}, lineage={"x": 1},
        )
        d = r.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION and d["kind"] == "rag_invocation"
        back = AuditRecord.from_dict(d)
        assert back.to_dict() == d
        assert back.retrieved_chunks[0].content_sha256() == RetrievedChunk("c", "u", "i").content_sha256()

    def test_from_dict_legacy_row(self):
        row = {"id": "x", "session_id": "s", "workspace_id": "w", "query": "q", "llm_answer": "a",
               "chain_signature": "z" * 64, "timestamp": "2026-01-01T00:00:00Z", "latency_ms": "7"}
        r = AuditRecord.from_dict(row)
        assert r.schema_version == 1 and r.sequence is None and r.latency_ms == 7
        assert r.timestamp.tzinfo is not None

    def test_from_dict_accepts_chunk_objects_and_apply_seal(self):
        r = AuditRecord.from_dict({"retrieved_chunks": [RetrievedChunk("c", "u", "i")]})
        assert r.retrieved_chunks[0].chunk_id == "i"
        r.apply_seal({"chain_id": "c", "sequence": "4", "prev_hash": "p", "record_hash": "h", "record_hmac": "m"})
        assert r.sequence == 4 and r.chain_id == "c" and r.record_hmac == "m"
        r.apply_seal({})
        assert r.sequence is None

    @pytest.mark.parametrize(
        "value",
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.12345+00:00", "2026-01-01T00:00:00.1+00:00",
         "2026-01-01 00:00:00", datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=timezone.utc), None, ""],
    )
    def test_parse_timestamp_variants(self, value):
        dt = _parse_timestamp(value)
        assert dt.tzinfo is not None

    def test_chunk_from_dict_defaults(self):
        c = RetrievedChunk.from_dict({"content": None, "similarity_score": "0.25"})
        assert c.content == "" and c.similarity_score == 0.25 and c.metadata == {}


class TestConfig:
    def test_defaults(self):
        cfg = RAGComplianceConfig()
        assert cfg.chain_id == "default" and cfg.sink_url == "" and cfg.retention_days is None
        assert cfg.default_actor == {} and cfg.hmac_key_bytes is None
        assert cfg.max_query_chars == 2000 and cfg.async_overflow == "drop"

    def test_from_env_full(self, monkeypatch):
        env = {
            "RAGCOMPLIANCE_SINK": "sqlite:///x.db",
            "RAGCOMPLIANCE_CHAIN_ID": "writer-1",
            "RAGCOMPLIANCE_CHAIN_HMAC_KEY": "secret",
            "RAGCOMPLIANCE_RETENTION_DAYS": "365",
            "RAGCOMPLIANCE_ASYNC_OVERFLOW": "BLOCK",
            "RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION": "2",
            "RAGCOMPLIANCE_RECORD_ERRORS": "yes",
            "RAGCOMPLIANCE_CAPTURE_PROMPTS": "1",
            "RAGCOMPLIANCE_MAX_QUERY_CHARS": "50",
            "RAGCOMPLIANCE_MAX_ANSWER_CHARS": "junk",
            "RAGCOMPLIANCE_ACTOR_ID": "svc-rag",
            "RAGCOMPLIANCE_ACTOR_TYPE": "service",
            "RAGCOMPLIANCE_MAX_PENDING_RUNS": "-3",
            "RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE": "true",
            "RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS": "https://a.example, https://b.example",
            "RAGCOMPLIANCE_REDACTION_PATTERNS": "email, ssn",
            "RAGCOMPLIANCE_ENABLED": "",
            "RAGCOMPLIANCE_ASYNC_MAX_QUEUE": "oops",
            "RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT": "oops",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        cfg = RAGComplianceConfig.from_env()
        assert cfg.sink_url == "sqlite:///x.db" and cfg.chain_id == "writer-1"
        assert cfg.hmac_key_bytes == b"secret" and cfg.retention_days == 365
        assert cfg.async_overflow == "block" and cfg.supabase_schema_version == 2
        assert cfg.record_errors and cfg.capture_prompts
        assert cfg.max_query_chars == 50 and cfg.max_answer_chars == 4000
        assert cfg.default_actor == {"id": "svc-rag", "type": "service"}
        assert cfg.max_pending_runs == 1
        assert cfg.dashboard_allow_any_workspace is True
        assert cfg.dashboard_cors_origins == ["https://a.example", "https://b.example"]
        assert cfg.redaction_patterns == ["email", "ssn"]
        assert cfg.enabled is True and cfg.async_max_queue == 1000 and cfg.async_shutdown_timeout == 5.0

    def test_bad_retention_ignored(self, monkeypatch):
        monkeypatch.setenv("RAGCOMPLIANCE_RETENTION_DAYS", "forever")
        assert RAGComplianceConfig.from_env().retention_days is None
