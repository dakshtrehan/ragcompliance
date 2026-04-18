"""Tests for the SOC 2 evidence report generator."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ragcompliance.soc2 import (
    CONTROLS,
    ReportStats,
    _compute_stats,
    _format_md,
    _parse_date,
    _sample_records,
    _verify_signature,
    generate_report,
)


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #


def _sign(query: str, chunks: list[dict], answer: str) -> str:
    payload = {
        "query": query,
        "chunks": [
            {
                "content": c.get("content", ""),
                "source_url": c.get("source_url", ""),
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in chunks
        ],
        "answer": answer,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _record(
    ts: str = "2026-02-15T10:00:00+00:00",
    query: str = "why is the sky blue",
    chunks: list[dict] | None = None,
    answer: str = "rayleigh scattering",
    model: str = "gpt-4o-mini",
    latency: int = 420,
    session: str = "sess-1",
    sign: bool = True,
) -> dict:
    chunks = chunks if chunks is not None else [
        {
            "content": "Rayleigh scattering explains blue skies.",
            "source_url": "sci://physics",
            "chunk_id": "c1",
            "similarity_score": 0.91,
        }
    ]
    sig = _sign(query, chunks, answer) if sign else ""
    return {
        "id": f"rec-{ts}",
        "timestamp": ts,
        "session_id": session,
        "workspace_id": "ws-test",
        "query": query,
        "retrieved_chunks": chunks,
        "llm_answer": answer,
        "model_name": model,
        "chain_signature": sig,
        "latency_ms": latency,
    }


# ------------------------------------------------------------------ #
# _parse_date                                                          #
# ------------------------------------------------------------------ #


class TestParseDate:
    def test_parses_iso_date_string(self):
        dt = _parse_date("2026-01-01")
        assert dt.year == 2026 and dt.month == 1 and dt.day == 1
        assert dt.tzinfo is not None

    def test_parses_full_iso_with_z(self):
        dt = _parse_date("2026-03-01T12:00:00Z")
        assert dt.tzinfo is not None

    def test_passes_datetime_through(self):
        now = datetime.now(timezone.utc)
        assert _parse_date(now) == now

    def test_naive_datetime_gets_utc(self):
        naive = datetime(2026, 2, 1, 12, 0, 0)
        result = _parse_date(naive)
        assert result.tzinfo is not None


# ------------------------------------------------------------------ #
# _compute_stats                                                       #
# ------------------------------------------------------------------ #


class TestComputeStats:
    def test_empty_records(self):
        s = _compute_stats([])
        assert s.record_count == 0
        assert s.integrity_pct == 100.0
        assert s.models_seen == {}

    def test_all_signed_is_100pct(self):
        recs = [_record(ts=f"2026-02-0{i+1}T10:00:00+00:00") for i in range(3)]
        s = _compute_stats(recs)
        assert s.record_count == 3
        assert s.signed_count == 3
        assert s.unsigned_count == 0
        assert s.integrity_pct == 100.0

    def test_mixed_signed_and_unsigned(self):
        recs = [
            _record(ts="2026-02-01T10:00:00+00:00", sign=True),
            _record(ts="2026-02-02T10:00:00+00:00", sign=False),
            _record(ts="2026-02-03T10:00:00+00:00", sign=True),
        ]
        s = _compute_stats(recs)
        assert s.signed_count == 2
        assert s.unsigned_count == 1
        assert s.integrity_pct == round(100 * 2 / 3, 2)

    def test_unique_sessions_are_deduped(self):
        recs = [
            _record(session="a"),
            _record(ts="2026-02-02T10:00:00+00:00", session="a"),
            _record(ts="2026-02-03T10:00:00+00:00", session="b"),
        ]
        assert _compute_stats(recs).unique_sessions == 2

    def test_avg_latency(self):
        recs = [_record(latency=100), _record(ts="2026-02-02T10:00:00+00:00", latency=300)]
        assert _compute_stats(recs).avg_latency_ms == 200.0

    def test_models_histogram(self):
        recs = [
            _record(model="gpt-4o"),
            _record(ts="2026-02-02T10:00:00+00:00", model="gpt-4o"),
            _record(ts="2026-02-03T10:00:00+00:00", model="claude-3-5"),
        ]
        s = _compute_stats(recs)
        assert s.models_seen == {"gpt-4o": 2, "claude-3-5": 1}

    def test_earliest_and_latest(self):
        recs = [
            _record(ts="2026-02-15T10:00:00+00:00"),
            _record(ts="2026-01-01T10:00:00+00:00"),
            _record(ts="2026-03-30T10:00:00+00:00"),
        ]
        s = _compute_stats(recs)
        assert s.earliest_ts == "2026-01-01T10:00:00+00:00"
        assert s.latest_ts == "2026-03-30T10:00:00+00:00"


# ------------------------------------------------------------------ #
# _verify_signature                                                    #
# ------------------------------------------------------------------ #


class TestVerifySignature:
    def test_valid_signature_verifies(self):
        assert _verify_signature(_record()) is True

    def test_tampered_answer_fails_verification(self):
        rec = _record()
        rec["llm_answer"] = "tampered answer"
        assert _verify_signature(rec) is False

    def test_tampered_query_fails_verification(self):
        rec = _record()
        rec["query"] = "totally different question"
        assert _verify_signature(rec) is False

    def test_missing_signature_fails(self):
        rec = _record(sign=False)
        assert _verify_signature(rec) is False

    def test_malformed_record_returns_false(self):
        # Pass in garbage. Should not explode.
        assert _verify_signature({"chain_signature": None}) is False


# ------------------------------------------------------------------ #
# _sample_records                                                      #
# ------------------------------------------------------------------ #


class TestSampleRecords:
    def test_empty_input_returns_empty(self):
        assert _sample_records([]) == []

    def test_returns_all_when_n_exceeds_count(self):
        recs = [_record(ts=f"2026-02-0{i+1}T10:00:00+00:00") for i in range(3)]
        assert len(_sample_records(recs, n=10)) == 3

    def test_returns_n_when_enough_records(self):
        recs = [_record(ts=f"2026-02-{i:02d}T10:00:00+00:00") for i in range(1, 21)]
        assert len(_sample_records(recs, n=5)) == 5

    def test_seed_is_deterministic(self):
        recs = [_record(ts=f"2026-02-{i:02d}T10:00:00+00:00") for i in range(1, 21)]
        a = _sample_records(recs, n=5, seed=42)
        b = _sample_records(recs, n=5, seed=42)
        assert [r["id"] for r in a] == [r["id"] for r in b]


# ------------------------------------------------------------------ #
# _format_md                                                            #
# ------------------------------------------------------------------ #


class TestFormatMd:
    def test_empty_period_produces_valid_md(self):
        stats = ReportStats(0, 0, 0, 0, None, None, None, {})
        md = _format_md(
            "ws-empty",
            _parse_date("2026-01-01"),
            _parse_date("2026-01-31"),
            stats,
            [],
        )
        assert "# RAG Audit Evidence — ws-empty" in md
        assert "No records in the reporting period" in md
        # Every control still appears in the matrix.
        for ctrl in CONTROLS:
            assert ctrl["id"] in md

    def test_sampled_table_contains_verification_column(self):
        recs = [_record(ts="2026-02-01T10:00:00+00:00")]
        stats = _compute_stats(recs)
        md = _format_md(
            "ws-test",
            _parse_date("2026-01-01"),
            _parse_date("2026-03-01"),
            stats,
            recs,
        )
        assert "Signature (first 16)" in md
        assert "Verified" in md
        # Signature verified => checkmark.
        assert "✅" in md

    def test_tampered_record_renders_x_mark(self):
        rec = _record()
        rec["llm_answer"] = "tampered"
        stats = _compute_stats([rec])
        md = _format_md(
            "ws-test",
            _parse_date("2026-01-01"),
            _parse_date("2026-03-01"),
            stats,
            [rec],
        )
        assert "❌" in md

    def test_header_includes_period_days(self):
        md = _format_md(
            "ws-test",
            _parse_date("2026-01-01"),
            _parse_date("2026-01-31"),
            ReportStats(0, 0, 0, 0, None, None, None, {}),
            [],
        )
        assert "30 days" in md

    def test_renders_all_controls(self):
        md = _format_md(
            "ws-test",
            _parse_date("2026-01-01"),
            _parse_date("2026-01-31"),
            ReportStats(0, 0, 0, 0, None, None, None, {}),
            [],
        )
        for ctrl in CONTROLS:
            assert ctrl["id"] in md
            assert ctrl["title"] in md


# ------------------------------------------------------------------ #
# generate_report (end-to-end with mocked storage)                     #
# ------------------------------------------------------------------ #


class _FakeSupabaseQuery:
    """A tiny fake that mirrors the chain supabase-py exposes:
    client.table(x).select(...).eq(...).gte(...).lte(...).order(...).limit(...).execute()"""

    def __init__(self, data: list[dict]):
        self._data = data

    def select(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def gte(self, *args, **kwargs):
        return self

    def lte(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return SimpleNamespace(data=self._data)


class _FakeClient:
    def __init__(self, data: list[dict]):
        self._data = data

    def table(self, name: str):
        return _FakeSupabaseQuery(self._data)


class _FakeStorage:
    """Minimal stand-in for AuditStorage for generator tests."""

    def __init__(self, records: list[dict], workspace_id: str = "ws-test"):
        self._client = _FakeClient(records)
        self.config = SimpleNamespace(
            table_name="rag_audit_logs",
            workspace_id=workspace_id,
        )


class TestGenerateReport:
    def test_end_to_end_with_mocked_storage(self):
        recs = [
            _record(ts="2026-02-01T10:00:00+00:00"),
            _record(ts="2026-02-10T10:00:00+00:00", query="q2", answer="a2"),
            _record(ts="2026-02-20T10:00:00+00:00", query="q3", answer="a3"),
        ]
        fake = _FakeStorage(recs)
        md = generate_report(
            workspace_id="ws-test",
            start="2026-01-01",
            end="2026-03-01",
            sample_size=3,
            seed=1,
            storage=fake,
        )
        assert "# RAG Audit Evidence — ws-test" in md
        assert "Chain invocations captured: **3**" in md
        # 100% integrity because every record is properly signed.
        assert "Signature integrity: **100.0%**" in md
        # Every control shows up.
        for ctrl in CONTROLS:
            assert ctrl["id"] in md

    def test_empty_workspace_still_renders(self):
        fake = _FakeStorage([])
        md = generate_report(
            workspace_id="ws-empty",
            start="2026-01-01",
            end="2026-01-31",
            storage=fake,
        )
        assert "Chain invocations captured: **0**" in md
        assert "No records in the reporting period" in md

    def test_rejects_inverted_date_range(self):
        fake = _FakeStorage([])
        with pytest.raises(ValueError, match="strictly after"):
            generate_report(
                workspace_id="ws",
                start="2026-03-01",
                end="2026-01-01",
                storage=fake,
            )

    def test_raises_when_storage_has_no_client(self):
        fake = _FakeStorage([])
        fake._client = None
        with pytest.raises(RuntimeError, match="live Supabase client"):
            generate_report(
                workspace_id="ws",
                start="2026-01-01",
                end="2026-01-31",
                storage=fake,
            )
