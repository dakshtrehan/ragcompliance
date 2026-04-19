"""
SOC 2 evidence report generator.

Produces a Markdown evidence package mapped to the Trust Services Criteria
controls most commonly scoped for an AI / RAG-backed system. The report is
NOT a SOC 2 attestation — only a licensed auditor can issue one. It's a
pre-compiled evidence bundle that makes audit prep faster by showing:

  * the volume and integrity of audit records captured in the period
  * workspace (tenant) isolation enforced by row-level security
  * retention posture (oldest and newest record dates)
  * a sampled set of signed chain records for the auditor to spot-check

Usage (CLI):

    python -m ragcompliance.soc2 \\
      --workspace acme-prod \\
      --start 2026-01-01 \\
      --end 2026-03-31 \\
      --out acme-q1-2026-evidence.md

Usage (programmatic):

    from ragcompliance.soc2 import generate_report
    markdown = generate_report(
        workspace_id="acme-prod",
        start="2026-01-01",
        end="2026-03-31",
    )
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import RAGComplianceConfig
from .storage import AuditStorage

logger = logging.getLogger(__name__)


# Trust Services Criteria we can speak to with audit-log data. We intentionally
# scope this to what ragcompliance has EVIDENCE for — writing claims about
# controls we can't back up with data is how you fail an audit.
CONTROLS: list[dict[str, str]] = [
    {
        "id": "CC6.1",
        "title": "Logical access and segmentation",
        "claim": (
            "Audit logs are segmented per workspace via row-level security in "
            "Supabase. A service-role key is required for cross-workspace "
            "access; application-level keys scope queries to a single "
            "workspace_id."
        ),
    },
    {
        "id": "CC7.2",
        "title": "Detection of anomalies",
        "claim": (
            "The middleware raises Slack alerts on four anomaly rules: "
            "retrieval_returned_zero_chunks, low_similarity, chain_slow, "
            "chain_errored. Alerts are fire-and-forget so an alerting outage "
            "cannot suppress the underlying audit record."
        ),
    },
    {
        "id": "CC8.1",
        "title": "Integrity of system output",
        "claim": (
            "Each chain invocation is signed with SHA-256 over the tuple "
            "(query, retrieved_chunks[i].content, retrieved_chunks[i].source_url, "
            "retrieved_chunks[i].chunk_id, llm_answer). This covers the exact "
            "text the model saw, the URIs it came from, and the answer "
            "produced — the fields an auditor reconstructs to answer 'what "
            "did the model see, and what did it say?'. Similarity scores and "
            "free-form chunk metadata are intentionally out of scope because "
            "they are retriever-implementation details that do not affect "
            "the answer's provenance. Any post-hoc modification of any "
            "covered field is detectable by recomputing the signature."
        ),
    },
    {
        "id": "A1.1",
        "title": "Availability of audit trail",
        "claim": (
            "The middleware writes audit records asynchronously with a "
            "bounded retry queue. Normal process exit drains pending records "
            "via an atexit hook; the drop path is logged, not silent."
        ),
    },
    {
        "id": "C1.1",
        "title": "Confidentiality via least privilege",
        "claim": (
            "The middleware stores only the audit payload required to prove "
            "chain integrity. Supabase row-level security restricts reads to "
            "the workspace's service role, preventing cross-tenant exposure."
        ),
    },
]


@dataclass
class ReportStats:
    record_count: int
    signed_count: int
    unsigned_count: int
    unique_sessions: int
    earliest_ts: str | None
    latest_ts: str | None
    avg_latency_ms: float | None
    models_seen: dict[str, int]

    @property
    def integrity_pct(self) -> float:
        if self.record_count == 0:
            return 100.0
        return round(100.0 * self.signed_count / self.record_count, 2)


# ------------------------------------------------------------------ #
# Core                                                                 #
# ------------------------------------------------------------------ #


def _parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_records(
    storage: AuditStorage,
    workspace_id: str,
    start: datetime,
    end: datetime,
    limit: int = 10_000,
) -> list[dict[str, Any]]:
    """Pull audit records for a workspace within a date range. Relies on
    AuditStorage's Supabase client."""
    if storage._client is None:  # type: ignore[attr-defined]
        raise RuntimeError(
            "SOC 2 report requires a live Supabase client. Check "
            "RAGCOMPLIANCE_SUPABASE_URL and _KEY."
        )
    client = storage._client  # type: ignore[attr-defined]
    q = (
        client.table(storage.config.table_name)
        .select("*")
        .eq("workspace_id", workspace_id)
        .gte("timestamp", start.isoformat())
        .lte("timestamp", end.isoformat())
        .order("timestamp", desc=True)
        .limit(limit)
    )
    result = q.execute()
    return result.data or []


def _compute_stats(records: list[dict[str, Any]]) -> ReportStats:
    if not records:
        return ReportStats(0, 0, 0, 0, None, None, None, {})

    signed = sum(
        1 for r in records
        if isinstance(r.get("chain_signature"), str) and len(r["chain_signature"]) == 64
    )
    unique_sessions = len({r.get("session_id") for r in records if r.get("session_id")})
    timestamps = [r.get("timestamp") for r in records if r.get("timestamp")]
    earliest = min(timestamps) if timestamps else None
    latest = max(timestamps) if timestamps else None

    latencies = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    models: dict[str, int] = {}
    for r in records:
        m = r.get("model_name") or "unknown"
        models[m] = models.get(m, 0) + 1

    return ReportStats(
        record_count=len(records),
        signed_count=signed,
        unsigned_count=len(records) - signed,
        unique_sessions=unique_sessions,
        earliest_ts=earliest,
        latest_ts=latest,
        avg_latency_ms=round(avg_latency, 1) if avg_latency is not None else None,
        models_seen=models,
    )


def _verify_signature(record: dict[str, Any]) -> bool:
    """Recompute the SHA-256 signature from the record payload and confirm it
    matches what was persisted. This is the exact check an auditor will want
    to run on the sampled rows."""
    try:
        payload = {
            "query": record.get("query", ""),
            "chunks": [
                {
                    "content": c.get("content", ""),
                    "source_url": c.get("source_url", ""),
                    "chunk_id": c.get("chunk_id", ""),
                }
                for c in record.get("retrieved_chunks", []) or []
            ],
            "answer": record.get("llm_answer", ""),
        }
        recomputed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return recomputed == record.get("chain_signature")
    except Exception:
        return False


def _sample_records(
    records: list[dict[str, Any]], n: int = 5, seed: int | None = None
) -> list[dict[str, Any]]:
    if not records:
        return []
    rng = random.Random(seed)
    if len(records) <= n:
        return list(records)
    return rng.sample(records, n)


def _format_md(
    workspace_id: str,
    start: datetime,
    end: datetime,
    stats: ReportStats,
    sampled: list[dict[str, Any]],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    period_days = max(1, (end - start).days)

    lines: list[str] = []
    lines.append(f"# RAG Audit Evidence — {workspace_id}")
    lines.append("")
    lines.append(
        f"**Reporting period:** {start.date().isoformat()} to {end.date().isoformat()} "
        f"({period_days} day{'s' if period_days != 1 else ''})"
    )
    lines.append(f"**Generated:** {generated_at}")
    lines.append("**Tool:** ragcompliance")
    lines.append("")
    lines.append(
        "> This document is an evidence package intended to accompany a SOC 2 "
        "audit. It is not itself a SOC 2 attestation — only a licensed auditor "
        "can issue one. The data below is pulled directly from the "
        "`rag_audit_logs` table for the named workspace and signed records are "
        "verifiable by recomputing the SHA-256 signature over "
        "`(query, retrieved_chunks, llm_answer)`."
    )
    lines.append("")

    # ---- Executive summary ----
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Chain invocations captured: **{stats.record_count}**")
    lines.append(f"- Unique sessions: **{stats.unique_sessions}**")
    lines.append(
        f"- Signature integrity: **{stats.integrity_pct}%** "
        f"({stats.signed_count} signed, {stats.unsigned_count} unsigned)"
    )
    if stats.avg_latency_ms is not None:
        lines.append(f"- Average end-to-end chain latency: **{stats.avg_latency_ms} ms**")
    if stats.earliest_ts and stats.latest_ts:
        lines.append(f"- Earliest record: `{stats.earliest_ts}`")
        lines.append(f"- Latest record: `{stats.latest_ts}`")
    if stats.models_seen:
        model_lines = ", ".join(
            f"{name} ({count})" for name, count in sorted(
                stats.models_seen.items(), key=lambda kv: -kv[1]
            )
        )
        lines.append(f"- Models observed: {model_lines}")
    lines.append("")

    # ---- Control matrix ----
    lines.append("## Control evidence matrix")
    lines.append("")
    lines.append("| Control | Title | Claim | Supporting evidence |")
    lines.append("|---|---|---|---|")
    for ctrl in CONTROLS:
        evidence = {
            "CC6.1": f"`workspace_id = '{workspace_id}'` RLS-scoped reads; {stats.record_count} records in scope.",
            "CC7.2": "See Slack alerter rules in the ragcompliance README.",
            "CC8.1": f"{stats.signed_count} signed records verifiable in the sample below.",
            "A1.1": "Async writer queue + atexit drain; see code in ragcompliance/storage.py.",
            "C1.1": "Supabase RLS + service-role key separation.",
        }.get(ctrl["id"], "")
        lines.append(
            f"| {ctrl['id']} | {ctrl['title']} | {ctrl['claim']} | {evidence} |"
        )
    lines.append("")

    # ---- Sample + signature verification ----
    lines.append("## Sampled records with signature verification")
    lines.append("")
    if not sampled:
        lines.append("_No records in the reporting period. Nothing to sample._")
    else:
        lines.append(
            f"The table below is a random sample of {len(sampled)} record(s). "
            "Each signature has been recomputed from the persisted payload; "
            "`verified = true` means the record has not been tampered with "
            "since it was written."
        )
        lines.append("")
        lines.append("| Timestamp | Session | Model | Latency (ms) | Signature (first 16) | Verified |")
        lines.append("|---|---|---|---|---|---|")
        for r in sampled:
            ok = _verify_signature(r)
            sig = (r.get("chain_signature") or "")[:16]
            lines.append(
                f"| {r.get('timestamp', '')} | `{r.get('session_id', '')}` | "
                f"{r.get('model_name', 'unknown')} | {r.get('latency_ms', '')} | "
                f"`{sig}...` | {'✅' if ok else '❌'} |"
            )
    lines.append("")

    # ---- Appendix: methodology ----
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Records were pulled from the `rag_audit_logs` Supabase table scoped "
        f"to `workspace_id = '{workspace_id}'` and filtered to timestamps in "
        f"`[{start.isoformat()}, {end.isoformat()}]`. The sample is drawn "
        "uniformly at random without replacement; the auditor can request a "
        "different seed or sample size by re-running the generator with "
        "`--sample N --seed K`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Generated by ragcompliance · https://github.com/dakshtrehan/ragcompliance*"
    )

    return "\n".join(lines) + "\n"


def generate_report(
    workspace_id: str,
    start: str | datetime,
    end: str | datetime,
    sample_size: int = 25,
    seed: int | None = None,
    config: RAGComplianceConfig | None = None,
    storage: AuditStorage | None = None,
) -> str:
    """Pull audit records for a workspace over a date range and return a
    Markdown evidence package."""
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt <= start_dt:
        raise ValueError("end must be strictly after start")

    cfg = config or RAGComplianceConfig.from_env()
    store = storage or AuditStorage(cfg)
    records = _load_records(store, workspace_id, start_dt, end_dt)
    stats = _compute_stats(records)
    sampled = _sample_records(records, n=sample_size, seed=seed)
    return _format_md(workspace_id, start_dt, end_dt, stats, sampled)


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragcompliance-soc2",
        description="Generate a SOC 2 evidence report for a workspace.",
    )
    parser.add_argument("--workspace", required=True, help="Workspace ID")
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2026-01-01")
    parser.add_argument("--end", required=True, help="ISO date, e.g. 2026-03-31")
    parser.add_argument("--sample", type=int, default=25, help="Records to sample")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic sample seed")
    parser.add_argument("--out", default=None, help="Output file (Markdown). Defaults to stdout.")

    args = parser.parse_args(argv)
    try:
        md = generate_report(
            workspace_id=args.workspace,
            start=args.start,
            end=args.end,
            sample_size=args.sample,
            seed=args.seed,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {args.out} ({len(md)} bytes)", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
