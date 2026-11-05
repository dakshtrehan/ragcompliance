"""SOC 2 evidence report generator.

Produces a Markdown evidence summary mapped to the Trust Services
Criteria (and ISO 27001 Annex A controls) that ragcompliance has data
for. The report is NOT a SOC 2 attestation; only a licensed auditor can
issue one. It is a pre compiled summary that shows:

* the volume of audit records captured in the period
* content signature verification across every record (not a sample)
* hash chain verification per workspace and chain
* retention posture (oldest and newest record)
* a sampled set of records for the auditor to spot check by hand

Usage (CLI)::

    ragcompliance soc2 --sink sqlite:///audit.db --workspace acme-prod \\
        --start 2026-01-01 --end 2026-03-31 --out acme-q1-2026-evidence.md

Usage (programmatic)::

    from ragcompliance.soc2 import generate_report
    markdown = generate_report(workspace_id="acme-prod", start="2026-01-01",
                               end="2026-03-31", storage=storage)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .chain import (
    RECORD_KIND_INVOCATION,
    VerificationResult,
    verify_content_signature,
    verify_records,
)
from .config import RAGComplianceConfig
from .controls import CONTROL_STATUS_LABELS, CONTROLS

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 25


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
    verified_count: int = 0
    failed_count: int = 0
    checkpoint_count: int = 0
    actors_seen: int = 0
    redaction_findings: dict[str, int] = field(default_factory=dict)
    chain: VerificationResult | None = None

    @property
    def integrity_pct(self) -> float:
        """Share of invocation records whose content signature verifies."""
        invocations = self.record_count - self.checkpoint_count
        if invocations <= 0:
            return 100.0
        return round(100.0 * self.verified_count / invocations, 2)


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
    storage: Any,
    workspace_id: str,
    start: datetime,
    end: datetime,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Pull every record for a workspace in a date range.

    Accepts an ``AuditStorage``, any ``AuditSink``, or a legacy object
    exposing a Supabase ``_client`` (0.1.x). Pagination is exhaustive;
    ``limit`` exists only for callers who explicitly want a cap.
    """
    iterator = getattr(storage, "iter_records", None)
    if callable(iterator):
        records = list(iterator(workspace_id=workspace_id, start=start, end=end, limit=limit))
        return records

    client = getattr(storage, "_client", None)
    if client is None:
        raise RuntimeError(
            "SOC 2 report needs a storage with iter_records() or a live Supabase client. "
            "Check RAGCOMPLIANCE_SINK or RAGCOMPLIANCE_SUPABASE_URL and _KEY."
        )
    from .sinks.supabase import unpack_row

    table = storage.config.table_name
    page = 1000
    offset = 0
    out: list[dict[str, Any]] = []
    while True:
        q = (
            client.table(table)
            .select("*")
            .eq("workspace_id", workspace_id)
            .gte("timestamp", start.isoformat())
            .lte("timestamp", end.isoformat())
            .order("timestamp")
            .range(offset, offset + page - 1)
        )
        rows = q.execute().data or []
        out.extend(unpack_row(r) for r in rows)
        if len(rows) < page or (limit is not None and len(out) >= limit):
            break
        offset += page
    return out[:limit] if limit is not None else out


def _compute_stats(records: list[dict[str, Any]], hmac_key: bytes | None = None) -> ReportStats:
    if not records:
        return ReportStats(0, 0, 0, 0, None, None, None, {}, chain=verify_records([]))

    signed = 0
    verified = 0
    failed = 0
    checkpoints = 0
    findings: dict[str, int] = {}
    actors: set[str] = set()
    for r in records:
        kind = r.get("kind") or RECORD_KIND_INVOCATION
        if kind != RECORD_KIND_INVOCATION:
            checkpoints += 1
            continue
        sig = r.get("chain_signature")
        if isinstance(sig, str) and len(sig) == 64:
            signed += 1
            if verify_content_signature(r):
                verified += 1
            else:
                failed += 1
        for k, v in ((r.get("extra") or {}).get("redaction_findings") or {}).items():
            findings[k] = findings.get(k, 0) + int(v)
        actor_id = (r.get("actor") or {}).get("id")
        if actor_id:
            actors.add(str(actor_id))

    invocations = len(records) - checkpoints
    unique_sessions = len({r.get("session_id") for r in records if r.get("session_id")})
    timestamps = [str(r["timestamp"]) for r in records if r.get("timestamp")]
    latencies = [
        float(r["latency_ms"])
        for r in records
        if r.get("latency_ms") is not None and (r.get("kind") or RECORD_KIND_INVOCATION) == RECORD_KIND_INVOCATION
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    models: dict[str, int] = {}
    for r in records:
        if (r.get("kind") or RECORD_KIND_INVOCATION) != RECORD_KIND_INVOCATION:
            continue
        m = r.get("model_name") or "unknown"
        models[m] = models.get(m, 0) + 1

    return ReportStats(
        record_count=len(records),
        signed_count=signed,
        unsigned_count=invocations - signed,
        unique_sessions=unique_sessions,
        earliest_ts=min(timestamps) if timestamps else None,
        latest_ts=max(timestamps) if timestamps else None,
        avg_latency_ms=round(avg_latency, 1) if avg_latency is not None else None,
        models_seen=models,
        verified_count=verified,
        failed_count=failed,
        checkpoint_count=checkpoints,
        actors_seen=len(actors),
        redaction_findings=findings,
        chain=verify_records(records, hmac_key=hmac_key),
    )


def _verify_signature(record: dict[str, Any]) -> bool:
    """Recompute the content signature and compare (0.1.x compatible name)."""
    return verify_content_signature(record)


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
    *,
    source: str = "audit sink",
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    period_days = max(1, (end - start).days)

    lines: list[str] = []
    lines.append(f"# RAG Audit Evidence: {workspace_id}")
    lines.append("")
    lines.append(
        f"**Reporting period:** {start.date().isoformat()} to {end.date().isoformat()} "
        f"({period_days} day{'s' if period_days != 1 else ''})"
    )
    lines.append(f"**Generated:** {generated_at}")
    lines.append(f"**Tool:** ragcompliance {__version__}")
    lines.append(f"**Source:** {source}")
    lines.append("")
    lines.append(
        "> This document is an evidence summary intended to accompany a SOC 2 or "
        "ISO 27001 audit. It is not itself an attestation or certificate; only a "
        "licensed auditor or accredited body can issue one. Every record in the "
        "period was re-verified: the content signature (SHA-256 over query, "
        "retrieved chunks and answer) and the hash chain (prev_hash, record_hash, "
        "sequence) were recomputed from the stored data."
    )
    lines.append("")

    # ---- Executive summary ----
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- Records in scope: **{stats.record_count}** "
                 f"({stats.record_count - stats.checkpoint_count} invocations, "
                 f"{stats.checkpoint_count} retention checkpoints)")
    lines.append(f"- Unique sessions: **{stats.unique_sessions}**")
    lines.append(f"- Distinct actors attributed: **{stats.actors_seen}**")
    lines.append(
        f"- Content signature integrity: **{stats.integrity_pct}%** "
        f"({stats.verified_count} verified, {stats.failed_count} failed, "
        f"{stats.unsigned_count} unsigned)"
    )
    chain = stats.chain
    if chain is not None:
        state = "intact" if chain.ok else f"BROKEN ({len(chain.problems)} problem(s))"
        lines.append(
            f"- Hash chain: **{state}** across {len(chain.chains)} chain(s); "
            f"{chain.legacy_unchained} legacy record(s) without chain fields"
        )
    if stats.avg_latency_ms is not None:
        lines.append(f"- Average end-to-end chain latency: **{stats.avg_latency_ms} ms**")
    if stats.earliest_ts and stats.latest_ts:
        lines.append(f"- Earliest record: `{stats.earliest_ts}`")
        lines.append(f"- Latest record: `{stats.latest_ts}`")
    if stats.models_seen:
        model_lines = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(stats.models_seen.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- Models observed: {model_lines}")
    if stats.redaction_findings:
        red = ", ".join(f"{k}: {v}" for k, v in sorted(stats.redaction_findings.items()))
        lines.append(f"- Redaction findings (values removed before storage): {red}")
    lines.append("")

    # ---- Chain detail ----
    if chain is not None and chain.chains:
        lines.append("## Hash chain verification")
        lines.append("")
        lines.append("| Workspace | Chain | Records | First seq | Last seq | Head hash (first 16) | Anchored by checkpoint | Result |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in chain.chains:
            lines.append(
                f"| {c.workspace_id} | {c.chain_id} | {c.records} | {c.first_sequence} | "
                f"{c.last_sequence} | `{c.head_hash[:16]}` | "
                f"{'yes' if c.anchored_by_checkpoint else 'no'} | "
                f"{'intact' if c.ok else 'BROKEN'} |"
            )
        if chain.problems:
            lines.append("")
            lines.append("Problems:")
            lines.append("")
            for p in chain.problems[:50]:
                lines.append(f"- seq {p.sequence} ({p.record_id}): {p.code}: {p.detail}")
            if len(chain.problems) > 50:
                lines.append(f"- ... and {len(chain.problems) - 50} more")
        lines.append("")

    # ---- Control matrix ----
    lines.append("## Control evidence matrix")
    lines.append("")
    lines.append("| Framework | Control | Title | Status | Evidence in this package |")
    lines.append("|---|---|---|---|---|")
    for ctrl in CONTROLS:
        lines.append(
            f"| {ctrl['framework']} | {ctrl['id']} | {ctrl['title']} | "
            f"{CONTROL_STATUS_LABELS[ctrl['status']]} | {ctrl['evidence']} |"
        )
    lines.append("")
    lines.append(
        "Status legend: 'Supported by the library' means the evidence is produced by "
        "ragcompliance itself; 'Operator dependent' means the control relies on how the "
        "library is deployed; 'Partially addressed' means the library covers part of the "
        "criterion. See docs/control-mapping.md for the caveats on each row."
    )
    lines.append("")

    # ---- Sample ----
    lines.append("## Sampled records for manual spot check")
    lines.append("")
    if not sampled:
        lines.append("_No records in the reporting period. Nothing to sample._")
    else:
        lines.append(
            f"The table below is a random sample of {len(sampled)} record(s) for an auditor "
            "to inspect by hand. Verification columns were recomputed from the stored payload."
        )
        lines.append("")
        lines.append("| Timestamp | Session | Actor | Model | Latency (ms) | Seq | Signature (first 16) | Content verified |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sampled:
            ok = _verify_signature(r)
            sig = (r.get("chain_signature") or "")[:16]
            actor = (r.get("actor") or {}).get("id", "")
            lines.append(
                f"| {r.get('timestamp', '')} | `{r.get('session_id', '')}` | {actor} | "
                f"{r.get('model_name') or 'unknown'} | {r.get('latency_ms', '')} | "
                f"{r.get('sequence', '')} | `{sig}` | {'yes' if ok else 'NO'} |"
            )
    lines.append("")

    # ---- Methodology ----
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"Records were read from the {source} scoped to workspace `{workspace_id}` and "
        f"filtered to timestamps in `[{start.isoformat()}, {end.isoformat()}]`, with "
        "exhaustive pagination. Every record's content signature and chain link was "
        "recomputed; the sample is drawn uniformly at random without replacement and can "
        "be reproduced with the same `--sample N --seed K`. Independent re-verification: "
        "`ragcompliance verify --sink <url> --workspace <id>`."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by ragcompliance, https://github.com/dakshtrehan/ragcompliance*")
    return "\n".join(lines) + "\n"


def generate_report(
    workspace_id: str,
    start: str | datetime,
    end: str | datetime,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int | None = None,
    config: RAGComplianceConfig | None = None,
    storage: Any | None = None,
    records: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Return a Markdown evidence summary for a workspace over a date range.

    Pass ``storage`` (an ``AuditStorage`` or sink) or ``records`` directly.
    """
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    if end_dt <= start_dt:
        raise ValueError("end must be strictly after start")

    cfg = config or RAGComplianceConfig.from_env()
    if records is not None:
        loaded = [r for r in records if _in_period(r, start_dt, end_dt, workspace_id)]
        source = "records supplied by the caller"
    else:
        if storage is None:
            from .storage import AuditStorage

            cfg.async_writes = False
            storage = AuditStorage(cfg)
        loaded = _load_records(storage, workspace_id, start_dt, end_dt)
        described = getattr(storage, "describe", None)
        sink_desc = described() if callable(described) else None
        sink_type = (sink_desc or {}).get("sink", {}) if isinstance(sink_desc, dict) else {}
        source = f"{(sink_type or {}).get('type', 'audit')} sink"
    stats = _compute_stats(loaded, hmac_key=cfg.hmac_key_bytes)
    sampled = _sample_records(loaded, n=sample_size, seed=seed)
    return _format_md(workspace_id, start_dt, end_dt, stats, sampled, source=source)


def _in_period(record: dict[str, Any], start: datetime, end: datetime, workspace_id: str) -> bool:
    from .models import _parse_timestamp

    if record.get("workspace_id") not in (None, "", workspace_id):
        return False
    ts = _parse_timestamp(record.get("timestamp"))
    return start <= ts <= end


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ragcompliance-soc2",
        description="Generate a SOC 2 / ISO 27001 evidence summary for a workspace.",
    )
    parser.add_argument("--workspace", required=True, help="Workspace ID")
    parser.add_argument("--start", required=True, help="ISO date, e.g. 2026-01-01")
    parser.add_argument("--end", required=True, help="ISO date, e.g. 2026-03-31")
    parser.add_argument("--sample", type=int, default=25, help="Records to sample")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic sample seed")
    parser.add_argument("--sink", default=None, help="Sink URL (defaults to RAGCOMPLIANCE_SINK)")
    parser.add_argument("--out", default=None, help="Output file (Markdown). Defaults to stdout.")

    args = parser.parse_args(argv)
    try:
        cfg = RAGComplianceConfig.from_env()
        cfg.async_writes = False
        if args.sink:
            cfg.sink_url = args.sink
        md = generate_report(
            workspace_id=args.workspace,
            start=args.start,
            end=args.end,
            sample_size=args.sample,
            seed=args.seed,
            config=cfg,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
