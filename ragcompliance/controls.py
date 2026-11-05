"""Control mapping: which product features support which audit criteria.

This is the single source for the control table rendered in the SOC 2
report, the evidence pack and ``docs/control-mapping.md``. Wording is
deliberately conservative:

* ``supported``: the library produces the evidence on its own.
* ``operator``: the library helps, but the control depends on how the
  operator deploys it (key management, access to the store, SSO).
* ``partial``: the library covers part of the criterion; the rest is
  out of scope for an audit logging component.

Nothing here is a certification claim. SOC 2 reports are issued by a
licensed CPA firm after examining the whole system; ISO 27001
certificates are issued by accredited bodies. This table says which
criteria a reviewer can expect ragcompliance to produce evidence for.
"""

from __future__ import annotations

from typing import Any

CONTROL_STATUS_LABELS = {
    "supported": "Supported by the library",
    "operator": "Operator dependent",
    "partial": "Partially addressed",
}

CONTROLS: list[dict[str, Any]] = [
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "CC6.1",
        "title": "Logical access security over protected information assets",
        "status": "operator",
        "feature": "Workspace scoped records, actor attribution, dashboard SSO (OIDC), "
        "dashboard reads pinned to the configured workspace.",
        "evidence": "actor field on every record; dashboard SSO configuration; "
        "RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE left unset.",
        "caveat": "Supabase row level security is bypassed by the service role key the "
        "library uses. Tenant isolation at the store depends on who holds that key.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "CC6.5",
        "title": "Disposal of data and software when no longer required",
        "status": "supported",
        "feature": "RetentionPolicy and purge_expired() with a chain checkpoint record.",
        "evidence": "retention_checkpoint records in the chain; retention.json in the "
        "evidence pack; purge output.",
        "caveat": "Purging is explicit (CLI or API call). Sinks with Object Lock refuse it.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "CC7.2",
        "title": "Monitoring of system components for anomalies",
        "status": "partial",
        "feature": "Slack alerts on zero retrieval, low similarity, slow chain, chain error.",
        "evidence": "Alert configuration (thresholds) and the alert payloads themselves.",
        "caveat": "Covers the RAG pipeline only, not the host infrastructure.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "CC7.3",
        "title": "Evaluation of security events to determine impact",
        "status": "partial",
        "feature": "Error records (RAGCOMPLIANCE_RECORD_ERRORS) with redacted context and "
        "lineage.status.",
        "evidence": "Records with lineage.status = error and extra.error.",
        "caveat": "Triage and impact assessment are human processes.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "CC8.1",
        "title": "Change management: integrity of system output",
        "status": "supported",
        "feature": "Content signature per record plus a hash chain (prev_hash, record_hash, "
        "sequence) per workspace and chain, optional HMAC seal, append-only triggers.",
        "evidence": "ragcompliance verify output; verification.json in the evidence pack; "
        "tool_version on every record.",
        "caveat": "A hash chain is tamper evident. Tamper resistance needs the HMAC key or "
        "a WORM store (S3 Object Lock) that the writer cannot modify.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "A1.1",
        "title": "Capacity management for availability of the audit trail",
        "status": "partial",
        "feature": "Bounded async queue, atexit drain, dropped_count counter, "
        "RAGCOMPLIANCE_ASYNC_OVERFLOW=block option.",
        "evidence": "dropped_count in the evidence pack manifest; storage.describe().",
        "caveat": "With the default drop policy the log is not guaranteed complete under "
        "sustained overload; the counter makes that visible.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "C1.1",
        "title": "Identification and protection of confidential information",
        "status": "supported",
        "feature": "Redaction before hashing (regex patterns, hooks, pseudonymisation), "
        "redaction findings per record, no query text in alerts by default.",
        "evidence": "extra.redaction_findings; redaction.json in the evidence pack.",
        "caveat": "Regex redaction misses free text names and addresses; use hooks for NER.",
    },
    {
        "framework": "SOC 2 TSC (2017)",
        "id": "PI1.4",
        "title": "Complete, accurate and timely outputs",
        "status": "partial",
        "feature": "Data lineage: query, chunk hashes, prompt hash, model parameters, "
        "answer hash, timing, truncation flags.",
        "evidence": "lineage field on every record.",
        "caveat": "The library records what the pipeline did; it does not judge accuracy.",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.5.33",
        "title": "Protection of records",
        "status": "supported",
        "feature": "Hash chain, HMAC seal, append-only sinks, Object Lock support.",
        "evidence": "ragcompliance verify; sink description in the evidence pack.",
        "caveat": "",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.8.15",
        "title": "Logging",
        "status": "supported",
        "feature": "One record per RAG invocation with actor, timestamps, query, retrieved "
        "content, answer, model and parameters.",
        "evidence": "records.jsonl in the evidence pack.",
        "caveat": "",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.8.16",
        "title": "Monitoring activities",
        "status": "partial",
        "feature": "Anomaly alerts and error records.",
        "evidence": "Alert configuration and payloads.",
        "caveat": "Pipeline level only.",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.8.10",
        "title": "Information deletion",
        "status": "supported",
        "feature": "Retention policy with checkpointed purge.",
        "evidence": "retention_checkpoint records; retention.json.",
        "caveat": "Operator schedules the purge.",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.8.11",
        "title": "Data masking",
        "status": "supported",
        "feature": "Redaction and pseudonymisation before storage.",
        "evidence": "redaction.json; extra.redaction_findings.",
        "caveat": "Pattern coverage is the operator's responsibility to review.",
    },
    {
        "framework": "ISO/IEC 27001:2022 Annex A",
        "id": "A.5.15",
        "title": "Access control",
        "status": "operator",
        "feature": "Actor attribution per record; dashboard SSO.",
        "evidence": "actor field; SSO configuration.",
        "caveat": "Access control to the store itself is outside the library.",
    },
]


def controls_markdown() -> str:
    """Render the mapping as a Markdown table (ASCII only)."""
    lines = [
        "| Framework | Control | Title | Status | Feature | Evidence | Caveat |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in CONTROLS:
        lines.append(
            "| {framework} | {id} | {title} | {status} | {feature} | {evidence} | {caveat} |".format(
                framework=c["framework"],
                id=c["id"],
                title=c["title"],
                status=CONTROL_STATUS_LABELS[c["status"]],
                feature=c["feature"],
                evidence=c["evidence"],
                caveat=c["caveat"] or "none",
            )
        )
    return "\n".join(lines) + "\n"


def controls_json() -> list[dict[str, Any]]:
    return [
        {**c, "status_label": CONTROL_STATUS_LABELS[c["status"]]} for c in CONTROLS
    ]
