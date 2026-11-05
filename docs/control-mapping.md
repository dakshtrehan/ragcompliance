# Control mapping (SOC 2 and ISO 27001)

This table maps ragcompliance features to the criteria auditors most often scope for an AI system that uses retrieval augmented generation. It is generated from `ragcompliance/controls.py`, which is also what the evidence pack and the summary report embed, so the three never drift.

**This is not a certification claim.** A SOC 2 report is issued by a licensed CPA firm after examining the whole system and its operation over a period; an ISO 27001 certificate is issued by an accredited certification body. ragcompliance is one component. The table says which criteria a reviewer can expect this component to produce evidence for, and how far that evidence goes.

Status legend:

* **Supported by the library**: the library produces the evidence on its own.
* **Operator dependent**: the library helps, but the control depends on how you deploy it (key management, who can reach the store, SSO).
* **Partially addressed**: the library covers part of the criterion; the rest is outside an audit logging component.

## SOC 2 Trust Services Criteria (2017)

| Control | Title | Status | Feature | Evidence | Caveat |
|---|---|---|---|---|---|
| CC6.1 | Logical access security over protected information assets | Operator dependent | Workspace scoped records, actor attribution, dashboard SSO (OIDC), dashboard reads pinned to the configured workspace. | actor field on every record; dashboard SSO configuration; RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE left unset. | Supabase row level security is bypassed by the service role key the library uses. Tenant isolation at the store depends on who holds that key. |
| CC6.5 | Disposal of data and software when no longer required | Supported by the library | RetentionPolicy and purge_expired() with a chain checkpoint record. | retention_checkpoint records in the chain; retention.json in the evidence pack; purge output. | Purging is explicit (CLI or API call). Sinks with Object Lock refuse it. |
| CC7.2 | Monitoring of system components for anomalies | Partially addressed | Slack alerts on zero retrieval, low similarity, slow chain, chain error. | Alert configuration (thresholds) and the alert payloads themselves. | Covers the RAG pipeline only, not the host infrastructure. |
| CC7.3 | Evaluation of security events to determine impact | Partially addressed | Error records (RAGCOMPLIANCE_RECORD_ERRORS) with redacted context and lineage.status. | Records with lineage.status = error and extra.error. | Triage and impact assessment are human processes. |
| CC8.1 | Change management: integrity of system output | Supported by the library | Content signature per record plus a hash chain (prev_hash, record_hash, sequence) per workspace and chain, optional HMAC seal, append-only triggers. | ragcompliance verify output; verification.json in the evidence pack; tool_version on every record. | A hash chain is tamper evident. Tamper resistance needs the HMAC key or a WORM store (S3 Object Lock) that the writer cannot modify. |
| A1.1 | Capacity management for availability of the audit trail | Partially addressed | Bounded async queue, atexit drain, dropped_count counter, RAGCOMPLIANCE_ASYNC_OVERFLOW=block option. | dropped_count in the evidence pack manifest; storage.describe(). | With the default drop policy the log is not guaranteed complete under sustained overload; the counter makes that visible. |
| C1.1 | Identification and protection of confidential information | Supported by the library | Redaction before hashing (regex patterns, hooks, pseudonymisation), redaction findings per record, no query text in alerts by default. | extra.redaction_findings; redaction.json in the evidence pack. | Regex redaction misses free text names and addresses; use hooks for NER. |
| PI1.4 | Complete, accurate and timely outputs | Partially addressed | Data lineage: query, chunk hashes, prompt hash, model parameters, answer hash, timing, truncation flags. | lineage field on every record. | The library records what the pipeline did; it does not judge accuracy. |

## ISO/IEC 27001:2022 Annex A

| Control | Title | Status | Feature | Evidence | Caveat |
|---|---|---|---|---|---|
| A.5.33 | Protection of records | Supported by the library | Hash chain, HMAC seal, append-only sinks, Object Lock support. | ragcompliance verify; sink description in the evidence pack. | none |
| A.8.15 | Logging | Supported by the library | One record per RAG invocation with actor, timestamps, query, retrieved content, answer, model and parameters. | records.jsonl in the evidence pack. | none |
| A.8.16 | Monitoring activities | Partially addressed | Anomaly alerts and error records. | Alert configuration and payloads. | Pipeline level only. |
| A.8.10 | Information deletion | Supported by the library | Retention policy with checkpointed purge. | retention_checkpoint records; retention.json. | Operator schedules the purge. |
| A.8.11 | Data masking | Supported by the library | Redaction and pseudonymisation before storage. | redaction.json; extra.redaction_findings. | Pattern coverage is the operator's responsibility to review. |
| A.5.15 | Access control | Operator dependent | Actor attribution per record; dashboard SSO. | actor field; SSO configuration. | Access control to the store itself is outside the library. |

## Criteria this component does not address

Auditors will also ask about change management of the application itself, vendor management for the model provider, incident response, business continuity, personnel security, and encryption at rest and in transit for the store. None of those are audit logging concerns and ragcompliance makes no claims about them. The [threat model](threat-model.md) lists the assumptions the integrity and confidentiality properties rest on.

## Reading the evidence

For each row, the evidence column names where in an evidence pack (see [Evidence packs](evidence.md)) the supporting data lives. `ragcompliance verify` reproduces the integrity result from the pack alone, and `records.jsonl` is the raw population an auditor can sample from.
