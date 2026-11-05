# Threat model

What ragcompliance protects, against whom, under which assumptions, and
what it does not protect. Read this before relying on it for an audit.

## Assets

1. **The audit records**: query, retrieved chunks, answer, actor, lineage.
   Confidential (they are a copy of user data) and integrity critical (they
   are the evidence).
2. **The chain state**: sequence numbers and hashes that make the records
   tamper evident.
3. **The HMAC key** (optional) and the pseudonymisation salt (optional).
4. **The evidence pack** handed to an auditor.

## Actors

| Actor | Capability assumed |
|-------|-------------------|
| Application developer | Runs the pipeline, chooses configuration, can read the store through the application credentials. |
| Store administrator | Full read and write on the database, bucket or file system that holds the records. |
| Dashboard user | Reaches the FastAPI dashboard. |
| Auditor | Receives an evidence pack, has the public `ragcompliance` package, does not have access to the live system. |
| Attacker | Any of the above roles, compromised, plus anyone who can intercept traffic to an alert webhook. |

## Integrity: what is detected

| Attack | Detected by |
|--------|-------------|
| Edit a stored record (query, chunks, answer, actor, lineage, timestamp) | `record_hash_mismatch`; edits to query, chunks or answer also `content_signature_mismatch` |
| Edit a record and recompute its hash | `broken_link` on the next record |
| Delete a record | `sequence_gap` and `broken_link` |
| Insert a forged record | `fork` (duplicate sequence) or gap |
| Reorder records | `broken_link` |
| Rewrite the whole chain consistently | `hmac_mismatch` when an HMAC key is configured and the attacker does not hold it; otherwise **not detected** by the chain alone (see below) |
| Delete the newest records (truncate the head) | **Not detected** by the chain alone; detected by comparing against a recorded head hash (`ragcompliance show`, evidence pack manifest) |
| Replace the records with a different valid chain | `hmac_mismatch` with a key; otherwise detected only against a recorded head hash |
| Tamper with an evidence pack | file hash mismatch, `pack_hash` mismatch, or chain problems on re-verification |

### Assumptions behind the integrity claims

* The HMAC key is stored outside the record store (secret manager, KMS,
  environment of the writer only). A key stored next to the records
  protects nothing.
* Head hashes are periodically recorded somewhere the writer and the store
  administrator cannot change (an evidence pack in a separate bucket, a
  ticket, a signed email). Without that, truncation is undetectable in any
  hash chain design.
* Where available, the store enforces append only semantics: the SQLite and
  PostgreSQL sinks install triggers; S3 Object Lock makes objects WORM. A
  store administrator can still drop a trigger; the HMAC and the recorded
  head hash are the controls that survive that.
* Clocks are reasonably accurate. Timestamps are taken from the writer and
  are included in the hashed payload, so they cannot be edited after the
  fact, but a wrong clock at write time is recorded as is.

## Confidentiality

| Risk | Control | Residual |
|------|---------|----------|
| Audit store holds PII and secrets from queries, documents and answers | Redaction before hashing (regex patterns, hooks, pseudonymisation); findings recorded per record | Regex misses free text names and addresses; the store must still be treated as sensitive |
| Query text leaks to a third party through alerts | Slack payloads carry ids and rule names only unless `RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true` | Opting in sends redacted text to the webhook |
| Dev mode prints records to stdout | Records printed are the redacted, sealed records; dev mode is off by default | Log aggregators capture stdout |
| Cross tenant read through the dashboard | Reads pinned to the configured workspace; `RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE` is an explicit opt in | With the opt in, SSO is mandatory and dashboard users see every tenant |
| Stored cross site scripting in the dashboard | Every value is escaped before insertion | none known |
| Formula injection in CSV export | Leading `=`, `+`, `-`, `@`, tab and CR are prefixed | none known |
| Supabase row level security | **Not a tenant isolation control for this library.** The service role key bypasses RLS. Isolation depends on who holds the key. | Operators wanting store level isolation per tenant should use separate projects or the PostgreSQL sink with per tenant roles |
| Secrets in logs | Billing readiness returns presence booleans only; DSNs are redacted in `describe()`; API keys are never part of lineage (whitelist) | Operator supplied `extra` is stored verbatim |

## Availability

| Risk | Control | Residual |
|------|---------|----------|
| Store unavailable | Bounded async queue, retry on the next record, writer thread survives exceptions | Records beyond the queue are dropped (default) or the caller blocks (`RAGCOMPLIANCE_ASYNC_OVERFLOW=block`); `dropped_count` makes the loss visible in the evidence pack |
| Audit bug takes down the application | Every callback and wrapper path is guarded; only the deliberate quota error escapes | none known |
| Misconfiguration silently drops records | Missing driver with credentials present raises at start; no sink at all logs once and `save()` returns `False` | Operators must read logs or run `ragcompliance selftest` |

## Out of scope

* Security of the LLM provider, the vector store, the host, the network.
* Authentication of the application's end users (the `actor` is whatever
  the application reports).
* Encryption at rest and in transit for the store (configure it on the
  store).
* Legal non-repudiation. A hash chain plus HMAC shows a record was not
  altered by anyone without the key; it does not identify who wrote it.
  For that, sign records with an asymmetric key you control (not provided
  in 1.0).

## Reporting

See `SECURITY.md` for the disclosure channel. Bugs that break any row of the
integrity or confidentiality tables above are the highest priority class.
