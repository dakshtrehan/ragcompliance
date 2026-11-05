# ragcompliance 0.1.8 audit

Scope: every module in `ragcompliance/`, every test in `tests/`, `README.md`,
`CHANGELOG.md`, `SECURITY.md`, `pyproject.toml`, the GitHub Actions workflow,
the two Supabase SQL files, the shipped example, and the `docs/` HTML site.
The audit was done by reading the code, running the suite (179 tests passing,
65 percent line coverage), running `ruff`, `mypy`, and the shipped example,
and comparing what the library can prove against what SOC 2 and ISO 27001
auditors ask for when they scope an AI system.

Each finding has an id, a severity, and a status. Status is one of
`fixed` (changed in the 1.0.0rc1 release), `documented` (behaviour kept
on purpose, now described honestly), or `deferred` (out of scope for 1.0,
tracked in the roadmap). Severity follows the usual scale: critical means
the integrity or confidentiality promise of the product is broken, high
means data is lost or exposed under realistic conditions, medium means a
correctness or operational defect, low means hygiene.

## 1. Bugs

| Id | Severity | Finding | Status |
|----|----------|---------|--------|
| B01 | high | `example_usage.py` never writes an audit record. `on_chain_start` synthesises a `run_id` when none is passed, but the later `on_retriever_end`, `on_llm_end` and `on_chain_end` calls arrive with `run_id=None`, cannot resolve to the synthesised root, and are silently dropped. The example exits 0 and prints nothing. The handler now resolves a missing `run_id` to the single pending root when exactly one root is in flight, and the example was rewritten against the new API. | fixed |
| B02 | high | `RAGComplianceHandler.on_chain_error` builds the alert payload from the raw, unredacted run state, so a failing chain sends the unredacted query to Slack even when `redact_pii=True`. The error path now redacts before alerting. | fixed |
| B03 | high | `LlamaIndexRAGComplianceHandler` has no redaction support at all. `RAGCOMPLIANCE_REDACT_PII=true` is documented as a global switch but only the LangChain handler honours it, so LlamaIndex users persist raw PII while believing it is scrubbed. Both handlers now share one `Auditor` core that applies the same redaction, attribution and lineage logic. | fixed |
| B04 | high | `AuditStorage` silently falls back to the stdout dev path when Supabase credentials are set but the `supabase` package is missing, and `save()` returns `True`. In production this means every audit record is lost while the application believes it is compliant. Misconfiguration now raises at construction time unless `dev_mode=True`. | fixed |
| B05 | high | `/api/logs`, `/api/logs/export.csv` and `/api/logs/export.json` accept an arbitrary `workspace_id` query parameter and forward it to Supabase using the service role key, which bypasses row level security. Any dashboard caller can read any tenant. Dashboard reads are now pinned to the configured workspace unless `RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true` is set explicitly. | fixed |
| B06 | high | The dashboard renders `r.query`, `r.model_name` and other record fields with `innerHTML` and no escaping. A user query such as `<img src=x onerror=...>` becomes stored cross site scripting against whoever opens the dashboard. All values are now escaped before insertion. | fixed |
| B07 | medium | `on_llm_end` reads `model_name` only from `llm_output["model_name"]`. `langchain-anthropic` reports the model under `model`, and streaming chat models can return no `llm_output` at all, so `model_name` is blank for a large share of real deployments. The handler now also reads `model`, the `invocation_params` passed to `on_llm_start` and `on_chat_model_start`, and `response_metadata`. | fixed |
| B08 | medium | `on_llm_end` falls back to `str(response)` when the result has no generations, so a Python repr of an `LLMResult` object is signed and stored as the answer. The fallback is now an empty string and the chain output coercion fills the answer. | fixed |
| B09 | medium | `_coerce_query` and `_coerce_answer` truncate to 2000 and 4000 characters with no marker. A signed record can therefore cover a truncated payload without any indication. Truncation is now explicit: the truncated flag and the original length are recorded in `lineage`, and the limit is configurable. | fixed |
| B10 | medium | `soc2._load_records` fetches at most 10000 rows and the report then says "Chain invocations captured: N" as if N were the population. Busy workspaces get an evidence report that silently excludes records. Loading now paginates until exhaustion. | fixed |
| B11 | medium | `soc2._compute_stats` counts a record as "signed" when `chain_signature` is a 64 character string, and the report headline calls that "Signature integrity: 100%". A record with a wrong but well formed signature counts as intact. Every record in scope is now actually verified and the headline reports verified, failed, and missing counts separately. | fixed |
| B12 | medium | `handler.raise_error = True` is needed so the deliberate quota `RuntimeError` reaches the caller, but it also means any unexpected exception inside a callback (odd input shapes, a custom redactor raising) propagates and kills the user's chain. All callback bodies are now wrapped so only the intentional quota error escapes. | fixed |
| B13 | medium | Every handler instance constructs its own `AuditStorage`, so each handler owns a Supabase client, a writer thread and an `atexit` hook. Applications that create a handler per request leak threads. Handlers now accept a shared `storage=` argument and `AuditStorage` is safe to share. | fixed |
| B14 | medium | `BillingManager.from_env()` is constructed inside every handler even when Stripe is not configured, and `check_query_quota` then issues a Supabase read on every chain start. A compliance handler should not add a billing round trip to the hot path unless billing is enabled. Billing is now only enabled when `STRIPE_SECRET_KEY` is set or a manager is passed explicitly. | fixed |
| B15 | medium | `end_trace` in the LlamaIndex handler and `on_chain_error` in the LangChain handler pass `self.extra` by reference into records, so a later mutation of the shared dict changes already written records in memory. A copy is now taken per record. | fixed |
| B16 | medium | Supabase inserts use `record.to_dict()` directly, so adding any field to `AuditRecord` breaks writes against existing tables (PostgREST rejects unknown columns). The Supabase sink now writes the legacy column set, places new fields in `extra` when the migration has not been applied, and uses dedicated columns once `supabase_migration_v1.sql` has run. | fixed |
| B17 | medium | The signing algorithm is copied in four places (`handler._sign_chain`, `llamaindex_handler._sign_chain`, `soc2._verify_signature`, `selftest._sign`). Any drift silently breaks verification. There is now one implementation in `ragcompliance.chain.content_signature`. | fixed |
| B18 | medium | `Redactor.redact_record_fields` exists but is not used by any handler; the handler reimplements it inline and skips chunk `metadata`, which frequently carries file paths, author emails and ticket ids. Redaction now runs through `Redactor.redact_record`, which covers metadata string values, and the inline copy is gone. | fixed |
| B19 | low | `_format_md` in `soc2.py` emits em dashes and emoji check marks, which do not render in many auditor tooling pipelines and contradict the project's own style rule. Replaced with plain ASCII. | fixed |
| B20 | low | `supabase_schema.sql` uses `create policy` without `drop policy if exists`, so re-running the schema fails. The migration files are now idempotent. | fixed |
| B21 | low | `auth.is_public_path` prefix matches `"/health/"`, so `/health/billing` is reachable without a session. This is intended (it is sanitised) but was undocumented. Documented in the threat model. | documented |
| B22 | low | The LlamaIndex handler routes events to the active trace through a thread local, which is correct for threaded use but not for `aquery` where several traces can interleave on one thread. Documented as a known limitation with a recommendation to use one handler per concurrent async query. | documented |
| B23 | low | `pytest-asyncio` emits a deprecation warning because `asyncio_default_fixture_loop_scope` is unset. Set in `pyproject.toml`. | fixed |

## 2. Security

| Id | Severity | Finding | Status |
|----|----------|---------|--------|
| S01 | critical | The only integrity control is a per record SHA-256 over query, chunks and answer. That detects edits inside a record, but deletion of a record, insertion of a forged record, or reordering is undetectable, so the "tamper evident" wording in the docs site overstated what the product proves. 1.0 adds a hash chain (`prev_hash`, `record_hash`, `sequence`, `chain_id`), a verifier CLI, optional HMAC sealing with a key held outside the store, append only triggers for SQLite and Postgres, and S3 Object Lock support. | fixed |
| S02 | high | PII leakage in logs. Query text is sent to Slack on every alert (`alerts._build_payload`), the dev mode path prints the full record to stdout, and the error path alerts with unredacted content (B02). Slack payloads no longer include query text unless `RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true`, the error path is redacted, and the dev mode printer uses the same redacted record. | fixed |
| S03 | high | Cross tenant read through the dashboard (B05) and stored XSS in the dashboard (B06). | fixed |
| S04 | high | README, SECURITY.md and the SOC 2 report claim "row level security per workspace", but the library always connects with the service role key, which bypasses RLS. Isolation therefore depends on the operator never exposing that key, not on the policy. Claims rewritten to say exactly that, and the control mapping marks tenant isolation as operator dependent. | fixed |
| S05 | medium | Secrets handling. `RAGCOMPLIANCE_SUPABASE_KEY`, `STRIPE_SECRET_KEY` and `RAGCOMPLIANCE_SESSION_SECRET` are read from the environment, which is acceptable, but `BillingManager.readiness()` returns the first seven characters of the Stripe key and the `_upsert` dev log line prints the full subscription row at info level. The readiness summary now returns only the mode, and the dev log is at debug level with ids only. | fixed |
| S06 | medium | Injection. Supabase access goes through PostgREST filters, so there is no SQL string building in 0.1.8. The new SQLite and Postgres sinks use parameterised statements exclusively, and identifiers (table names) are validated against `^[A-Za-z_][A-Za-z0-9_]*$` before being interpolated. Covered by tests. | fixed |
| S07 | medium | The dashboard enables CORS for all origins with credentials style headers allowed. With SSO enabled the session cookie is `SameSite=Lax`, which mitigates but does not eliminate cross origin reads. CORS origins are now configurable via `RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS` and default to none. | fixed |
| S08 | medium | `/billing/subscription/{workspace_id}` returns Stripe customer and subscription ids for any workspace when SSO is off (the default). Documented as part of the "dashboard is open by default" posture and now pinned to the configured workspace like the other read endpoints. | fixed |
| S09 | low | `/logout` is a GET, so a cross site request can log a user out. Low impact; documented. | documented |
| S10 | low | No `py.typed` and no signed release artefacts. 1.0 ships `py.typed`, `RELEASE.md` includes `twine check` and a reproducible build step, and the CI builds the artefacts on every push. Sigstore signing is deferred. | fixed |

## 3. API design

| Id | Finding | Status |
|----|---------|--------|
| A01 | `langchain-core` and `stripe` are hard dependencies. A team that only uses the Anthropic SDK pulls in LangChain and a payments SDK to write an audit log. Both are now optional extras (`langchain`, `billing`); the handler imports lazily. This is a packaging change and is called out in the migration guide. | fixed |
| A02 | `requires-python >= 3.11` excludes a large share of enterprise deployments. 1.0 supports 3.9 through 3.13 and the CI matrix proves it. | fixed |
| A03 | Capture, redaction, signing, storage, billing and alerting are all implemented inside the LangChain callback and again inside the LlamaIndex callback. There is no framework neutral core, so a plain SDK user has nothing to call. 1.0 adds `ragcompliance.Auditor` with a `trace()` context manager; the callbacks and the OpenAI and Anthropic wrappers are thin adapters over it. | fixed |
| A04 | `AuditStorage` is Supabase only, with the dev mode printer as the single alternative. Storage is now a `AuditSink` protocol with JSONL, SQLite, Postgres, S3 compatible, Supabase and in memory implementations, selected with `RAGCOMPLIANCE_SINK`. `AuditStorage` remains as the façade that owns the async queue. | fixed |
| A05 | `AuditRecord` has no `from_dict`, no schema version, and no place for who asked (actor) or how the answer was produced (lineage). Added `actor`, `lineage`, `schema_version`, `tool_version`, chain fields, and `from_dict`. Existing fields and `to_dict` keys are unchanged. | fixed |
| A06 | `generate_report` requires a live Supabase client even though the caller may already hold the records. It now accepts any sink or an explicit list of records. | fixed |
| A07 | The FastAPI app builds its storage and billing objects at import time, which makes it untestable and triggers network calls on `import ragcompliance.app`. There is now a `create_app()` factory and the module level `app` is built lazily from it. | fixed |
| A08 | Four console scripts for one tool (`ragcompliance-selftest`, `ragcompliance-soc2`, plus the new verifier and exporter). 1.0 adds a single `ragcompliance` entry point with subcommands and keeps the old names as aliases. | fixed |
| A09 | Configuration is split between `RAGComplianceConfig` and ad hoc `os.getenv` calls in `handler.py`, `alerts.py` and `auth.py`. The handler and sink settings are now on the config object; alerting and SSO keep their own small config classes because they are optional subsystems. | fixed |

## 4. Missing features compared with what auditors ask for

The table lists the question an auditor asks, what 0.1.8 could show, and
what 1.0 adds. Nothing here is a certification claim; see
`docs/control-mapping.md` for the conservative mapping.

| Auditor question | 0.1.8 | 1.0 |
|------------------|-------|-----|
| Are audit logs immutable, or at least tamper evident, and can you prove it? | Per record content hash only. Deletion and reordering undetectable. | Hash chain per `chain_id`, verifier CLI with machine readable output, optional HMAC seal, append only triggers (SQLite, Postgres), S3 Object Lock, verification result embedded in every evidence pack. |
| What is your retention policy and how is it enforced? | None. Records live until someone deletes them by hand. | `RetentionPolicy` with `purge_expired()` on every sink that supports deletion, a retention checkpoint record written into the chain so the verifier can explain the gap, and the policy exported in the evidence pack. |
| Who issued each query? | `session_id` only, chosen by the application. | `actor` on every record (id, type, auth method, ip, tenant), set per handler or per invocation through LangChain metadata, LlamaIndex payload, or the SDK wrapper call. |
| Can you trace an answer back to the exact retrieved content and model configuration? | Query, chunks with content and source, answer, model name. | Adds per chunk SHA-256, retriever name, prompt SHA-256 (and optionally prompt text), model parameters, framework and library versions, start and end timestamps, truncation flags. |
| Can you hand us an evidence package we can verify independently? | Markdown report with a random sample. | Zipped evidence pack: all records in scope as JSONL, chain verification result, summary report, control mapping, retention and redaction configuration, SHA-256 manifest of every file and a pack hash. Independently verifiable with `ragcompliance verify`. |
| Is sensitive data minimised in the log? | Opt in regex redaction, LangChain only. | Redaction in every integration, redaction hooks for custom logic, pseudonymisation helper, metadata coverage, redaction configuration recorded in the evidence pack. |
| How do you know the tool itself was not changed? | Nothing. | `tool_version` on every record, CI with lint, types and 90 percent coverage gate, reproducible build and `twine check` in the release process. |

## 5. Test and tooling gaps

| Id | Finding | Status |
|----|---------|--------|
| T01 | 65 percent line coverage. `app.py` and `selftest.py` have zero coverage, `storage.py` 48 percent, `billing.py` 66 percent. | fixed, suite now above 90 percent with a CI gate |
| T02 | No property based tests. The integrity primitive is exactly the kind of code that benefits from them. | fixed, `hypothesis` suite for the hash chain |
| T03 | CI runs only 3.11 and 3.12, no `mypy`, no coverage, no build check. | fixed, matrix 3.9 to 3.13, ruff, mypy, pytest with coverage gate, build and `twine check` |
| T04 | `mypy` reports 13 errors; `ruff` with a reasonable rule set reports 73. | fixed, both clean |
| T05 | Tests reach into private state (`handler._runs`) heavily. Kept where it documents the per run design, but new tests assert on public behaviour. | documented |
