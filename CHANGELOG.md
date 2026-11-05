# Changelog

All notable changes to `ragcompliance` are recorded here.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0rc1] - 2026-11-05

Release candidate for 1.0. The theme is "evidence an auditor can verify":
a tamper evident hash chain, pluggable storage, a framework neutral core,
first class SDK wrappers, and a zipped evidence pack. Everything found in
the pre release audit (`AUDIT.md`) is fixed or documented; design choices
are logged in `DECISIONS.md`. Public API and record keys from 0.1.x are
preserved and old records still verify. Behaviour changes that can affect
an upgrade are listed under "Changed" and in `docs/migration.md`.

### Added
- `ragcompliance.chain`: canonical JSON, the 0.1.x compatible content
  signature, per record hash chain (`chain_id`, `sequence`, `prev_hash`,
  `record_hash`), optional HMAC seal (`RAGCOMPLIANCE_CHAIN_HMAC_KEY`), and
  a verifier that reports edits, broken links, gaps, forks, duplicate ids,
  unanchored starts and HMAC mismatches per `(workspace, chain)`.
- Pluggable sinks selected by `RAGCOMPLIANCE_SINK`: JSONL (cross process
  `flock`), SQLite (append only triggers, `BEGIN IMMEDIATE`), PostgreSQL
  (`psycopg` 3, advisory lock, append only trigger, `postgres` extra), S3
  compatible (`boto3`, Object Lock aware, `s3` extra), Supabase (two table
  layouts), in memory. `AuditStorage` is now a façade over a sink and
  gains `iter_records`, `verify`, `purge_expired`, `count`, `describe`
  and a `dropped_count` counter.
- `RetentionPolicy` and checkpointed purges (`storage.purge_expired()`,
  `ragcompliance purge`). A `retention_checkpoint` record anchors the
  surviving chain so verification explains the gap instead of failing.
- `Auditor`: framework neutral core with a `trace()` context manager,
  `bind()` for thread local actor and context, actor attribution, data
  lineage (chunk hashes, retriever, prompt hash, model parameters, usage,
  timing, truncation flags), redaction before hashing, optional error
  records (`RAGCOMPLIANCE_RECORD_ERRORS`), optional prompt capture
  (`RAGCOMPLIANCE_CAPTURE_PROMPTS`).
- `AuditRecord` fields: `kind`, `actor`, `lineage`, `schema_version`,
  `tool_version`, `chain_id`, `sequence`, `prev_hash`, `record_hash`,
  `record_hmac`; `AuditRecord.from_dict()`; `RetrievedChunk.to_dict()`,
  `from_dict()`, `content_sha256()`.
- Integrations package: `ragcompliance.integrations.langchain`
  (`audited_chain`), `.llamaindex` (`install`), `.openai`
  (`AuditedOpenAI`: chat completions and responses, sync, async,
  streaming) and `.anthropic` (`AuditedAnthropic`: messages, sync, async,
  streaming; drops sampling parameters the installed SDK rejects and
  records them in `lineage.model_params_dropped`).
- Per invocation attribution in LangChain through
  `metadata["ragcompliance_actor"]` / `actor_id` / `user_id`, and
  `metadata["ragcompliance_extra"]`.
- Redaction: hooks (`Redactor(hooks=[...])`), callable replacements,
  `pseudonymize(salt)`, `iban` pattern, `Redactor.redact_record()` that
  also covers chunk metadata and captured prompts, `describe()`.
- Evidence pack exporter (`ragcompliance.evidence`, `ragcompliance
  evidence`): zip with `manifest.json` (file hashes and `pack_hash`),
  `records.jsonl`, `verification.json`, `report.md`, `controls.json`,
  `controls.md`, `retention.json`, `redaction.json`, `README.txt`.
  `verify_evidence_pack()` and `ragcompliance verify --evidence`.
- `ragcompliance.controls`: the SOC 2 and ISO 27001 control mapping used by
  the report, the evidence pack and `docs/control-mapping.md`.
- Unified CLI `ragcompliance` (`verify`, `evidence`, `soc2`, `purge`,
  `show`, `selftest`) and `ragcompliance-verify`. Old scripts kept.
- Dashboard: `create_app()` factory, `/api/verify`, actor and chain columns.
- `py.typed`, `mypy` clean, `ruff` clean, `hypothesis` property tests for
  the chain, test coverage above 90 percent with a CI gate, CI matrix for
  Python 3.9 to 3.13, build and `twine check` in CI.
- mkdocs documentation under `docs/` (quickstart, hash chain, sinks,
  redaction, retention, lineage, integrations, evidence, control mapping,
  threat model, configuration, CLI, dashboard, examples, migration, FAQ),
  runnable examples under `examples/`, `RELEASE.md`, `AUDIT.md`,
  `DECISIONS.md`, `supabase_migration_v1.sql`, `sql/postgres_schema.sql`.

### Changed
- Packaging: the core has no required dependencies. `langchain-core` moved
  to the `langchain` extra and `stripe` to the `billing` extra. Python
  3.9 is now the floor (was 3.11). Version is read from
  `ragcompliance/_version.py`.
- No sink configured and not in dev mode: `save()` logs once, counts the
  drop and returns `False` (was `True`). Credentials present but driver
  missing raises `ConfigurationError` (was a silent fallback). Dev mode now
  uses an in memory sink and prints the sealed record.
- Slack alert payloads no longer include query text by default
  (`RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true` restores it); they include the
  record id and actor id.
- Dashboard reads ignore the caller supplied `workspace_id` unless
  `RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true`; CORS is off unless
  `RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS` is set; all values are HTML
  escaped; CSV cells are protected against formula injection.
- Billing is only constructed when `STRIPE_SECRET_KEY` is set or a manager
  is passed; the handler no longer makes a Supabase read per chain start
  for everyone else. `BillingReadiness.summary` exposes booleans
  (`secret_key_set`, `price_ids_set`) instead of key prefixes.
- `on_chain_error` alerts with the redacted state. `LlamaIndexRAGComplianceHandler`
  applies redaction (it did not before) and accepts `redactor`, `storage`,
  `actor`.
- Model name capture reads `llm_output.model`, `invocation_params` and
  `response_metadata` in addition to `llm_output.model_name`. An empty
  `LLMResult` no longer stores the object repr as the answer.
- Query and answer truncation is recorded in `lineage.truncation` and the
  limits are configurable.
- SOC 2 report verifies every record (not a 64 character length check),
  paginates exhaustively, reports chain verification, uses ASCII only, and
  accepts any storage with `iter_records()` or a `records=` list.
- Handler callbacks are guarded: only `QuotaExceededError` (a
  `RuntimeError` subclass) propagates; other exceptions are logged.
- A missing `run_id` on a direct callback call resolves to the single
  pending root, so driving the handler by hand writes a record.
- `supabase_schema.sql` is idempotent and documents that the service role
  key bypasses row level security.

### Fixed
- The shipped example never wrote a record (missing `run_id` routing).
- Unredacted query sent to Slack on the error path.
- LlamaIndex handler ignored `RAGCOMPLIANCE_REDACT_PII`.
- Silent audit loss when Supabase credentials were set without the driver.
- Cross tenant reads through the dashboard; stored XSS in the dashboard.
- Model name blank for `langchain-anthropic` and streaming chat models.
- `self.extra` shared by reference across records.
- Em dashes removed from code, docs and generated reports.

### Removed
- Nothing public. `AuditStorage._client` is now a read only compatibility
  property.

## [0.1.8]: 2026-05-02

This release adds opt-in PII / PHI redaction inside the audit handler so
sensitive values never reach storage. The chain signature is computed
over the redacted payload, which means an auditor reproducing the
SHA-256 from a persisted record never needs access to raw secrets.
Off by default; turning it on is a one-line config change.

### Added
- `ragcompliance.redaction` module with a `Redactor` class and a
  per-pattern hit registry. Built-in patterns: `email`, `ssn`,
  `credit_card` (Luhn-validated), `phone_us`, `ipv4`,
  `aws_access_key`, `openai_key`, `anthropic_key`, `bearer_token`.
  Operators can register custom `Pattern` objects on top.
- New config fields: `redact_pii: bool = False`,
  `redaction_patterns: list[str] | None`,
  `redaction_replacement: str = "[REDACTED:{name}]"`. All three are
  bound to env vars (`RAGCOMPLIANCE_REDACT_PII`,
  `RAGCOMPLIANCE_REDACTION_PATTERNS`,
  `RAGCOMPLIANCE_REDACTION_REPLACEMENT`).
- New `redactor` keyword on `RAGComplianceHandler.__init__` for
  passing a fully constructed redactor (or any object with a
  compatible `.redact()` method).
- Per-record `extra["redaction_findings"]` dict surfaces a
  per-pattern hit count so dashboards and SOC 2 evidence reports
  can show which records had sensitive data and what kind, without
  ever storing the raw value.
- `tests/test_redaction.py`: 27 tests covering pattern correctness
  (including SSN invalid-prefix exclusion, Luhn rejection of
  non-cards, IPv4 octet bounds, anthropic/openai prefix priority),
  redactor behavior (custom patterns, custom replacement, unknown
  pattern raises), and end-to-end handler integration verifying the
  signature recomputes from the redacted record alone.

### Changed
- `RAGComplianceHandler.on_chain_end` now runs the redactor (when
  configured) BEFORE `_sign_chain`. Hot path is unchanged for
  workspaces that leave `redact_pii=False`: `self.redactor` is
  `None` and the redaction block short-circuits to a no-op.

### Internal
- `_bump(dst, src)` module-level helper in `handler.py` for merging
  per-pattern hit counts. Kept module-private so the public API
  surface is unchanged.

## [0.1.7]: 2026-04-19

This release is a credibility and polish pass. No handler behaviour
changes. The library surface, signatures, and audit semantics are
identical to 0.1.6; everything that moved is either packaging, docs,
or operator tooling around the install.

### Added
- `SECURITY.md` at the repo root documenting supported versions, the
  private disclosure channel (daksh.trehan@hotmail.com), response
  SLAs, scope, and safe-harbour terms. Required for responsible
  disclosure and for landing a GitHub Security Advisory.
- `ragcompliance-selftest` console script (new module
  `ragcompliance.selftest`) that verifies an install is wired
  correctly: import + version readback, optional extras probe
  (supabase, langchain_core, llama_index_core, fastapi, authlib),
  required env vars for non-dev-mode operation,
  `RAGComplianceConfig.from_env()` load, and a dev-mode storage
  round-trip that confirms SHA-256 signature recomputation matches
  what was stored. Red / yellow / green severity, exits non-zero on
  red failures, `--json` for CI pipelines, `--dev-mode` to force the
  stdout path on a fresh machine.
- `ragcompliance-soc2` console script wired to the existing
  `ragcompliance.soc2._main` entry point. Operators no longer need to
  know the module path to generate an evidence report; it is now a
  flat CLI.
- Docs site FAQ section (six questions: LangSmith delta, PHI / PII,
  legal signature, encryption, Supabase-only, Haystack / DSPy).
  Answers the questions the first wave of readers asked by email
  rather than forcing everyone to find the answer in the source.
- Comparison strip on the landing page: "Works with", "Complements",
  "Storage". Disambiguates the stack position before a reader
  scrolls to the problem statement, so LangSmith users stop treating
  this as a competitor.
- Billing reference-implementation callout in the docs. The Stripe
  plans and prices shipped in `ragcompliance.billing` are a
  ready-to-run reference operators can fork into their own billing
  layer, not a paid tier of this library. This was already true;
  the callout makes it unambiguous.
- Changelog URL in `[project.urls]` and a footer link on the landing
  page so readers have a one-click path to per-release notes.

### Changed
- Em-dash sweep across `README.md`, `docs/index.html`, and
  `docs/docs.html`. All `-` (U+2014) and `-` (U+2013) characters in
  prose have been replaced with semicolons, parentheses, colons, or
  sentence splits. Zero dashes remain across the three surfaces.
- Landing page now uses dynamic shields.io badges for PyPI version
  and GitHub Actions CI status instead of hardcoded `v0.1.4` / `145
  tests` strings. Every future release updates the badges
  automatically and removes a class of stale-version bugs.
- Landing page test-count references updated from 126 / 145 to 152
  (current suite). Pilot-customer language removed; replaced with
  "power user" where appropriate. Testimonial quotes reframed as
  illustrative customer-question archetypes ("the
  audit-reconstruction ask", "the evidence-shape ask", "the
  answer-drift ask") with a lede that names them as the questions
  compliance, internal audit, and GRC teams actually bring.
- "40 to 60 percent of RAG projects never reach production" stat
  softened to "A lot of RAG projects stall before production". The
  original number is widely cited but not sourced to a study the
  author can defend in an auditor conversation, so it's out.
- Dashboard mockup on the landing page now carries an `example /
  mock data` label and an `aria-label` so no reader mistakes the
  screenshot for live production data.
- Example `model_name` in `README.md` unified to `gpt-4o-mini`
  everywhere (was a mix of `gpt-4` and `gpt-4o-mini`), matching the
  most common starter model in the current OpenAI pricing tier.

### Internal
- None.

## [0.1.6]: 2026-04-19

### Changed
- SOC 2 evidence report default sample size raised from 5 to 25. This
  matches what a real compliance reviewer expects to see on a quarterly
  spot-check. The previous default of 5 was symbolic; on a workspace
  with 50k records it gave roughly a 1-in-10,000 shot of surfacing a
  rare tamper event. The CLI default (`--sample`) and the Python API
  default (`generate_report(..., sample_size=25)`) both updated.
- README and docs site now distinguish "handler overhead in isolation"
  (<1ms, ~38µs p50) from "end-to-end chain latency" (dominated by
  retriever and LLM). No behavior change; the prior claim was correct
  but ambiguous enough that a reader could misread the full-chain
  number as the handler's contribution.

### Added
- README "Sample size and confidence" subsection explaining the
  hypergeometric tradeoff between sample size and detection probability
  for SOC 2 evidence audits, plus a programmatic note on exhaustive
  verification for deeper due-diligence runs.
- 2 new tests in `tests/test_soc2.py::TestDefaults` that assert the
  `generate_report` default and the CLI `--sample` default stay at 25.
  Regression guards so a future drive-by change does not silently
  revert the default and surface stale evidence to auditors.

## [0.1.5]: 2026-04-19

### Fixed
- **Retrieved chunks silently dropped on `langchain-core >= 1.3.0`.**
  The handler only overrode `on_chain_start`, so `on_retriever_start`
  fired with a `parent_run_id` that was never recorded in
  `_run_parents`. When the matching `on_retriever_end` fired,
  `_resolve_root` could not walk up to the tracked root audit state and
  every retrieved chunk was silently dropped from the record. The
  record still saved with a valid SHA-256 signature but
  `retrieved_chunks` was an empty list: a silent correctness bug for
  any RAG chain using the recommended LCEL pattern. Earlier LangChain
  versions happened to double-fire `on_chain_start` for retrievers so
  `on_chain_start`'s parent registration covered them incidentally;
  `langchain-core 1.3.0` tightened the callback surface and removed
  that side channel. `on_retriever_start`, `on_llm_start`,
  `on_chat_model_start`, and `on_tool_start` now all register their
  `run_id`/`parent_run_id` pairs through a shared
  `_register_descendant` helper so every `*_end` event can route back
  to the correct audit state.

### Added
- 5 new tests in `tests/test_handler_retrieval.py`: LCEL baseline, batch
  with per-invocation chunks, deep-nested retriever, `on_llm_start`
  parent-registration guard, and 10-thread concurrent invoke with
  chunks. All five fail against the v0.1.4 handler (verified by
  monkey-patching the new overrides back to `BaseCallbackHandler`'s
  no-ops), so they are real regression guards.

### Changed
- `on_chain_start` now delegates inner-runnable parent tracking to the
  new `_register_descendant` helper. Pure refactor: same behavior as
  v0.1.4 for the chain-start path.

## [0.1.4]: 2026-04-19

### Fixed
- **`chain.batch([q1, q2, ...])` correctness.** The handler previously
  kept per-invocation state on the instance (`_root_run_id`, `_query`,
  `_chunks`, `_llm_answer`, `_model_name`, `_start_time`) and latched
  onto the outermost chain via `_root_run_id is None`. Inside a
  `batch()` call LangChain fires `on_chain_start` for every invocation
  before any `on_chain_end`, so invocations 2..N were silently
  ignored: the first query's record would be written with the last
  query's answer and every other query was dropped. This is now fixed
 : each root run gets its own `_RunState` keyed by `run_id` and all N
  records are written with matching query/chunks/answer triples.
- **Shared-handler thread safety.** A single `RAGComplianceHandler` is
  now safe to share across concurrent `chain.invoke` / `chain.ainvoke`
  calls. State lives in a `dict` keyed by root `run_id`, guarded by a
  `threading.Lock`. Same fix applied to
  `LlamaIndexRAGComplianceHandler`, which uses a `threading.local` to
  route events to the right per-trace state.
- **Defensive `storage.save()`.** The built-in Supabase storage
  catches its own errors, but a user-supplied custom storage backend
  could previously raise straight through the handler and take down
  the host chain. `storage.save(record)` is now wrapped in
  `try/except` in both handlers, with a `logger.error` that names the
  session id and tells custom-backend authors to stop raising.

### Added
- `RAGCOMPLIANCE_MAX_PENDING_RUNS` (default 10000): soft cap on the
  number of in-flight root run states the handler will hold. When
  exceeded (typically because `on_chain_end` was never delivered: a
  crashed worker or misconfigured callbacks), the oldest pending state
  is evicted with a single warning log, and descendant parent mappings
  are dropped with it.
- 10 new tests in `tests/test_handler_batch.py` covering batch 3
  queries, concurrent 10 threads, nested LCEL saves-once, inner events
  routing to root state, concurrent 20-thread invoke, interleaved
  events, pending soft cap with descendant cleanup, storage.save
  raising, junk env values, and a hot-path microbench guard.
- 3 new tests in `tests/test_llamaindex_handler_trace.py` covering
  concurrent traces, end_trace cleanup, and storage-raise containment.

### Changed
- Thread-safety note in the README now reads "safe to share" instead of
  "one handler per invocation", with a `chain.batch` example and a
  pointer to the soft-cap env var.

## [0.1.3]: 2026-04-18

### Added
- Stripe live-mode readiness probe.
  - `BillingManager.readiness()` returns a `BillingReadiness` dataclass
    indicating `ready` / `warnings` / `errors`, detected Stripe mode
    (`test` / `live` / `unknown`), and sanitized key-prefix hints.
  - `/health/billing` endpoint returns `200` when ready, `503` otherwise,
    with the same payload. Safe to scrape: no secrets leaked.
- Operator-facing "Going live (Stripe)" runbook in the README covering
  the four steps to cut over from test to live Stripe keys, including
  the webhook secret swap and DNS/TLS sanity checks.
- Landing page and docs site at
  [www.dakshtrehan.com/ragcompliance](https://www.dakshtrehan.com/ragcompliance/),
  shipped via GitHub Pages from `/docs`.
- Thread-safety note on both handlers (superseded in v0.1.4: the
  handlers are now safe to share).
- `AuditStorage.get_by_id(record_id, workspace_id=...)` for workspace-scoped
  single-record lookups backed by an indexed `.eq('id', ...)` Supabase query.
- README subsection "What the signature covers" spelling out which fields
  are inside the SHA-256 and which are intentionally out of scope.
- CHANGELOG.md (this file).

### Changed
- `/api/logs/detail/{record_id}` now fetches via `get_by_id` instead of
  paging through the last 500 records and searching in memory. Records
  older than the previous window are now reachable; the endpoint still
  404s on unknown ids.
- SOC 2 CC8.1 claim rewritten to enumerate the exact signed fields so an
  auditor knows what integrity the signature does and does not provide.

### Fixed
- Ruff lint: removed unused imports (`fastapi.FastAPI`, `os`,
  `authlib.integrations.starlette_client.OAuth`, `ragcompliance.auth`,
  `datetime.timedelta`, `unittest.mock.MagicMock`) and a spurious f-string.

## [0.1.2]: 2026-04

### Added
- SOC 2 evidence report generator (`ragcompliance.soc2`) that pulls live
  audit records, recomputes signatures on a random sample, and renders a
  Markdown report mapped to CC6.1, CC7.2, CC8.1, A1.1, C1.1 with a
  methodology section. Not an attestation: it's the evidence pack an
  auditor asks for on day one.
- Opt-in OIDC SSO on the dashboard via the `sso` extra
  (`pip install ragcompliance[sso]`). `/health` and `/stripe/webhook`
  stay open by design; everything else 401s when SSO is configured and
  the session is anonymous.
- Slack alerts on four anomaly rules: `retrieval_returned_zero_chunks`,
  `low_similarity`, `chain_slow`, `chain_errored`. Alerts are
  fire-and-forget so an alerting outage cannot suppress the audit
  record itself.
- Async audit writes with a bounded queue, atexit-based shutdown drain,
  and a `flush()` method for tests and explicit app-shutdown hooks.
  Chain hot path no longer blocks on Supabase.

## [0.1.1]: 2026-04

### Fixed
- LCEL pipelines fire `on_chain_start` / `on_chain_end` for every
  sub-runnable. The handler now latches the outermost chain via
  `run_id` / `parent_run_id` so state from nested runnables no longer
  overwrites mid-chain and produces half-built audit records.
- Quota enforcement: when a workspace is over its plan quota and
  `enforce_quota=True`, the `RuntimeError` raised in `on_chain_start`
  now actually propagates out and blocks the chain
  (`raise_error = True`).
- Stripe webhook handler normalizes incoming event objects to plain
  dicts before indexing, fixing spurious `TypeError` on live-mode
  events from Stripe CLI.
- Billing period reset wired into the quota check so usage counters
  actually roll over when Stripe's `current_period_end` passes.
- LlamaIndex handler payload keys aligned with the LangChain handler so
  the SHA-256 signature is identical across both integrations.

## [0.1.0]: 2026-03

### Added
- Initial release of RAGCompliance.
- LangChain callback handler (`RAGComplianceHandler`) that captures
  query, retrieved chunks with source URLs and similarity scores, LLM
  answer, model name, and latency, then signs everything with SHA-256
  and writes one row per invocation to the `rag_audit_logs` table.
- `AuditStorage` backed by Supabase with row-level security per
  workspace, dev-mode fallback that prints records to stdout, and
  `query()` for paged reads.
- LlamaIndex callback handler
  (`ragcompliance.llamaindex_handler.LlamaIndexRAGComplianceHandler`)
  mirroring the LangChain capture shape via the optional
  `ragcompliance[llamaindex]` extra.
- FastAPI dashboard (`ragcompliance.app`) with stats cards, recent
  logs, filterable CSV / JSON export, detail endpoint, and a liveness
  probe. Installed via `ragcompliance[dashboard]`.
- Stripe billing and quota metering: tier-aware quotas, checkout
  session creation, webhook-driven subscription state, and a
  self-hostable paid-tier UI as a reference implementation.
- `RAGComplianceConfig` with `from_env()` and sensible defaults so the
  middleware works out of the box in dev.

[Unreleased]: https://github.com/dakshtrehan/ragcompliance/compare/v1.0.0rc1...HEAD
[1.0.0rc1]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v1.0.0rc1
[0.1.8]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.8
[0.1.7]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.7
[0.1.6]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.6
[0.1.5]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.5
[0.1.4]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.4
[0.1.3]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.3
[0.1.2]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.2
[0.1.1]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.1
[0.1.0]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.0
