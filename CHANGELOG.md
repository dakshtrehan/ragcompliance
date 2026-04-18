# Changelog

All notable changes to `ragcompliance` are recorded here.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Thread-safety note on both handlers documenting that per-run state is
  instance-local. Callers should create one handler per chain invocation /
  task.
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

## [0.1.3] — 2026-04-18

### Added
- Stripe live-mode readiness probe.
  - `BillingManager.readiness()` returns a `BillingReadiness` dataclass
    indicating `ready` / `warnings` / `errors`, detected Stripe mode
    (`test` / `live` / `unknown`), and sanitized key-prefix hints.
  - `/health/billing` endpoint returns `200` when ready, `503` otherwise,
    with the same payload. Safe to scrape — no secrets leaked.
- Operator-facing "Going live (Stripe)" runbook in the README covering
  the four steps to cut over from test to live Stripe keys, including
  the webhook secret swap and DNS/TLS sanity checks.
- Landing page and docs site at
  [www.dakshtrehan.com/ragcompliance](https://www.dakshtrehan.com/ragcompliance/),
  shipped via GitHub Pages from `/docs`.

## [0.1.2] — 2026-04

### Added
- SOC 2 evidence report generator (`ragcompliance.soc2`) that pulls live
  audit records, recomputes signatures on a random sample, and renders a
  Markdown report mapped to CC6.1, CC7.2, CC8.1, A1.1, C1.1 with a
  methodology section. Not an attestation — it's the evidence pack an
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

## [0.1.1] — 2026-04

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

## [0.1.0] — 2026-03

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

[Unreleased]: https://github.com/dakshtrehan/ragcompliance/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.3
[0.1.2]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.2
[0.1.1]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.1
[0.1.0]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.0
