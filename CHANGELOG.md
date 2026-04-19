# Changelog

All notable changes to `ragcompliance` are recorded here.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] — 2026-04-19

### Changed
- README and docs site now distinguish "handler overhead in isolation"
  (<1ms, ~38µs p50) from "end-to-end chain latency" (dominated by
  retriever and LLM). No behavior change; the prior claim was correct
  but ambiguous enough that a reader could misread the full-chain
  number as the handler's contribution.

## [0.1.5] — 2026-04-19

### Fixed
- **Retrieved chunks silently dropped on `langchain-core >= 1.3.0`.**
  The handler only overrode `on_chain_start`, so `on_retriever_start`
  fired with a `parent_run_id` that was never recorded in
  `_run_parents`. When the matching `on_retriever_end` fired,
  `_resolve_root` could not walk up to the tracked root audit state and
  every retrieved chunk was silently dropped from the record. The
  record still saved with a valid SHA-256 signature but
  `retrieved_chunks` was an empty list — a silent correctness bug for
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
  new `_register_descendant` helper. Pure refactor — same behavior as
  v0.1.4 for the chain-start path.

## [0.1.4] — 2026-04-19

### Fixed
- **`chain.batch([q1, q2, ...])` correctness.** The handler previously
  kept per-invocation state on the instance (`_root_run_id`, `_query`,
  `_chunks`, `_llm_answer`, `_model_name`, `_start_time`) and latched
  onto the outermost chain via `_root_run_id is None`. Inside a
  `batch()` call LangChain fires `on_chain_start` for every invocation
  before any `on_chain_end`, so invocations 2..N were silently
  ignored: the first query's record would be written with the last
  query's answer and every other query was dropped. This is now fixed
  — each root run gets its own `_RunState` keyed by `run_id` and all N
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
  exceeded (typically because `on_chain_end` was never delivered — a
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
- Thread-safety note on both handlers (superseded in v0.1.4 — the
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

[Unreleased]: https://github.com/dakshtrehan/ragcompliance/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.5
[0.1.4]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.4
[0.1.3]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.3
[0.1.2]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.2
[0.1.1]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.1
[0.1.0]: https://github.com/dakshtrehan/ragcompliance/releases/tag/v0.1.0
