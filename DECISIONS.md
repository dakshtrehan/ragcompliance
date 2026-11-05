# Decision log

Decisions taken while preparing the 1.0.0rc1 release candidate. Each entry
records the choice, the alternatives considered, and why. Newest at the
bottom.

## D01. Hash chain lives in the sink, not in the handler

The chain fields (`sequence`, `prev_hash`, `record_hash`) are assigned by
the sink inside its own critical section, reading its own tail. The
alternative was to compute them in the handler and pass a sealed record
down. That breaks as soon as two processes write to the same store,
because neither knows the other's tail. Putting the seal next to the
write lets SQLite use `BEGIN IMMEDIATE`, Postgres use an advisory lock,
and JSONL use `flock`, so the chain stays linear across processes. The
pure functions that compute the hashes live in `ragcompliance.chain` and
are the only implementation.

## D02. One chain per `chain_id`, default `"default"`

Sinks that cannot serialise writers (Supabase through PostgREST, S3) can
still fork a chain if two processes share a `chain_id`. Rather than
pretend otherwise, the chain identity is explicit and the verifier checks
each chain independently and reports forks. Operators running several
writers against Supabase or S3 set `RAGCOMPLIANCE_CHAIN_ID` per process.
The documentation says this plainly.

## D03. HMAC sealing is optional and keyed from the environment

A hash chain is tamper evident, not tamper proof: someone with write
access to the whole store can rewrite the chain. The optional
`RAGCOMPLIANCE_CHAIN_HMAC_KEY` adds `record_hmac`, which an attacker
without the key cannot forge. It is optional because key management is
the operator's problem and a mandatory key would make the default install
fail. The threat model explains both levels.

## D04. Retention purges write a checkpoint record into the chain

Deleting expired records necessarily breaks a hash chain at the deletion
point. The purge therefore appends a `retention_checkpoint` record that
names the hash of the last purged record and the count removed. The
verifier treats the first surviving record's `prev_hash` as a trust
anchor when a matching checkpoint exists, and reports it as such instead
of as a break. This is honest about what can be proven after a purge.

## D05. `langchain-core` and `stripe` become optional extras

A plain SDK user should not install LangChain and a payments SDK to
write an audit log. This is the one packaging change that can bite an
existing user: `pip install ragcompliance` no longer pulls
`langchain-core`. The migration guide says to install
`ragcompliance[langchain]`. The import paths and class names are
unchanged.

## D06. Python 3.9 floor

The request asked for a 3.9 to 3.13 matrix. All modules use
`from __future__ import annotations` and avoid 3.10 syntax at runtime.
Optional integrations are skipped on interpreters where the upstream
package does not install.

## D07. Supabase schema compatibility through `extra`

Adding columns to `AuditRecord` would break inserts against existing
`rag_audit_logs` tables, because PostgREST rejects unknown columns. The
Supabase sink writes the 0.1.x column set and nests the new fields under
`extra["_ragcompliance"]` until the operator runs
`supabase_migration_v1.sql`, after which it uses dedicated columns. The
reader understands both layouts. No existing deployment breaks on upgrade.

## D08. Slack alerts stop including query text by default

Sending user queries to a third party webhook is data egress that
auditors flag. The default is now to omit the query. The old behaviour is
one environment variable away (`RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true`)
and listed as a behaviour change in the migration guide.

## D09. Dashboard reads are pinned to the configured workspace

The dashboard uses the service role key, so a caller supplied
`workspace_id` was a cross tenant read. The parameter is now ignored
unless `RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true`. Operators who
run one dashboard over many tenants can opt in and are told in the docs
that SSO is then mandatory.

## D10. Billing stays in the package but is off unless configured

Removing the Stripe reference implementation would be a breaking change
for the operators who use it. It stays, behind the `billing` extra, and
the handler only constructs a `BillingManager` when `STRIPE_SECRET_KEY`
is set or one is passed in. The hot path no longer touches billing for
everyone else.

## D11. mkdocs lives beside the legacy GitHub Pages site

`docs/` already hosts the marketing site that GitHub Pages serves. Moving
it would break live URLs. `mkdocs.yml` uses `docs_dir: docs` and
`exclude_docs` to skip the HTML, XML and image assets, so `mkdocs build`
produces a clean site from the Markdown while the Pages deployment keeps
working. Switching Pages to the mkdocs output is an operator decision
noted in `RELEASE.md`.

## D12. Anthropic wrapper drops `temperature` with a warning

Anthropic SDK 2.x defaults to API version `2026-10-21`, which rejects
`temperature`. The installed SDK (2.5.0) does not even have the parameter
on `messages.create`. The wrapper removes it from the call, logs a
warning once, and records the fact in `lineage.model_params.dropped`
so the audit record still shows what the application asked for. The
alternative, passing it through and letting the API fail, would turn an
audit wrapper into a source of production errors.

## D13. Version 1.0.0rc1, not 1.0.0

The request asked for a release candidate. The version is `1.0.0rc1`
so PyPI treats it as a pre release and `pip install ragcompliance` keeps
resolving to 0.1.8 until the final tag. `RELEASE.md` has the exact
commands for both the candidate and the final.

## D14. Coverage gate at 90 percent, measured on the full suite

Optional integrations are tested with the real client libraries where
they install (OpenAI and Anthropic SDKs with an `httpx` mock transport,
`moto` for S3) and with fakes elsewhere (Postgres through an injected
connection factory, since no database server is available in CI). The
gate is enforced in CI with `--cov-fail-under=90`.

## D15. Evidence pack is a zip with a manifest, not a PDF

Auditors ingest files into their own tooling. A zip with JSONL records, a
JSON verification result, Markdown summaries and a SHA-256 manifest can
be checked with standard tools and re-verified with the CLI. A PDF would
look finished and prove nothing.

## D16. Dropped records are counted, not hidden

When the async queue overflows, the record is dropped as before (the
alternative, blocking the chain, is worse for availability), but the
storage now keeps a `dropped_count`, logs once per threshold, and the
evidence pack reports the counter, so an auditor sees that the log is
not guaranteed complete under overload.

## D17. Optional extras that are installed but broken are reported, not fatal

On Python 3.9 a transitive dependency of llama-index-core fails at import
with a `TypeError` rather than an `ImportError`. The handler module's
import guard, the selftest's extras probe and the test suite's collection
hook all treat any exception during an optional import as "not available
here". The alternative, letting one broken optional package crash the
selftest or the whole test run, would hide the state of everything else.
The CI matrix therefore runs the core on 3.9 and skips the LlamaIndex
tests there, which the job output shows as skipped rather than failed.
