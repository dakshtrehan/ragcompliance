# Migration from 0.1.x to 1.0

The public API is backward compatible: `RAGComplianceHandler`,
`LlamaIndexRAGComplianceHandler`, `RAGComplianceConfig`, `AuditStorage`,
`AuditRecord`, `RetrievedChunk`, `Redactor`, `SlackAlerter`,
`BillingManager` and the environment variables all keep their names and
behaviour. Records written by 0.1.x still verify (`chain_signature` is
unchanged). The changes below are the ones that can affect an upgrade.

## 1. Install the extras you use

`langchain-core` and `stripe` are no longer installed by default. The core
package has no dependencies.

```bash
pip install "ragcompliance[langchain]"            # was: pip install ragcompliance
pip install "ragcompliance[langchain,supabase]"   # was: pip install "ragcompliance[supabase]"
pip install "ragcompliance[billing]"              # if you use the Stripe reference implementation
```

Symptom if you forget: `ImportError: langchain-core is not installed`
when constructing `RAGComplianceHandler`.

## 2. No sink means no record

0.1.x without Supabase credentials and without dev mode logged a warning
and `save()` returned `True` while storing nothing. 1.0 logs one warning,
counts the record in `storage.dropped_count` and returns `False`. Nothing
raises. If you relied on the silent path for local development, set
`RAGCOMPLIANCE_DEV_MODE=true` (now an in-memory sink plus stdout) or point
`RAGCOMPLIANCE_SINK` at a file.

Supabase credentials present but the `supabase` package missing now raises
`ConfigurationError` at construction instead of silently falling back.

## 3. Slack alerts no longer include the query text

Set `RAGCOMPLIANCE_SLACK_INCLUDE_QUERY=true` to restore it. Payloads now
carry the record id and actor id.

## 4. Dashboard reads are pinned to the configured workspace

`/api/logs?workspace_id=...` and the exports ignore the parameter unless
`RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true`. Multi tenant
dashboards must opt in, and should turn on SSO. CORS is now off unless
`RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS` is set.

## 5. Billing is off unless configured

The handler no longer constructs a `BillingManager` (and no longer issues a
Supabase read per chain start) unless `STRIPE_SECRET_KEY` is set or you
pass `billing=`. `BillingReadiness.summary` reports `secret_key_set` and
`price_ids_set` booleans instead of key prefixes.

## 6. Record shape

`AuditRecord.to_dict()` gains keys: `kind`, `actor`, `lineage`,
`schema_version`, `tool_version`, `chain_id`, `sequence`, `prev_hash`,
`record_hash`, `record_hmac`. Existing keys are unchanged. Code that
iterates over the dict and expects only the 0.1.x keys should ignore the
additions. `AuditRecord.from_dict()` is new.

## 7. Supabase table

Nothing to do for the default layout: new fields are nested under
`extra._ragcompliance` so inserts keep working against the 0.1.x table.
Run `supabase_migration_v1.sql` and set
`RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION=2` to get dedicated columns, the
unique index that keeps concurrent writers linear, the append only trigger
and purge support. The reader understands both layouts and the migration
back-fills existing nested rows.

## 8. SOC 2 report

`generate_report()` now verifies every record (not just checks that a
signature is 64 characters) and reports verified, failed and unsigned
counts separately. Headings and wording changed (no em dashes or emoji),
so anything that parsed the Markdown should use the evidence pack's
`verification.json` instead. It accepts any storage with `iter_records()`
or a `records=` list; the legacy Supabase `_client` path still works.

## 9. Internal attributes

`AuditStorage._client` is now a read only property that returns the
Supabase client when that is the sink. Tests or code that assigned to it
should construct `AuditStorage(config, sink=SupabaseSink(client=...))`.
Handler internals (`_runs`, `_RunState`) are preserved, but `_RunState` is
now a `Trace` subclass.

## 10. Python versions

1.0 supports Python 3.9 to 3.13 (0.1.x required 3.11).

## New things worth turning on

```bash
RAGCOMPLIANCE_SINK=sqlite:////var/lib/ragcompliance/audit.db    # or postgresql://, s3://
RAGCOMPLIANCE_CHAIN_HMAC_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
RAGCOMPLIANCE_REDACT_PII=true
RAGCOMPLIANCE_RETENTION_DAYS=365
RAGCOMPLIANCE_RECORD_ERRORS=true
```

Then `ragcompliance verify` and `ragcompliance evidence`.
