# Configuration

Every setting is a field on `RAGComplianceConfig` and an environment
variable read by `RAGComplianceConfig.from_env()`.

## Storage and chain

| Variable | Field | Default | Meaning |
|----------|-------|---------|---------|
| `RAGCOMPLIANCE_SINK` | `sink_url` | empty | Sink URL (`jsonl://`, `sqlite://`, `postgresql://`, `s3://`, `supabase://`, `memory://`). Empty means Supabase when its credentials are set, otherwise dev mode or no persistence. |
| `RAGCOMPLIANCE_SUPABASE_URL`, `RAGCOMPLIANCE_SUPABASE_KEY` | `supabase_url`, `supabase_key` | empty | Supabase credentials (service role key). |
| `RAGCOMPLIANCE_TABLE_NAME` | `table_name` | `rag_audit_logs` | Supabase table. |
| `RAGCOMPLIANCE_SUPABASE_SCHEMA_VERSION` | `supabase_schema_version` | `1` | `2` after running `supabase_migration_v1.sql`. |
| `RAGCOMPLIANCE_WORKSPACE_ID` | `workspace_id` | `default` | Tenant or application identifier; one chain per workspace. |
| `RAGCOMPLIANCE_CHAIN_ID` | `chain_id` | `default` | Chain name inside a workspace; set per writer for sinks that cannot serialise writers. |
| `RAGCOMPLIANCE_CHAIN_HMAC_KEY` | `chain_hmac_key` | empty | HMAC key for `record_hmac`. Keep it out of the store. |
| `RAGCOMPLIANCE_RETENTION_DAYS` | `retention_days` | unset | Retention period for `purge_expired()` and `ragcompliance purge`. |
| `RAGCOMPLIANCE_ENABLED` | `enabled` | `true` | `false` disables writes entirely. |
| `RAGCOMPLIANCE_DEV_MODE` | `dev_mode` | `false` | In-memory sink plus stdout print when no sink is configured; tolerates missing drivers. |
| `RAGCOMPLIANCE_ASYNC_WRITES` | `async_writes` | `true` | Background writer thread. |
| `RAGCOMPLIANCE_ASYNC_MAX_QUEUE` | `async_max_queue` | `1000` | Queue bound. |
| `RAGCOMPLIANCE_ASYNC_OVERFLOW` | `async_overflow` | `drop` | `drop` (count and log) or `block`. |
| `RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT` | `async_shutdown_timeout` | `5.0` | Seconds to drain on exit. |

## Capture

| Variable | Field | Default | Meaning |
|----------|-------|---------|---------|
| `RAGCOMPLIANCE_RECORD_ERRORS` | `record_errors` | `false` | Persist an error record when an invocation fails. |
| `RAGCOMPLIANCE_CAPTURE_PROMPTS` | `capture_prompts` | `false` | Store prompt text in lineage (the hash is always stored). |
| `RAGCOMPLIANCE_MAX_QUERY_CHARS` | `max_query_chars` | `2000` | Truncation limit for coerced queries; recorded in lineage when hit. |
| `RAGCOMPLIANCE_MAX_ANSWER_CHARS` | `max_answer_chars` | `4000` | Same for answers. |
| `RAGCOMPLIANCE_ACTOR_ID`, `RAGCOMPLIANCE_ACTOR_TYPE` | `actor_id`, `actor_type` | empty, `service` | Default actor for every record. |
| `RAGCOMPLIANCE_MAX_PENDING_RUNS` | `max_pending_runs` | `10000` | Soft cap on in-flight handler state. |

## Redaction

| Variable | Field | Default |
|----------|-------|---------|
| `RAGCOMPLIANCE_REDACT_PII` | `redact_pii` | `false` |
| `RAGCOMPLIANCE_REDACTION_PATTERNS` | `redaction_patterns` | all built in patterns except `iban` |
| `RAGCOMPLIANCE_REDACTION_REPLACEMENT` | `redaction_replacement` | `[REDACTED:{name}]` |

## Billing (reference implementation, optional extra `billing`)

| Variable | Meaning |
|----------|---------|
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID_TEAM`, `STRIPE_PRICE_ID_ENTERPRISE`, `APP_BASE_URL` | Stripe wiring. Billing is only active when `STRIPE_SECRET_KEY` is set. |
| `RAGCOMPLIANCE_ENFORCE_QUOTA` | `true` raises `QuotaExceededError` before the model runs when over quota; default warns. |

## Alerts

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAGCOMPLIANCE_SLACK_WEBHOOK_URL` | empty | Enables alerts. |
| `RAGCOMPLIANCE_SLACK_DASHBOARD_URL` | empty | Adds a link to the record. |
| `RAGCOMPLIANCE_SLACK_MIN_SIMILARITY` | `0.3` | `low_similarity` threshold. |
| `RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS` | `10000` | `chain_slow` threshold. |
| `RAGCOMPLIANCE_SLACK_MAX_QUEUE` | `100` | Alert queue bound. |
| `RAGCOMPLIANCE_SLACK_INCLUDE_QUERY` | `false` | Include the (redacted) query text in the payload. |

## Dashboard

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE` | `false` | Honour the `workspace_id` query parameter. Turn SSO on first. |
| `RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS` | empty | Comma separated allowed origins; empty disables CORS. |
| `RAGCOMPLIANCE_OIDC_ISSUER`, `RAGCOMPLIANCE_OIDC_CLIENT_ID`, `RAGCOMPLIANCE_OIDC_CLIENT_SECRET`, `RAGCOMPLIANCE_OIDC_REDIRECT_URI` | empty | All four turn on OIDC SSO. |
| `RAGCOMPLIANCE_OIDC_ALLOWED_DOMAINS` | empty | Email domain allowlist. |
| `RAGCOMPLIANCE_SESSION_SECRET` | empty | Session cookie signing key; set it in production. |

## Programmatic

```python
from ragcompliance import RAGComplianceConfig
config = RAGComplianceConfig(sink_url="sqlite:///audit.db", chain_hmac_key="...", redact_pii=True)
```

Boolean variables accept `1`, `true`, `yes`, `on` (case insensitive).
Invalid integers fall back to the default rather than failing at import.
