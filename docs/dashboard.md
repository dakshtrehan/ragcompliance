# Dashboard

```bash
pip install "ragcompliance[dashboard]"
uvicorn ragcompliance.app:app
```

Or build it yourself: `from ragcompliance.app import create_app`.

| Endpoint | Purpose |
|----------|---------|
| `GET /` | HTML dashboard: stats, chain state, recent records |
| `GET /health` | Liveness probe with version |
| `GET /api/logs` | Recent records (newest first), filter by `session_id` |
| `GET /api/logs/detail/{id}` | One record |
| `GET /api/logs/export.csv`, `GET /api/logs/export.json` | Exports |
| `GET /api/summary` | Aggregates including `dropped_count` |
| `GET /api/verify` | Chain verification result for the workspace |
| `GET /health/billing`, `GET /api/plans`, `POST /billing/checkout`, `POST /stripe/webhook`, `GET /billing/subscription/{workspace_id}` | Billing reference implementation (503 when the `billing` extra is not installed) |

## Tenant scoping

Every read is pinned to `RAGCOMPLIANCE_WORKSPACE_ID`. The `workspace_id`
query parameter is ignored unless
`RAGCOMPLIANCE_DASHBOARD_ALLOW_ANY_WORKSPACE=true`. Turn on SSO before
enabling that: the dashboard reads the store with the service credentials,
so a dashboard user with that flag sees every tenant.

## SSO

Set the four `RAGCOMPLIANCE_OIDC_*` variables and the dashboard requires a
signed in session for everything except `/health`, `/health/billing`,
`/login`, `/auth/callback`, `/logout` and `/stripe/webhook`. Set
`RAGCOMPLIANCE_SESSION_SECRET` in production.

## Hardening notes

* Record values are HTML escaped; CSV cells that start with a formula
  character are prefixed.
* CORS is off unless `RAGCOMPLIANCE_DASHBOARD_CORS_ORIGINS` lists origins.
* `/logout` is a GET, so a cross site request can log a user out; low
  impact, noted for completeness.
