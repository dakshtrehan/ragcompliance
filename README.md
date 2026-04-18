# RAGCompliance

[![PyPI](https://img.shields.io/pypi/v/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![CI](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml/badge.svg)](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Audit trail middleware for RAG pipelines in regulated industries.

40 to 60 percent of RAG projects never reach production, not because the retrieval is bad but because compliance teams cannot sign off on a black box. RAGCompliance wraps any LangChain or LlamaIndex retrieval call and logs the full chain: query, retrieved chunks (with source URLs and similarity scores), LLM answer, and a SHA-256 signature tying them together. State lives in Supabase with row-level security per workspace. Drop-in, no chain rewrites.

## Quickstart

```bash
pip install ragcompliance
pip install "ragcompliance[supabase,dashboard]"     # persistence + dashboard
pip install "ragcompliance[llamaindex]"             # optional LlamaIndex support
```

Copy `.env.example` to `.env` and fill in your values:

```bash
RAGCOMPLIANCE_SUPABASE_URL=https://your-project.supabase.co
RAGCOMPLIANCE_SUPABASE_KEY=your-service-role-key
RAGCOMPLIANCE_WORKSPACE_ID=your-workspace-id
RAGCOMPLIANCE_DEV_MODE=true   # logs to stdout in local dev
```

Run the SQL schemas once in your Supabase SQL editor:

```sql
-- paste supabase_schema.sql  (audit log table + RLS)
-- paste supabase_migration_billing.sql  (billing + usage RPC)
```

## Usage (LangChain)

```python
from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

config = RAGComplianceConfig.from_env()
handler = RAGComplianceHandler(config=config, session_id="user-abc")

answer = chain.invoke(
    {"query": "What does section 4.2 of the contract say?"},
    config={"callbacks": [handler]},
)
```

## Usage (LlamaIndex)

```python
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from ragcompliance import RAGComplianceConfig
from ragcompliance.llamaindex_handler import LlamaIndexRAGComplianceHandler

handler = LlamaIndexRAGComplianceHandler(
    config=RAGComplianceConfig.from_env(),
    session_id="user-abc",
)
Settings.callback_manager = CallbackManager([handler])

# Now any query engine runs under the audit handler.
response = query_engine.query("What does section 4.2 say?")
```

Every invocation writes an audit record like this:

```json
{
  "id": "uuid",
  "session_id": "user-abc",
  "workspace_id": "my-workspace",
  "query": "What does section 4.2 of the contract say?",
  "retrieved_chunks": [
    {
      "content": "Section 4.2 defines indemnification...",
      "source_url": "https://storage/contract-v3.pdf",
      "chunk_id": "chunk-042",
      "similarity_score": 0.94
    }
  ],
  "llm_answer": "Section 4.2 covers indemnification obligations...",
  "model_name": "gpt-4",
  "chain_signature": "a3f8c2d1...",
  "timestamp": "2026-04-10T06:00:00Z",
  "latency_ms": 1240
}
```

## Dashboard

```bash
pip install "ragcompliance[dashboard]"
uvicorn ragcompliance.app:app --reload
```

Open `http://localhost:8000` for the audit dashboard. It ships with:

| Endpoint | Purpose |
|---|---|
| `GET /` | HTML dashboard (stats cards + recent logs + export buttons) |
| `GET /health` | Liveness probe |
| `GET /api/logs` | Paginated audit records (JSON) |
| `GET /api/logs/detail/{id}` | Single record |
| `GET /api/logs/export.csv` | CSV export with filters |
| `GET /api/logs/export.json` | JSON file export with filters |
| `GET /api/summary` | Aggregate stats |
| `GET /api/plans` | Available billing plans |
| `POST /billing/checkout` | Start a Stripe Checkout session |
| `POST /stripe/webhook` | Stripe event receiver (checkout, subscription, invoice) |
| `GET /billing/subscription/{workspace_id}` | Current subscription + usage |

## Billing

Two plans:

| Tier | Price | Queries / month | Extras |
|---|---|---|---|
| Team | $49 / mo | 10,000 | CSV/JSON export, email support |
| Enterprise | $199 / mo | Unlimited | SSO, custom retention, SOC 2 on request |

Start a checkout from your app:

```python
import requests

r = requests.post(
    "https://your-dashboard.example.com/billing/checkout",
    json={"workspace_id": "my-workspace", "tier": "team"},
)
checkout_url = r.json()["checkout_url"]
# Redirect the user to checkout_url
```

Quota enforcement is soft by default (the chain logs a warning if the workspace is over its limit). Set `RAGCOMPLIANCE_ENFORCE_QUOTA=true` to hard-block instead.

## Why RAGCompliance

| Problem | RAGCompliance |
|---|---|
| Compliance team cannot audit RAG decisions | Full chain logged and signed |
| "Which document did the LLM use?" | Source URL + chunk ID per retrieval |
| "Did the answer change over time?" | SHA-256 signature per chain run |
| Multi-tenant SaaS | Row-level security per workspace |
| Works with existing stack | Drop-in callback for LangChain or LlamaIndex, no chain rewrites |

## Deploy

The dashboard is a single FastAPI app. The fastest path is Render's one-click from a repo:

1. Create a new Web Service on https://render.com, pointing at this repo.
2. Build command: `pip install -e ".[supabase,dashboard,llamaindex]"`
3. Start command: `uvicorn ragcompliance.app:app --host 0.0.0.0 --port $PORT`
4. Copy every variable from `.env.example` into Render's environment settings.
5. After the service is live, update the Stripe webhook endpoint to `https://<your-render-url>/stripe/webhook`.

Fly.io, Railway, Cloud Run all work identically; the app is a stateless container.

## Development

```bash
git clone https://github.com/dakshtrehan/ragcompliance
cd ragcompliance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,supabase,dashboard,llamaindex]"
pytest -v
```

## Roadmap

- [x] LangChain callback handler
- [x] LlamaIndex callback handler
- [x] Dashboard export to CSV / JSON
- [x] Stripe billing + quota metering
- [ ] Slack alerts for anomalous queries
- [ ] SOC 2 report template generator
- [ ] SSO (SAML / OIDC) on the dashboard

## License

MIT
