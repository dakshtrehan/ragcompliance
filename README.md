# RAGCompliance

[![PyPI](https://img.shields.io/pypi/v/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![CI](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml/badge.svg)](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Audit trail middleware for RAG pipelines in regulated industries.

40 to 60 percent of RAG projects never reach production, not because the retrieval is bad but because compliance teams cannot sign off on a black box. RAGCompliance wraps any LangChain or LlamaIndex retrieval call and logs the full chain: query, retrieved chunks (with source URLs and similarity scores), LLM answer, and a SHA-256 signature tying them together. State lives in Supabase with row-level security per workspace. Drop-in, no chain rewrites.

## Quickstart

Install with the Supabase extra (this is the one you want — without it, audit logs only print to stdout):

```bash
pip install "ragcompliance[supabase]"
```

Optional extras:

```bash
pip install "ragcompliance[supabase,dashboard]"     # + FastAPI dashboard
pip install "ragcompliance[supabase,llamaindex]"    # + LlamaIndex handler
```

Supported frameworks: `langchain-core >= 0.2` (LangChain 0.2+ and all LCEL chains), `llama-index-core >= 0.10`. Python 3.11 or newer.

Create a free Supabase project at https://supabase.com, then run these once in the SQL editor:

```sql
-- paste supabase_schema.sql           (audit log table + RLS)
-- paste supabase_migration_billing.sql (billing + usage RPC)
```

Copy `.env.example` to `.env` and fill in your values:

```bash
RAGCOMPLIANCE_SUPABASE_URL=https://your-project.supabase.co
RAGCOMPLIANCE_SUPABASE_KEY=your-service-role-key
RAGCOMPLIANCE_WORKSPACE_ID=your-workspace-id  # one per tenant/customer
RAGCOMPLIANCE_DEV_MODE=false                  # true = log to stdout, false = write to Supabase
RAGCOMPLIANCE_ENFORCE_QUOTA=false             # true = raise RuntimeError when over limit
```

`workspace_id` is how RAGCompliance isolates audit logs across tenants. One workspace per customer in a multi-tenant SaaS, or one per app for internal use. Row-level security keeps rows from leaking across workspaces.

## Usage (LangChain)

Drop the handler into any existing chain via `config={"callbacks": [handler]}`. Here's a complete runnable example using an OpenAI LLM and a pre-built retriever:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

# Your existing retriever (FAISS, Chroma, Pinecone, etc.)
retriever = my_vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)
llm = ChatOpenAI(model="gpt-4o-mini")

chain = (
    {"context": retriever | RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs)),
     "query": RunnablePassthrough()}
    | prompt
    | llm
)

handler = RAGComplianceHandler(
    config=RAGComplianceConfig.from_env(),
    session_id="user-abc",
)

answer = chain.invoke(
    "What does section 4.2 of the contract say?",
    config={"callbacks": [handler]},
)
```

The handler captures the full chain — query, all retrieved chunks with source URLs and similarity scores, the LLM answer, model name, and latency — signs it with SHA-256, and writes one row per chain invocation to `rag_audit_logs`.

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

Quota enforcement is soft by default (the chain logs a warning if the workspace is over its limit). Set `RAGCOMPLIANCE_ENFORCE_QUOTA=true` to hard-block instead — the handler will raise `RuntimeError` before the LLM runs.

Query counters reset automatically at each billing period rollover. The reset is driven by Stripe's `customer.subscription.updated` webhook, with a self-healing fallback in `check_query_quota` that forces a reset if the stored period end falls into the past (so a dropped webhook can never permanently lock a workspace out).

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

- [x] LangChain callback handler (LCEL-safe, outermost-chain latching)
- [x] LlamaIndex callback handler (SYNTHESIZE-based answer capture)
- [x] Dashboard export to CSV / JSON
- [x] Stripe billing + quota metering with period-rollover reset
- [x] Fail-closed quota enforcement (`RAGCOMPLIANCE_ENFORCE_QUOTA=true`)
- [ ] Slack alerts for anomalous queries
- [ ] SOC 2 report template generator
- [ ] SSO (SAML / OIDC) on the dashboard
- [ ] Async audit writes (fire-and-forget path for latency-sensitive chains)

## License

MIT
