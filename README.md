# RAGCompliance

[![PyPI](https://img.shields.io/pypi/v/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![CI](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml/badge.svg)](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Audit trail middleware for RAG pipelines in regulated industries.

**→ Website & full docs: [www.dakshtrehan.com/ragcompliance](https://www.dakshtrehan.com/ragcompliance/)**

RAGCompliance is drop-in audit trail middleware for retrieval-augmented generation pipelines built on LangChain or LlamaIndex.

A lot of RAG projects stall before production, not because the retrieval is bad but because compliance teams cannot sign off on a black box. RAGCompliance wraps any LangChain or LlamaIndex retrieval call and logs the full chain: query, retrieved chunks (with source URLs and similarity scores), LLM answer, and a SHA-256 signature tying them together. State lives in Supabase with row-level security per workspace. Drop-in, no chain rewrites.

## Quickstart

Install with the Supabase extra (this is the one you want, without it audit logs only print to stdout):

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
RAGCOMPLIANCE_ASYNC_WRITES=true               # fire-and-forget audit inserts (default)
RAGCOMPLIANCE_ASYNC_MAX_QUEUE=1000            # bounded in-memory buffer
RAGCOMPLIANCE_MAX_PENDING_RUNS=10000          # soft cap on in-flight run state (batch / concurrent safety)
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

The handler captures the full chain (query, all retrieved chunks with source URLs and similarity scores, the LLM answer, model name, and latency), signs it with SHA-256, and writes one row per chain invocation to `rag_audit_logs`.

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

> **Thread safety.** As of v0.1.4, a single handler is safe to share across concurrent `chain.invoke` / `chain.ainvoke` calls and across `chain.batch([...])`. State is kept per root `run_id` behind a lock, so events from different invocations cannot interleave. If `on_chain_end` is never delivered for some runs (crashed worker, misconfigured callbacks), the oldest pending state is evicted once `RAGCOMPLIANCE_MAX_PENDING_RUNS` (default 10000) is exceeded.

#### Batch support

```python
handler = RAGComplianceHandler(config=RAGComplianceConfig.from_env())

# All three queries land as three separate audit records, one per invocation,
# each with its own query, chunks, answer, and signature.
answers = chain.batch(
    [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}],
    config={"callbacks": [handler]},
)
```

The same holds for `chain.abatch(...)` and for running `chain.invoke(...)` concurrently across threads or asyncio tasks with a shared handler.

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
  "model_name": "gpt-4o-mini",
  "chain_signature": "a3f8c2d1...",
  "timestamp": "2026-04-10T06:00:00Z",
  "latency_ms": 1240
}
```

### What the signature covers

`chain_signature` is a SHA-256 over a JSON payload containing exactly these fields, in a stable order:

- `query`: the user's question string as received by the chain
- for each retrieved chunk: `content`, `source_url`, `chunk_id`
- `llm_answer`: the model's final answer string

These are the fields that answer *"what did the model see, and what did it say?"*. That is the chain of custody auditors actually want to verify.

Intentionally out of scope: `similarity_score`, free-form chunk `metadata`, `model_name`, `latency_ms`, `timestamp`, `session_id`. These are useful observability fields but they're either retriever-implementation details or metadata about the run, not statements about the answer's provenance. Tampering with them does not invalidate the answer, and including them would make the signature brittle across retriever versions.

A coarser-grained signature that also covers retriever scores is on the roadmap as "Signature coverage v2" for workspaces that want stricter retrieval-level integrity.

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

## Self-host and optional paid support

RAGCompliance is MIT-licensed and free to self-host. No dashboard call-home, no locked features, no per-seat fees. Clone, `pip install -e .`, point it at your Supabase, done.

If you'd rather not run it yourself, I offer a few kinds of paid help around the project:

- **Integration review.** I read your RAG pipeline, tell you exactly what to wire in, and leave you with a passing end-to-end audit trail. Flat fee, one-week engagement.
- **SOC 2 evidence prep.** I use the built-in evidence generator against your live data, sample-verify signatures, and hand you an auditor-ready pack mapped to CC6.1, CC7.2, CC8.1, A1.1, C1.1.
- **Custom features on contract.** New retrievers, new alert rules, custom redaction, BYO object storage, private benchmarks. Scoped per engagement.
- **Operated dashboard.** If you don't want to run the FastAPI dashboard, Supabase, and alerting yourself, I can host it for you under your own domain.

Reach out at [daksh.trehan@hotmail.com](mailto:daksh.trehan@hotmail.com?subject=RAGCompliance) with what you're trying to ship.

The Stripe billing + quota-metering code stays in the repo as a **reference implementation**. If you run RAGCompliance for a team and want to charge downstream users for quota, the plumbing is all there. It is not a paid tier of this project.

### Billing reference implementation

For operators running RAGCompliance as an internal product, here's how the billing plumbing fits together. Start a checkout from your app:

```python
import requests

r = requests.post(
    "https://your-dashboard.example.com/billing/checkout",
    json={"workspace_id": "my-workspace", "tier": "team"},
)
checkout_url = r.json()["checkout_url"]
# Redirect the user to checkout_url
```

Quota enforcement is soft by default (the chain logs a warning if the workspace is over its limit). Set `RAGCOMPLIANCE_ENFORCE_QUOTA=true` to hard-block instead; the handler will raise `RuntimeError` before the LLM runs.

Query counters reset automatically at each billing period rollover. The reset is driven by Stripe's `customer.subscription.updated` webhook, with a self-healing fallback in `check_query_quota` that forces a reset if the stored period end falls into the past (so a dropped webhook can never permanently lock a workspace out).

### Going live (Stripe)

Flipping the dashboard from test mode to live mode is a four-step runbook. RAGCompliance ships with a readiness pre-flight so you don't find out you shipped a `prod_…` where a `price_…` belongs on a Saturday night.

1. In the Stripe Dashboard, switch to **Live** mode. Re-create the Team and Enterprise products + recurring prices (live mode is a separate universe, test-mode IDs do not carry over). Copy the two `price_live_…` IDs.
2. Update your deployment env vars:

   ```bash
   STRIPE_SECRET_KEY=sk_live_...
   STRIPE_WEBHOOK_SECRET=whsec_...       # from the live webhook endpoint
   STRIPE_PRICE_ID_TEAM=price_live_...
   STRIPE_PRICE_ID_ENTERPRISE=price_live_...
   APP_BASE_URL=https://dash.example.com  # must not be localhost in live mode
   ```

3. In the Stripe Dashboard under **Developers → Webhooks**, create a new live-mode endpoint at `https://<your-dash>/stripe/webhook` subscribed to `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, and `invoice.paid`. Paste the signing secret into `STRIPE_WEBHOOK_SECRET`.
4. Hit the readiness probe to verify everything is wired:

   ```bash
   curl https://<your-dash>/health/billing
   ```

   A fully-configured live deployment returns `{"ok": true, "mode": "live", ...}` with a 200. Any misconfiguration (missing webhook secret, `prod_…` pasted where `price_…` belongs, localhost base URL in live mode, Supabase not reachable) comes back as 503 with an `issues` list. The response sanitises every secret (only prefixes like `sk_live…` ever leak), so it's safe to hit from a status page or uptime monitor.

Programmatic callers get the same structure via `BillingManager.readiness()` returning a `BillingReadiness` dataclass.

## Latency

Handler overhead is under 1ms at p50 (~38µs measured in isolation on a clean hot path). End-to-end chain latency depends on your retriever, LLM, and prompt; the handler's contribution is a small constant added on top of whatever your chain does.

Audit writes are fire-and-forget by default. `save()` enqueues the record onto a bounded in-memory queue and a single daemon worker drains it into Supabase, so the chain's hot path never blocks on audit I/O. In benchmarks, per-chain audit-write overhead drops from roughly 1.2s (sync Supabase RTT) to well under 1ms (enqueue only), a three to four order of magnitude improvement.

If Supabase is unreachable, records buffer in memory up to `RAGCOMPLIANCE_ASYNC_MAX_QUEUE` (default 1000) and then drop with a log warning rather than leak memory. On normal process exit an `atexit` hook drains pending records within `RAGCOMPLIANCE_ASYNC_SHUTDOWN_TIMEOUT` seconds (default 5). You can also call `handler.storage.flush()` explicitly in tests or your own shutdown path. Set `RAGCOMPLIANCE_ASYNC_WRITES=false` if you need a strictly synchronous write (for example, tests that inspect storage mid-chain).

## Anomaly alerts

Set `RAGCOMPLIANCE_SLACK_WEBHOOK_URL` to a Slack incoming-webhook URL (or any compatible receiver: Discord, Teams via shim, your own HTTP endpoint) and the handler will fire async alerts when a chain looks unhealthy. Four rules today, all with env-configurable thresholds:

| Rule | Fires when |
|---|---|
| `retrieval_returned_zero_chunks` | The retriever returned no documents |
| `low_similarity` | The best matching chunk scored below `RAGCOMPLIANCE_SLACK_MIN_SIMILARITY` (default 0.3) |
| `chain_slow` | End-to-end latency exceeded `RAGCOMPLIANCE_SLACK_SLOW_CHAIN_MS` (default 10000) |
| `chain_errored` | LangChain or LlamaIndex raised before the chain completed |

Alerts post on a separate daemon worker with a bounded queue, so Slack outages can't back-pressure your chain. When the queue fills, alerts drop with a log warning. Set `RAGCOMPLIANCE_SLACK_DASHBOARD_URL` to include a `View in dashboard` link in each payload.

## SSO on the dashboard

The dashboard ships wide open by default so local dev stays frictionless. Set four env vars and SSO turns on via standards OIDC discovery: Google Workspace, Okta, Auth0, Microsoft Entra, and Authentik all just work.

```bash
pip install "ragcompliance[dashboard,sso]"
```

```bash
RAGCOMPLIANCE_OIDC_ISSUER=https://accounts.google.com
RAGCOMPLIANCE_OIDC_CLIENT_ID=your-client-id
RAGCOMPLIANCE_OIDC_CLIENT_SECRET=your-client-secret
RAGCOMPLIANCE_OIDC_REDIRECT_URI=https://dash.example.com/auth/callback
RAGCOMPLIANCE_OIDC_ALLOWED_DOMAINS=acme.com,acme.co.uk   # optional allowlist
RAGCOMPLIANCE_SESSION_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
```

Once wired on, every dashboard route except `/health`, `/login`, `/auth/callback`, `/logout`, and `/stripe/webhook` requires a signed-in session. Browsers get a 302 redirect to `/login`; API clients get 401 so scripted access surfaces cleanly. The allowed domains list is optional; leave it blank to permit any email that signs in through the IdP, or lock down to a corporate domain.

## SOC 2 evidence

Most compliance teams cannot sign off on a RAG pipeline without a written trail of what was retrieved, what was answered, and proof that the trail hasn't been tampered with. The built-in evidence generator produces a Markdown report mapped to the Trust Services Criteria controls that RAGCompliance actually has data for (CC6.1, CC7.2, CC8.1, A1.1, C1.1), including a signature-verified sample an auditor can spot-check.

```bash
python -m ragcompliance.soc2 \
  --workspace acme-prod \
  --start 2026-01-01 \
  --end 2026-03-31 \
  --sample 25 --seed 42 \
  --out acme-q1-2026-evidence.md
```

The report pulls records straight from the `rag_audit_logs` table, computes integrity stats (signed vs unsigned, unique sessions, avg latency, models observed), recomputes the SHA-256 signature on a random sample, and renders the control matrix and methodology section. It is not itself a SOC 2 attestation (only a licensed auditor can issue one) but it cuts the audit-prep back-and-forth from weeks to minutes.

Programmatic access is the same pipeline without argparse:

```python
from ragcompliance.soc2 import generate_report

md = generate_report(
    workspace_id="acme-prod",
    start="2026-01-01",
    end="2026-03-31",
    sample_size=25,
    seed=42,
)
```

### Sample size and confidence

The evidence report recomputes SHA-256 signatures on a random sample of records from the period. The default is 25 records, suitable for a quarterly compliance spot-check. For deeper due-diligence runs, raise it:

```bash
python -m ragcompliance.soc2 --workspace acme-prod \
    --start 2026-01-01 --end 2026-03-31 \
    --sample 100 --seed 42
```

Sampling is random but seeded for reproducibility via `--seed`, so an auditor re-running with the same inputs gets the same sample. A given run may not surface a specific tampered record if the tamper rate is low and the sample size is small; the relationship is the standard hypergeometric one. For exhaustive verification across the full period, pass a `sample_size` equal to the total record count or loop `_verify_signature` over every record programmatically.

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

See [CHANGELOG.md](./CHANGELOG.md) for what's already shipped. What's coming next:

- [ ] **BYO object storage.** Write the raw query / chunks / answer payload to S3 / GCS / Azure Blob under a customer-owned KMS key, so Supabase only holds metadata and the signature. Keeps sensitive text out of the shared database entirely.
- [ ] **PII redaction pre-audit.** Opt-in hook that runs a local regex + NER pass over the query and retrieved chunks before the record is signed, so audit trails don't become a secondary PII leak.
- [ ] **Anthropic and Bedrock parity.** The LangChain integration already works with any chat model, but we want first-class coverage (and fixture tests) for `ChatAnthropic` and `ChatBedrock` so compliance-heavy teams don't have to verify the capture path themselves.
- [ ] **Reranker audit.** Capture the reranker step for pipelines that use one (Cohere / Jina / cross-encoders), so the audit record tells the story of *why* a chunk ended up in the final context, not just *which* chunk.
- [ ] **Signature coverage v2.** Opt-in stricter signature that also covers `similarity_score` and reranker outputs, for workspaces that need retrieval-level integrity, not just answer-level integrity.

Have a use case that isn't on this list? Open a GitHub issue or discussion; the roadmap is driven by what users actually need, not what's on my whiteboard.

## License

MIT
