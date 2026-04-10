# RAGCompliance

> Audit trail middleware for RAG pipelines in regulated industries.

40–60% of RAG projects never reach production — not because the retrieval is bad, but because compliance teams can't sign off on a black box. RAGCompliance wraps any LangChain or LlamaIndex retrieval call and logs the full chain: query → retrieved chunks (with sources and similarity scores) → LLM answer → SHA-256 signature. Stored in Supabase with row-level security per workspace.

## Quickstart

```bash
pip install ragcompliance
pip install ragcompliance[supabase]  # for Supabase persistence
```

Set your environment variables:

```bash
RAGCOMPLIANCE_SUPABASE_URL=https://your-project.supabase.co
RAGCOMPLIANCE_SUPABASE_KEY=your-service-role-key
RAGCOMPLIANCE_WORKSPACE_ID=your-workspace-id
RAGCOMPLIANCE_DEV_MODE=true  # logs to stdout instead of Supabase (local dev)
```

Run the Supabase schema (once):

```sql
-- paste contents of supabase_schema.sql into your Supabase SQL editor
```

## Usage

Drop the handler into any LangChain chain:

```python
from ragcompliance import RAGComplianceHandler, RAGComplianceConfig

config = RAGComplianceConfig.from_env()
handler = RAGComplianceHandler(config=config, session_id="user-abc")

# Works with any LangChain chain
answer = chain.invoke(
    {"query": "What does section 4.2 of the contract say?"},
    config={"callbacks": [handler]}
)
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
pip install ragcompliance[dashboard]
uvicorn ragcompliance.dashboard.app:app --reload
```

Open `http://localhost:8000` for the audit dashboard. API endpoints at `/api/logs`, `/api/summary`, and `/api/logs/{id}`.

## Why RAGCompliance

| Problem | RAGCompliance |
|---|---|
| Compliance team can't audit RAG decisions | Full chain logged and signed |
| "Which document did the LLM use?" | Source URL + chunk ID per retrieval |
| "Did the answer change?" | SHA-256 signature per chain run |
| Multi-tenant SaaS | Row-level security per workspace |
| Works with existing stack | Drop-in LangChain callback, no chain rewrites |

## Roadmap

- [ ] LlamaIndex callback handler
- [ ] Export to CSV / JSON
- [ ] Slack alerts for anomalous queries
- [ ] SOC 2 report template generator

## License

MIT
