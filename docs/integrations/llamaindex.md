# LlamaIndex

Install: `pip install "ragcompliance[llamaindex]"` (`llama-index-core >= 0.10`).

```python
from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.llamaindex import LlamaIndexRAGComplianceHandler, install

# registers on llama_index.core.Settings.callback_manager, keeping existing handlers
handler = install(config=RAGComplianceConfig.from_env(), session_id="user-42")

with handler.auditor.bind(actor={"id": "alice@example.com", "type": "user"}):
    response = query_engine.query("What does clause 4.2 say?")
```

Or build the handler yourself and add it to a `CallbackManager`.

## What is captured

| Event | Captured |
|-------|----------|
| `start_trace` | opens a trace; picks up actor, chunks and extra bound with `auditor.bind` |
| `QUERY` start | query string |
| `LLM` start | prompt (formatted prompt or message list, hashed), model parameters from the serialised LLM |
| `RETRIEVE` end | every node as a chunk (`source`, `file_path` or `url`; `chunk_id` or node id; score) |
| `LLM` end | answer and model name (completion, response or raw model) |
| `SYNTHESIZE` end | final response text (overrides the LLM answer; this is the canonical answer in refine and compact modes) |
| `EXCEPTION` | the failure is recorded on `end_trace` |
| `end_trace` | finalise: redaction, lineage, signing, storage |

Redaction, attribution and lineage behave exactly as in LangChain because
both handlers share the `Auditor` core.

## Concurrency

Per trace state is keyed by the `trace_id` LlamaIndex passes to
`start_trace` and `end_trace`; event callbacks carry no trace id, so the
active trace is remembered per thread. Threaded use of one handler is
safe. Several concurrent `aquery` calls on one thread can interleave; use
one handler per concurrent async query in that case. When exactly one
trace is live, events from any thread route to it.
