# LangChain

Install: `pip install "ragcompliance[langchain]"` (`langchain-core >= 0.2`).

```python
from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.langchain import RAGComplianceHandler, audited_chain

handler = RAGComplianceHandler(
    config=RAGComplianceConfig.from_env(),
    session_id="user-42",
    actor={"id": "svc-rag", "type": "service"},     # default for every record
)

chain = audited_chain(my_chain, handler)          # same as my_chain.with_config(callbacks=[handler])
chain.invoke(
    {"query": "What does clause 4.2 say?"},
    config={"metadata": {"ragcompliance_actor": {"id": "alice@example.com", "type": "user"},
                         "ragcompliance_extra": {"ticket": "T-1234"}}},
)
```

Or pass the handler per call: `config={"callbacks": [handler]}`.

## What is captured

| Callback | Captured |
|----------|----------|
| `on_chain_start` (root) | query (from `query`, `question`, `input`, `human_input`, `prompt` or the whole input), actor and extra from metadata, tags, chain name |
| `on_retriever_start` / `on_retriever_end` | retriever class name, every document as a chunk (`source` or `url`, `chunk_id` or `id`, `score` or `similarity_score`, remaining metadata) |
| `on_llm_start` / `on_chat_model_start` | prompt text (hashed, stored only with `RAGCOMPLIANCE_CAPTURE_PROMPTS`), whitelisted `invocation_params` |
| `on_llm_end` | answer, model name (`llm_output.model_name`, `llm_output.model`, or `response_metadata`), token usage |
| `on_chain_end` (root) | fallback answer from the outputs, then finalise |
| `on_chain_error` (root) | redacted alert; error record when `RAGCOMPLIANCE_RECORD_ERRORS=true` |

## Concurrency

Per invocation state is keyed by LangChain's root `run_id` behind a lock. One
handler is safe across `batch`, `abatch`, threads and asyncio tasks. If
`on_chain_end` is never delivered for some runs, the oldest pending state is
evicted once `RAGCOMPLIANCE_MAX_PENDING_RUNS` (default 10000) is exceeded,
with one warning.

## Errors inside the handler

`raise_error = True` is required so the deliberate quota error reaches your
code when `RAGCOMPLIANCE_ENFORCE_QUOTA=true`. Every other exception inside a
callback is logged and swallowed, so a malformed response object or a buggy
custom redactor degrades the audit record instead of taking down the chain.

## Calling the handler directly

Driving the callbacks by hand without `run_id` arguments (tests, scripts)
works when a single root is pending: the missing id resolves to that root.
With several pending roots the ambiguous event is dropped.

## Sharing storage

Every handler owns an `AuditStorage` by default. Applications that create a
handler per request should share one:

```python
storage = AuditStorage(config)
handler = RAGComplianceHandler(config=config, storage=storage, session_id=request.user.id)
```
