# Attribution and lineage

## Who asked: `actor`

Every record has an `actor` dict. Merge order, later wins:

1. `RAGCOMPLIANCE_ACTOR_ID` / `RAGCOMPLIANCE_ACTOR_TYPE` (a default for the process, for example a service account),
2. `actor=` on the handler, `Auditor` or SDK wrapper,
3. `auditor.bind(actor=...)` for a block of code on the current thread,
4. per invocation: LangChain `metadata={"ragcompliance_actor": {...}}` (or `actor_id` / `user_id` keys), `trace(actor=...)`, or the `actor=` keyword on a wrapped SDK call.

Recommended keys: `id`, `type` (`user`, `service`, `system`), `auth_method`,
`ip`, `tenant`. Nothing is validated, so put what your identity provider
gives you.

The dashboard shows `actor.id`, the evidence report counts distinct actors,
and Slack alerts include it.

## How the answer was produced: `lineage`

| Key | Content |
|-----|---------|
| `status` | `ok` or `error` |
| `framework`, `framework_version` | `langchain`, `llamaindex`, `openai`, `anthropic` or `direct`, and the library version seen |
| `tool_version`, `python_version` | ragcompliance and interpreter versions |
| `started_at`, `ended_at` | ISO timestamps; `latency_ms` on the record is their difference |
| `query_sha256`, `answer_sha256` | hashes of the stored (redacted) text |
| `chunks` | per chunk: `chunk_id`, `source_url`, `content_sha256`, `similarity_score` |
| `retrievers` | retriever class names seen (LangChain) or `llamaindex.retrieve` |
| `prompt_sha256` | hash of the rendered prompt or message list; `prompt_text` too when `RAGCOMPLIANCE_CAPTURE_PROMPTS=true` |
| `model_params` | whitelisted invocation parameters: model, temperature, max tokens, top_p, top_k, stop, seed, n, penalties, response_format |
| `model_params_dropped` | parameters the Anthropic wrapper removed because the installed SDK rejects them |
| `usage` | token counts when the provider reports them |
| `truncation` | present when the query or answer exceeded `RAGCOMPLIANCE_MAX_QUERY_CHARS` / `_MAX_ANSWER_CHARS`; records original and stored lengths |
| `chain`, `tags` | LangChain root runnable name and tags |
| `streamed` | `true` when the SDK wrapper captured a streamed response |

Everything not in the whitelist (API keys, headers, raw request bodies) is
deliberately excluded.

## Reconstructing an answer

Given a record an auditor can answer, in order:

1. Who asked, when, through which application (`actor`, `started_at`, `session_id`, `lineage.framework`).
2. Exactly which passages the model was given (`retrieved_chunks[*].content`, `source_url`, `chunk_id`, and `content_sha256` to match against the source corpus).
3. How the prompt was built and with which parameters (`prompt_sha256`, `model_params`, `model_name`).
4. What came back (`llm_answer`, `usage`).
5. Whether any of the above changed since it was written (`chain_signature`, `record_hash`, chain verification).
