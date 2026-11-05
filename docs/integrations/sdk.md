# OpenAI and Anthropic SDKs

For applications that call a provider SDK directly, without a framework.
The wrapper intercepts `create` calls, opens a trace with the last user
message as the query, forwards the call unchanged (minus audit only
arguments), records the answer, model and usage, and finalises. Sync and
async clients and streaming responses are supported. Everything else on the
client passes through untouched.

Audit only keyword arguments accepted by every wrapped call:

| Argument | Meaning |
|----------|---------|
| `retrieved_chunks=` | the context your application retrieved (`RetrievedChunk`, dicts, LangChain documents or plain strings); the SDK never sees it |
| `actor=` | per call attribution |
| `audit_extra=` | merged into the record's `extra` |

The same values can be bound for a block with
`client.auditor.bind(actor=..., chunks=...)`.

## OpenAI

Install: `pip install "ragcompliance[openai]"`.

```python
from openai import OpenAI
from ragcompliance.integrations.openai import AuditedOpenAI

client = AuditedOpenAI(OpenAI(), session_id="user-42")

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "Answer from the context."},
              {"role": "user", "content": "What does clause 4.2 say?"}],
    retrieved_chunks=chunks,
    actor={"id": "alice@example.com"},
)

resp = client.responses.create(model="gpt-5-mini", input="Summarise clause 4.2", retrieved_chunks=chunks)

for event in client.chat.completions.create(model="gpt-4o-mini", messages=msgs, stream=True):
    ...   # the record is written when the stream is exhausted (or the with block exits)
```

`AsyncOpenAI` works the same way; `await client.chat.completions.create(...)`.

## Anthropic

Install: `pip install "ragcompliance[anthropic]"`.

```python
from anthropic import Anthropic
from ragcompliance.integrations.anthropic import AuditedAnthropic

client = AuditedAnthropic(Anthropic(), session_id="user-42")
msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=512,
    system="Answer from the context only.",
    messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
    retrieved_chunks=chunks,
)
```

### Sampling parameters and API version 2026-10-21

Anthropic SDK 2.x defaults to API version `2026-10-21`, which rejects
`temperature`. The installed 2.x SDK does not even accept `temperature`,
`top_p` or `top_k` on `messages.create`. The wrapper inspects the installed
`create` signature and removes any of those three it does not accept, logs
a warning once per parameter per process, and records what was removed in
`lineage.model_params_dropped` so the audit record still shows what the
application asked for. If a future SDK accepts a parameter again, it is
passed through unchanged. Pass `drop_unsupported_params=False` to disable
this and let the SDK raise.

`messages.stream(...)` (the context manager helper) is passed through
without auditing; use `create(stream=True)` for an audited stream.

## Failures

A provider error records the failure (redacted alert, and an error record
when `RAGCOMPLIANCE_RECORD_ERRORS=true`) and re-raises. A stream that fails
midway records the partial text with `lineage.status = error`.
