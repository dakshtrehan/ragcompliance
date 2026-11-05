# Quickstart

## Install

The core has no dependencies. Add the extras you use:

```bash
pip install ragcompliance                       # core: Auditor, sinks (JSONL, SQLite, memory), CLI
pip install "ragcompliance[langchain]"          # LangChain callback handler
pip install "ragcompliance[llamaindex]"         # LlamaIndex callback handler
pip install "ragcompliance[openai]"             # OpenAI SDK wrapper
pip install "ragcompliance[anthropic]"          # Anthropic SDK wrapper
pip install "ragcompliance[postgres]"           # PostgreSQL sink (psycopg 3)
pip install "ragcompliance[s3]"                 # S3 compatible sink (boto3)
pip install "ragcompliance[supabase]"           # Supabase sink
pip install "ragcompliance[dashboard,sso]"      # FastAPI dashboard with optional OIDC
pip install "ragcompliance[all]"                # everything
```

Python 3.9 to 3.13.

## Pick a sink

Set one environment variable:

```bash
export RAGCOMPLIANCE_SINK="sqlite:////var/lib/ragcompliance/audit.db"
```

Other URLs: `jsonl:///path/audit.jsonl`, `postgresql://user:pw@host/db`,
`s3://bucket/prefix`, `supabase://` (with the Supabase credentials set),
`memory://`. See [Sinks](sinks.md). Without a sink and without
`RAGCOMPLIANCE_DEV_MODE=true`, records are not persisted and `save()` returns
`False`; the library never pretends to have stored something.

## Standalone (any framework)

```python
from ragcompliance import Auditor, RAGComplianceConfig, RetrievedChunk

auditor = Auditor(RAGComplianceConfig.from_env(), session_id="user-42")

with auditor.trace("What does clause 4.2 say?", actor={"id": "alice@example.com"}) as t:
    chunks = my_retriever.search("clause 4.2")
    t.add_chunks(chunks)            # RetrievedChunk, dicts, LangChain documents or strings
    t.set_prompt(prompt_text)        # optional: the prompt hash is recorded, text only if enabled
    answer = my_llm(prompt_text)
    t.set_answer(answer, model_name="my-model", params={"temperature": 0.2})
```

If the block raises, the failure is alerted (redacted) and, when
`RAGCOMPLIANCE_RECORD_ERRORS=true`, stored as an error record.

## LangChain

```python
from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.langchain import RAGComplianceHandler, audited_chain

handler = RAGComplianceHandler(config=RAGComplianceConfig.from_env(), session_id="user-42")
chain = audited_chain(my_chain, handler)

chain.invoke(
    {"query": "What does clause 4.2 say?"},
    config={"metadata": {"ragcompliance_actor": {"id": "alice@example.com", "type": "user"}}},
)
```

One handler is safe to share across `batch`, `abatch`, threads and async
tasks. Details in [LangChain](integrations/langchain.md).

## LlamaIndex

```python
from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.llamaindex import install

handler = install(config=RAGComplianceConfig.from_env(), session_id="user-42")

with handler.auditor.bind(actor={"id": "alice@example.com", "type": "user"}):
    response = query_engine.query("What does clause 4.2 say?")
```

## OpenAI or Anthropic SDK

```python
from openai import OpenAI
from ragcompliance.integrations.openai import AuditedOpenAI

client = AuditedOpenAI(OpenAI(), session_id="user-42")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
    retrieved_chunks=my_chunks,              # audit only, never sent to the API
    actor={"id": "alice@example.com"},       # audit only
)
```

```python
from anthropic import Anthropic
from ragcompliance.integrations.anthropic import AuditedAnthropic

client = AuditedAnthropic(Anthropic(), session_id="user-42")
msg = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=512,
    messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
    retrieved_chunks=my_chunks,
)
```

The Anthropic wrapper drops `temperature` (and other sampling parameters the
installed SDK does not accept, since API version 2026-10-21 rejects them)
and records the requested value in the audit record. See
[OpenAI and Anthropic SDKs](integrations/sdk.md).

## Turn on redaction

```bash
export RAGCOMPLIANCE_REDACT_PII=true
```

Every query, chunk, chunk metadata value, answer and captured prompt is run
through the redactor before hashing, so the stored record verifies without
the raw values. See [Redaction](redaction.md).

## Verify and export

```bash
ragcompliance verify                                # uses RAGCOMPLIANCE_SINK
ragcompliance show                                  # per chain counts and head hashes
ragcompliance evidence --workspace default --start 2026-01-01 --end 2026-03-31 --out q1.zip
ragcompliance verify --evidence q1.zip              # independent re-verification
```

Keep the head hash from `ragcompliance show` somewhere the writer cannot
modify (a ticket, a signed email). A hash chain cannot detect truncation of
its newest records on its own; a recorded head hash can. See
[Threat model](threat-model.md).
