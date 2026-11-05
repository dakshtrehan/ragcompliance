# ragcompliance

[![PyPI](https://img.shields.io/pypi/v/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![CI](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml/badge.svg)](https://github.com/dakshtrehan/ragcompliance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ragcompliance.svg)](https://pypi.org/project/ragcompliance/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/ragcompliance)](https://pepy.tech/project/ragcompliance)

Tamper evident audit trails and audit ready evidence for retrieval augmented
generation (RAG) pipelines. One record per invocation: who asked, what was
retrieved, what the model was told, what it answered, and a hash chain that
makes any later edit, deletion or insertion detectable. Export a zipped
evidence pack an auditor can verify with one command.

Works as a LangChain or LlamaIndex callback, as a wrapper around the OpenAI
and Anthropic SDKs, or standalone through the `Auditor` API. Core has no
dependencies; everything else is an optional extra. Python 3.9 to 3.13.

Full documentation: [www.dakshtrehan.com/ragcompliance](https://www.dakshtrehan.com/ragcompliance/) and the `docs/` folder (`mkdocs serve`).

## Install

```bash
pip install ragcompliance                       # core: Auditor, JSONL/SQLite/memory sinks, CLI
pip install "ragcompliance[langchain]"          # LangChain callback handler
pip install "ragcompliance[llamaindex]"         # LlamaIndex callback handler
pip install "ragcompliance[openai]"             # OpenAI SDK wrapper
pip install "ragcompliance[anthropic]"          # Anthropic SDK wrapper
pip install "ragcompliance[postgres]"           # PostgreSQL sink
pip install "ragcompliance[s3]"                 # S3 compatible sink (Object Lock aware)
pip install "ragcompliance[supabase]"           # Supabase sink (the 0.1.x backend)
pip install "ragcompliance[dashboard,sso]"      # FastAPI dashboard with optional OIDC
pip install "ragcompliance[all]"
```

## Quickstart

Choose a sink and, optionally, an HMAC key and redaction:

```bash
export RAGCOMPLIANCE_SINK="sqlite:////var/lib/ragcompliance/audit.db"
export RAGCOMPLIANCE_CHAIN_HMAC_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export RAGCOMPLIANCE_REDACT_PII=true
```

### Standalone

```python
from ragcompliance import Auditor, RAGComplianceConfig, RetrievedChunk

auditor = Auditor(RAGComplianceConfig.from_env(), session_id="user-42")

with auditor.trace("What does clause 4.2 say?", actor={"id": "alice@example.com", "type": "user"}) as t:
    t.add_chunks(my_retriever.search("clause 4.2"))          # RetrievedChunk, dicts, documents or strings
    answer = my_llm(prompt)
    t.set_answer(answer, model_name="my-model", params={"temperature": 0.2})
```

### LangChain

```python
from ragcompliance.integrations.langchain import RAGComplianceHandler, audited_chain

handler = RAGComplianceHandler(session_id="user-42")
chain = audited_chain(my_chain, handler)
chain.invoke({"query": "What does clause 4.2 say?"},
             config={"metadata": {"ragcompliance_actor": {"id": "alice@example.com"}}})
```

### LlamaIndex

```python
from ragcompliance.integrations.llamaindex import install

handler = install(session_id="user-42")               # registers on Settings.callback_manager
with handler.auditor.bind(actor={"id": "alice@example.com"}):
    response = query_engine.query("What does clause 4.2 say?")
```

### OpenAI or Anthropic SDK

```python
from openai import OpenAI
from ragcompliance.integrations.openai import AuditedOpenAI

client = AuditedOpenAI(OpenAI(), session_id="user-42")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
    retrieved_chunks=my_chunks,          # audit only, stripped before the API call
    actor={"id": "alice@example.com"},   # audit only
)
```

```python
from anthropic import Anthropic
from ragcompliance.integrations.anthropic import AuditedAnthropic

client = AuditedAnthropic(Anthropic(), session_id="user-42")
msg = client.messages.create(model="claude-sonnet-5-5", max_tokens=512,
                             messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
                             retrieved_chunks=my_chunks)
```

The Anthropic wrapper drops `temperature` (and `top_p`, `top_k`) when the
installed SDK rejects them, as SDK 2.x does by default with API version
2026-10-21, and records the requested value in the audit record.

### Verify and export

```bash
ragcompliance verify                       # hash chain and content signatures, exit 1 on any break
ragcompliance show                         # per chain counts and head hashes
ragcompliance evidence --workspace default --start 2026-01-01 --end 2026-03-31 --out q1.zip
ragcompliance verify --evidence q1.zip     # independent re-verification of the pack
ragcompliance purge --workspace default    # apply RAGCOMPLIANCE_RETENTION_DAYS, with a chain checkpoint
ragcompliance selftest                     # is this install wired correctly
```

## What a record looks like

```json
{
  "id": "0d4f...", "session_id": "user-42", "workspace_id": "acme",
  "actor": {"id": "alice@example.com", "type": "user"},
  "query": "What does clause 4.2 say?",
  "retrieved_chunks": [{"content": "Clause 4.2 ...", "source_url": "contracts/msa.pdf",
                        "chunk_id": "p3-c1", "similarity_score": 0.91, "metadata": {}}],
  "llm_answer": "Clause 4.2 covers indemnification.",
  "model_name": "gpt-4o-mini-2026",
  "lineage": {"framework": "langchain", "retrievers": ["VectorStoreRetriever"],
              "prompt_sha256": "...", "model_params": {"temperature": 0.2},
              "chunks": [{"chunk_id": "p3-c1", "content_sha256": "..."}],
              "usage": {"total_tokens": 412}, "started_at": "...", "ended_at": "..."},
  "chain_signature": "a3f8...",
  "chain_id": "default", "sequence": 1198, "prev_hash": "90ab...", "record_hash": "c1d2...",
  "record_hmac": "e4f5...", "timestamp": "2026-04-10T06:00:00+00:00", "latency_ms": 1240,
  "schema_version": 2, "tool_version": "1.0.0rc1", "extra": {"redaction_findings": {"email": 1}}
}
```

## Features

| Area | What you get |
|------|--------------|
| Integrity | Per record content signature plus a hash chain (`prev_hash`, `record_hash`, `sequence`) per workspace and chain; optional HMAC seal with a key kept outside the store; append only triggers in SQLite and PostgreSQL; S3 Object Lock support; verifier CLI with machine readable output |
| Sinks | JSONL, SQLite, PostgreSQL, S3 compatible, Supabase, in memory. Pluggable through `AuditSink` |
| Redaction | Regex patterns (email, SSN, cards with Luhn, phones, IPs, API keys, bearer tokens, IBAN), hooks for NER or dictionaries, pseudonymisation, applied before hashing in every integration, findings per record |
| Attribution | `actor` on every record, set per process, per handler, per block, or per invocation |
| Lineage | Chunk hashes, retriever, prompt hash, model parameters, usage, timing, truncation flags, framework and library versions |
| Retention | `RetentionPolicy`, explicit purge with a checkpoint record the verifier understands |
| Evidence | Zipped pack: records, verification result, summary report, control mapping, retention and redaction config, SHA-256 manifest and `pack_hash` |
| Safety | Audit bugs never take down the pipeline; misconfiguration fails loudly; dropped records are counted and reported |

## What it does not claim

ragcompliance is not a SOC 2 report or an ISO 27001 certificate; those are
issued by auditors after examining your whole system. The
[control mapping](docs/control-mapping.md) says which criteria this component
produces evidence for and how far that goes, and the
[threat model](docs/threat-model.md) states the assumptions (keep the HMAC
key out of the store, record head hashes, treat the store as sensitive even
with redaction on).

## Dashboard, alerts, billing

* `uvicorn ragcompliance.app:app` serves a small dashboard with exports and a
  live chain verification endpoint. Reads are pinned to the configured
  workspace; OIDC SSO is a few environment variables away. See
  [docs/dashboard.md](docs/dashboard.md).
* `RAGCOMPLIANCE_SLACK_WEBHOOK_URL` turns on anomaly alerts (zero retrieval,
  low similarity, slow chain, chain error). Payloads carry ids, not query
  text, unless you opt in.
* The Stripe billing and quota code stays in the repo as a reference
  implementation behind the `billing` extra. It is inactive unless
  `STRIPE_SECRET_KEY` is set.

## Upgrading from 0.1.x

Public API and record keys are preserved and old records still verify.
`langchain-core` and `stripe` became optional extras, no sink now means
`save()` returns `False`, Slack alerts omit query text, and the dashboard
pins reads to one workspace. Full list in
[docs/migration.md](docs/migration.md).

## Development

```bash
git clone https://github.com/dakshtrehan/ragcompliance
cd ragcompliance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check ragcompliance tests && mypy && pytest --cov=ragcompliance
mkdocs serve
```

Design notes for the 1.0 release are in [AUDIT.md](AUDIT.md) (what was
wrong in 0.1.8 and what changed) and [DECISIONS.md](DECISIONS.md).

## Self host and paid help

MIT licensed, free to self host, no call home. If you would rather have
help, the maintainer offers integration reviews, evidence preparation and
custom work; email [daksh.trehan@hotmail.com](mailto:daksh.trehan@hotmail.com?subject=RAGCompliance).

## License

MIT
