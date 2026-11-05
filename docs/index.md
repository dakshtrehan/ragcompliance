# ragcompliance

Tamper evident audit trails and audit ready evidence for retrieval augmented
generation (RAG) pipelines. One record per invocation: who asked, what was
retrieved, what the model was told, what it answered, and a hash chain that
makes any later edit, deletion or insertion detectable.

ragcompliance is a Python library with no required dependencies. It plugs
into LangChain and LlamaIndex as a callback handler, wraps the OpenAI and
Anthropic SDKs directly, or runs standalone through the `Auditor` API.

## What you get

| Need | Feature |
|------|---------|
| Prove the log was not edited | Per record content signature plus a hash chain (`prev_hash`, `record_hash`, `sequence`) per workspace, optional HMAC seal, append only sinks |
| Store it where you already store things | JSONL file, SQLite, PostgreSQL, S3 compatible object storage (with Object Lock), Supabase, in memory |
| Keep sensitive data out of the log | Regex redaction before hashing, hooks for custom logic, pseudonymisation, findings per record |
| Know who asked | `actor` on every record, set per handler, per invocation, or for a block of code |
| Trace an answer back to its inputs | Lineage: chunk hashes, retriever, prompt hash, model parameters, usage, timing, truncation flags |
| Enforce retention | `RetentionPolicy` and a checkpointed purge the verifier understands |
| Hand something to an auditor | Zipped evidence pack with records, verification result, summary report, control mapping and a manifest hash |
| Verify independently | `ragcompliance verify` against a sink, a JSONL file or an evidence pack |

## Sixty second tour

```python
from ragcompliance import Auditor, RAGComplianceConfig, RetrievedChunk

config = RAGComplianceConfig(sink_url="sqlite:///audit.db", async_writes=False)
auditor = Auditor(config, session_id="user-42", actor={"id": "alice@example.com", "type": "user"})

with auditor.trace("What does clause 4.2 say?") as t:
    t.add_chunk(RetrievedChunk("Clause 4.2 ...", "contracts/msa.pdf", "p3-c1", 0.91))
    answer = "Clause 4.2 covers indemnification."   # your model call goes here
    t.set_answer(answer, model_name="my-model")
```

```bash
ragcompliance verify --sink sqlite:///audit.db
ragcompliance evidence --sink sqlite:///audit.db --workspace default \
    --start 2026-01-01 --end 2026-03-31 --out q1-evidence.zip
```

## Where to go next

* [Quickstart](quickstart.md) for each integration.
* [Hash chain and verification](hash-chain.md) for exactly what is proven and what is not.
* [Threat model](threat-model.md) before you rely on it for an audit.
* [Control mapping](control-mapping.md) for how features line up with SOC 2 and ISO 27001 criteria. It is deliberately conservative and claims no certification.
* [Migration from 0.1.x](migration.md) if you are upgrading.
