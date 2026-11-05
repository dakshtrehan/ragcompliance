# Examples

Runnable scripts live in `examples/`. None of them call a paid API: model
calls are faked, SDK clients are pointed at an in-process mock transport,
and every script writes to a temporary SQLite or JSONL file and verifies it
at the end. They are executed by the test suite, so they stay working.

| Script | Shows |
|--------|-------|
| `examples/01_standalone_auditor.py` | `Auditor.trace()` with redaction, attribution, a JSONL sink, verification and tamper detection |
| `examples/02_langchain_fake_llm.py` | A real LCEL chain with an in-memory retriever and a fake LLM, audited through `RAGComplianceHandler` |
| `examples/03_llamaindex_mock.py` | Driving the LlamaIndex handler the way `CallbackManager` does |
| `examples/04_openai_wrapper_mock.py` | `AuditedOpenAI` against the real SDK with an httpx mock transport |
| `examples/05_anthropic_wrapper_mock.py` | `AuditedAnthropic`, including the dropped `temperature` parameter |
| `examples/06_retention_and_evidence.py` | Retention purge with checkpoint, evidence pack export and independent verification |

Run one:

```bash
pip install "ragcompliance[dev]"
python examples/01_standalone_auditor.py
```

Each script prints what it did and exits non-zero if verification fails.
