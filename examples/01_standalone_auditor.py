"""Standalone Auditor: trace, redact, attribute, verify, detect tampering.

Runs offline. Writes to a temporary JSONL file.

    python examples/01_standalone_auditor.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ragcompliance import Auditor, RAGComplianceConfig, RetrievedChunk, verify_records


def fake_llm(prompt: str) -> str:
    """Stand in for a model call. No network."""
    return "Clause 4.2 requires each party to indemnify the other against third party claims."


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(
        sink_url=f"jsonl://{tmp}/audit.jsonl",
        workspace_id="demo",
        async_writes=False,          # synchronous so the file is complete when we read it
        redact_pii=True,             # redact before hashing
        chain_hmac_key="demo-key",   # in production: a secret kept outside the store
    )
    auditor = Auditor(config, session_id="demo-session", actor={"id": "svc-demo", "type": "service"},
                      billing=None, alerter=None)

    chunks = [
        RetrievedChunk("Clause 4.2: each party shall indemnify the other ... contact legal@example.com",
                       "contracts/msa-v3.pdf", "p3-c1", 0.94, {"page": 3}),
        RetrievedChunk("Indemnification survives termination for three years.",
                       "contracts/msa-v3.pdf", "p3-c2", 0.88, {"page": 3}),
    ]

    for i, question in enumerate(["What does clause 4.2 say?", "Does indemnification survive termination?"]):
        with auditor.trace(question, actor={"id": f"user-{i}@example.com", "type": "user"}) as t:
            t.add_chunks(chunks)
            prompt = "Context:\n" + "\n".join(c.content for c in chunks) + f"\n\nQuestion: {question}"
            t.set_prompt(prompt)
            t.set_answer(fake_llm(prompt), model_name="fake-model", params={"temperature": 0.0})

    records = list(auditor.storage.iter_records(workspace_id="demo"))
    print(f"wrote {len(records)} records to {tmp / 'audit.jsonl'}")
    first = records[0]
    print("actor      :", first["actor"])
    print("redacted   :", first["retrieved_chunks"][0]["content"][-30:])
    print("findings   :", first["extra"].get("redaction_findings"))
    print("lineage    :", json.dumps({k: first["lineage"][k] for k in ("framework", "model_params", "prompt_sha256")}))
    print("chain      :", first["sequence"], first["prev_hash"][:12], first["record_hash"][:12])

    result = auditor.storage.verify(workspace_id="demo")
    print("verify     :", "intact" if result.ok else "BROKEN", f"({result.records_checked} records)")

    tampered = [dict(r) for r in records]
    tampered[0]["llm_answer"] = "Clause 4.2 says nothing about indemnity."
    detect = verify_records(tampered, hmac_key=b"demo-key")
    print("tampered   :", sorted({p.code for p in detect.problems}))
    auditor.close()
    return 0 if result.ok and not detect.ok else 1


if __name__ == "__main__":
    sys.exit(main())
