"""AuditedOpenAI against the real openai SDK with an in-process mock transport.

No network, no API key, no cost.

    pip install "ragcompliance[openai]" httpx
    python examples/04_openai_wrapper_mock.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import httpx
from openai import OpenAI

from ragcompliance import RAGComplianceConfig, RetrievedChunk
from ragcompliance.integrations.openai import AuditedOpenAI

FAKE_COMPLETION = {
    "id": "chatcmpl-demo", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini-demo",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Clause 4.2 covers mutual indemnification."}}],
    "usage": {"prompt_tokens": 42, "completion_tokens": 9, "total_tokens": 51},
}


def main() -> int:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=FAKE_COMPLETION))
    raw = OpenAI(api_key="not-a-real-key", http_client=httpx.Client(transport=transport))

    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(sink_url=f"sqlite:///{tmp}/audit.db", workspace_id="demo", async_writes=False)
    client = AuditedOpenAI(raw, config=config, session_id="demo", billing=None, alerter=None)

    chunks = [RetrievedChunk("Clause 4.2: mutual indemnification.", "contracts/msa-v3.pdf", "p3-c1", 0.94)]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Answer from the context."},
                  {"role": "user", "content": "What does clause 4.2 say?"}],
        temperature=0.2,
        retrieved_chunks=chunks,                       # audit only
        actor={"id": "alice@example.com", "type": "user"},
    )
    print("answer :", resp.choices[0].message.content)

    record = list(client.auditor.storage.iter_records(workspace_id="demo"))[0]
    print("query  :", record["query"])
    print("model  :", record["model_name"], record["lineage"]["model_params"], record["lineage"]["usage"])
    print("chunks :", [c["chunk_id"] for c in record["retrieved_chunks"]])
    result = client.auditor.storage.verify(workspace_id="demo")
    print("verify :", "intact" if result.ok else "BROKEN")
    client.auditor.close()
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
