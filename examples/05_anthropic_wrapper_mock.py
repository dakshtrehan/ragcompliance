"""AuditedAnthropic against the real anthropic SDK with a mock transport.

Shows the dropped `temperature` parameter: Anthropic SDK 2.x targets API
version 2026-10-21, which rejects it. No network, no API key.

    pip install "ragcompliance[anthropic]"
    python examples/05_anthropic_wrapper_mock.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:  # anthropic 2.x is built against the httpx2 package
    import httpx2 as httpx  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]
from anthropic import Anthropic

from ragcompliance import RAGComplianceConfig, RetrievedChunk
from ragcompliance.integrations.anthropic import AuditedAnthropic

FAKE_MESSAGE = {
    "id": "msg_demo", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Clause 4.2 covers mutual indemnification."}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 40, "output_tokens": 9},
}


def main() -> int:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json=FAKE_MESSAGE)

    raw = Anthropic(api_key="not-a-real-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(sink_url=f"sqlite:///{tmp}/audit.db", workspace_id="demo", async_writes=False)
    client = AuditedAnthropic(raw, config=config, session_id="demo", billing=None, alerter=None)

    msg = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        system="Answer from the context only.",
        messages=[{"role": "user", "content": "What does clause 4.2 say?"}],
        temperature=0.2,     # dropped if the installed SDK rejects it, recorded in lineage
        retrieved_chunks=[RetrievedChunk("Clause 4.2: mutual indemnification.", "contracts/msa-v3.pdf", "p3-c1", 0.94)],
        actor={"id": "alice@example.com", "type": "user"},
    )
    print("answer           :", msg.content[0].text)
    print("sent temperature :", "temperature" in seen[0])

    record = list(client.auditor.storage.iter_records(workspace_id="demo"))[0]
    print("model params     :", record["lineage"]["model_params"])
    print("dropped params   :", record["lineage"].get("model_params_dropped"))
    result = client.auditor.storage.verify(workspace_id="demo")
    print("verify           :", "intact" if result.ok else "BROKEN")
    client.auditor.close()
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
