"""Driving the LlamaIndex handler the way CallbackManager does, offline.

    pip install "ragcompliance[llamaindex]"
    python examples/03_llamaindex_mock.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import NodeWithScore, TextNode

from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.llamaindex import LlamaIndexRAGComplianceHandler


class FakeResponse:
    response = "Clause 4.2 covers mutual indemnification."


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(sink_url=f"sqlite:///{tmp}/audit.db", workspace_id="demo",
                                 async_writes=False, redact_pii=True)
    handler = LlamaIndexRAGComplianceHandler(config=config, session_id="demo", billing=None, alerter=None)

    # In a real app you register the handler on Settings.callback_manager and
    # LlamaIndex fires these events itself. Here we fire them by hand.
    with handler.auditor.bind(actor={"id": "alice@example.com", "type": "user"}):
        handler.start_trace("query-1")
        handler.on_event_start(CBEventType.QUERY, payload={EventPayload.QUERY_STR: "What does clause 4.2 say? (ref 123-45-6789)"})
        handler.on_event_end(CBEventType.RETRIEVE, payload={EventPayload.NODES: [
            NodeWithScore(node=TextNode(text="Clause 4.2: mutual indemnification.",
                                        metadata={"file_path": "contracts/msa-v3.pdf", "chunk_id": "p3-c1"}), score=0.94),
        ]})
        handler.on_event_start(CBEventType.LLM, payload={EventPayload.SERIALIZED: {"model": "fake-model", "temperature": 0.0}})
        handler.on_event_end(CBEventType.SYNTHESIZE, payload={EventPayload.RESPONSE: FakeResponse()})
        handler.end_trace("query-1", {})

    record = list(handler.storage.iter_records(workspace_id="demo"))[0]
    print("query  :", record["query"])            # SSN redacted
    print("actor  :", record["actor"])
    print("model  :", record["model_name"], record["lineage"]["model_params"])
    result = handler.storage.verify(workspace_id="demo")
    print("verify :", "intact" if result.ok else "BROKEN")
    handler.storage.close()
    return 0 if result.ok and "[REDACTED:SSN]" in record["query"] else 1


if __name__ == "__main__":
    sys.exit(main())
