"""A real LCEL chain audited by RAGComplianceHandler, with no network.

Uses langchain-core's fake chat model and an in-memory retriever.

    pip install "ragcompliance[langchain]"
    python examples/02_langchain_fake_llm.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ragcompliance import RAGComplianceConfig
from ragcompliance.integrations.langchain import RAGComplianceHandler, audited_chain


class TinyRetriever(BaseRetriever):
    """Returns the same two documents for any query."""

    def _get_relevant_documents(self, query: str, *, run_manager=None):  # type: ignore[override]
        return [
            Document(page_content="Clause 4.2: mutual indemnification.",
                     metadata={"source": "contracts/msa-v3.pdf", "chunk_id": "p3-c1", "score": 0.94}),
            Document(page_content="Indemnification survives termination for three years.",
                     metadata={"source": "contracts/msa-v3.pdf", "chunk_id": "p3-c2", "score": 0.88}),
        ]


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(sink_url=f"sqlite:///{tmp}/audit.db", workspace_id="demo", async_writes=False)
    handler = RAGComplianceHandler(config=config, session_id="demo", billing=None, alerter=None)

    prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
    llm = FakeListChatModel(responses=["Clause 4.2 covers mutual indemnification.", "Yes, for three years."])
    chain = (
        {"context": TinyRetriever() | RunnableLambda(lambda docs: "\n".join(d.page_content for d in docs)),
         "query": RunnablePassthrough()}
        | prompt
        | llm
    )
    chain = audited_chain(chain, handler, metadata={"ragcompliance_actor": {"id": "alice@example.com", "type": "user"}})

    answers = chain.batch(["What does clause 4.2 say?", "Does indemnification survive termination?"])
    print("answers :", [a.content for a in answers])

    records = list(handler.storage.iter_records(workspace_id="demo"))
    for r in records:
        print(f"seq {r['sequence']}: {r['query']!r} -> {r['llm_answer']!r}; chunks={len(r['retrieved_chunks'])}; "
              f"actor={r['actor']['id']}; retrievers={r['lineage'].get('retrievers')}")
    result = handler.storage.verify(workspace_id="demo")
    print("verify  :", "intact" if result.ok else "BROKEN")
    handler.storage.close()
    return 0 if result.ok and len(records) == 2 and all(len(r["retrieved_chunks"]) == 2 for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
