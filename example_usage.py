"""
RAGCompliance — end-to-end example using dev mode (no Supabase needed locally).
Run with: python example_usage.py
"""

import os
os.environ["RAGCOMPLIANCE_DEV_MODE"] = "true"

from langchain_core.documents import Document
from langchain_core.outputs import Generation, LLMResult
from ragcompliance import RAGComplianceConfig, RAGComplianceHandler

# --- Setup ---
config = RAGComplianceConfig.from_env()
handler = RAGComplianceHandler(config=config, session_id="demo-session-001")

# --- Simulate a RAG chain ---
print("Simulating RAG chain...\n")

# 1. Chain starts with a query
handler.on_chain_start({}, {"query": "What are the indemnification terms in section 4.2?"})

# 2. Retriever returns chunks
docs = [
    Document(
        page_content="Section 4.2 states that each party shall indemnify the other against third-party claims...",
        metadata={"source": "https://storage/contract-v3.pdf", "chunk_id": "chunk-042", "score": 0.94},
    ),
    Document(
        page_content="Indemnification obligations shall survive termination of this agreement for 3 years.",
        metadata={"source": "https://storage/contract-v3.pdf", "chunk_id": "chunk-043", "score": 0.88},
    ),
]
handler.on_retriever_end(docs)

# 3. LLM produces an answer
generation = Generation(text="Section 4.2 covers mutual indemnification obligations, requiring each party to defend the other against third-party claims. These obligations survive contract termination for 3 years.")
handler.on_llm_end(LLMResult(generations=[[generation]], llm_output={"model_name": "gpt-4"}))

# 4. Chain ends — audit record is built, signed, and saved
handler.on_chain_end({"answer": "Section 4.2 covers mutual indemnification..."})

print("\nDone. Check the output above for the full audit record.")
print("When Supabase is connected, this record gets persisted with row-level security.")
