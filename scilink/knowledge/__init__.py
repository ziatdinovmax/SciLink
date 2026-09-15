"""Knowledge layer — vector store, RAG engine, and knowledge synthesis.

Sits above ``scilink.parsers``: agents ingest documents with ``parsers`` and
embed/retrieve/generate over them here. This package does not import
``scilink.agents`` at module scope — ``run_rag`` takes instructions as a
parameter, and ``synthesis`` keeps a single lazy in-function import.

- ``KnowledgeBase`` — embedding + FAISS index + retrieval.
- ``retrieve_context`` / ``run_rag`` — the shared RAG engine.
- ``synthesize_knowledge`` — distill analysis results into reusable knowledge.
"""

from .knowledge_base import KnowledgeBase
from .rag_engine import retrieve_context, run_rag, parse_json_from_response
from .synthesis import synthesize_knowledge

__all__ = [
    "KnowledgeBase",
    "retrieve_context",
    "run_rag",
    "parse_json_from_response",
    "synthesize_knowledge",
]
