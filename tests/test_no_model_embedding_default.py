"""No embedding model is the default and means an INTENTIONAL keyword-only
(BM25) knowledge base — no embedding client, no dense index, no failed API
call — while naming a model still gives a dense index. Session constructors
default to None; the store compat warning reads cleanly for a keyword-only
session."""
import logging

import pytest

from scilink.knowledge.knowledge_base import KnowledgeBase
from scilink.knowledge.kb_store import embedding_compat_warning

CHUNKS = [{"text": "Annealing MoS2 at 400 C improved the Raman A1g/E2g ratio.", "source": "n.md"},
          {"text": "Above 500 C sulfur loss produced a defect band near 227 cm-1.", "source": "n.md"}]


def test_no_model_is_keyword_only_with_no_client_and_no_dense_index(caplog):
    caplog.set_level(logging.INFO)
    kb = KnowledgeBase(embedding_model=None, api_key="unused")
    assert kb.embedding_client is None and kb.embedding_model_name is None
    kb.build(CHUNKS)
    assert kb.index is None and len(kb.chunks) == 2          # chunks kept, no dense index
    assert "keyword-only" in caplog.text.lower()
    hits = kb.retrieve("what temperature improved the Raman ratio?", top_k=1)   # degrades to BM25
    assert hits and "400 C" in hits[0]["text"]
    # blank string is treated the same as None
    assert KnowledgeBase(embedding_model="").embedding_client is None


def test_no_model_build_makes_no_embedding_call(monkeypatch):
    kb = KnowledgeBase(embedding_model=None)
    # any attempt to embed would blow up here — the keyword path must not call it
    kb.embedding_client = type("Boom", (), {"embed_content": lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedded!"))})()
    kb.embedding_client = None   # restore the real no-client state
    kb.build(CHUNKS)             # must not raise
    assert kb.index is None


def test_naming_a_model_builds_a_dense_client():
    kb = KnowledgeBase(embedding_model="text-embedding-3-small", api_key="k", use_litellm=True)
    assert kb.embedding_client is not None and kb.embedding_model_name == "text-embedding-3-small"


def test_compat_warning_reads_cleanly_for_a_keyword_only_session():
    m = {"name": "kb1", "embedding_model": "gemini-embedding-001"}
    w = embedding_compat_warning(m, None)                    # keyword-only session
    assert w and "no embedding model" in w and "gemini-embedding-001" in w and "None" not in w
    assert embedding_compat_warning({"name": "kb2", "embedding_model": "unknown"}, None) is None
    assert embedding_compat_warning(None, None) is None
    # an explicit mismatch still warns as before
    assert "fall back to keyword" in embedding_compat_warning(m, "text-embedding-3-small")


def test_session_constructors_default_to_no_model(tmp_path):
    from scilink.agents.planning_agents.planning_agent import PlanningAgent
    a = PlanningAgent(model_name="bedrock/us.anthropic.claude-opus-4-8",
                      kb_base_path=str(tmp_path / "kb" / "k"), output_dir=str(tmp_path))
    assert a.kb_docs.embedding_client is None and a.kb_code.embedding_client is None
    import inspect
    for mod, cls in (("planning_orchestrator", "PlanningOrchestratorAgent"),
                     ("meta_orchestrator", "MetaOrchestratorAgent")):
        m = __import__(f"scilink.agents.{'planning_agents' if 'planning' in mod else 'meta_agent'}.{mod}",
                       fromlist=[cls])
        sig = inspect.signature(getattr(m, cls).__init__)
        assert sig.parameters["embedding_model"].default is None
