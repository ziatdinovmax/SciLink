"""Build-time leg of the KB degradation ladder.

Live failure: a Bedrock-only session (no Google key) uploaded two PDFs;
generate_initial_plan triggered a KB build, the Gemini embedding call
raised, and the WHOLE plan call aborted — even though the documents had
already been read and the query-time ladder (dense -> BM25 ->
no-context) would have coped had the KB existed. Build now degrades to
a KEYWORD-ONLY KB (chunks kept, dense index dropped, loud warning);
save/load round-trips it; retrieve() itself falls to BM25 so every
call site gets the ladder; content gates count chunks, not vectors.
"""

from types import SimpleNamespace

import pytest

from scilink.knowledge.knowledge_base import KnowledgeBase


class NoKeyEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_content(self, **kw):
        self.calls += 1
        raise RuntimeError(
            "Missing Gemini API key. Set the GEMINI_API_KEY or "
            "GOOGLE_API_KEY environment variable.")


CHUNKS = [
    {"text": "The controllability map plots intervention timing against "
             "endpoint distribution shift.", "metadata": {"source": "a"}},
    {"text": "Bayesian optimization selects the next perturbation from "
             "the acquisition function.", "metadata": {"source": "a"}},
    {"text": "Opentrons handoff limits throughput to twelve samples per "
             "day between the two laboratories.", "metadata": {"source": "b"}},
]


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.embedding_client = NoKeyEmbedder()
    kb.embedding_model_name = "gemini-embedding-001"
    kb.index = None
    kb.chunks = []
    kb.sources = []
    kb.repo_maps = {}
    return kb


def test_build_degrades_to_keyword_only(capsys):
    kb = make_kb()
    kb.build(list(CHUNKS))                      # must NOT raise
    out = capsys.readouterr().out
    assert "KEYWORD-ONLY" in out
    assert kb.index is None and len(kb.chunks) == 3


def test_keyword_only_kb_retrieves_via_bm25():
    kb = make_kb()
    kb.build(list(CHUNKS))
    hits = kb.retrieve("throughput Opentrons handoff", top_k=1)
    assert hits and "Opentrons" in hits[0]["text"]
    assert kb.embedding_client.calls == 1       # only the failed build call


def test_save_load_roundtrip_keyword_only(tmp_path):
    kb = make_kb()
    kb.build(list(CHUNKS))
    idx = tmp_path / "kb.faiss"
    ch = tmp_path / "kb.json"
    src = tmp_path / "kb.sources.json"
    kb.save(str(idx), str(ch), sources_path=str(src))
    assert not idx.exists()                     # no dense index written

    kb2 = make_kb()
    ok = kb2.load(str(idx), str(ch), sources_path=str(src))
    assert ok and kb2.index is None and len(kb2.chunks) == 3
    hits = kb2.retrieve("acquisition function Bayesian", top_k=1)
    assert hits and "Bayesian" in hits[0]["text"]


def test_partial_dense_state_is_dropped_entirely():
    """An existing dense index + a failed incremental build would
    silently search a stale subset — the whole index must drop so BM25
    over ALL chunks is the single source of truth."""
    kb = make_kb()
    kb.index = SimpleNamespace(ntotal=5)        # pre-existing dense index
    kb.build(list(CHUNKS))
    assert kb.index is None
    assert len(kb.chunks) == 3


def test_content_gates_count_chunks_not_vectors():
    from pathlib import Path
    pa = Path("scilink/agents/planning_agents/planning_agent.py").read_text()
    assert pa.count("keyword-only KBs count as content") == 3
    ot = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert "kb_available = bool(self.orch.planner.kb_code.chunks)" in ot


def test_routing_guidance_on_knowledge_paths():
    from pathlib import Path
    ot = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    assert ot.count("additional_context, not") >= 2   # both plan tools
    assert ot.count("CORPORA too large to read") >= 2
