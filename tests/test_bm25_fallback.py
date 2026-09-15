"""Offline tests for BM25 sparse retrieval and the degradation tiers in
retrieve_context (dense -> BM25 -> no-context)."""

import logging

import pytest

from scilink.knowledge.sparse_retrieval import (
    bm25_top_k, build_bm25_state, tokenize,
)
from scilink.knowledge.rag_engine import retrieve_context


def _chunk(text, source="doc.md"):
    return {"text": text, "metadata": {"source": source, "content_type": "doc"}}


CORPUS = [
    _chunk("Lithium recovery from produced water using LDH sorbents "
           "achieved 91 percent recovery with high Mg Li selectivity.",
           "lithium_note.md"),
    _chunk("Iodine membranes recovered 78 percent of iodide from Permian "
           "brine at 32 mg/L.", "iodine_note.md"),
    _chunk("The DOE criticality matrix ranks gallium and rare earths as "
           "supply-critical materials.", "doe_assessment.pdf"),
]


class TestBM25:
    def test_relevant_chunk_ranks_first(self):
        hits, _ = bm25_top_k(CORPUS, "lithium sorbent recovery selectivity")
        assert hits and "Lithium" in hits[0]["text"]

        hits, _ = bm25_top_k(CORPUS, "iodide membrane brine")
        assert hits and "Iodine" in hits[0]["text"]

    def test_no_shared_terms_returns_nothing(self):
        hits, _ = bm25_top_k(CORPUS, "quantum entanglement qubit decoherence")
        assert hits == []

    def test_top_k_respected_and_state_reused(self):
        hits, state = bm25_top_k(CORPUS, "recovery produced water", top_k=2)
        assert len(hits) <= 2
        hits2, state2 = bm25_top_k(CORPUS, "criticality matrix", state=state)
        assert state2 is state          # same corpus -> reused
        assert hits2 and "criticality" in hits2[0]["text"].lower()

    def test_state_invalidated_on_corpus_change(self):
        _, state = bm25_top_k(CORPUS, "lithium")
        bigger = CORPUS + [_chunk("boron extraction resin")]
        hits, state2 = bm25_top_k(bigger, "boron resin", state=state)
        assert state2 is not state
        assert hits and "boron" in hits[0]["text"]

    def test_empty_corpus(self):
        hits, _ = bm25_top_k([], "anything")
        assert hits == []

    def test_tokenizer_keeps_numbers(self):
        assert "91" in tokenize("achieved 91% recovery at pH 6.8")

    def test_scores_finite_on_degenerate_docs(self):
        state = build_bm25_state(["", "a", "a a a"])
        assert state["n"] == 3          # no crash on empty documents


class _Index:
    ntotal = 3


class _DenseBrokenKB:
    """Dense retrieval raises (missing provider key); sparse works."""
    index = _Index()
    chunks = CORPUS

    def retrieve(self, query, top_k=10):
        raise RuntimeError("Missing Gemini API key")

    def retrieve_sparse(self, query, top_k=10):
        hits, _ = bm25_top_k(self.chunks, query, top_k=top_k)
        return hits


class _AllBrokenKB(_DenseBrokenKB):
    def retrieve_sparse(self, query, top_k=10):
        raise RuntimeError("no chunks loaded")


class _HealthyKB(_DenseBrokenKB):
    def retrieve(self, query, top_k=10):
        return [CORPUS[2]]              # dense path returns its own answer


class TestRetrieveContextTiers:
    def test_dense_failure_falls_back_to_bm25(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = retrieve_context(_DenseBrokenKB(), "lithium sorbent recovery")
        assert "lithium_note.md" in out          # real grounding, not empty
        assert "BM25" in caplog.text

    def test_both_tiers_failing_degrades_to_empty(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = retrieve_context(_AllBrokenKB(), "lithium")
        assert out == ""
        assert "without retrieved context" in caplog.text

    def test_dense_success_never_touches_sparse(self):
        out = retrieve_context(_HealthyKB(), "anything")
        assert "doe_assessment.pdf" in out

    def test_bm25_no_match_yields_empty_context(self):
        out = retrieve_context(_DenseBrokenKB(), "qubit decoherence entanglement")
        assert out == ""
