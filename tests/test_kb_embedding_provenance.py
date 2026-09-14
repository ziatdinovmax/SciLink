"""#631 — a dense index queried with a different embedding model.

Live: a Bedrock/Titan web session showed the dense path ("Retrieving top
10 ...") and then "Dense KB retrieval failed ()" — an EMPTY message —
before every TEA query fell back to BM25. The empty message is FAISS's
Python wrapper rejecting a query whose dimension differs from the index's
with a bare ``assert``: the index had been built by another embedding
model (a named store KB built earlier, or the store manifest's model on
the add-documents path) and nothing recorded that. Now the index carries
its provenance, a mismatch is explained once and switches the KB to
keyword retrieval up front, a legacy index without provenance gets a
named dimension error instead of ``()``, and no query is embedded twice
for an index it can never match.

  conda run -n scilink python -m pytest tests/test_kb_embedding_provenance.py -v
"""
import json
import logging

import numpy as np
import pytest

from scilink.knowledge.knowledge_base import (
    EmbeddingDimensionMismatch, KnowledgeBase, describe_error)
from scilink.knowledge.rag_engine import retrieve_context

CHUNKS = [
    {"text": "Titanium coatings improve cathode cycling stability.",
     "metadata": {"source": "a.pdf", "content_type": "text"}},
    {"text": "Cost of alumina precursor per kilogram in 2025.",
     "metadata": {"source": "b.pdf", "content_type": "text"}},
    {"text": "Opentrons handoff limits throughput per day.",
     "metadata": {"source": "c.pdf", "content_type": "text"}},
]


class FakeEmbedder:
    def __init__(self, dim):
        self.dim, self.calls = dim, 0

    def embed_content(self, model, content, task_type=None):
        self.calls += 1
        items = content if isinstance(content, list) else [content]
        return {"embedding": [np.random.default_rng(len(t)).random(self.dim).tolist()
                              for t in items]}


def make_kb(model, dim):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.embedding_client = FakeEmbedder(dim)
    kb.embedding_model_name = model
    kb.index = None
    kb.chunks = []
    kb.sources = []
    kb.repo_maps = {}
    kb.index_built_with = None
    kb._dense_disabled_reason = None
    return kb


@pytest.fixture
def built(tmp_path):
    """A KB built by 'model-A' (4-d) and saved as the planning agent saves
    its docs KB (default_kb_docs.* — the prefix the meta's cache copy globs)."""
    kb = make_kb("model-A", 4)
    kb.build(list(CHUNKS))
    prefix = tmp_path / "default_kb_docs"
    paths = (str(prefix.with_suffix(".faiss")), str(prefix.with_suffix(".json")),
             str(prefix.with_suffix(".sources.json")))
    kb.save(paths[0], paths[1], sources_path=paths[2])
    return tmp_path, paths


def _load(paths, model, dim):
    kb = make_kb(model, dim)
    assert kb.load(paths[0], paths[1], sources_path=paths[2])
    return kb


def test_save_records_index_provenance(built):
    tmp_path, paths = built
    meta = json.loads((tmp_path / "default_kb_docs.meta.json").read_text())
    assert meta == {"embedding_model": "model-A", "dimension": 4, "vectors": 3}
    # The sidecar shares the KB prefix, so the meta's kb_cache copy
    # (glob "default_kb_*") carries the provenance along.
    assert tmp_path / "default_kb_docs.meta.json" in set(tmp_path.glob("default_kb_*"))


def test_same_model_keeps_dense_retrieval(built, caplog):
    _, paths = built
    kb = _load(paths, "model-A", 4)
    assert kb.index_built_with == "model-A" and kb._dense_disabled_reason is None
    with caplog.at_level(logging.WARNING):
        ctx = retrieve_context(kb, "precursor cost", top_k=2)
    assert ctx and kb.embedding_client.calls == 1    # dense: query embedded
    assert "Dense KB retrieval failed" not in caplog.text


def test_model_mismatch_is_explained_once_and_uses_keyword_search(built, caplog, capsys):
    """The live shape, with provenance: no embedding call, no '()' message."""
    _, paths = built
    with caplog.at_level(logging.WARNING):
        kb = _load(paths, "model-B", 3)
    assert kb._dense_disabled_reason
    assert "built with 'model-A'" in caplog.text
    assert "embeds with 'model-B'" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ctx = retrieve_context(kb, "precursor cost per kilogram", top_k=2)
        ctx2 = retrieve_context(kb, "Opentrons throughput", top_k=1)
    assert "precursor" in ctx and "Opentrons" in ctx2
    assert kb.embedding_client.calls == 0        # never embeds for this index
    assert "Dense KB retrieval failed" not in caplog.text   # no failure — a mode
    assert "retrieving via BM25" in capsys.readouterr().out


def test_legacy_index_without_provenance_names_the_dimension_mismatch(built, caplog):
    """Index files written before the sidecar existed: the first query
    still has to embed, but the failure is named, and the KB stops
    embedding queries for an index they can never match."""
    tmp_path, paths = built
    (tmp_path / "default_kb_docs.meta.json").unlink()
    kb = _load(paths, "model-B", 3)
    assert kb.index_built_with is None and kb._dense_disabled_reason is None
    with caplog.at_level(logging.WARNING):
        ctx = retrieve_context(kb, "precursor cost per kilogram", top_k=2)
        retrieve_context(kb, "again", top_k=2)
    assert "precursor" in ctx
    assert "failed ()" not in caplog.text
    assert "EmbeddingDimensionMismatch" in caplog.text
    assert "3-d" in caplog.text and "4-d" in caplog.text
    assert "embeds with 'model-B'" in caplog.text
    assert kb.embedding_client.calls == 1        # second query: no re-embed


def test_appending_with_another_model_is_a_named_error(built):
    """Adding documents to a foreign index used to die with a bare
    AssertionError from FAISS; now it says what is wrong."""
    _, paths = built
    kb = _load(paths, "model-B", 3)
    kb._dense_disabled_reason = None             # force the append path
    with pytest.raises(EmbeddingDimensionMismatch, match="new document embeddings are 3-d"):
        kb.build([{"text": "more", "metadata": {"source": "d"}}])


def test_describe_error_never_renders_empty():
    assert describe_error(AssertionError()) == "AssertionError: AssertionError()"
    assert describe_error(RuntimeError("boom")) == "RuntimeError: boom"


def test_rebuild_after_mismatch_reenables_dense(built):
    """A fresh build with the session's model clears the mismatch."""
    _, paths = built
    kb = _load(paths, "model-B", 3)
    kb.index = None                              # rebuild from scratch
    kb.chunks = []
    kb.build(list(CHUNKS))
    assert kb.index_built_with == "model-B" and kb._dense_disabled_reason is None
    assert kb.retrieve("precursor", top_k=1)
