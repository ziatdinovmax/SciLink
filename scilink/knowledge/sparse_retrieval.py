"""Model-free sparse retrieval (Okapi BM25) over in-memory KB chunks.

The dense-retrieval escape hatch: a knowledge base's chunk text is stored
alongside its vector index, so when the embedding provider that built the
index is unavailable (missing key, provider switch), keyword retrieval
still works — over any KB, built by any embedding model. Quality sits
below dense retrieval (no paraphrase matching) but far above the
no-retrieval fallback it replaces as the middle degradation tier.

Pure Python, no dependencies; linear scan scoring is comfortably fast for
the corpus sizes KBs hold (~10^3-10^4 chunks).
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Standard Okapi constants: k1 saturates term frequency, b scales the
# document-length normalization.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; numbers kept (element symbols,
    concentrations and units matter in scientific corpora)."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_state(chunk_texts: List[str]) -> Dict[str, Any]:
    """Precompute the corpus statistics BM25 scoring needs."""
    docs = [Counter(tokenize(t)) for t in chunk_texts]
    doc_lens = [sum(c.values()) for c in docs]
    n = len(docs)
    df: Counter = Counter()
    for c in docs:
        df.update(c.keys())
    return {
        "docs": docs,
        "doc_lens": doc_lens,
        "avgdl": (sum(doc_lens) / n) if n else 0.0,
        "idf": {t: math.log(1 + (n - d + 0.5) / (d + 0.5))
                for t, d in df.items()},
        "n": n,
    }


def bm25_scores(state: Dict[str, Any], query: str) -> List[float]:
    """Okapi BM25 score of every document in ``state`` against ``query``."""
    q_terms = tokenize(query)
    idf, avgdl = state["idf"], state["avgdl"] or 1.0
    scores = [0.0] * state["n"]
    for i, (doc, dl) in enumerate(zip(state["docs"], state["doc_lens"])):
        s = 0.0
        for t in q_terms:
            tf = doc.get(t)
            if not tf:
                continue
            s += idf.get(t, 0.0) * (tf * (_K1 + 1)) / (
                tf + _K1 * (1 - _B + _B * dl / avgdl))
        scores[i] = s
    return scores


def bm25_top_k(chunks: List[Dict[str, Any]], query: str, top_k: int = 5,
               state: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]],
                                                      Dict[str, Any]]:
    """Top-k chunks (same raw dicts dense retrieval returns) by BM25.

    Returns ``(hits, state)`` — pass the state back in on subsequent calls
    over the same corpus to skip re-tokenization. Zero-score documents are
    excluded: a chunk sharing no terms with the query is not a match.
    """
    if not chunks:
        return [], {"n": 0}
    if state is None or state.get("n") != len(chunks):
        state = build_bm25_state([c.get("text", "") for c in chunks])
    scores = bm25_scores(state, query)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    hits = [chunks[i] for i in ranked[:top_k] if scores[i] > 0.0]
    return hits, state
