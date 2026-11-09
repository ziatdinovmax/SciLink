"""faiss, imported the one way every knowledge-base module uses.

On macOS the torch and faiss-cpu wheels each bundle libomp.dylib, and
``import scilink`` loads torch. ``scilink/__init__.py`` allows the duplicate
runtime (``KMP_DUPLICATE_LIB_OK``), which stops the abort at the first
search, but faiss's own runtime still cannot start a thread team beside
torch's (``OMP: Error #179``, a crash on any search of more than one row).
One faiss thread avoids the team; a knowledge-base index is small enough
that a single thread costs nothing noticeable. Elsewhere faiss is untouched.
"""

import sys

_pinned = False


def load_faiss():
    """``import faiss``, pinned to one OpenMP thread on macOS."""
    global _pinned
    import faiss
    if sys.platform == "darwin" and not _pinned:
        faiss.omp_set_num_threads(1)
        _pinned = True
    return faiss
