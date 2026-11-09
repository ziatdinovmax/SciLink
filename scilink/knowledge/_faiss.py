"""faiss, imported the one way every knowledge-base module uses.

On macOS the torch and faiss-cpu wheels each bundle libomp.dylib, and
``import scilink`` loads torch. ``scilink/__init__.py`` allows the duplicate
runtime (``KMP_DUPLICATE_LIB_OK``), which stops the abort at the first
search, but faiss's own runtime still cannot start a thread team beside
torch's (``OMP: Error #179``, a crash on any search of more than one row).
One faiss thread avoids the team; a knowledge-base index is small enough
that a single thread costs nothing noticeable. Elsewhere faiss is untouched.

The thread count is an OpenMP setting OF THE CALLING THREAD: a pin made on
the thread that loaded the index does not reach a chat turn's thread that
searches it. So ``load_faiss`` pins on every call, and a search calls it
on its own thread first.
"""

import sys


def load_faiss():
    """``import faiss``, pinned to one OpenMP thread on macOS for the
    calling thread. Cheap; call it on the thread that is about to use faiss."""
    import faiss
    if sys.platform == "darwin":
        faiss.omp_set_num_threads(1)
    return faiss
