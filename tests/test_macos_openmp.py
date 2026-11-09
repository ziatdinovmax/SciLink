"""``import scilink`` then a faiss search must not kill the process.

On macOS the torch and faiss-cpu wheels each bundle libomp.dylib; with both
loaded, the first faiss search aborted with ``OMP: Error #15`` (exit 134)
although ``scilink`` imports torch first. Allowing the duplicate runtime
alone turned that into ``OMP: Error #179`` on any search of more than one
row; with faiss on one thread (``knowledge/_faiss.load_faiss``) both pass.
Run in a fresh interpreter: the failure kills the process, and the variable
must be set before either runtime starts.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("faiss")

_SEARCH = (
    "import scilink, numpy as np, torch\n"
    "from scilink.knowledge._faiss import load_faiss\n"
    "faiss = load_faiss()\n"
    "i = faiss.IndexFlatL2(64)\n"
    "i.add(np.random.rand(5000, 64).astype('float32'))\n"
    "for rows in (1, 2, 50):\n"
    "    d, n = i.search(np.random.rand(rows, 64).astype('float32'), 3)\n"
    "    assert n.shape == (rows, 3)\n"
    "assert torch.randn(300, 300).mm(torch.randn(300, 300)).shape == (300, 300)\n"
    "print('ok')\n"
)


def _run(env):
    return subprocess.run([sys.executable, "-c", _SEARCH], capture_output=True, text=True,
                          env=env, timeout=300)


def test_a_faiss_search_after_import_scilink_does_not_abort():
    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    r = _run(env)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


@pytest.mark.skipif(sys.platform != "darwin", reason="the variable is set on macOS only")
def test_a_value_the_user_set_is_kept():
    out = subprocess.run(
        [sys.executable, "-c", "import os; os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'; import scilink; "
                               "print(os.environ['KMP_DUPLICATE_LIB_OK'])"],
        capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("FALSE"), out.stderr[-2000:]
