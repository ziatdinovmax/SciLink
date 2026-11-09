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
    env = {k: v for k, v in os.environ.items()
           if k not in ("KMP_DUPLICATE_LIB_OK", "_SCILINK_SET_KMP_DUPLICATE_LIB_OK")}
    r = _run(env)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


@pytest.mark.skipif(sys.platform != "darwin", reason="the variable is set on macOS only")
def test_a_value_the_user_set_is_kept():
    out = subprocess.run(
        [sys.executable, "-c", "import os; os.environ['KMP_DUPLICATE_LIB_OK']='FALSE'; import scilink; "
                               "print(os.environ['KMP_DUPLICATE_LIB_OK'])"],
        capture_output=True, text=True, timeout=300)
    assert out.stdout.strip().endswith("FALSE"), out.stderr[-2000:]


_SEARCH_ON_A_WORKER_THREAD = (
    "import threading, scilink, numpy as np\n"
    "from scilink.knowledge.knowledge_base import KnowledgeBase\n"
    "from scilink.knowledge._faiss import load_faiss\n"
    "faiss = load_faiss()\n"                       # loaded and pinned on the main thread
    "kb = KnowledgeBase.__new__(KnowledgeBase)\n"
    "kb.index = faiss.IndexFlatL2(64)\n"
    "kb.index.add(np.random.rand(20000, 64).astype('float32'))\n"
    "out = []\n"
    "def worker():\n"
    "    load_faiss()\n"                            # what retrieve() does before its search
    "    d, n = kb.index.search(np.random.rand(200, 64).astype('float32'), 3)\n"
    "    out.append(n.shape)\n"
    "t = threading.Thread(target=worker); t.start(); t.join()\n"
    "assert out == [(200, 3)]\n"
    "print('ok')\n"
)


def test_a_search_on_another_thread_than_the_load_does_not_crash():
    """The thread count is per OpenMP thread: a pin made where the index was
    loaded does not reach a chat turn's thread that searches it. The search
    path pins on its own thread."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("KMP_DUPLICATE_LIB_OK", "_SCILINK_SET_KMP_DUPLICATE_LIB_OK")}
    r = subprocess.run([sys.executable, "-c", _SEARCH_ON_A_WORKER_THREAD], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0 and "ok" in r.stdout, (r.returncode, r.stderr[-2000:])


def test_retrieve_pins_the_searching_thread():
    """The one dense search site calls load_faiss on its own thread first."""
    import inspect
    from scilink.knowledge.knowledge_base import KnowledgeBase
    src = inspect.getsource(KnowledgeBase)
    search = src.index("self.index.search(")
    assert "load_faiss()" in src[max(0, search - 300):search]


def test_the_allowance_scilink_set_does_not_reach_a_generated_script():
    """Set by SciLink for its own process, never passed to a generated script;
    a value the user set is passed as before."""
    from scilink.executors import sandbox_env
    ours = {"PATH": "/bin", "KMP_DUPLICATE_LIB_OK": "TRUE", "_SCILINK_SET_KMP_DUPLICATE_LIB_OK": "1"}
    env = sandbox_env(source=ours)
    assert "KMP_DUPLICATE_LIB_OK" not in env and "_SCILINK_SET_KMP_DUPLICATE_LIB_OK" not in env
    assert sandbox_env(source={"PATH": "/bin", "KMP_DUPLICATE_LIB_OK": "TRUE"})["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    # the site escape hatch can still pass it deliberately
    hatch = {**ours, "SCILINK_SANDBOX_ENV": "KMP_DUPLICATE_LIB_OK"}
    assert sandbox_env(source=hatch)["KMP_DUPLICATE_LIB_OK"] == "TRUE"
