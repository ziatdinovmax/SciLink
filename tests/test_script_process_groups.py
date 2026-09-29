"""A timed-out or stopped script takes the processes it started with it.

ScriptExecutor killed only its direct child. A generated script that starts
workers (a loky or multiprocessing pool for per-pixel fits, a solver it shells
out to) left them running after a timeout or a Stop, holding memory and CPU
after the analysis had moved on. Scripts now run in their own session and
process group, and the whole group is ended.
"""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from scilink import executors as ex

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")

# The script starts a grandchild that would sleep for a minute, records its
# pid, and then sleeps itself.
SCRIPT = textwrap.dedent("""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open("grandchild.pid", "w").write(str(child.pid))
    time.sleep(60)
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); check it is not one.
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(out.strip()) and not out.strip().startswith("Z")


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _grandchild(tmp_path: Path, timeout: float = 15.0) -> int:
    f = tmp_path / "grandchild.pid"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if f.exists() and f.read_text().strip():
            return int(f.read_text())
        time.sleep(0.05)
    raise AssertionError("the script never started its grandchild")


def test_a_timeout_ends_the_grandchild_too(tmp_path):
    result = ex.ScriptExecutor(timeout=3).execute_script(SCRIPT, working_dir=str(tmp_path))
    assert result["status"] == "error" and "timed out" in result["message"]
    assert _wait_gone(_grandchild(tmp_path))


def test_a_stop_ends_the_grandchild_too(tmp_path):
    results = {}
    t = threading.Thread(target=lambda: results.update(
        r=ex.ScriptExecutor(timeout=120).execute_script(SCRIPT, working_dir=str(tmp_path))))
    t.start()
    gc = _grandchild(tmp_path)
    ex.kill_subprocesses_for_thread(t.ident)      # what a user Stop calls
    t.join(timeout=20)
    assert not t.is_alive()
    assert results["r"]["status"] == "error"
    assert _wait_gone(gc)


def test_an_interrupted_wait_ends_the_script_tree(tmp_path):
    """Ctrl-C no longer reaches a script in its own session, so an interrupt
    while waiting on it must end it explicitly."""
    real = subprocess.Popen.communicate

    def interrupted(self, *a, **k):
        _grandchild(tmp_path)
        raise KeyboardInterrupt

    with mock.patch.object(subprocess.Popen, "communicate", interrupted):
        with pytest.raises(KeyboardInterrupt):
            ex.ScriptExecutor(timeout=120).execute_script(SCRIPT, working_dir=str(tmp_path))
    assert _wait_gone(_grandchild(tmp_path))
    assert subprocess.Popen.communicate is real


def test_exit_cleanup_ends_registered_script_trees(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", SCRIPT], cwd=tmp_path, start_new_session=True)
    ex._register_subprocess(proc)
    gc = _grandchild(tmp_path)
    ex._kill_all_registered()
    assert _wait_gone(gc) and proc.poll() is not None


def test_a_process_sharing_our_group_is_killed_alone(tmp_path):
    """A registered process that is not a group leader (started without a new
    session) is signalled by itself: the caller's own group is never hit."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    assert os.getpgid(proc.pid) == os.getpgrp()
    ex._kill_process_tree(proc, grace=0.5)
    assert proc.poll() is not None               # and this test process is still here


def test_a_successful_script_is_unaffected(tmp_path):
    r = ex.ScriptExecutor(timeout=30).execute_script("print('hello')", working_dir=str(tmp_path))
    assert r["status"] == "success" and r["stdout"].strip() == "hello"
