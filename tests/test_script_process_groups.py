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


# ── review follow-ups ─────────────────────────────────────────────────────

# The script starts a helper that INHERITS its stdout/stderr and exits at
# once: communicate() waits for EOF the helper holds, and the leader is a
# zombie. On macOS getpgid() of that zombie raises ESRCH.
LEADER_EXITS = textwrap.dedent("""
    import subprocess, sys
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    open("grandchild.pid", "w").write(str(child.pid))
""")

# A helper that does NOT hold the pipes: the script exits 0 and returns.
DETACHED = textwrap.dedent("""
    import subprocess, sys
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open("grandchild.pid", "w").write(str(child.pid))
    print("done")
""")


def test_a_timeout_ends_a_helper_after_the_script_itself_exited(tmp_path):
    t0 = time.time()
    result = ex.ScriptExecutor(timeout=3).execute_script(LEADER_EXITS, working_dir=str(tmp_path))
    assert result["status"] == "error" and "timed out" in result["message"]
    assert time.time() - t0 < 15
    assert _wait_gone(_grandchild(tmp_path))


def test_a_stop_ends_a_helper_after_the_script_itself_exited(tmp_path):
    results = {}
    t = threading.Thread(target=lambda: results.update(
        r=ex.ScriptExecutor(timeout=120).execute_script(LEADER_EXITS, working_dir=str(tmp_path))))
    t.start()
    gc = _grandchild(tmp_path)
    time.sleep(0.5)                                  # the leader has exited by now
    t0 = time.time()
    ex.kill_subprocesses_for_thread(t.ident)
    t.join(timeout=20)
    assert not t.is_alive() and time.time() - t0 < 10
    assert _wait_gone(gc)


def test_a_detached_helper_does_not_outlive_a_successful_script(tmp_path):
    r = ex.ScriptExecutor(timeout=30).execute_script(DETACHED, working_dir=str(tmp_path))
    assert r["status"] == "success" and r["stdout"].strip() == "done"
    assert _wait_gone(_grandchild(tmp_path))


@pytest.mark.parametrize("signame", ["SIGHUP", "SIGTERM"])
def test_a_hangup_or_terminate_of_the_parent_ends_the_script_tree(tmp_path, signame):
    """A new session no longer receives the terminal's SIGHUP (window closed,
    SSH dropped); Python dies of SIGHUP/SIGTERM without running atexit."""
    import signal
    parent_code = textwrap.dedent(f"""
        import sys
        from scilink import executors as ex
        ex.ScriptExecutor(timeout=120).execute_script({SCRIPT!r}, working_dir={str(tmp_path)!r})
    """)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    parent = subprocess.Popen([sys.executable, "-c", parent_code], env=env)
    gc = _grandchild(tmp_path, timeout=30)
    os.kill(parent.pid, getattr(signal, signame))
    parent.wait(timeout=20)
    assert parent.returncode == -getattr(signal, signame)     # still dies of the same signal
    assert _wait_gone(gc)


def test_a_warm_run_that_times_out_takes_its_helper_with_it(tmp_path):
    w = ex.WarmScriptExecutor(timeout=3)
    try:
        r = w.execute_script(SCRIPT, working_dir=str(tmp_path))
        assert r["status"] == "error"
        assert _wait_gone(_grandchild(tmp_path))
    finally:
        w.close()


def test_a_stop_during_a_warm_run_takes_its_helper_with_it(tmp_path):
    w = ex.WarmScriptExecutor(timeout=120)
    results = {}
    try:
        t = threading.Thread(target=lambda: results.update(r=w.execute_script(SCRIPT, working_dir=str(tmp_path))))
        t.start()
        gc = _grandchild(tmp_path, timeout=30)
        ex.kill_subprocesses_for_thread(t.ident)
        t.join(timeout=20)
        assert not t.is_alive()
        assert _wait_gone(gc)
    finally:
        w.close()


def test_the_signal_handler_cannot_deadlock_on_the_registry_lock():
    """A SIGTERM landing while the main thread holds the registry lock ran a
    handler that waited for that same lock forever."""
    # In a subprocess with a timeout, so a regression fails instead of
    # hanging the suite.
    code = ("from scilink import executors as ex\n"
            "with ex._active_subprocesses_lock:\n"      # the main thread is inside _register_subprocess
            "    ex._kill_all_registered()\n"          # what the handler does, on this same thread
            "print('ok')")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "ok", out.stderr[-400:]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_does_not_inherit_the_parents_scripts(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    ex._mark_own_group(proc)
    ex._register_subprocess(proc)
    try:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                                # child: the registry must be empty
            os.write(w, str(sum(len(v) for v in ex._active_subprocesses.values())).encode())
            os._exit(0)
        os.waitpid(pid, 0)
        assert os.read(r, 16) == b"0"
        assert proc.poll() is None                  # the parent's script is untouched
    finally:
        ex._unregister_subprocess(proc)
        ex._kill_process_tree(proc, grace=0.2)
