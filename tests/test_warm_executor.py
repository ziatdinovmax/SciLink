"""A long-lived interpreter for replaying a verified script: imports paid once,
everything else as a fresh process would do it."""

import os
import time

import pytest

from scilink.executors import ScriptExecutor, WarmScriptExecutor

SCRIPT = '''
import json, os, sys
import numpy as np
np.save("out.npy", np.arange(3))
print("RESULT:" + json.dumps({"cwd": os.path.basename(os.getcwd()), "pid": os.getpid(), "argv0": os.path.basename(sys.argv[0])[-3:]}))
print("to stderr", file=sys.stderr)
'''


@pytest.fixture
def warm():
    ex = WarmScriptExecutor(timeout=30)
    yield ex
    ex.close()


def _payload(res):
    import json
    return json.loads(res["stdout"].split("RESULT:")[1].splitlines()[0])


def test_same_result_shape_own_working_directory_one_interpreter(warm, tmp_path):
    cold = ScriptExecutor(timeout=30).execute_script(SCRIPT, working_dir=str(tmp_path / "cold"))
    a = warm.execute_script(SCRIPT, working_dir=str(tmp_path / "a"))
    b = warm.execute_script(SCRIPT, working_dir=str(tmp_path / "b"))
    assert cold["status"] == a["status"] == b["status"] == "success"
    assert set(a) == set(cold) and "to stderr" in a["stderr"]
    assert (_payload(a)["cwd"], _payload(b)["cwd"]) == ("a", "b")
    assert (tmp_path / "a" / "out.npy").is_file() and (tmp_path / "b" / "out.npy").is_file()
    assert _payload(a)["pid"] == _payload(b)["pid"] != os.getpid()        # one worker, not this process
    assert not list((tmp_path / "a").glob("*.py")) and not list((tmp_path / "a").glob(".warm_*"))
    assert os.getcwd() != str(tmp_path / "b")                              # the caller's cwd is never touched


def test_imports_are_paid_once(warm, tmp_path):
    slow = "import time\nimport scilink_warm_probe\n"
    (tmp_path / "scilink_warm_probe.py").write_text("import time\ntime.sleep(0.6)\n")
    script = f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\n" + slow + "print('ok')\n"
    t = []
    for i in range(3):
        t0 = time.perf_counter()
        assert warm.execute_script(script, working_dir=str(tmp_path / f"r{i}"))["status"] == "success"
        t.append(time.perf_counter() - t0)
    assert t[0] > 0.55 and max(t[1:]) < 0.3


def test_a_failing_script_is_a_failed_run_and_the_worker_lives_on(warm, tmp_path):
    bad = warm.execute_script("raise ValueError('no peak here')", working_dir=str(tmp_path / "x"))
    assert bad["status"] == "error" and "ValueError: no peak here" in bad["message"]
    assert "return code 1" in bad["message"]
    exited = warm.execute_script("import sys\nsys.exit(3)", working_dir=str(tmp_path / "y"))
    assert exited["status"] == "error" and "return code 3" in exited["message"]
    assert warm.execute_script("import sys\nsys.exit(0)", working_dir=str(tmp_path / "z"))["status"] == "success"
    assert warm.execute_script(SCRIPT, working_dir=str(tmp_path / "w"))["status"] == "success"


def test_the_timeout_is_hard_and_the_next_run_gets_a_new_worker(warm, tmp_path):
    first = _payload(warm.execute_script(SCRIPT, working_dir=str(tmp_path / "a")))["pid"]
    t0 = time.perf_counter()
    res = warm.execute_script("import time\ntime.sleep(60)", working_dir=str(tmp_path / "s"), timeout=1)
    assert res == {"status": "error", "message": "Script execution timed out after 1 seconds."}
    assert time.perf_counter() - t0 < 10
    after = _payload(warm.execute_script(SCRIPT, working_dir=str(tmp_path / "b")))["pid"]
    assert after != first


def test_one_run_cannot_change_the_next_ones_environment_or_path(warm, tmp_path):
    warm.execute_script("import os, sys\nos.environ['LEAK'] = '1'\nsys.path.insert(0, '/nowhere')\nos.chdir('/')",
                        working_dir=str(tmp_path / "a"))
    res = warm.execute_script("import os, sys\nprint('RESULT:' + __import__('json').dumps("
                              "{'leak': os.environ.get('LEAK'), 'path': '/nowhere' in sys.path, "
                              "'cwd': os.path.basename(os.getcwd()), 'pid': 0, 'argv0': ''}))",
                              working_dir=str(tmp_path / "b"))
    assert _payload(res)["leak"] is None and _payload(res)["path"] is False and _payload(res)["cwd"] == "b"


def test_a_worker_is_retired_after_max_runs(tmp_path):
    ex = WarmScriptExecutor(timeout=30, max_runs=2)
    try:
        pids = [_payload(ex.execute_script(SCRIPT, working_dir=str(tmp_path / f"r{i}")))["pid"] for i in range(3)]
        assert pids[0] == pids[1] != pids[2]
    finally:
        ex.close()
