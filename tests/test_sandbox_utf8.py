"""Generated scripts run in UTF-8 on every platform.

The executor wrote the script file as UTF-8 but ran it with the platform's
console encoding (cp1252 on Windows), so a ``print`` of a check mark raised
``UnicodeEncodeError`` and cost an attempt. ``sandbox_env`` now puts the
child in UTF-8 mode, and every executor decodes the child's pipes as UTF-8
(``errors="replace"``). UTF-8 mode is set ALONE: it gives the child's stdout
``utf-8`` with ``surrogateescape``, so a file name that is not valid UTF-8
(surrogate-escaped by ``os.listdir``) prints too; ``PYTHONIOENCODING=utf-8``
beside it would make stdout strict and that print raise. The non-UTF-8
console is simulated with ``PYTHONIOENCODING=ascii`` and a C locale in the
test's own environment — the locale passes the sandbox allowlist, the
parent's ``PYTHONIOENCODING`` is dropped, and UTF-8 mode governs.

  conda run -n scilink python -m pytest tests/test_sandbox_utf8.py -q
"""
import json
import os

import pytest

from scilink import executors
from scilink.executors import (SANDBOX_UTF8_ENV, ScriptExecutor, WarmScriptExecutor,
                              run_generated_script, sandbox_env)

RESULT = {"model_type": "✓ Gaussian (≥ 1 peak, σ ≠ 0)", "parameters": {"peak_1": {"center": 144.0}}}
SCRIPT = ('print("✓ fit done, R² ≥ 0.99, σ ≠ 0")\n'
          'import json; print("FIT_RESULTS_JSON:" + json.dumps(' + repr(RESULT) + ', ensure_ascii=False))\n')
# A file name that is not valid UTF-8, as ``os.listdir`` hands it to a script
# under UTF-8 mode (surrogate-escaped), and the bare surrogate itself.
SURROGATE_SCRIPT = ('import os\n'
                    'print("listed:", os.fsdecode(b"spectrum_\\x80.csv"))\n'
                    'print("\\udc80")\n'
                    'print("FIT_RESULTS_JSON:" + \'{"ok": true}\')\n')


def _parsed(res):
    return json.loads(res["stdout"].split("FIT_RESULTS_JSON:", 1)[1].splitlines()[0])


@pytest.fixture
def ascii_console(monkeypatch):
    """A parent whose console cannot hold non-ASCII, as cp1252 cannot hold ✓."""
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    monkeypatch.delenv("PYTHONUTF8", raising=False)


def test_sandbox_env_drops_the_parents_pythonioencoding_and_sets_utf8_mode():
    env = sandbox_env(source={"PATH": "/bin", "PYTHONIOENCODING": "cp1252", "LC_ALL": "C"})
    assert env["PYTHONUTF8"] == "1" and "PYTHONIOENCODING" not in env
    assert env["LC_ALL"] == "C"                                   # the locale itself still passes
    # the site escape hatch does not bring it back either
    src = {"PYTHONIOENCODING": "cp1252", "SCILINK_SANDBOX_ENV": "PYTHONIOENCODING"}
    assert "PYTHONIOENCODING" not in sandbox_env(source=src)
    # an executor's explicit extras stay in charge
    assert sandbox_env({"PYTHONIOENCODING": "latin-1"}, source={})["PYTHONIOENCODING"] == "latin-1"


def test_without_the_forced_utf8_a_non_ascii_print_fails(tmp_path, ascii_console, monkeypatch):
    """The failure the fix removes: the child took the parent's console
    encoding (its PYTHONIOENCODING passed the allowlist, no UTF-8 mode) and
    the script's own print raised."""
    original = executors.sandbox_env

    def inherited(extra=None, source=None):
        env = original(extra, source)
        env.pop("PYTHONUTF8", None)
        env["PYTHONIOENCODING"] = os.environ["PYTHONIOENCODING"]
        return env

    monkeypatch.setattr(executors, "sandbox_env", inherited)
    res = ScriptExecutor(timeout=60).execute_script(SCRIPT, str(tmp_path))
    assert res["status"] == "error", res
    assert "UnicodeEncodeError" in res["message"]


def test_cold_executor_runs_non_ascii_under_an_ascii_console(tmp_path, ascii_console):
    res = ScriptExecutor(timeout=60).execute_script(SCRIPT, str(tmp_path))
    assert res["status"] == "success", res
    assert "✓ fit done, R² ≥ 0.99" in res["stdout"]              # reached the parent intact
    assert _parsed(res) == RESULT                                 # and the parser


def test_warm_executor_runs_non_ascii_under_an_ascii_console(tmp_path, ascii_console):
    ex = WarmScriptExecutor(timeout=60)
    try:
        res = ex.execute_script(SCRIPT, str(tmp_path))
        assert res["status"] == "success", res
        assert ex.cold_fallbacks == 0
        assert _parsed(res) == RESULT and "✓" in res["stdout"]
    finally:
        ex.close()


def test_run_generated_script_decodes_utf8(tmp_path, ascii_console):
    path = tmp_path / "s.py"
    path.write_text(SCRIPT, encoding="utf-8")
    proc = run_generated_script(path, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "✓" in proc.stdout and json.loads(proc.stdout.split("FIT_RESULTS_JSON:", 1)[1]) == RESULT


def test_a_surrogate_escaped_name_prints_under_the_sandbox_env(tmp_path, ascii_console):
    """UTF-8 mode gives the child's stdout ``surrogateescape``: a file name that
    is not valid UTF-8 prints instead of raising, and reaches the parent as a
    replacement character with the result beside it."""
    res = ScriptExecutor(timeout=60).execute_script(SURROGATE_SCRIPT, str(tmp_path))
    assert res["status"] == "success", res
    assert "listed: spectrum_\ufffd.csv" in res["stdout"]
    assert _parsed(res) == {"ok": True}
    path = tmp_path / "s.py"
    path.write_text(SURROGATE_SCRIPT, encoding="utf-8")
    proc = run_generated_script(path, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "listed: spectrum_\ufffd.csv" in proc.stdout


def test_a_surrogate_escaped_name_prints_in_the_warm_executor(tmp_path, ascii_console):
    ex = WarmScriptExecutor(timeout=60)
    try:
        res = ex.execute_script(SURROGATE_SCRIPT, str(tmp_path))
        assert res["status"] == "success", res
        assert ex.cold_fallbacks == 0
        assert _parsed(res) == {"ok": True}
    finally:
        ex.close()


def test_a_strict_pythonioencoding_would_make_that_print_raise(tmp_path, ascii_console, monkeypatch):
    """Why UTF-8 mode is set alone: ``PYTHONIOENCODING=utf-8`` beside it makes
    the child's stdout strict, and the same print raises and spends the attempt."""
    monkeypatch.setattr(executors, "SANDBOX_UTF8_ENV", {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    res = ScriptExecutor(timeout=60).execute_script(SURROGATE_SCRIPT, str(tmp_path))
    assert res["status"] == "error", res
    assert "UnicodeEncodeError" in res["message"]


def test_the_forced_variable_is_utf8_mode_alone():
    assert SANDBOX_UTF8_ENV == {"PYTHONUTF8": "1"}
