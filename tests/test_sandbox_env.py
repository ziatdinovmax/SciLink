"""A generated script gets an allowlisted environment and optional limits.

It used to inherit the agent process's whole environment — vendor keys, a
Bedrock token, the web access token — which nothing a script legitimately
does needs. Both executors now build the child's environment from
``sandbox_env`` and apply ``SCILINK_SANDBOX_*`` limits in the child.
"""
import json
import os

import pytest

from scilink.executors import ScriptExecutor, WarmScriptExecutor, sandbox_env

SECRETS = {
    "ANTHROPIC_API_KEY": "sk-a", "OPENAI_API_KEY": "sk-o", "GOOGLE_API_KEY": "g",
    "GEMINI_API_KEY": "g2", "AWS_BEARER_TOKEN_BEDROCK": "bt", "AWS_SECRET_ACCESS_KEY": "s",
    "AWS_ACCESS_KEY_ID": "id", "FUTUREHOUSE_API_KEY": "fh", "SCILINK_API_KEY": "proxy",
    "SCILINK_WEB_TOKEN": "web", "HF_TOKEN": "hf", "DB_PASSWORD": "pw",
}
KEPT = {"PATH": "/bin", "HOME": "/h", "SCILINK_HOME": "/h/.scilink", "SCILINK_MODELS": "/m",
        "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/p",
        "MPLBACKEND": "Agg", "https_proxy": "http://proxy:3128", "LC_ALL": "C"}

ENV_SCRIPT = 'import json, os; print("ENV:" + json.dumps(dict(os.environ)))'


def _child_env(res):
    return json.loads(res["stdout"].split("ENV:")[1].splitlines()[0])


def test_allowlist_keeps_the_stack_and_drops_every_secret():
    env = sandbox_env(source={**SECRETS, **KEPT})
    assert env == KEPT


def test_explicit_extras_and_the_site_escape_hatch():
    src = {**KEPT, "LAB_LICENSE": "x", "OTHER": "y", "SCILINK_SANDBOX_ENV": "LAB_LICENSE, MISSING"}
    env = sandbox_env({"MP_API_KEY": "mp"}, source=src)
    assert env["MP_API_KEY"] == "mp" and env["LAB_LICENSE"] == "x"
    assert "OTHER" not in env and "MISSING" not in env


def test_cold_executor_child_sees_no_vendor_key(tmp_path, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    res = ScriptExecutor(timeout=60, mp_api_key="mp-key").execute_script(ENV_SCRIPT, str(tmp_path))
    assert res["status"] == "success", res
    env = _child_env(res)
    assert not (set(SECRETS) & set(env)), set(SECRETS) & set(env)
    assert env["SCILINK_HOME"] == str(tmp_path) and env["MP_API_KEY"] == "mp-key"
    assert "PATH" in env


def test_warm_executor_worker_sees_no_vendor_key(tmp_path, monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    ex = WarmScriptExecutor(timeout=60)
    try:
        res = ex.execute_script(ENV_SCRIPT, str(tmp_path))
        assert res["status"] == "success", res
        assert not (set(SECRETS) & set(_child_env(res)))
    finally:
        ex.close()


@pytest.mark.skipif(os.name == "nt", reason="rlimits are POSIX")
def test_file_size_limit_is_enforced_in_the_child(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_SANDBOX_FILE_MB", "1")
    script = 'open("big.bin", "wb").write(b"x" * (3 << 20)); print("wrote")'
    res = ScriptExecutor(timeout=60).execute_script(script, str(tmp_path))
    assert res["status"] == "error", res
    assert (tmp_path / "big.bin").stat().st_size <= 1 << 20


@pytest.mark.skipif(os.name == "nt", reason="rlimits are POSIX")
def test_no_limit_by_default_and_bad_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("SCILINK_SANDBOX_FILE_MB", raising=False)
    monkeypatch.setenv("SCILINK_SANDBOX_MEM_MB", "lots")
    script = 'open("big.bin", "wb").write(b"x" * (3 << 20)); print("wrote")'
    res = ScriptExecutor(timeout=60).execute_script(script, str(tmp_path))
    assert res["status"] == "success", res


def test_platform_essentials_pass_through():
    """What an interpreter needs to start on each platform is never dropped."""
    src = {"SYSTEMROOT": "C:\\Windows", "COMSPEC": "cmd.exe", "PATHEXT": ".EXE", "USERPROFILE": "C:\\U",
           "LOCALAPPDATA": "C:\\L", "__CF_USER_TEXT_ENCODING": "0x1F5:0:0", "LD_LIBRARY_PATH": "/opt/lib",
           "ASE_VASP_COMMAND": "vasp", "VASP_PP_PATH": "/pp", "SLURM_JOB_ID": "7"}
    assert sandbox_env(source=src) == src
