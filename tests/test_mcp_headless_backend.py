"""The MCP server enforces a headless matplotlib backend itself (was
integration rule 2 of every client: MCP tools run off the main thread, and
macOS's GUI backend kills the call). Runs in subprocesses so each case sees
a clean matplotlib import."""
import os
import subprocess
import sys

PY = sys.executable

CASE = """
import os, sys
{setup}
from scilink.mcp_server import create_server
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
create_server(api_key="sk-dummy", mode="plan", session_dir=None)
import matplotlib
print(matplotlib.get_backend(), os.environ.get("MPLBACKEND"))
"""


def _run(setup):
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    env.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
    r = subprocess.run([PY, "-c", CASE.format(setup=setup)], capture_output=True,
                       text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()[-1]


def test_unset_defaults_to_agg():
    backend, env = _run("").split()
    assert backend.lower() == "agg" and env == "Agg"


def test_gui_backend_forced_to_agg():
    out = _run("import matplotlib; matplotlib.use('MacOSX' if sys.platform=='darwin' else 'TkAgg')")
    assert out.split()[0].lower() == "agg"


def test_explicit_headless_choice_respected():
    out = _run("os.environ['MPLBACKEND'] = 'svg'")
    assert out.split()[0].lower() == "svg"
