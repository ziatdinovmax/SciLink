"""The ASE bundle's default_run_command: a Python launch for the driver script.

The ASE runner emits a plain Python script (run_md.py / run_relax.py), so the
run command is `<python> {script}` — no engine binary. This checks the command
uses the SAME interpreter SciLink runs in (sys.executable, not a bare "python")
and is discoverable through the skill registry like any other engine's
default_run_command (MLIP-execution-in-loop, Blocker 3).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.skills._shared._registry import get_tool_function  # noqa: E402
from scilink.skills.molecular_dynamics.ase.ase import (  # noqa: E402
    default_run_command,
)


def test_default_run_command_uses_sys_executable():
    cmd = default_run_command()
    assert cmd == f"{sys.executable} {{script}}"
    # not a bare "python": the run must use THIS environment so the MLIP
    # backend the script imports is the one installed here.
    assert sys.executable in cmd
    assert "{script}" in cmd


def test_default_run_command_fills_script():
    cmd = default_run_command(script="run_md.py")
    assert cmd == f"{sys.executable} run_md.py"


def test_discoverable_via_registry():
    # The registry auto-discovers the bundle; resolving by name must return the
    # same callable, usable exactly like lammps's default_run_command.
    fn = get_tool_function("default_run_command", active_skills=["ase"])
    assert fn is default_run_command
    assert fn() == f"{sys.executable} {{script}}"
