"""The FutureHouse literature stack (edison-client -> ldp -> fhlmi) is
OPTIONAL: a broken or absent chain must not brick agent initialization.

Regression for a Windows report: a version-skewed fhlmi raised
"No module named 'lmi.config'" during `from edison_client import ...`,
which the old module-scope guard re-raised — taking down the whole UI
("Failed to initialize agent") over an ancillary feature. Import-time
failures now defer to CONSTRUCTION, which raises a clean, actionable
RuntimeError (same pattern as the meta agent's guarded `ase` import).
"""

import subprocess
import sys

import pytest


_BROKEN_STACK_SCRIPT = '''
import sys
sys.modules["edison_client"] = None  # simulate the broken FutureHouse chain

# 1. Every module-scope importer of lit_agents must still import.
import scilink.agents.exp_agents.analysis_orchestrator_tools  # noqa: F401
from scilink.agents.lit_agents import OwlLiteratureAgent, MoleculesAgent

# 2. Constructing a literature agent fails with the actionable message.
for cls, kwargs in ((OwlLiteratureAgent, {"api_key": "x"}),
                    (MoleculesAgent, {"api_key": "x"})):
    try:
        cls(**kwargs)
        raise SystemExit(f"{cls.__name__} should have raised")
    except RuntimeError as e:
        assert "Literature features are unavailable" in str(e), str(e)
        assert "pip install -U edison-client ldp fhlmi fhaviary" in str(e)
print("OK")
'''


def test_broken_literature_stack_does_not_brick_agent_init():
    r = subprocess.run([sys.executable, "-c", _BROKEN_STACK_SCRIPT],
                       capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"
    assert "OK" in r.stdout


def test_intact_stack_constructs_normally():
    # In this (healthy) environment the guard is a no-op.
    from scilink.agents.lit_agents.literature_agent import (
        EdisonClient, _EDISON_IMPORT_ERROR, _require_edison,
    )
    assert EdisonClient is not None and _EDISON_IMPORT_ERROR is None
    _require_edison()  # must not raise
