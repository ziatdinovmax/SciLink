"""The simulation orchestrator's system prompt routes surgical input edits to
`edit_file` and larger revisions to `apply_input_adjustments` (#439 Tier 1)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.simulation_orchestrator import (  # noqa: E402
    _SYSTEM_PROMPT_BODY)


def test_system_prompt_routes_surgical_edits():
    body = _SYSTEM_PROMPT_BODY.lower()
    # Both tools are named, and the guidance distinguishes them.
    assert "edit_file" in body
    assert "apply_input_adjustments" in body
    assert "surgical" in body or "byte-exact" in body
    # The canonical example (a small exact input change) is present.
    assert "encut" in body
