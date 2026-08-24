"""Every orchestrator system prompt carries the house prose-style rule
(no "load-bearing", no "not merely X, but Y"), and no prompt teaches
those phrases as vocabulary."""

import re
from pathlib import Path

import pytest

from scilink.utils.prose_style import PROSE_STYLE_RULE


def test_rule_names_both_banned_moves():
    assert '"load-bearing"' in PROSE_STYLE_RULE
    assert "not merely X, but Y" in PROSE_STYLE_RULE


def _prompts():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode, get_system_prompt as analysis_sp)
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel, get_system_prompt as planning_sp)
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationMode, get_system_prompt as sim_sp)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaMode, get_system_prompt as meta_sp)
    return {
        "analysis": analysis_sp(AnalysisMode.AUTONOMOUS),
        "planning": planning_sp(AutonomyLevel.AUTONOMOUS),
        "simulation": sim_sp(SimulationMode.AUTONOMOUS),
        "meta": meta_sp(MetaMode.AUTONOMOUS),
    }


def test_all_four_system_prompts_carry_the_rule():
    for name, prompt in _prompts().items():
        assert PROSE_STYLE_RULE in prompt, name


def test_no_prompt_teaches_the_phrase_as_vocabulary():
    """The phrases may appear only where they are being BANNED — never as a
    rubric term or descriptive vocabulary the model could imitate."""
    for f in (Path("scilink/agents/planning_agents/instruct.py"),
              Path("scilink/agents/exp_agents/instruct.py")):
        text = f.read_text()
        for m in re.finditer(r"load.bearing", text, re.I):
            ctx = text[max(0, m.start() - 200):m.end() + 200]
            assert "never" in ctx or "Never" in ctx, (
                f"{f.name}: 'load-bearing' used outside a ban context:\n{ctx}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
