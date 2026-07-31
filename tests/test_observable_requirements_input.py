"""Piece 1 of the observable-requirements contract: target observables are a
first-class co-input to Generate.

Covers the engine-neutral render helper, that the typed ``required_observables``
parameter is threaded through every input-generation entry point, and that a
declared observable actually reaches the MD planning prompt and the DFT prompt
(present when passed, absent when not — so the default path is unchanged).

No API keys or LLM calls: the one LLM hop (MD planning) is monkeypatched to
capture the prompt; the DFT prompt builder is pure.
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.contradictions import (  # noqa: E402
    Requirement, format_requirements_for_prompt,
)
import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
from scilink.agents.sim_agents.md_simulation_agent import MDSimulationAgent  # noqa: E402
from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent  # noqa: E402


REQS = [
    Requirement("mass density", "signal_present", {}),
    Requirement("shear viscosity", "cadence", {"quantity": "stress", "every": "1 step"}),
]


class TestRenderHelper:
    def test_empty_is_blank(self):
        assert format_requirements_for_prompt([]) == ""
        assert format_requirements_for_prompt(None) == ""

    def test_renders_observable_kind_and_params(self):
        out = format_requirements_for_prompt(REQS)
        assert "mass density [signal_present]" in out
        assert "shear viscosity [cadence]" in out
        assert "quantity=stress" in out and "every=1 step" in out


class TestThreadedSignatures:
    """Every input-generation entry point accepts required_observables."""

    @pytest.mark.parametrize("func", [
        sp._generate_inputs,
        sp._run_workflow_once,
        MDSimulationAgent.generate_simulation,
        MDSimulationAgent.plan_simulation,
        MDSimulationAgent._generate_md_input,
        PeriodicDFTAgent.generate_inputs,
        PeriodicDFTAgent._build_prompt,
    ])
    def test_has_required_observables_param(self, func):
        assert "required_observables" in inspect.signature(func).parameters

    def test_run_complete_workflow_forwards_kwargs(self):
        # It forwards **kwargs to _run_workflow_once, so the typed co-input rides
        # through without an explicit param.
        params = inspect.signature(sp.run_complete_workflow).parameters
        assert any(p.kind == p.VAR_KEYWORD for p in params.values())


class TestReachesPrompt:
    def test_md_planning_prompt_includes_observables(self, tmp_path):
        agent = MDSimulationAgent(working_dir=str(tmp_path), api_key="test-key")
        captured = {}

        def fake_json(prompt):
            captured["p"] = prompt
            return {}

        agent._generate_json = fake_json
        info = {"element_counts": {"O": 2, "H": 4}, "atom_count": 6}

        agent.plan_simulation("equilibrate water", info, required_observables=REQS)
        assert "TARGET OBSERVABLES" in captured["p"]
        assert "shear viscosity [cadence]" in captured["p"]

        captured.clear()
        agent.plan_simulation("equilibrate water", info)
        assert "TARGET OBSERVABLES" not in captured["p"]

    def test_dft_prompt_includes_observables(self):
        agent = PeriodicDFTAgent(api_key="test-key")
        with_obs = agent._build_prompt("POSCAR-CONTENT", "relax the cell", "vasp",
                                       required_observables=REQS)
        assert "Target observables" in with_obs
        assert "mass density [signal_present]" in with_obs

        without = agent._build_prompt("POSCAR-CONTENT", "relax the cell", "vasp")
        assert "Target observables" not in without


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
