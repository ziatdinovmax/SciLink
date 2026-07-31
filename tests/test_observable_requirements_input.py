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


class TestGateChecksList:
    """Piece 2: the pre-run coverage gate checks a passed-in list rather than
    re-inferring it from the goal."""

    def test_assess_signature_has_param(self):
        from scilink.agents.sim_agents.critics import RunCritic
        assert "required_observables" in inspect.signature(
            RunCritic.assess).parameters

    def test_refinement_context_carries_observables(self):
        from scilink.agents.sim_agents.refinement import RefinementContext
        ctx = RefinementContext(research_goal="x", required_observables=REQS)
        assert ctx.required_observables == REQS

    def test_gate_prompt_uses_authoritative_list(self, tmp_path):
        from scilink.agents.sim_agents.critics import RunCritic
        (tmp_path / "log.lammps").write_text("run completed\n")
        critic = RunCritic(api_key="test-key")
        captured = {}

        def fake_json(prompt):
            captured["p"] = prompt
            return {"status": "success", "run_status": "succeeded",
                    "verdict": "good"}

        critic._generate_json = fake_json

        critic.assess(str(tmp_path), "compute the mass density",
                      check_observables=True, required_observables=REQS)
        assert "authoritative" in captured["p"]
        assert "shear viscosity [cadence]" in captured["p"]
        assert "Determine which physical properties" not in captured["p"]

        captured.clear()
        critic.assess(str(tmp_path), "compute the mass density",
                      check_observables=True)
        assert "Determine which physical properties" in captured["p"]
        assert "authoritative" not in captured["p"]


def _fake_detector(deck_text, signal):
    table = {
        "stress": {"present": True, "interval_steps": 10},
        "trajectory": {"present": False, "interval_steps": None},
    }
    return table.get(signal, {"present": False, "interval_steps": None})


class TestDeterministicCheckers:
    """Piece 3: engine-neutral signal_present + cadence checkers, delegating
    detection to a supplied engine tool."""

    def test_signal_present_flags_absent(self):
        from scilink.agents.sim_agents.contradictions import check_requirements
        reqs = [Requirement("Zn RDF", "signal_present", {"signal": "trajectory"})]
        c = check_requirements(reqs, {"deck": "x", "signal_detector": _fake_detector})
        assert len(c) == 1 and "trajectory" in c[0].message

    def test_signal_present_passes_when_logged(self):
        from scilink.agents.sim_agents.contradictions import check_requirements
        reqs = [Requirement("viscosity", "signal_present", {"signal": "stress"})]
        c = check_requirements(reqs, {"deck": "x", "signal_detector": _fake_detector})
        assert c == []

    def test_cadence_flags_undersampled(self):
        from scilink.agents.sim_agents.contradictions import check_requirements
        reqs = [Requirement("viscosity", "cadence",
                            {"signal": "stress", "max_interval_steps": 1})]
        c = check_requirements(reqs, {"deck": "x", "signal_detector": _fake_detector})
        assert len(c) == 1 and "every 10 steps" in c[0].message

    def test_cadence_passes_when_dense_enough(self):
        from scilink.agents.sim_agents.contradictions import check_requirements
        reqs = [Requirement("viscosity", "cadence",
                            {"signal": "stress", "max_interval_steps": 50})]
        c = check_requirements(reqs, {"deck": "x", "signal_detector": _fake_detector})
        assert c == []

    def test_degrades_without_detector(self):
        from scilink.agents.sim_agents.contradictions import check_requirements
        reqs = [Requirement("viscosity", "signal_present", {"signal": "stress"})]
        c = check_requirements(reqs, {"deck": "x"})  # no detector -> defer to LLM
        assert c == []


class TestLammpsDetector:
    """Piece 3: the LAMMPS engine realization of detect_signal_logging."""

    def _fn(self):
        from scilink.skills._shared._registry import get_tool_function
        return get_tool_function("detect_signal_logging", active_skills=["lammps"])

    def test_trajectory_from_dump(self):
        deck = "units real\ndump 1 all custom 500 t.dump id x y z\n"
        assert self._fn()(deck_text=deck, signal="trajectory") == {
            "present": True, "interval_steps": 500}

    def test_thermo_signal_present_and_cadence(self):
        deck = "thermo 100\nthermo_style custom step temp press density\n"
        r = self._fn()(deck_text=deck, signal="density")
        assert r == {"present": True, "interval_steps": 100}

    def test_stress_absent_when_not_logged(self):
        deck = "thermo 100\nthermo_style custom step temp\n"
        assert self._fn()(deck_text=deck, signal="stress")["present"] is False

    def test_stress_present_via_compute_and_ave(self):
        deck = ("thermo 100\nthermo_style custom step temp\n"
                "compute p all pressure thermo_temp\n"
                "fix v all ave/correlate 5 100 1000 c_p[1]\n")
        r = self._fn()(deck_text=deck, signal="stress")
        assert r["present"] is True and r["interval_steps"] == 1000


class TestGateDeterministicLayer:
    """Piece 3: the gate's deterministic layer resolves the real engine tool and
    blocks the union with the LLM layer."""

    def _ctx(self, reqs):
        from scilink.agents.sim_agents.refinement import RefinementContext
        return RefinementContext(research_goal="compute viscosity",
                                 engine="lammps", required_observables=reqs)

    def test_blocks_missing_signal(self):
        from scilink.agents.sim_agents.refinement import _deterministic_coverage
        ctx = self._ctx([Requirement("shear viscosity", "signal_present",
                                     {"signal": "stress"})])
        deck = "thermo 100\nthermo_style custom step temp\n"  # no stress logged
        blocking = _deterministic_coverage(ctx, deck)
        assert len(blocking) == 1 and "stress" in blocking[0]

    def test_passes_when_signal_present(self):
        from scilink.agents.sim_agents.refinement import _deterministic_coverage
        ctx = self._ctx([Requirement("shear viscosity", "signal_present",
                                     {"signal": "stress"})])
        deck = ("thermo 100\nthermo_style custom step temp press\n")  # press logged
        assert _deterministic_coverage(ctx, deck) == []

    def test_no_observables_no_block(self):
        from scilink.agents.sim_agents.refinement import _deterministic_coverage
        ctx = self._ctx(None)
        assert _deterministic_coverage(ctx, "anything") == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
