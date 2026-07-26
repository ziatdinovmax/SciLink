"""RunCritic can attribute a converged-but-wrong result to the force field.

The post-run critic already rates a converged run "poor" when the result is
suspect. These tests cover the added ability to name the ROOT cause of such a
result as ``force_field`` — a physically unsound number no input-deck change
can fix — parallel to the existing ``structure`` cause, and to judge "poor" by
reasoning about consistency with known physical behaviour rather than only
convergence. This is the hook the (future) reparameterization fixer routes on.

No LLM call (the model is stubbed); no API key required.
"""

import json
import logging

from scilink.agents.sim_agents.critics import RunCritic


def _stub_run_critic(fake_text: str):
    """RunCritic with the LLM stubbed to record the prompt and return canned
    JSON — same shape as tests/test_lammps_prose_critic.py::_stub_critic."""
    obj = RunCritic.__new__(RunCritic)
    obj.logger = logging.getLogger("test_run_critic_physics")
    obj.futurehouse_api_key = None
    captured = {}

    class _Model:
        def generate_content(self, prompt, generation_config=None):
            captured["prompt"] = prompt

            class _Resp:
                text = fake_text

            return _Resp()

    obj.model = _Model()
    return obj, captured


def test_prompt_offers_force_field_cause_and_physics_reasoning(tmp_path):
    rc, captured = _stub_run_critic(
        json.dumps({"run_status": "succeeded", "verdict": "good",
                    "reasoning": "ok"})
    )
    rc.assess(output_dir=str(tmp_path),
              research_goal="Equilibrium density of a molecular liquid")
    prompt = captured["prompt"]
    # The force-field cause is offered alongside the existing ones...
    assert "force_field" in prompt
    assert '"deck" | "structure" | "force_field"' in prompt
    # ...and "poor" is framed as a reasoning-about-physics judgement, not just
    # a convergence check (kept principle-level — no domain/property tokens).
    assert "known physical behaviour" in prompt


def test_converged_force_field_verdict_passes_through(tmp_path):
    # A run that completed cleanly but is physically wrong: poor verdict,
    # cause attributed to the force field, no deck patch offered.
    rc, _ = _stub_run_critic(json.dumps({
        "run_status": "succeeded",
        "verdict": "poor",
        "failure_class": "force_field",
        "reasoning": "Computed property contradicts the components' known "
                     "behaviour; the parameters, not the deck, are at fault.",
        "suggested_fixes": None,
    }))
    report = rc.assess(
        output_dir=str(tmp_path),
        research_goal="Mixture density vs composition",
    )
    assert report["status"] == "success"
    assert report["verdict"] == "poor"
    assert report["failure_class"] == "force_field"
    # No deck fix is invented for a force-field cause (like "structure").
    assert report.get("suggested_fixes") is None
