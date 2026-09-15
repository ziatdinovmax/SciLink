"""Engine-neutral critics work for a PROSE-ONLY engine (LAMMPS).

LAMMPS ships no deterministic critic bundle tools — no ``snapshot_run`` /
``check_input_syntax`` TOOL_SPECs, only ``lammps.md`` prose. These tests
show the engine-neutral ``InputValidator`` / ``RunCritic`` produce a
skill-grounded review for LAMMPS with ZERO critic-code changes and ZERO
bundle tools, relying on the graceful ``LookupError`` fallbacks. This is
the extensibility floor: adding an engine is a ``skill.md``; the critics
ground on its prose, and skill graduation grows that prose over time.

No LLM call (the model is stubbed); no API key required.
"""

import json
import logging

import pytest

from scilink.agents.sim_agents.critics import (
    InputValidator,
    RunCritic,
    _drop_vacuous_fix,
    _has_runnable_content,
    _run_deterministic_syntax_check,
    _snapshot_run_outputs,
)

LAMMPS_SCRIPT = (
    "units metal\n"
    "atom_style atomic\n"
    "boundary p p p\n"
    "read_data in.data\n"
    "pair_style eam/alloy\n"
    "pair_coeff * * Cu.eam.alloy Cu\n"
    "timestep 0.001\n"
    "fix 1 all nvt temp 300 300 0.1\n"
    "run 10000\n"
)


def _stub_critic(cls, fake_text: str):
    """Build a critic instance with the LLM stubbed to return fake_text.

    Bypasses __init__ (no real model / key); installs a model whose
    generate_content records the prompt and returns canned JSON.
    """
    obj = cls.__new__(cls)
    obj.logger = logging.getLogger("test_lammps")
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


# ── Graceful fallbacks: LAMMPS has no deterministic bundle tools ──

class TestProseOnlyFallbacks:
    def test_no_lammps_syntax_tool_falls_back_to_empty(self):
        # No check_input_syntax registered for lammps → empty, no raise.
        issues = _run_deterministic_syntax_check(
            {"in.lammps": LAMMPS_SCRIPT}, "lammps"
        )
        assert issues == []

    def test_no_lammps_snapshot_tool_falls_back_to_note(self, tmp_path):
        snap = _snapshot_run_outputs(str(tmp_path), "lammps")
        # No snapshot_run registered for lammps → a note, not a crash.
        assert "note" in snap


# ── The critics ground on lammps.md prose (zero critic changes) ──

class TestSkillProseGrounding:
    def test_input_validator_uses_lammps_validation_prose(self):
        iv, captured = _stub_critic(
            InputValidator,
            json.dumps({"validation_status": "passes", "suggested_adjustments": []}),
        )
        report = iv.validate(
            input_files={"in.lammps": LAMMPS_SCRIPT},
            system_description="Cu bulk, NVT at 300 K",
            skill="lammps",
            domain="molecular_dynamics",
        )
        # The lammps.md `validation` prose was injected into the prompt.
        assert "Command ordering" in captured["prompt"]
        assert "pair_coeff before pair_style" in captured["prompt"]
        # No deterministic syntax tool → empty syntax_check, still succeeds.
        assert report["status"] == "success"
        assert report["syntax_check"] == []

    def test_run_critic_uses_lammps_interpretation_prose(self, tmp_path):
        rc, captured = _stub_critic(
            RunCritic,
            json.dumps({"run_status": "succeeded", "verdict": "good",
                        "reasoning": "Energy stable; temperature on target."}),
        )
        report = rc.assess(
            output_dir=str(tmp_path),
            research_goal="Cu bulk NVT equilibration at 300 K",
            skill="lammps",
            domain="molecular_dynamics",
        )
        # The lammps.md `interpretation` prose was injected into the prompt.
        assert "Lost atoms" in captured["prompt"]
        assert report["status"] == "success"

    def test_same_critic_classes_serve_vasp_and_lammps(self):
        # The proof: one InputValidator class, no per-engine subclass.
        iv_lammps, cap_l = _stub_critic(
            InputValidator, json.dumps({"validation_status": "passes",
                                        "suggested_adjustments": []}))
        iv_lammps.validate({"in.lammps": LAMMPS_SCRIPT}, "Cu bulk",
                           skill="lammps", domain="molecular_dynamics")
        iv_vasp, cap_v = _stub_critic(
            InputValidator, json.dumps({"validation_status": "passes",
                                        "suggested_adjustments": []}))
        iv_vasp.validate({"INCAR": "ISPIN = 2\nENCUT = 400\n"}, "Fe BCC",
                         skill="vasp", domain="periodic_dft")
        # Each got its own engine's prose from the same class.
        assert "Command ordering" in cap_l["prompt"]          # lammps validation
        assert "ENCUT" in cap_v["prompt"] or "ISMEAR" in cap_v["prompt"]  # vasp validation


class TestVacuousFixGuard:
    """The post-run critic must not return a no-op (all-comment) deck as a fix."""

    def test_has_runnable_content(self):
        assert _has_runnable_content("units real\nrun 0\n") is True
        # all comments / blank -> no runnable content
        assert _has_runnable_content("# just\n# comments\n\n   \n") is False
        assert _has_runnable_content("! INCAR-style comment only\n") is False
        # one real command among comments still counts
        assert _has_runnable_content("# note\nkspace_style pppm 1e-4\n") is True

    def test_drop_vacuous_fix_nulls_all_comment_deck(self):
        report = {"suggested_fixes": {"run.lammps": "# recommended ordering:\n# read_data ...\n"}}
        _drop_vacuous_fix(report)
        assert report["suggested_fixes"] is None
        assert "no executable commands" in report.get("diagnostic_notes", "")

    def test_drop_vacuous_fix_keeps_real_deck(self):
        deck = "units real\natom_style full\nread_data system.data\nkspace_style pppm 1e-4\nrun 0\n"
        report = {"suggested_fixes": {"run.lammps": deck}}
        _drop_vacuous_fix(report)
        assert report["suggested_fixes"] == {"run.lammps": deck}

    def test_drop_vacuous_fix_noop_when_no_fixes(self):
        report = {"suggested_fixes": None}
        _drop_vacuous_fix(report)
        assert report["suggested_fixes"] is None
