"""Offline tests for the hyperspectral phase-2 expansion (issue #327):

- HS-1: per-target verification records from RunDynamicAnalysisController
  (happy path level 0; retry ladder climbing to hot and succeeding),
- the T=2 staging gate on those records (memory monkeypatched),
- HS-2: literature-context injection into the planning prompt.
"""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ["UNSAFE_EXECUTION_OK"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

from qc_golden.harness import Rule, ScriptedModel  # noqa: E402

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (  # noqa: E402
    RunDynamicAnalysisController,
    SelectRefinementTargetController,
    _hs_attempt_entry,
    _render_attempt_history,
    _retry_annealing_level,
    _retry_stage_label,
)
from scilink.agents.exp_agents.instruct import (  # noqa: E402
    SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS,
)

logging.basicConfig(level=logging.INFO)

_QC_MARKER = SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS.split("{")[0].strip()[:60]

GOOD_CODE = """\
def analyze_feature(data, energy_axis):
    import numpy as np
    return {
        "maps": {"mean_map": np.asarray(data).mean(axis=2)},
        "units": "a.u.",
        "description": "per-pixel mean intensity",
    }
"""

BROKEN_CODE = """\
def analyze_feature(data, energy_axis):
    raise ValueError("global fit saturated - method inadequate")
"""


def _parse(resp):
    return json.loads(resp.text), None


def _make_state(tmp_path, extra=None):
    cube = np.random.default_rng(0).random((6, 5, 12))
    state = {
        "refinement_decision": {
            "refinement_needed": True,
            "requires_custom_code": True,
            "targets": [{"type": "custom_code",
                         "description": "map the mean intensity per pixel"}],
        },
        "hspy_data": cube,
        "original_hspy_data": cube,
        "system_info": {},
        "settings": {"output_dir": str(tmp_path)},
        "iteration_title": "Test_Iter",
        "analysis_images": [],
        "error_dict": None,
    }
    state.update(extra or {})
    return state


def _run_controller(tmp_path, codegen_responses):
    rules = [
        Rule("qc", _QC_MARKER, [{"valid": True}], repeat_last=True),
        Rule("codegen", "analyze_feature", codegen_responses, repeat_last=True),
    ]
    model = ScriptedModel(rules)
    ctrl = RunDynamicAnalysisController(
        model=model, logger=logging.getLogger("hs_test"),
        generation_config=None, safety_settings=None, parse_fn=_parse,
        executor_timeout=60,
    )
    state = _make_state(tmp_path)
    state = ctrl.execute(state)
    return state, model


def test_happy_path_record_level0(tmp_path):
    state, model = _run_controller(tmp_path, [{"code": GOOD_CODE}])
    recs = state.get("dynamic_analysis_records")
    assert recs and len(recs) == 1
    rec = recs[0]
    assert rec["task_success"] is True
    assert rec["salvaged"] is False
    assert "analyze_feature" in rec["script"]
    qh = rec["quality_history"]
    assert qh["approved"] is True
    assert qh["final_passed_fraction"] == 1.0
    assert qh["threshold"] == RunDynamicAnalysisController.SUCCESS_THRESHOLD
    iters = qh["verification_iterations"]
    assert len(iters) == 1
    assert iters[0]["annealing_level"] == 0
    assert iters[0]["issues"] == []
    # results still committed normally
    assert state["custom_analysis_metadata_list"][0]["name"] == "mean_map"


def test_retry_ladder_reaches_hot_and_succeeds(tmp_path):
    # 3 broken attempts (levels 0, 1, 1) then success at level 2 (hot).
    responses = [{"code": BROKEN_CODE}] * 3 + [{"code": GOOD_CODE}]
    state, model = _run_controller(tmp_path, responses)
    rec = state["dynamic_analysis_records"][0]
    qh = rec["quality_history"]
    assert rec["task_success"] is True
    assert qh["approved"] is True
    levels = [it["annealing_level"] for it in qh["verification_iterations"]]
    assert levels == [0, 1, 1, 2]
    # failed attempts recorded the escalation applied next + the error
    first = qh["verification_iterations"][0]
    assert first["fix_applied"] == "patch the logic/math"
    assert any("method inadequate" in i["problem"] for i in first["issues"])
    third = qh["verification_iterations"][2]
    assert third["fix_applied"] == "abandon the method family"
    # hot success = staging-eligible record
    assert max(levels) >= 2


def test_all_attempts_fail_record_not_approved(tmp_path):
    responses = [{"code": BROKEN_CODE}]
    state, model = _run_controller(tmp_path, responses)
    rec = state["dynamic_analysis_records"][0]
    assert rec["task_success"] is False
    assert rec["quality_history"]["approved"] is False
    assert state.get("dynamic_analysis_failed") is True
    assert len(rec["quality_history"]["verification_iterations"]) == \
        RunDynamicAnalysisController.MAX_RETRIES


def test_retry_helpers():
    assert [_retry_annealing_level(n) for n in (0, 1, 2, 3, 4)] == [0, 1, 1, 2, 2]
    assert _retry_stage_label(0) == ""
    assert _retry_stage_label(3) == "abandon the method family"
    e = _hs_attempt_entry(1, 0.5, ["mean_map: looks like noise"], "question the method",
                          error="boom")
    assert e["issues_found"] == [
        {"location": "mean_map", "problem": "looks like noise"},
        {"location": "execution", "problem": "boom"},
    ]


def test_attempt_history_block():
    """First attempt gets no block; later reviews see the trajectory."""
    assert _render_attempt_history([]) == ""
    entries = [
        _hs_attempt_entry(0, None,
                          ["Au_Thickness: 40 um implausible for sputtered film"],
                          "patch the logic/math", error="Required outputs failed"),
        _hs_attempt_entry(1, 0.5, [], "question the method",
                          error="Required outputs failed: ['Au_Thickness']"),
    ]
    block = _render_attempt_history(entries)
    assert "### PRIOR ATTEMPTS ON THIS TASK" in block
    assert "Attempt 1 (level 0, maps passed n/a)" in block
    assert "Attempt 2 (level 1, maps passed 0.50)" in block
    assert "implausible for sputtered film" in block
    assert "converging on the same measured magnitude" in block


def test_staging_gate_on_records(monkeypatch, tmp_path):
    """The agent staging hook stages approved+hot records and skips others."""
    from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa

    staged_calls = []

    import scilink.skills._shared._staging as _staging_mod

    def _fake_label(domain, kind, deviation, llm_call, tmpl):
        return "xray-edge-mapping"

    def _fake_stage(domain, technique, record):
        staged_calls.append((domain, technique, record))
        return f"sid-{len(staged_calls)}"

    monkeypatch.setattr("scilink.skills.loader.memory_enabled", lambda: True)
    monkeypatch.setattr(_staging_mod, "assign_technique_label", _fake_label)
    monkeypatch.setattr(_staging_mod, "stage_solution", _fake_stage)
    monkeypatch.delenv("SCILINK_T2_AUTODISTILL", raising=False)

    agent = hsa.HyperspectralAnalysisAgent.__new__(hsa.HyperspectralAnalysisAgent)
    agent.logger = logging.getLogger("hs_stage_test")
    agent.model = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text="label"))
    agent.generation_config = None
    agent.safety_settings = None
    agent.output_dir = tmp_path

    def _rec(approved, max_level, script="def analyze_feature(): pass"):
        return {
            "target": "edge jump map",
            "required_outputs": ["edge_jump"],
            "task_success": approved,
            "salvaged": False,
            "script": script,
            "quality_history": {
                "approved": approved,
                "final_passed_fraction": 1.0,
                "verification_iterations": [
                    {"annealing_level": lvl} for lvl in range(max_level + 1)
                ],
            },
        }

    records = [
        _rec(True, 2),            # approved + hot → staged
        _rec(True, 1),            # approved but never hot → skipped
        _rec(False, 2),           # hot but not approved → skipped
        {**_rec(True, 2), "script": None},  # no script → skipped
    ]
    staged = agent._maybe_stage_t2_solutions(records, {"skills_loaded": []})
    assert staged == ["sid-1"]
    assert staged_calls[0][0] == "hyperspectral"
    assert staged_calls[0][2]["working_script"].startswith("def analyze_feature")


def test_literature_context_reaches_planning_prompt(tmp_path):
    rules = [Rule("plan", "refinement", [
        {"refinement_needed": False, "reasoning": "no", "targets": []}
    ], repeat_last=True)]
    model = ScriptedModel(rules)
    ctrl = SelectRefinementTargetController(
        model=model, logger=logging.getLogger("hs_lit_test"),
        generation_config=None, safety_settings=None, parse_fn=_parse,
    )
    state = _make_state(tmp_path, extra={
        "literature_context": "LITFACT-XYZ-789: the K-edge jump ratio is the standard measure.",
    })
    ctrl.execute(state)
    assert len(model.calls) == 1
    prompt = model.calls[0]["prompt"]
    assert "LITFACT-XYZ-789" in prompt
    assert "--- Literature Context ---" in prompt


def test_no_literature_no_section(tmp_path):
    rules = [Rule("plan", "refinement", [
        {"refinement_needed": False, "reasoning": "no", "targets": []}
    ], repeat_last=True)]
    model = ScriptedModel(rules)
    ctrl = SelectRefinementTargetController(
        model=model, logger=logging.getLogger("hs_lit_test2"),
        generation_config=None, safety_settings=None, parse_fn=_parse,
    )
    ctrl.execute(_make_state(tmp_path))
    assert "--- Literature Context ---" not in model.calls[0]["prompt"]
