"""Two gaps that let a hyperspectral run end without the quantities asked for.

#809: a boolean (or integer) map from the generated script crashed the
dashboard's percentile scaling, and the exception failed the WHOLE attempt,
discarding every other map in it. Maps are now float before any check or
plot, and a map whose figure cannot be drawn fails its own review only.

#810: a measurement target planned with an empty ``required_outputs`` gave
the gate nothing to hold an attempt to, so an attempt returning only
diagnostic maps was committed as the task's result. When the run has an
objective, the planner is asked once to declare them; a target that already
declares them, or a run with no objective, is untouched.

Through the real controllers (``RunDynamicAnalysisController``,
``SelectRefinementTargetController``) with scripted models.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ["UNSAFE_EXECUTION_OK"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

from qc_golden.harness import Rule, ScriptedModel  # noqa: E402
from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc  # noqa: E402
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (  # noqa: E402
    RunDynamicAnalysisController,
    SelectRefinementTargetController,
    _as_float_maps,
    _required_outputs_rules,
)
from scilink.agents.exp_agents.instruct import SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS  # noqa: E402

_QC_MARKER = SPECTROSCOPY_VISUAL_QC_INSTRUCTIONS.split("{")[0].strip()[:60]


def _parse(resp):
    return json.loads(resp.text), None


# ── #809: map dtype ──────────────────────────────────────────────────────

def test_float_maps_pass_through_as_the_same_objects():
    a = np.random.default_rng(0).random((3, 4))
    out = _as_float_maps({"V": a})
    assert out["V"] is a                       # the working path is untouched


@pytest.mark.parametrize("value, expected", [
    (np.array([[True, False]]), np.array([[1.0, 0.0]])),
    (np.array([[1, 2]], dtype=np.int64), np.array([[1.0, 2.0]])),
    (np.array([[1, 2]], dtype=np.uint8), np.array([[1.0, 2.0]])),
    ([[0.5, 1.5]], np.array([[0.5, 1.5]])),
    (np.array([[1, 2]], dtype=object), np.array([[1.0, 2.0]])),
])
def test_non_float_numeric_maps_become_float(value, expected):
    out = _as_float_maps({"M": value})["M"]
    assert out.dtype.kind == "f"
    np.testing.assert_array_equal(out, expected)


def test_a_non_numeric_map_is_left_for_the_existing_checks():
    weird = np.array([["a", "b"]], dtype=object)
    assert _as_float_maps({"W": weird})["W"] is weird


def _cube():
    rng = np.random.default_rng(0)
    return rng.random((6, 5, 12)), np.linspace(0.0, 11.0, 12)


def _run_dynamic(tmp_path, code, required=None, description="map the mean intensity per pixel"):
    rules = [
        Rule("qc", _QC_MARKER, [{"valid": True}], repeat_last=True),
        Rule("codegen", "analyze_feature", [{"code": code}], repeat_last=True),
    ]
    model = ScriptedModel(rules)
    ctrl = RunDynamicAnalysisController(model=model, logger=logging.getLogger("t809"),
                                        generation_config=None, safety_settings=None,
                                        parse_fn=_parse, executor_timeout=60)
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    cube, E = _cube()
    target = {"type": "custom_code", "description": description}
    if required is not None:
        target["required_outputs"] = required
    state = ctrl.execute({
        "refinement_decision": {"refinement_needed": True, "requires_custom_code": True,
                                "targets": [target]},
        "hspy_data": cube, "original_hspy_data": cube, "energy_axis": E, "system_info": {},
        "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T",
        "analysis_images": [], "error_dict": None, "max_verification_iterations": 0})
    return state, model


MASK_AND_VALUE = """\
def analyze_feature(data, energy_axis):
    import numpy as np
    d = np.asarray(data)
    v = d.mean(axis=2)
    return {"maps": {"Value": v, "Detection_Mask": v > np.median(v)},
            "units": "a.u.", "description": "mean and a detection mask"}
"""


def test_a_boolean_mask_beside_a_required_map_commits_both(tmp_path):
    state, model = _run_dynamic(tmp_path, MASK_AND_VALUE, required=["Value"])
    rec = state["dynamic_analysis_records"][0]
    assert rec["task_success"] is True
    names = {m["name"] for m in state["custom_analysis_metadata_list"]}
    assert {"Value", "Detection_Mask"} <= names
    assert sum(c["rule"] == "codegen" for c in model.calls) == 1   # no retry spent


def test_a_figure_that_cannot_be_drawn_fails_only_its_map(tmp_path, monkeypatch):
    real = hc.tools.create_feature_dashboard

    def flaky(result_map, feature_name, *a, **k):
        if feature_name == "Detection_Mask":
            raise RuntimeError("renderer exploded")
        return real(result_map, feature_name, *a, **k)

    monkeypatch.setattr(hc.tools, "create_feature_dashboard", flaky)
    state, _ = _run_dynamic(tmp_path, MASK_AND_VALUE, required=["Value"])
    assert state["dynamic_analysis_records"][0]["task_success"] is True
    names = {m["name"] for m in state["custom_analysis_metadata_list"]}
    assert "Value" in names and "Detection_Mask" not in names


def test_a_float_only_attempt_is_unchanged(tmp_path):
    """Request parity for the common case: float maps, one codegen call, committed."""
    code = MASK_AND_VALUE.replace("v > np.median(v)", "v * 2.0")
    state, model = _run_dynamic(tmp_path, code, required=["Value"])
    assert state["dynamic_analysis_records"][0]["task_success"] is True
    assert sum(c["rule"] == "codegen" for c in model.calls) == 1


# ── #810: required outputs ───────────────────────────────────────────────

_DECL_MARKER = "names no `required_outputs`"


def _plan(required):
    t = {"type": "custom_code", "value": None, "description": "Per-pixel fit of the main band: center and area maps."}
    if required is not None:
        t["required_outputs"] = required
    return {"refinement_needed": True, "reasoning": "fit needed", "targets": [t]}


def _run_planner(tmp_path, plan, declaration=None, objective="Map where the material is and how its band position varies."):
    rules = []
    if declaration is not None:
        rules.append(Rule("declare", _DECL_MARKER, [declaration], repeat_last=True))
    rules.append(Rule("plan", "refinement", [plan], repeat_last=True))
    model = ScriptedModel(rules)
    ctrl = SelectRefinementTargetController(model=model, logger=logging.getLogger("t810"),
                                            generation_config=None, safety_settings=None,
                                            parse_fn=_parse)
    cube, E = _cube()
    state = {"hspy_data": cube, "original_hspy_data": cube, "energy_axis": E, "system_info": {},
             "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T",
             "analysis_images": [], "error_dict": None}
    if objective:
        state["analysis_objective"] = objective
    state = ctrl.execute(state)
    return state, model


def test_an_empty_declaration_is_asked_for_once_and_applied(tmp_path):
    state, model = _run_planner(tmp_path, _plan([]),
                                {"required_outputs": ["Band_Center", "Band_Area"], "reason": "named"})
    target = state["refinement_decision"]["targets"][0]
    assert target["required_outputs"] == ["Band_Center", "Band_Area"]
    assert [c["rule"] for c in model.calls] == ["plan", "declare"]
    decl = model.calls[1]["prompt"]
    assert "Map where the material is" in decl and "Per-pixel fit of the main band" in decl
    assert _required_outputs_rules() in decl          # the planner's own rules, verbatim


def test_a_missing_field_counts_as_empty(tmp_path):
    state, _ = _run_planner(tmp_path, _plan(None), {"required_outputs": ["Band_Center"], "reason": "r"})
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == ["Band_Center"]


def test_a_declared_target_gets_no_extra_call(tmp_path):
    state, model = _run_planner(tmp_path, _plan(["Peak_Center"]),
                                {"required_outputs": ["SHOULD_NOT_APPEAR"], "reason": "r"})
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == ["Peak_Center"]
    assert [c["rule"] for c in model.calls] == ["plan"]


def test_no_objective_no_extra_call(tmp_path):
    state, model = _run_planner(tmp_path, _plan([]), {"required_outputs": ["X"], "reason": "r"}, objective=None)
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == []
    assert [c["rule"] for c in model.calls] == ["plan"]


def test_an_exploratory_answer_keeps_the_list_empty(tmp_path):
    state, _ = _run_planner(tmp_path, _plan([]), {"required_outputs": [], "reason": "exploratory"})
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == []


def test_invalid_keys_are_dropped_and_duplicates_collapsed(tmp_path):
    state, _ = _run_planner(tmp_path, _plan([]),
                            {"required_outputs": ["Band_Center", "Band_Center", "bad key!", 3, " Band_Area "]})
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == ["Band_Center", "Band_Area"]


@pytest.mark.parametrize("reply", ["not json", {"required_outputs": "Band_Center"}, {"reason": "no field"}])
def test_a_bad_declaration_keeps_the_plan_as_it_was(tmp_path, reply):
    rules_reply = reply if isinstance(reply, dict) else {"_raw": reply}
    state, _ = _run_planner(tmp_path, _plan([]), rules_reply)
    assert state["refinement_decision"]["targets"][0]["required_outputs"] == []
    assert state["refinement_decision"]["requires_custom_code"] is True


def test_the_rules_markers_still_exist_in_the_planner_prompt():
    """The declaration quotes the planner's rules; if the prompt's headings
    move, this must fail rather than send an empty rules block."""
    rules = _required_outputs_rules()
    assert rules.startswith("**Required outputs") and len(rules) > 500
