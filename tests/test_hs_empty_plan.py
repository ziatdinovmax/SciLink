"""A hyperspectral plan with nothing to run is asked for again, then recorded.

With decomposition skipped, the plan's `custom_code` target is the only
stage that measures anything. A plan without one used to run nothing and
end as a clean success with a report written from no data. The planner now
asks once more, and if the plan still has nothing to run the dynamic-analysis
step records it, so the run is no longer a clean success.

  conda run -n scilink python -m pytest tests/test_hs_empty_plan.py -q
"""
import json
import logging
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    RunDynamicAnalysisController, SelectRefinementTargetController)

import test_ungated_outputs as tu

EMPTY = {"refinement_needed": True, "reasoning": "a custom_code target is required",
         "targets": []}
GOOD = {"refinement_needed": True, "reasoning": "map the band",
        "targets": [{"type": "custom_code", "description": "fit the band per pixel",
                     "required_outputs": ["Peak_Map"]}]}
ZOOM = {"refinement_needed": True, "reasoning": "zoom",
        "targets": [{"type": "spatial", "description": "zoom on the bright region"}]}


class Model:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.sent = []

    def generate_content(self, contents, **kw):
        self.sent.append(contents)
        return json.dumps(next(self.answers))


def _plan(model, skip=True):
    ctrl = SelectRefinementTargetController(
        model, logging.getLogger("t"), None, None, lambda r: (json.loads(r), None))
    cube, _ = tu._cube()
    state = {"hspy_data": cube, "analysis_images": [], "iteration_title": "T",
             "system_info": {"experiment": {"technique": "spectrum image"}}}
    if skip:
        state["skip_decomposition"] = True
    else:
        state["analysis_images"] = [{"label": "Component 1", "data": b"x"}]
    return ctrl.execute(state)


def _text(contents):
    return "\n".join(p for p in contents if isinstance(p, str))


def test_an_empty_plan_is_asked_for_again_and_the_second_answer_is_used():
    model = Model(EMPTY, GOOD)
    state = _plan(model)
    assert len(model.sent) == 2
    retry = _text(model.sent[1])
    assert "YOUR PREVIOUS ANSWER" in retry and "custom_code" in retry
    assert _text(model.sent[0]) in retry           # the same evidence, plus the answer
    decision = state["refinement_decision"]
    assert decision["requires_custom_code"] is True
    assert [t["type"] for t in decision["targets"]] == ["custom_code"]


def test_a_second_empty_answer_keeps_the_first():
    model = Model(EMPTY, EMPTY)
    state = _plan(model)
    assert len(model.sent) == 2
    assert state["refinement_decision"]["targets"] == []
    assert state["refinement_decision"]["requires_custom_code"] is False


def test_a_retry_that_fails_keeps_the_first():
    class Broken(Model):
        def generate_content(self, contents, **kw):
            self.sent.append(contents)
            if len(self.sent) == 2:
                raise RuntimeError("provider down")
            return json.dumps(EMPTY)
    model = Broken()
    state = _plan(model)
    assert len(model.sent) == 2
    assert state["refinement_decision"]["targets"] == []


def test_a_plan_with_a_custom_code_target_is_asked_once():
    model = Model(GOOD)
    state = _plan(model)
    assert len(model.sent) == 1
    assert state["refinement_decision"]["requires_custom_code"] is True


def test_a_refinement_with_decomposition_is_never_re_asked():
    """With decomposition maps, no target (or a zoom) is a legitimate answer."""
    for answer in (EMPTY, ZOOM):
        model = Model(answer)
        _plan(model, skip=False)
        assert len(model.sent) == 1


def _run_dynamic(state):
    ctrl = RunDynamicAnalysisController(
        model=Model(), logger=logging.getLogger("t"), generation_config=None,
        safety_settings=None, parse_fn=lambda r: (json.loads(r), None), executor_timeout=60)
    return ctrl.execute(state)


def test_nothing_to_run_without_decomposition_is_recorded():
    state = _run_dynamic({"skip_decomposition": True,
                          "refinement_decision": {"targets": [], "requires_custom_code": False}})
    notes = state["degradation_notes"]
    assert len(notes) == 1
    assert notes[0]["kind"] == "withheld" and notes[0]["confidence"] == "none"
    assert "plan had nothing to compute" in notes[0]["caveat"]


def test_nothing_to_run_after_decomposition_is_not_recorded():
    """Decomposition maps are the result there; no refinement is no failure."""
    state = _run_dynamic({"refinement_decision": {"targets": [], "requires_custom_code": False}})
    assert "degradation_notes" not in state


def test_a_failed_planner_call_says_so():
    ctrl = SelectRefinementTargetController(
        Model("not json"), logging.getLogger("t"), None, None,
        lambda r: (None, {"error": "unparseable"}))
    cube, _ = tu._cube()
    state = ctrl.execute({"skip_decomposition": True, "hspy_data": cube,
                          "analysis_images": [], "iteration_title": "T"})
    assert state["refinement_decision"]["planning_failed"] is True
    state = _run_dynamic(state)
    assert "planner failed" in state["degradation_notes"][0]["caveat"]


def test_locked_plans_make_no_planner_call():
    """Replays and locked targets build their plan without the model."""
    class NoCalls:
        def generate_content(self, *a, **k):
            raise AssertionError("the planner called the model")
    ctrl = SelectRefinementTargetController(
        NoCalls(), logging.getLogger("t"), None, None, lambda r: (json.loads(r), None))
    for extra in ({"reuse_records": [{"script": "def analyze_feature(d, a): return {}",
                                      "target": "replay", "required_outputs": ["M"]}]},
                  {"locked_targets": [{"target": "map the band", "required_outputs": ["M"]}]}):
        state = ctrl.execute({"skip_decomposition": True, "analysis_images": [], **extra})
        assert state["refinement_decision"]["requires_custom_code"] is True


def test_nothing_measured_makes_the_run_partial(tmp_path, monkeypatch):
    """The note reaches the run's status: partial, confidence none, with the caveat."""
    import numpy as np
    from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa
    note = {"kind": "withheld", "confidence": "none", "missing_required": [],
            "caveat": "The analysis plan had nothing to compute, so no maps or "
                      "measurements were produced."}
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_auto_select_skills",
                        lambda self, *a, **k: [])
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_run_analysis_pipeline",
                        lambda self, *a, **k: ({"detailed_analysis": "x", "scientific_claims": [],
                                                "degradation_notes": [note]}, None))
    cube = tmp_path / "cube.npy"
    np.save(cube, np.ones((4, 4, 8), dtype=np.float32))
    agent = hsa.HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(tmp_path / "out"),
                                           enable_human_feedback=False)
    res = agent.analyze(str(cube), system_info={"technique": "EELS"})
    assert res["status"] == "partial"
    assert res["confidence"] == "none"
    assert note["caveat"] in res["warnings"]
