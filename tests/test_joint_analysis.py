"""#754: a series whose method needs every unit at once runs as ONE analysis
over all of them; and no prompt role may push a script towards constructing
the inputs a method needs.

Observed live: the series mode ran each measurement of a joint method as its
own unit, each unit script constructed the measurements it lacked from an
assumed parameter and measured the assumption back, and the result was posted
as verified."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scilink.agents.exp_agents import _joint
from scilink.agents.exp_agents._input_integrity import (CODEGEN_PRINCIPLE, CONFORMANCE_PRINCIPLE,
                                                         JUDGE_PRINCIPLE, VERIFIER_PRINCIPLE)


@pytest.fixture(autouse=True)
def _sandboxed(monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------

def test_only_a_declared_joint_shape_is_joint():
    assert _joint.analysis_shape_of({"analysis_shape": "joint"}) == "joint"
    assert _joint.analysis_shape_of({"analysis_shape": " Joint "}) == "joint"
    assert _joint.analysis_shape_of({"series_analysis_plan": {"analysis_shape": "joint"}}) == "joint"
    for plan in ({}, {"analysis_shape": "per_unit"}, {"analysis_shape": "jointly-ish"}, None, "joint",
                 {"analysis_shape": None}):
        assert _joint.analysis_shape_of(plan) == "per_unit"


def test_units_are_staged_with_a_manifest_and_described_to_the_script(tmp_path):
    units = [{"path": str(tmp_path / f"unit_{b}.csv"), "control_value": b} for b in (1, 2, 5)]
    for u in units:
        Path(u["path"]).write_text("x")
    manifest = _joint.stage_joint_units(units, tmp_path / "joint",
                                        to_array=lambda src: np.arange(6.0).reshape(3, 2))
    rows = _joint.read_manifest(manifest)
    assert [r["control_value"] for r in rows] == [1, 2, 5]
    assert all(Path(r["path"]).name.startswith("joint_unit_") and r["shape"] == [3, 2] for r in rows)
    assert all(np.load(r["path"]).shape == (3, 2) for r in rows)
    block = _joint.joint_units_block(manifest, control_name="condition")
    assert "JOINT ANALYSIS: 3 measurements" in block and "condition = 5" in block
    assert "must load EVERY file" in block
    # without a loader (a cube too large to copy) the source path is listed as is
    m2 = _joint.stage_joint_units(units, tmp_path / "joint2")
    assert [r["path"] for r in _joint.read_manifest(m2)] == [u["path"] for u in units]


# ---------------------------------------------------------------------------
# The redirect, through each agent's real analyze()
# ---------------------------------------------------------------------------

class _Planner:
    """Stands in for the pipeline: the first (series) run "plans" a shape;
    the re-entered joint run stops at once, reporting what it was given."""

    def __init__(self, shape, single_key):
        self.shape, self.single_key, self.seen = shape, single_key, []

    def execute(self, state):
        self.seen.append({"single": state.get(self.single_key), "manifest": state.get("joint_manifest")})
        if state.get("joint_manifest"):
            state["error_dict"] = {"error": "joint run reached", "manifest": state["joint_manifest"]}
        elif self.shape:
            state["analysis_shape"] = self.shape
        return state


class _Stop:
    def __init__(self):
        self.reached = False

    def execute(self, state):
        self.reached = True
        state["error_dict"] = {"error": "per-unit series continued"}
        return state


def _curves(tmp_path, n=3):
    paths = []
    for k in range(n):
        x = np.linspace(0, 200, 200)
        p = tmp_path / f"curve_{k}.csv"
        np.savetxt(p, np.column_stack([x, 1.0 / (1 + np.exp(-(x - 70 - 10 * k) / 5))]),
                   delimiter=",", header="x,y", comments="")
        paths.append(str(p))
    return paths


@pytest.mark.parametrize("shape", ["joint", None])
def test_a_curve_series_planned_joint_runs_once_over_every_unit(tmp_path, monkeypatch, shape):
    from scilink.agents.exp_agents import curve_fitting_agent as cfa
    planner, stop = _Planner(shape, "is_single_spectrum"), _Stop()
    monkeypatch.setattr(cfa, "create_unified_curve_fitting_pipeline", lambda *a, **k: [planner, stop])
    ag = cfa.CurveFittingAgent(api_key="sk-dummy", output_dir=str(tmp_path / "out"), enable_human_feedback=False)
    paths = _curves(tmp_path)
    res = ag.analyze(paths, series_metadata={"variable": "condition", "values": [1, 2, 5]})
    if shape == "joint":
        # the series run stopped after planning; ONE run followed, single, with every unit as an input
        assert [s["single"] for s in planner.seen] == [False, True] and not stop.reached
        rows = _joint.read_manifest(planner.seen[1]["manifest"])
        assert [r["source"] for r in rows] == paths and [r["control_value"] for r in rows] == [1, 2, 5]
        assert all(np.load(r["path"]).shape[-1] in (2, 200) or np.load(r["path"]).shape[0] == 2 for r in rows)
        assert res["error"]["error"] == "joint run reached"
        assert res["analysis_shape"] == "joint" and [u["source"] for u in res["joint_units"]] == paths
    else:
        # no declaration: the series mode as on main, nothing staged
        assert [s["single"] for s in planner.seen] == [False] and stop.reached
        assert res["error"]["error"] == "per-unit series continued" and "analysis_shape" not in res
        assert not (tmp_path / "out" / "joint_inputs").exists()


@pytest.mark.parametrize("shape", ["joint", None])
def test_an_image_series_planned_joint_runs_once_over_every_unit(tmp_path, monkeypatch, shape):
    from scilink.agents.exp_agents import image_analysis_agent as iaa
    planner, stop = _Planner(shape, "is_single_image"), _Stop()
    monkeypatch.setattr(iaa, "create_unified_image_analysis_pipeline", lambda *a, **k: [planner, stop])
    ag = iaa.ImageAnalysisAgent(api_key="sk-dummy", output_dir=str(tmp_path / "out"), enable_human_feedback=False)
    paths = []
    for k in range(3):
        p = tmp_path / f"frame_{k}.npy"
        np.save(p, np.random.default_rng(k).random((32, 32)))
        paths.append(str(p))
    res = ag.analyze(paths, series_metadata={"variable": "dose", "values": [0, 1, 2]})
    if shape == "joint":
        assert [s["single"] for s in planner.seen] == [False, True] and not stop.reached
        rows = _joint.read_manifest(planner.seen[1]["manifest"])
        assert [r["source"] for r in rows] == paths and all(r["shape"] == [32, 32] for r in rows)
        assert res["analysis_shape"] == "joint"
    else:
        assert [s["single"] for s in planner.seen] == [False] and stop.reached
        assert "analysis_shape" not in res


@pytest.mark.parametrize("shape", ["joint", None])
def test_a_cube_series_planned_joint_runs_once_over_every_dataset(tmp_path, monkeypatch, shape):
    import test_hs_series as hs
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    calls = hs._Calls()
    hs._install_fake_pipeline(monkeypatch, calls)
    seen = []
    fake = HyperspectralAnalysisAgent._run_analysis_pipeline

    def record(self, *a, **k):
        seen.append(getattr(self, "_joint_manifest", None))
        return fake(self, *a, **k)
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_run_analysis_pipeline", record)

    class Planner(hs._FakeModel):
        def generate_content(self, contents, **kw):
            text = "\n".join(c for c in contents if isinstance(c, str))
            if "Series Regime Planning" in text and shape:
                assert '"analysis_shape"' in text          # the planner was told the field exists
                self.calls.llm.append(text)
                return json.dumps({"observations": "scouted", "analysis_shape": shape})
            return super().generate_content(contents, **kw)
    agent, out = hs._agent(tmp_path, calls, monkeypatch)
    agent.model = Planner(calls)
    paths = hs._cubes(tmp_path)
    res = agent.analyze(paths, system_info=dict(hs.AXIS),
                        series_metadata={"variable": "temperature", "values": [300, 350, 400, 450, 500, 550],
                                         "unit": "K"})
    if shape == "joint":
        assert len(calls.pipeline) == 1 and seen[0] is not None      # one analysis, not anchor + replays
        rows = _joint.read_manifest(seen[0])
        assert [r["path"] for r in rows] == [str(p) for p in paths]    # cubes listed by path, not copied
        assert res["analysis_shape"] == "joint" and len(res["joint_units"]) == 6
    else:
        assert len(calls.pipeline) == 6 and not any(seen) and "analysis_shape" not in res


# ---------------------------------------------------------------------------
# Measured inputs only: every role's prompt carries its principle
# ---------------------------------------------------------------------------

def test_every_role_of_every_agent_carries_its_principle():
    from scilink.agents.exp_agents import instruct as I
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    from scilink.agents.exp_agents.controllers.image_analysis_controllers import UnifiedImageProcessingController
    codegen = [I.FITTING_SCRIPT_INSTRUCTIONS, I.FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
               I.BANK_EDIT_ADAPT_INSTRUCTIONS, I.IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
               I.IMAGE_ANALYSIS_SCRIPT_REFINEMENT_PROMPT, I.IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
               UnifiedImageProcessingController.USER_FEEDBACK_SCRIPT_PROMPT]
    assert all(CODEGEN_PRINCIPLE in t for t in codegen)
    assert all(CONFORMANCE_PRINCIPLE in t for t in (I.PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
                                                   I.IMAGE_ANALYSIS_PLAN_CONFORMANCE_CHECK_INSTRUCTIONS))
    assert all(JUDGE_PRINCIPLE in t for t in (UnifiedSeriesProcessingController.BEST_OF_N_JUDGE_PROMPT,
                                             UnifiedSeriesProcessingController.JUDGE_PROMPT,
                                             UnifiedImageProcessingController.JUDGE_PROMPT,
                                             I.IMAGE_ANALYSIS_BEST_OF_N_SELECTION_PROMPT))
    # the principle sits up front, ahead of any response footer
    for t in codegen:
        assert t.index(CODEGEN_PRINCIPLE) < 400


def test_the_correction_prompt_actually_sent_carries_the_principle(tmp_path):
    """The retry path, from a pipeline built by the real factory: what the
    correction call sends, not only the constant."""
    import logging
    from scilink.agents.exp_agents.pipelines.curve_fitting_pipelines import create_unified_curve_fitting_pipeline
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    sent = []

    class Model:
        def generate_content(self, prompt, *a, **k):
            sent.append(prompt if isinstance(prompt, str) else "\n".join(map(str, prompt)))
            raise RuntimeError("captured")
    pipe = create_unified_curve_fitting_pipeline(
        model=Model(), logger=logging.getLogger("t"), generation_config=None, safety_settings=None,
        parse_fn=lambda *a, **k: None, store_fn=lambda *a, **k: None, plot_fn=lambda *a, **k: None,
        executor=None, output_dir=str(tmp_path), load_skills_fn=lambda *a, **k: None)
    ctrl = next(c for c in pipe if isinstance(c, UnifiedSeriesProcessingController))
    with pytest.raises(Exception):
        ctrl._correct_script({"locked_fitting_config": {"physical_model": "a joint model"}}, "print(1)", "boom")
    assert sent and CODEGEN_PRINCIPLE in sent[0]


def test_the_verifiers_carry_the_principle():
    """All three verifiers whose rejection drives a retry: curve and image
    append it beside the tool-scrutiny principle, hyperspectral fills it into
    the same slot."""
    import inspect
    from scilink.agents.exp_agents.controllers import (curve_fitting_controllers as C,
                                                       image_analysis_controllers as IC,
                                                       hyperspectral_controllers as H)
    assert "VERIFIER_PRINCIPLE" in inspect.getsource(C.UnifiedSeriesProcessingController._verify_fit_with_llm)
    assert "VERIFIER_PRINCIPLE" in inspect.getsource(IC.UnifiedImageProcessingController._verify_quality)
    assert H.VERIFIER_PRINCIPLE == VERIFIER_PRINCIPLE


def test_a_joint_runs_checks_and_repairs_are_told_its_inputs_are_the_contract(tmp_path):
    """Live: the joint script read every unit, the conformance check called
    the extra files a breach of the data-loading rules, and the correction
    cut it back to one unit. The conformance and correction prompts actually
    sent carry the joint contract; a per-unit run's prompts are unchanged."""
    import logging
    from scilink.agents.exp_agents.pipelines.curve_fitting_pipelines import create_unified_curve_fitting_pipeline
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import UnifiedSeriesProcessingController
    sent = []

    class Model:
        def generate_content(self, *a, **k):
            c = k.get("contents", a[0] if a else None)
            sent.append(c if isinstance(c, str) else "\n".join(map(str, c)))
            raise RuntimeError("captured")
    pipe = create_unified_curve_fitting_pipeline(
        model=Model(), logger=logging.getLogger("t"), generation_config=None, safety_settings=None,
        parse_fn=lambda *a, **k: None, store_fn=lambda *a, **k: None, plot_fn=lambda *a, **k: None,
        executor=None, output_dir=str(tmp_path), load_skills_fn=lambda *a, **k: None)
    ctrl = next(c for c in pipe if isinstance(c, UnifiedSeriesProcessingController))
    units = [{"path": str(tmp_path / f"u{k}.csv"), "control_value": k} for k in range(3)]
    manifest = _joint.stage_joint_units(units, tmp_path / "joint", to_array=lambda s: np.zeros((2, 5)))
    cfg = {"physical_model": "a joint model", "analysis_approach": "joint"}
    for joint in (True, False):
        sent.clear()
        state = {"locked_fitting_config": cfg, **({"joint_manifest": str(manifest)} if joint else {})}
        ctrl._check_plan_conformance(state, "print(1)")
        with pytest.raises(Exception):
            ctrl._correct_script(state, "print(1)", "boom")
        assert len(sent) == 2
        for prompt in sent:
            assert ("Joint analysis contract" in prompt) is joint
            assert ("joint_unit_0002.npy" in prompt) is joint
