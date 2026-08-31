"""Offline tests: hyperspectral locked-script replay (harmonized re-run).

The gap this closes (found live): the fusion stage can PRESCRIBE "re-run the
sibling cubes with a single harmonized pipeline" but nothing could EXECUTE
it — every branch regenerated its own script with its own segmentation and
continuum choices, confounding cross-dataset magnitude comparisons.

Now `analyze(prior_analysis_paths=[...], reuse_locked_script=True)` replays a
prior run's APPROVED dynamic-analysis script(s) verbatim: no planning LLM
call, no codegen, decomposition bypassed, retry budget forced to 0 so a
failure can never be silently regenerated into a different method. Per-map
QC still verifies the outputs on the new dataset.

  conda run -n scilink python -m pytest tests/test_hs_locked_replay.py -q
"""
import json
import logging
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
# Agent construction probes for an execution sandbox; none exists in CI.
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc

LOGGER = logging.getLogger("test.locked_replay")

SCRIPT = '''
def analyze_feature(data, axis):
    m = data.mean(axis=2)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d"}
'''

AXIS_OK = {
    "axis_spec": {
        "axis_2": {"name": "wavelength", "units": "nm", "start": 400, "end": 900},
    }
}


class _ExplodingModel:
    """Any LLM call in replay mode is a regression — fail loudly."""

    def generate_content(self, *a, **k):
        raise AssertionError("LLM was called during locked-script replay")


def _records(script=SCRIPT, approved=True):
    return [{"target": "mean map (donor)", "task_success": approved,
             "required_outputs": ["Mean_Map"], "script": script,
             "quality_history": {"approved": approved}}]


def _replay_state(tmp_path, records=None):
    return {
        "hspy_data": np.random.rand(5, 5, 8),
        "original_hspy_data": np.random.rand(5, 5, 8),
        "system_info": dict(AXIS_OK),
        "energy_axis": np.linspace(400, 900, 8),
        "settings": {"output_dir": str(tmp_path)},
        "reuse_records": records if records is not None else _records(),
        "max_verification_iterations": 0,
        "iteration_title": "T",
        "analysis_objective": "obj",
    }


# ---------------------------------------------------------------------------
# Plan short-circuit: no planning LLM call, targets carry the scripts
# ---------------------------------------------------------------------------

def test_select_refinement_short_circuits_without_llm(tmp_path):
    ctrl = hc.SelectRefinementTargetController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    state = ctrl.execute(_replay_state(tmp_path))
    dec = state["refinement_decision"]
    assert dec["requires_custom_code"] is True
    assert len(dec["targets"]) == 1
    t = dec["targets"][0]
    assert t["type"] == "custom_code"
    assert t["supplied_script"] == SCRIPT
    assert t["required_outputs"] == ["Mean_Map"]


def test_decomposition_bypassed_without_llm(tmp_path):
    ctrl = hc.DecompositionController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, settings={"output_dir": str(tmp_path)},
        preprocessor=None, parse_fn=lambda r: ({}, None))
    state = ctrl.execute(_replay_state(tmp_path))
    assert state["skip_decomposition"] is True
    assert state["preprocessing_mask"].shape == (5, 5)


# ---------------------------------------------------------------------------
# Execution: supplied script runs verbatim, zero generation calls
# ---------------------------------------------------------------------------

def _run_dynamic(tmp_path, records, monkeypatch, model=None):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    plan = hc.SelectRefinementTargetController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    state = plan.execute(_replay_state(tmp_path, records))
    ctrl = hc.RunDynamicAnalysisController(
        model or _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    return ctrl.execute(state)


def test_supplied_script_runs_verbatim_zero_llm_calls(tmp_path, monkeypatch):
    state = _run_dynamic(tmp_path, _records(), monkeypatch)
    names = [m["name"] for m in state.get("custom_analysis_metadata_list") or []]
    assert names == ["Mean_Map"]
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    assert rec["locked_replay"] is True
    assert rec["replay_verbatim"] is True
    assert rec["script"] == SCRIPT          # byte-identical to the donor's


def test_repaired_replay_flagged_not_verbatim(tmp_path, monkeypatch):
    """A broken supplied script may be mechanically repaired, but the record
    must say the run is no longer byte-comparable to the donor."""
    broken = SCRIPT.replace("axis=2)", "axis=2")   # syntax error
    calls = []

    class _RepairModel:
        def generate_content(self, contents, **kw):
            calls.append(str(contents))
            return json.dumps({"code": SCRIPT})

    state = _run_dynamic(tmp_path, _records(script=broken), monkeypatch,
                         model=_RepairModel())
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    assert rec["locked_replay"] is True
    assert rec["replay_verbatim"] is False
    assert len(calls) == 1 and "MECHANICAL CORRECTION" in calls[0]


# ---------------------------------------------------------------------------
# analyze() entry contract (no LLM reached — errors happen before any call)
# ---------------------------------------------------------------------------

def _agent():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    return HyperspectralAnalysisAgent(api_key="sk-dummy",
                                      model_name="claude-opus-4-6")


def test_reuse_without_paths_is_refused(tmp_path):
    ag = _agent()
    np.save(tmp_path / "cube.npy", np.random.rand(4, 4, 6))
    res = ag.analyze(str(tmp_path / "cube.npy"), system_info=dict(AXIS_OK),
                     reuse_locked_script=True)
    assert res["status"] == "error"
    assert "prior_analysis_paths" in res["error"]["error"]


def test_reuse_with_no_approved_records_is_refused(tmp_path):
    ag = _agent()
    np.save(tmp_path / "cube.npy", np.random.rand(4, 4, 6))
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "dynamic_analysis_records.json").write_text(
        json.dumps(_records(approved=False)))
    res = ag.analyze(str(tmp_path / "cube.npy"), system_info=dict(AXIS_OK),
                     prior_analysis_paths=[str(prior)],
                     reuse_locked_script=True)
    assert res["status"] == "error"
    assert "No approved prior script" in res["error"]["error"]


def test_loader_finds_records_in_nested_result_dirs(tmp_path):
    ag = _agent()
    nested = tmp_path / "results" / "analysis_x"
    nested.mkdir(parents=True)
    (nested / "dynamic_analysis_records.json").write_text(
        json.dumps(_records() + _records(approved=False)))
    recs = ag._load_prior_dynamic_records([str(tmp_path)])
    assert len(recs) == 1 and recs[0]["script"] == SCRIPT
    # direct file path also works
    recs2 = ag._load_prior_dynamic_records(
        [str(nested / "dynamic_analysis_records.json")])
    assert len(recs2) == 1


# ---------------------------------------------------------------------------
# #518: a harmonized replay must reproduce the donor's fit-mask SCOPING
# (re-derived on the follower cube from the record's persisted recipe), or
# be explicitly flagged degraded — never silently full-frame.
# ---------------------------------------------------------------------------

E518 = 6
SPEC_LOCAL = [0.0, 0.0, 10.0, 10.0, 0.0, 0.0]   # localized component endmember


def _mask_cube(h=24, w=24):
    """Flat background with a bright 3x3 corner region carrying SPEC_LOCAL —
    projection onto SPEC_LOCAL separates region from background at half-max."""
    cube = np.ones((h, w, E518))
    cube[:3, :3, :] = np.array(SPEC_LOCAL) * 5.0
    return cube


def _recipe(spec=SPEC_LOCAL, idx=1):
    return {"fit_scope": "component_mask", "mask_component_index": idx,
            "component_spectrum": list(spec), "mask_fraction": 0.05}


MASKED_SCRIPT = '''
def analyze_feature(data, axis, fit_mask=None):
    assert fit_mask is not None, "fit_mask missing on a mask-scoped replay"
    assert 0 < fit_mask.sum() < fit_mask.size, "mask does not scope the frame"
    m = data.mean(axis=2) * (1.0 * fit_mask)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d"}
'''

DECLARES_ONLY_SCRIPT = '''
def analyze_feature(data, axis, fit_mask=None):
    m = data.mean(axis=2)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d"}
'''


def _mask_state(tmp_path, records):
    state = _replay_state(tmp_path, records)
    state["hspy_data"] = _mask_cube()
    state["original_hspy_data"] = _mask_cube()
    state["energy_axis"] = np.linspace(400, 900, E518)
    return state


def test_script_declares_fit_mask_detector():
    assert hc._script_declares_fit_mask(MASKED_SCRIPT) is True
    assert hc._script_declares_fit_mask(DECLARES_ONLY_SCRIPT) is True
    assert hc._script_declares_fit_mask(SCRIPT) is False
    assert hc._script_declares_fit_mask("") is False


def test_rebuild_mask_from_recipe_localizes():
    mask = hc._rebuild_fit_mask_from_recipe(
        _mask_cube(), _recipe(), (24, 24), LOGGER)
    assert mask is not None and mask.dtype == bool
    assert mask[0, 0] and not mask[23, 23]          # region in, far corner out
    assert 0 < mask.sum() < mask.size / 2


def test_rebuild_mask_channel_mismatch_returns_none():
    bad = _recipe(spec=[1.0] * 9)                    # 9 channels vs cube's 6
    assert hc._rebuild_fit_mask_from_recipe(
        _mask_cube(), bad, (24, 24), LOGGER) is None


def test_replay_target_carries_recipe():
    ctrl = hc.SelectRefinementTargetController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    recs = _records(script=MASKED_SCRIPT)
    recs[0]["fit_mask_recipe"] = _recipe()
    state = ctrl.execute(_mask_state("unused", recs))
    t = state["refinement_decision"]["targets"][0]
    assert t["fit_mask_recipe"] == _recipe()
    assert t["fit_scope"] == "component_mask"
    assert t["mask_component_index"] == 1


def test_replay_rederives_mask_and_reproduces_scoping(tmp_path, monkeypatch):
    """Recipe present -> mask re-derived on the follower cube, the supplied
    script actually receives it (it asserts so), record stays clean and
    carries the recipe forward for chained replays."""
    recs = _records(script=MASKED_SCRIPT)
    recs[0]["fit_mask_recipe"] = _recipe()
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    plan = hc.SelectRefinementTargetController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    state = plan.execute(_mask_state(tmp_path, recs))
    ctrl = hc.RunDynamicAnalysisController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    state = ctrl.execute(state)
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    assert rec["replay_verbatim"] is True
    assert "replay_scope_degraded" not in rec
    assert rec["fit_mask_recipe"]["mask_component_index"] == 1
    assert rec["fit_mask_recipe"]["component_spectrum"] == SPEC_LOCAL
    assert 0 < rec["fit_mask_recipe"]["mask_fraction"] < 0.5


def test_replay_without_recipe_flags_degraded(tmp_path, monkeypatch):
    """Pre-#518 donor record (mask-dependent script, no recipe) -> loud
    degraded flag on the record; the script still runs (full-frame)."""
    state = _run_dynamic_masked(tmp_path, _records(script=DECLARES_ONLY_SCRIPT),
                                monkeypatch)
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    assert "NOT reproduced" in rec["replay_scope_degraded"]
    assert "fit_mask_recipe" not in rec


def test_replay_unusable_recipe_flags_degraded(tmp_path, monkeypatch):
    """Recipe present but unusable on this cube (channel mismatch) ->
    degraded flag, full-frame execution."""
    recs = _records(script=DECLARES_ONLY_SCRIPT)
    recs[0]["fit_mask_recipe"] = _recipe(spec=[1.0] * 9)
    state = _run_dynamic_masked(tmp_path, recs, monkeypatch)
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    assert "could not be re-derived" in rec["replay_scope_degraded"]


def _run_dynamic_masked(tmp_path, records, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    plan = hc.SelectRefinementTargetController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    state = plan.execute(_mask_state(tmp_path, records))
    ctrl = hc.RunDynamicAnalysisController(
        _ExplodingModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    return ctrl.execute(state)


def test_donor_record_persists_recipe(tmp_path, monkeypatch):
    """Fresh (non-replay) component-mask target -> the approved record
    persists the cube-independent mask recipe (#518 spot 3)."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    state = _mask_state(tmp_path, None)
    del state["reuse_records"]
    # Donor-side decomposition state: component 1 is the localized region.
    amap = np.zeros((24, 24)); amap[:3, :3] = 1.0
    state["final_abundance_maps"] = np.dstack([amap, np.ones((24, 24)) * 0.1])
    state["final_components"] = np.array([SPEC_LOCAL, [1.0] * E518])
    state["refinement_decision"] = {
        "refinement_needed": True, "requires_custom_code": True,
        "targets": [{"type": "custom_code", "description": "masked mean map",
                     "required_outputs": ["Mean_Map"],
                     "fit_scope": "component_mask",
                     "mask_component_index": 1}],
    }

    class _CodegenModel:
        def generate_content(self, *a, **k):
            return json.dumps({"code": MASKED_SCRIPT})

    ctrl = hc.RunDynamicAnalysisController(
        _CodegenModel(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None))
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    state = ctrl.execute(state)
    rec = (state.get("dynamic_analysis_records") or [])[0]
    assert rec["task_success"] is True
    recipe = rec["fit_mask_recipe"]
    assert recipe["fit_scope"] == "component_mask"
    assert recipe["mask_component_index"] == 1
    assert recipe["component_spectrum"] == SPEC_LOCAL
    assert 0 < recipe["mask_fraction"] < 0.5
    # Round-trip: the persisted record is exactly what a follower replays.
    replayed = json.loads(json.dumps(rec, default=str))
    assert replayed["fit_mask_recipe"]["component_spectrum"] == SPEC_LOCAL


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
