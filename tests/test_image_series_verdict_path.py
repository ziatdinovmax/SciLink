"""The board's verdict on IMAGE series runs, through the real image path —
the twin of test_series_verdict_path.py for the curve agent.

Drives `UnifiedImageProcessingController.execute` (the real
`_process_single_image` for every follower, a fake executor printing the
analysis JSON it is assigned), `_detect_outliers`, `stamp_profile`,
`ImageAdaptiveRefitController.execute`, `_save_analysis_scripts` and
`ImageAnalysisAgent._compile_results`, stubbing only the anchor/refit QC
loop. The image refit touches only `analysis_failed` units (and the
below-threshold ones the detector flags), so the curve's M1/M2 case has no
image twin; the cases that exist on this path are here.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scilink.agents.exp_agents._verification_record import (analysis_verdict, legacy_series_verdict,
                                                             series_anchor_unit, series_recipes)
from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    ImageAdaptiveRefitController, UnifiedImageProcessingController)


def _image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((16, 16)).astype(np.float32)


class FakeExecutor:
    """Prints the analysis JSON assigned to the unit (by working directory)
    and drops a visualization; ``None`` fails the unit's script."""
    timeout = 30

    def __init__(self, score_by_name):
        self.score_by_name = score_by_name
        self.calls = []

    def execute_script(self, script, working_dir=None, timeout=None, **kw):
        wd = Path(working_dir)
        name = next((n for n in self.score_by_name if n in wd.as_posix()), None)
        self.calls.append((name, script))
        score = self.score_by_name.get(name, 0.9)
        if score is None:
            return {"status": "error", "stdout": "", "stderr": "boom", "message": "script failed: boom"}
        (wd / "visualization.png").write_bytes(b"png")
        out = {"analysis_type": f"analysis of {script}", "extracted_features": {},
               "quality_metrics": {"quality_score": score}, "summary": "ok"}
        return {"status": "success", "stdout": "IMAGE_ANALYSIS_RESULTS_JSON:" + json.dumps(out),
                "stderr": "", "message": ""}


def _canned_anchor(name, idx, *, score, approved, script, warning=None, failed=False):
    if failed:
        return {"index": idx, "name": name, "data_path": f"stack_index_{idx}", "success": False,
                "error": "all attempts failed", "extracted_features": {}, "quality_metrics": {},
                "script": None, "script_errors": []}
    res = {"index": idx, "name": name, "data_path": f"stack_index_{idx}", "success": True, "error": None,
           "analysis_type": f"analysis of {script}", "extracted_features": {},
           "quality_metrics": {"quality_score": score}, "summary": "ok", "saved_arrays": {},
           "visualization_path": None, "visualization_bytes": None, "statistics": {},
           "script": script, "script_errors": [],
           "quality_history": {"final_score": score, "threshold": 0.7, "approved": approved,
                               "verification_iterations": [{"score": score, "annealing_level": 0}],
                               "script_errors": [], "judge_reasoning": None,
                               **({"approved_by": "verifier"} if approved else {})}}
    if warning:
        res["quality_warning"] = warning
    return res


def _common(tmp_path, executor):
    return dict(model=MagicMock(), logger=logging.getLogger("image_series_path"), generation_config=None,
                safety_settings=None, parse_fn=lambda r: (json.loads(r.text), None), executor=executor,
                script_instructions="", correction_instructions="", quality_instructions="",
                output_dir=str(tmp_path), image_to_bytes_fn=lambda arr: b"img")


def run_series(tmp_path, monkeypatch, *, names, anchors, follower_score, refits=None, regimes=None):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    executor = FakeExecutor(follower_score)
    ctrl = UnifiedImageProcessingController(**_common(tmp_path, executor))

    def fake_best_of_n(state, image_data, data_path, image_name, image_idx, **kw):
        return _canned_anchor(image_name, image_idx, **anchors[image_name])
    monkeypatch.setattr(ctrl, "_execute_and_verify_best_of_n", fake_best_of_n)
    monkeypatch.setattr(ctrl, "_generate_analysis_script", lambda *a, **k: "FRESH: np.load('data.npy')")
    monkeypatch.setattr(ctrl, "_check_plan_conformance", lambda state, script: None, raising=False)

    state = {"num_images": len(names), "is_single_image": False,
             "image_stack": np.stack([_image(i) for i in range(len(names))]),
             "locked_analysis_config": {"analysis_type": "M1"}, "system_info": {}}
    if regimes:
        state["series_analysis_plan"] = {"regimes": [
            {"name": f"R{k + 1}", "image_indices": idxs} for k, idxs in enumerate(regimes)]}
        state["regime_configs"] = {i: {"analysis_type": "M1"} for idxs in regimes for i in idxs}
    state = ctrl.execute(state)
    refitter = ImageAdaptiveRefitController(**_common(tmp_path, executor))

    def fake_refit(state, image_data, data_path, image_name, image_idx, **kw):
        spec = (refits or {}).get(image_name)
        if spec is None:
            return {"success": False, "error": "no refit canned", "index": image_idx, "name": image_name}
        return _canned_anchor(image_name, image_idx, **spec)
    monkeypatch.setattr(refitter._processing_helper, "_execute_and_verify", fake_refit)
    state = refitter.execute(state)
    return state, executor


def compile_results(tmp_path, state):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = object.__new__(ImageAnalysisAgent)
    agent.output_dir = Path(tmp_path)
    agent.logger = logging.getLogger("image_series_path.agent")
    agent._validate_scientific_claims = lambda claims: claims
    state.setdefault("synthesis_result", {"detailed_analysis": "d", "scientific_claims": [{"claim": "c"}]})
    agent._save_analysis_scripts(state)
    return agent._compile_results(state)


NAMES = ["image_0000", "image_0001", "image_0002"]
OK = {"score": 0.9, "approved": True, "script": "M1"}
SALVAGED = {"score": 0.5, "approved": False, "script": "M1", "warning": "score 0.50 below threshold 0.70"}


def _units(results):
    return [(u["name"], u.get("success"), u.get("role"), u.get("adaptively_refitted", False), u.get("fitted_from"),
             (u.get("unit_verdict") or {}).get("verified"), u.get("regime")) for u in results["individual_results"]]


def test_clean_image_series_verifies(tmp_path, monkeypatch):
    state, ex = run_series(tmp_path, monkeypatch, names=NAMES, anchors={"image_0000": OK},
                           follower_score={"image_0001": 0.88, "image_0002": 0.9})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[0][2] == "anchor" and all(u[4] == "locked_script" for u in units[1:]), units
    assert [n for n, _ in ex.calls] == ["image_0001", "image_0002"]
    v = analysis_verdict(results)
    assert v["verified"], (v, units)
    assert series_anchor_unit(results) == "image_0000"
    assert [(r["regime"], r["unit"], r["verified"], r["script"]) for r in series_recipes(results)] == [
        ("default", "image_0000", True, "M1")]
    assert sorted(p.name for p in (tmp_path / "scripts").glob("*.py")) == [f"{n}.py" for n in NAMES]


def test_salvaged_image_anchor_stays_provisional_through_followers(tmp_path, monkeypatch):
    state, _ = run_series(tmp_path, monkeypatch, names=NAMES, anchors={"image_0000": SALVAGED},
                          follower_score={"image_0001": 0.88, "image_0002": 0.9})
    results = compile_results(tmp_path, state)
    v = analysis_verdict(results)
    assert not v["verified"] and v["reason"].startswith("salvaged best-available result"), v


def test_failed_image_anchor_makes_fresh_code_followers(tmp_path, monkeypatch):
    state, _ = run_series(tmp_path, monkeypatch, names=NAMES,
                          anchors={"image_0000": {"score": 0.0, "approved": False, "script": None, "failed": True}},
                          follower_score={"image_0001": 0.9, "image_0002": 0.9})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert all(u[4] == "fresh_code" for u in units[1:]), units
    v = analysis_verdict(results)
    assert not v["verified"] and "without a locked recipe" in v["reason"], v


def test_two_image_regimes_judge_their_own_recipes(tmp_path, monkeypatch):
    names = [f"image_{i:04d}" for i in range(6)]
    regimes = [[0, 1, 2], [3, 4, 5]]
    # regime 1 salvaged (followers on it), regime 2 clean → blocked on a regime-1 unit
    state, _ = run_series(tmp_path / "a", monkeypatch, names=names, regimes=regimes,
                          anchors={"image_0000": SALVAGED, "image_0003": {**OK, "script": "M3"}},
                          follower_score={n: 0.9 for n in names})
    results = compile_results(tmp_path / "a", state)
    units = _units(results)
    assert [u[6] for u in units] == ["R1", "R1", "R1", "R2", "R2", "R2"], units
    v = analysis_verdict(results)
    assert not v["verified"] and "image_0000" in v["reason"], v
    assert [(r["regime"], r["unit"], r["verified"]) for r in series_recipes(results)] == [
        ("R1", "image_0000", False), ("R2", "image_0003", True)]
    # both clean → verified
    state, _ = run_series(tmp_path / "b", monkeypatch, names=names, regimes=regimes,
                          anchors={"image_0000": OK, "image_0003": {**OK, "script": "M3"}},
                          follower_score={n: 0.9 for n in names})
    assert analysis_verdict(compile_results(tmp_path / "b", state))["verified"]


def test_failed_image_follower_refit_is_judged_by_its_own_gate(tmp_path, monkeypatch):
    names4 = [f"image_{i:04d}" for i in range(4)]
    state, _ = run_series(tmp_path, monkeypatch, names=names4, anchors={"image_0000": OK},
                          follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                          refits={"image_0002": {"score": 0.85, "approved": True, "script": "M2"}})
    results = compile_results(tmp_path, state)
    units = _units(results)
    assert units[2][3] is True and units[2][5] is True, units
    assert analysis_verdict(results)["verified"]
    assert (tmp_path / "scripts" / "image_0002.py").read_text() == "M2"
    assert sorted(p.name for p in (tmp_path / "scripts").glob("*.py")) == [f"{n}.py" for n in names4]
    # a refit that stays salvaged is a salvaged row
    state, _ = run_series(tmp_path / "s", monkeypatch, names=names4, anchors={"image_0000": OK},
                          follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                          refits={"image_0002": {"score": 0.5, "approved": False, "script": "M2", "warning": "below"}})
    v = analysis_verdict(compile_results(tmp_path / "s", state))
    assert not v["verified"] and "image_0002" in v["reason"], v


def test_image_parity_of_stamped_and_legacy_verdicts(tmp_path, monkeypatch):
    names6 = [f"image_{i:04d}" for i in range(6)]
    names4 = [f"image_{i:04d}" for i in range(4)]
    regimes = [[0, 1, 2], [3, 4, 5]]
    S = {
        "clean": dict(names=NAMES, anchors={"image_0000": OK}, follower_score={"image_0001": 0.88, "image_0002": 0.9}),
        "salvaged_anchor": dict(names=NAMES, anchors={"image_0000": SALVAGED}, follower_score={"image_0001": 0.88, "image_0002": 0.9}),
        "failed_anchor": dict(names=NAMES, anchors={"image_0000": {"score": 0.0, "approved": False, "script": None, "failed": True}},
                              follower_score={"image_0001": 0.9, "image_0002": 0.9}),
        "two_regimes_r1_salvaged": dict(names=names6, regimes=regimes, anchors={"image_0000": SALVAGED, "image_0003": {**OK, "script": "M3"}},
                                        follower_score={n: 0.9 for n in names6}),
        "two_regimes_clean": dict(names=names6, regimes=regimes, anchors={"image_0000": OK, "image_0003": {**OK, "script": "M3"}},
                                  follower_score={n: 0.9 for n in names6}),
        "failed_follower_refit_ok": dict(names=names4, anchors={"image_0000": OK},
                                         follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                                         refits={"image_0002": {"score": 0.85, "approved": True, "script": "M2"}}),
        "failed_follower_refit_salvaged": dict(names=names4, anchors={"image_0000": OK},
                                               follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                                               refits={"image_0002": {"score": 0.5, "approved": False, "script": "M2", "warning": "below"}}),
    }
    #: The one intended difference: the legacy rule held a FOLLOWER refit only
    #: to "finished, not unverified" (a round-3 relaxation); the stamp judges
    #: every refit by its own gate, so a refit that stayed salvaged is a
    #: salvaged row in the table — the same reasoning as for a salvaged
    #: anchor refit.
    allowed = {"failed_follower_refit_salvaged"}
    differ = {}
    for i, (name, kw) in enumerate(S.items()):
        state, _ = run_series(tmp_path / f"p{i}", monkeypatch, **kw)
        results = compile_results(tmp_path / f"p{i}", state)
        assert all(isinstance(u.get("unit_verdict"), dict) for u in results["individual_results"] if u.get("success")), name
        st, lg = analysis_verdict(results), legacy_series_verdict(results)
        if st["verified"] != lg["verified"]:
            differ[name] = (st, lg)
    assert set(differ) <= allowed, differ
    assert differ["failed_follower_refit_salvaged"][0]["verified"] is False      # the stricter answer wins
