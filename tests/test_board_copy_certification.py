"""#753: a replay of the swarm board's COPY of a recipe is certified (or
flagged) the way a replay of the run is, because the copy carries the run's
certification reference in its sidecar; a copy with none says why.

Through the real paths: post_delegation writes the copy and its sidecar, the
curve reuse (`curve._replay`, the real qc_try_reuse) replays the COPY on new
data, and the hyperspectral reader takes the reference from a single cube's
copy."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_regime_choice as rc                                   # noqa: E402
import test_series_verdict_path as curve                          # noqa: E402

from scilink.agents.exp_agents import _replay                      # noqa: E402
from scilink.agents.exp_agents._verification_record import (       # noqa: E402
    interpretation_checked_by, prior_recipe_candidates, recipe_sidecar)
from scilink.agents.meta_agent import board as board_mod          # noqa: E402
from scilink.agents.meta_agent.board import Board                 # noqa: E402


def _anatase_reference():
    """An anatase regime's reference as the curve driver stamps it: its
    units' curves and what its units found."""
    curves = [tuple(rc.spectrum(rc.ANATASE, seed=s).T) for s in (11, 12, 13)]
    samples = [_replay.identity_features(rc.auto_detect_parameters(rc.ANATASE, seed=s)) for s in (21, 22, 23)]
    x_range = float(curves[0][0].max() - curves[0][0].min())
    return _replay.curve_certification_reference(_replay.drift_state_of_curves(curves), x_range, samples)


def _post_series_copy(tmp_path, ref):
    board = Board(tmp_path / "meta")
    row = {"analysis_id": "s1", "status": "success", "verified": True, "reason": "ok", "series": True,
           "agent_name": "CurveFittingAgent",
           "recipes": [{"regime": "anatase", "unit": "spectrum_0000", "index": 0, "verified": True, "reason": "ok",
                        "script": "LOW", **({"certification_reference": ref} if ref else {})}]}
    board_mod.post_delegation(board, {"index": 2, "label": "series", "mode": "analysis", "status": "success"},
                              {"analyses": [row]})
    rec = next(r for r in board.records() if r["kind"] == "recipe")
    return board, rec, Path(rec["payload"]["path"])


def test_the_copy_carries_the_reference_in_its_sidecar_not_in_the_board_record(tmp_path):
    ref = _anatase_reference()
    board, rec, copy = _post_series_copy(tmp_path, ref)
    assert "certification_reference" not in rec["payload"]                 # a board record stays small
    assert "certification_reference" not in (board.path.read_text())
    side = recipe_sidecar(copy)
    assert side["certification_reference"] == json.loads(json.dumps(ref))
    cands = prior_recipe_candidates(copy.parent, single_name="fitting_script.py", named=copy)
    assert cands[0]["certification_reference"]["kind"] == "curve"


def test_a_replay_of_the_copy_is_certified_on_its_regime_and_flagged_off_it(tmp_path, monkeypatch):
    """Before #753 both replays came back {"checked": false} with no reason:
    the copy had nothing to check against."""
    _, _, copy = _post_series_copy(tmp_path, _anatase_reference())
    res, _, _, _ = curve._replay(tmp_path / "same", monkeypatch, {"LOW": 0.99}, prior=copy,
                                 data=rc.spectrum(rc.ANATASE, seed=31),
                                 extra_params={"LOW": rc.auto_detect_parameters(rc.ANATASE, seed=32)})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["identity"]["checked"] and rv["identity"]["within"]
    assert rv["state_distance"] is not None and interpretation_checked_by(rv) is True
    res, _, _, _ = curve._replay(tmp_path / "other", monkeypatch, {"LOW": 0.99}, prior=copy,
                                 data=rc.spectrum(rc.RUTILE, seed=33),
                                 extra_params={"LOW": rc.auto_detect_parameters(rc.RUTILE, seed=34)})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good"                                          # the gate alone cannot tell
    assert rv.get("state_flag") or rv["identity"].get("flagged")            # the carried reference can
    assert interpretation_checked_by(rv) is False


def test_a_copy_with_no_reference_says_why(tmp_path, monkeypatch):
    _, _, copy = _post_series_copy(tmp_path, None)
    res, _, _, _ = curve._replay(tmp_path / "r", monkeypatch, {"LOW": 0.99}, prior=copy,
                                 data=rc.spectrum(rc.ANATASE, seed=41),
                                 extra_params={"LOW": rc.auto_detect_parameters(rc.ANATASE, seed=42)})
    rv = res["reuse_validity"]
    assert rv["identity"] == {"checked": False, "reason": _replay.NO_REFERENCE}
    assert rv["state_check"] == "skipped: " + _replay.NO_REFERENCE
    assert interpretation_checked_by(rv) is False


def test_a_single_runs_copy_carries_its_reference_curve_and_cube(tmp_path):
    board = Board(tmp_path / "meta")
    ref = _anatase_reference()
    run = tmp_path / "curve_run"
    (run / "scripts").mkdir(parents=True)
    (run / "scripts" / "fitting_script.py").write_text("ONE")
    (run / "analysis_results.json").write_text(json.dumps({"status": "success", "certification_reference": ref}))
    cube = tmp_path / "cube_run"
    cube.mkdir()
    maps = {"kind": "maps", "reference_maps": {"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0,
                                                                 "coverage": 1.0}}}
    (cube / "dynamic_analysis_records.json").write_text(json.dumps([{"target": "t", "task_success": True,
                                                                     "script": "def analyze_feature(d, a): pass"}]))
    (cube / "analysis_results.json").write_text(json.dumps({"status": "success", "certification_reference": maps}))
    for k, (aid, out) in enumerate((("c1", run), ("h1", cube)), 1):
        board_mod.post_delegation(board, {"index": k, "label": f"item {k}", "mode": "analysis", "status": "success"},
                                  {"analyses": [{"analysis_id": aid, "status": "success", "verified": True,
                                                 "reason": "ok", "output_directory": str(out),
                                                 "agent_name": "a"}]})
    recs = {r["payload"]["analysis_id"]: Path(r["payload"]["path"]) for r in board.records() if r["kind"] == "recipe"}
    assert recipe_sidecar(recs["c1"])["certification_reference"]["kind"] == "curve"
    # the hyperspectral reader takes the maps from a single cube's copy (it has no map gate)
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    import logging
    ag = HyperspectralAnalysisAgent.__new__(HyperspectralAnalysisAgent)
    ag.logger = logging.getLogger("t")
    assert ag._recipe_sidecar_reference([str(recs["h1"])]) == maps["reference_maps"]


def test_the_curve_driver_stamps_each_regimes_reference(tmp_path):
    """Where the series is done, each regime's recipe records the reference
    from ITS units: their curves (the state) and their parameters (identity);
    a single run's recipe records one from its one unit."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _restamp_regimes
    rows = []
    for i, (peaks, regime) in enumerate([(rc.ANATASE, "a"), (rc.ANATASE, "a"), (rc.RUTILE, "r"), (rc.RUTILE, "r")]):
        d = tmp_path / f"spectrum_{i:04d}"
        d.mkdir()
        np.save(d / "data.npy", rc.spectrum(peaks, seed=50 + i))
        rows.append({"index": i, "success": True, "regime": regime,
                     "parameters": rc.auto_detect_parameters(peaks, seed=60 + i)})
    recipes = {"a": {"unit": "spectrum_0000"}, "r": {"unit": "spectrum_0002"}}
    _restamp_regimes(tmp_path, recipes, rows)
    for name in ("a", "r"):
        ref = recipes[name]["certification_reference"]
        assert ref["kind"] == "curve" and ref["identity"]["n_units"] == 2 and ref["drift_state"]["seed"]
    single = {"default": {"unit": "spectrum_0000"}}
    _restamp_regimes(tmp_path, single, [{**rows[0], "regime": None}])
    assert single["default"]["certification_reference"]["identity"]["n_units"] == 1
