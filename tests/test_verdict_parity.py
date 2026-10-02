"""Parity for #712 PR A: the verdict an agent STAMPS when its result is
final equals the verdict `main` reconstructed from the result's shapes —
identical `{verified, reason}`, no allow-list — over result shapes built by
the agents' own code: the curve and image series harnesses (real series and
refit controllers, real compile), single runs through the real compile with
the QC engine's record shapes and the real reuse path, a hyperspectral cube
through the real `analyze()` (strict replay, no model) and hyperspectral
series rows through the real row builder.

The reference is `reconstructed_verdict` on the result with the new stamps
removed: the top-level `verdict` everywhere, and the hyperspectral rows'
`unit_verdict` (main had none; the curve and image units were already
stamped in stage 2, so theirs stay).

PR B's allow-list for THIS parity (stamp vs reconstruction) is empty:
- #699 changes when a script is called broken, not how a result is judged;
- #710 changes which regime recipe a reuse tries first; the verdict of the
  result it produces is read as before;
- #711 moves the identity check INTO the replay gate, so a drifted replay is
  ``poor`` on the result itself (``reuse_validity.verdict``) and the stamp and
  the reconstruction read the same thing (the case is here, `reused_drifted`);
  what #711 changes beyond the gate is the CLAIMS' status on the board,
  which is not a verdict — see tests/test_identity_check.py.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_series_verdict_path as curve  # noqa: E402
import test_image_series_verdict_path as image  # noqa: E402
import test_hs_locked_replay as hs  # noqa: E402

from scilink.agents.exp_agents._verification_record import (  # noqa: E402
    analysis_verdict, final_verdict_record, reconstructed_verdict)
from scilink.agents.exp_agents._replay import is_verdict_record  # noqa: E402


def _reference(final: dict, *, hs_rows: bool = False) -> dict:
    """`main`'s reading of the same result: the stamps this PR adds removed."""
    plain = copy.deepcopy(final)
    plain.pop("verdict", None)
    if hs_rows:
        for row in plain.get("individual_results") or []:
            row.pop("unit_verdict", None)
    return reconstructed_verdict(plain)


def _check(final: dict, label: str, *, hs_rows: bool = False):
    assert is_verdict_record(final.get("verdict")), f"{label}: no stamp"
    read = analysis_verdict(final)
    # the two decision fields (#711) ride beside the verdict; the verdict itself is the parity
    assert set(read) == {"verified", "reason", "decided_by", "interpretation_checked"}, label
    stamped = {"verified": read["verified"], "reason": read["reason"]}
    assert stamped == {"verified": final["verdict"]["verified"], "reason": final["verdict"]["reason"]}, label
    assert stamped == _reference(final, hs_rows=hs_rows), label
    assert final["verdict"]["decided_by"] in ("qc_gate", "replay_gate", "recipe", "excluded", "none"), label


# ------------------------------------------------------------- curve series
CURVE_SERIES = {
    "clean": dict(names=curve.NAMES, anchors={"spectrum_0000": curve.OK},
                  follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
    "salvaged_anchor": dict(names=curve.NAMES, anchors={"spectrum_0000": curve.SALVAGED},
                            follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
    "cut_anchor": dict(names=curve.NAMES, anchors={"spectrum_0000": {**curve.OK, "unverified": True, "approved": False}},
                       follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
    "failed_anchor": dict(names=curve.NAMES, anchors={"spectrum_0000": {"r2": 0.0, "approved": False, "script": None, "failed": True}},
                          follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
    "two_regimes": dict(names=[f"spectrum_{i:04d}" for i in range(6)], regimes=[[0, 1, 2], [3, 4, 5]],
                        anchors={"spectrum_0000": curve.OK, "spectrum_0003": {**curve.OK, "script": "M3"}},
                        follower_r2={f"spectrum_{i:04d}": 0.97 for i in range(6)}),
    "two_regimes_launder": dict(names=[f"spectrum_{i:04d}" for i in range(6)], regimes=[[0, 1, 2], [3, 4, 5]],
                                anchors={"spectrum_0000": curve.SALVAGED, "spectrum_0003": {**curve.OK, "script": "M3"}},
                                follower_r2={f"spectrum_{i:04d}": 0.97 for i in range(6)},
                                refits={"spectrum_0000": {"r2": 0.99, "approved": True, "script": "M2"}}),
    "anchor_refit_m1_m2": dict(names=curve.NAMES, anchors={"spectrum_0000": {**curve.OK, "r2": 0.92}},
                               follower_r2={"spectrum_0001": 0.93, "spectrum_0002": 0.97},
                               refits={"spectrum_0000": {"r2": 0.99, "approved": True, "script": "M2"},
                                       "spectrum_0001": {"r2": 0.79, "approved": False, "script": "M2"}}),
    "follower_refit_ok": dict(names=[f"spectrum_{i:04d}" for i in range(4)], anchors={"spectrum_0000": curve.OK},
                              follower_r2={"spectrum_0001": 0.97, "spectrum_0002": None, "spectrum_0003": 0.96},
                              refits={"spectrum_0002": {"r2": 0.98, "approved": True, "script": "M2"}}),
    "follower_refit_salvaged": dict(names=[f"spectrum_{i:04d}" for i in range(4)], anchors={"spectrum_0000": curve.OK},
                                    follower_r2={"spectrum_0001": 0.97, "spectrum_0002": None, "spectrum_0003": 0.96},
                                    refits={"spectrum_0002": {"r2": 0.80, "approved": False, "script": "M2",
                                                              "warning": "R² = 0.8000 below threshold 0.95"}}),
    "good_reuse": dict(names=curve.NAMES, anchors={"spectrum_0000": {**curve.OK, "reused": "good"}},
                       follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
    "failed_reuse_rederived": dict(names=curve.NAMES, anchors={"spectrum_0000": {**curve.SALVAGED, "reused": "failed"}},
                                   follower_r2={"spectrum_0001": 0.97, "spectrum_0002": 0.98}),
}


@pytest.mark.parametrize("name", sorted(CURVE_SERIES))
def test_curve_series_stamp_equals_the_reconstruction(tmp_path, monkeypatch, name):
    spec = dict(CURVE_SERIES[name])
    state, _ = curve.run_series(tmp_path, monkeypatch, **spec)
    final = curve.compile_results(tmp_path, state)
    _check(final, f"curve series {name}")


# ------------------------------------------------------------- curve single
def _single_curve(tmp_path, unit):
    """A single-spectrum result through the real compile, from the QC
    engine's record shapes (``_canned_anchor``) or a real reuse result."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    import logging
    agent = object.__new__(CurveFittingAgent)
    agent.output_dir = Path(tmp_path)
    agent.logger = logging.getLogger("parity.single")
    agent._validate_scientific_claims = lambda claims: claims
    agent._maybe_stage_t2_solutions = lambda state: []
    agent._bank_series_scripts = lambda state: None
    state = {"is_single_spectrum": True, "num_spectra": 1, "series_results": [unit],
             "fit_results": {"model_type": unit.get("model_type"), "parameters": unit.get("fitted_parameters") or {},
                             "fit_quality": unit.get("fit_quality") or {}},
             "synthesis_result": {"detailed_analysis": "d", "scientific_claims": [{"claim": "c"}]},
             "task_mode": "fitting", "spectrum_names": ["spectrum_0000"]}
    agent._save_fitting_scripts(state)
    return agent._compile_results(state)


CURVE_SINGLE = {
    "approved": dict(r2=0.98, approved=True, script="M1"),
    "soft_band_approved": dict(r2=0.92, approved=True, script="M1"),
    "salvaged": dict(r2=0.80, approved=False, script="M1", warning="R² = 0.8000 below threshold 0.95"),
    "judge_picked": dict(r2=0.90, approved=True, script="M1", judge_warning="Judge selected this as best available"),
    "budget_cut": dict(r2=0.98, approved=False, script="M1", unverified=True),
    "failed": dict(r2=0.0, approved=False, script=None, failed=True),
    "reused_good": dict(r2=0.98, approved=True, script="M1", reused="good"),
    "reused_poor": dict(r2=0.80, approved=False, script="M1", reused="poor"),
}


@pytest.mark.parametrize("name", sorted(CURVE_SINGLE))
def test_curve_single_run_stamp_equals_the_reconstruction(tmp_path, name):
    unit = curve._canned_anchor("spectrum_0000", 0, **CURVE_SINGLE[name])
    final = _single_curve(tmp_path, unit)
    _check(final, f"curve single {name}")


def test_curve_single_run_through_the_real_reuse_path(tmp_path, monkeypatch):
    """The reuse verdict the gate writes (good / poor / failed → re-derive)
    reaches the compile and the stamp through the real qc_try_reuse — and a
    replay whose identity drifted beyond the anchor's spread (#711) is poor
    on the result itself, so stamp and reconstruction agree on it too."""
    for label, scripts in (("good", {"LOW": 0.80, "HIGH": 0.985}), ("poor", {"LOW": 0.70, "HIGH": 0.75})):
        res, ex, _, item = curve._replay(tmp_path / label, monkeypatch, scripts)
        unit = {**res, "name": "spectrum_0000", "index": 0, "success": True}
        final = _single_curve(tmp_path / label / "compiled", unit)
        assert final["reuse_validity"]["verdict"] == label
        _check(final, f"curve single reuse {label}")
    import test_regime_choice as rc
    prior = rc.prior_two_regime_run(tmp_path / "drift", names=True)
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    res, ex, _, _ = curve._replay(tmp_path / "reused_drifted", monkeypatch, {"LOW": 0.999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=2),
                                  extra_params={"LOW": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=2)})
    final = _single_curve(tmp_path / "reused_drifted" / "compiled", {**res, "name": "spectrum_0000", "index": 0, "success": True})
    assert final["reuse_validity"]["verdict"] == "poor" and not final["verdict"]["verified"]
    _check(final, "curve single reuse drifted (identity)")


# ------------------------------------------------------------- image series
IMAGE_SERIES = {
    "clean": dict(names=image.NAMES, anchors={"image_0000": image.OK}, follower_score={"image_0001": 0.88, "image_0002": 0.9}),
    "salvaged_anchor": dict(names=image.NAMES, anchors={"image_0000": image.SALVAGED}, follower_score={"image_0001": 0.88, "image_0002": 0.9}),
    "failed_anchor": dict(names=image.NAMES, anchors={"image_0000": {"score": 0.0, "approved": False, "script": None, "failed": True}},
                          follower_score={"image_0001": 0.9, "image_0002": 0.9}),
    "two_regimes": dict(names=[f"image_{i:04d}" for i in range(6)], regimes=[[0, 1, 2], [3, 4, 5]],
                        anchors={"image_0000": image.OK, "image_0003": {**image.OK, "script": "M3"}},
                        follower_score={f"image_{i:04d}": 0.9 for i in range(6)}),
    "follower_refit_ok": dict(names=[f"image_{i:04d}" for i in range(4)], anchors={"image_0000": image.OK},
                              follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                              refits={"image_0002": {"score": 0.85, "approved": True, "script": "M2"}}),
    "follower_refit_salvaged": dict(names=[f"image_{i:04d}" for i in range(4)], anchors={"image_0000": image.OK},
                                    follower_score={"image_0001": 0.9, "image_0002": None, "image_0003": 0.88},
                                    refits={"image_0002": {"score": 0.5, "approved": False, "script": "M2", "warning": "below"}}),
}


@pytest.mark.parametrize("name", sorted(IMAGE_SERIES))
def test_image_series_stamp_equals_the_reconstruction(tmp_path, monkeypatch, name):
    state, _ = image.run_series(tmp_path, monkeypatch, **dict(IMAGE_SERIES[name]))
    final = image.compile_results(tmp_path, state)
    _check(final, f"image series {name}")


# --------------------------------------------------------- hyperspectral
def test_hyperspectral_cube_stamp_through_the_real_analyze(tmp_path, monkeypatch):
    """The whole analyze() of a strict replay (no model call), success and
    failure: the stamp is placed after the status is decided."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    np.save(tmp_path / "cube.npy", hs._peak_cube(center=660.0, seed=1))
    ok = hs._strict_agent(tmp_path, "frame").analyze(
        str(tmp_path / "cube.npy"), system_info=dict(hs.AXIS_OK),
        prior_analysis_paths=[str(hs._anchor_dir(tmp_path))], reuse_locked_script=True, strict_replay=True,
        replay_reference={"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0, "coverage": 1.0}})
    assert ok["status"] == "success"
    _check(ok, "hs cube strict replay (good)")
    assert ok["verdict"]["verified"] and ok["verdict"]["decided_by"] == "replay_gate"
    # the saved file carries the same stamp
    saved = json.loads((tmp_path / "frame" / "analysis_results.json").read_text())
    assert saved.get("verdict", {}).get("verified") is True
    broken = hs.PEAK_SCRIPT.replace("w = data", "w = undefined_name + data")
    (tmp_path / "b").mkdir()
    bad = hs._strict_agent(tmp_path, "frame2").analyze(
        str(tmp_path / "cube.npy"), system_info=dict(hs.AXIS_OK),
        prior_analysis_paths=[str(hs._anchor_dir(tmp_path / "b", broken))], reuse_locked_script=True, strict_replay=True)
    assert bad["status"] != "success"
    if is_verdict_record(bad.get("verdict")):
        _check(bad, "hs cube strict replay (failed)")
        assert not bad["verdict"]["verified"]


def test_hyperspectral_series_rows_stamp_equals_the_legacy_row_rule():
    """Rows from the real row builder over cube results of every outcome: the
    stamp on each row aggregates to what main's row rule read."""
    from scilink.agents.exp_agents.controllers.hyperspectral_series import build_series_row
    rec_ok = {"target": "t", "task_success": True, "script": "s", "quality_history": {"approved": True}}
    rec_bad = {"target": "u", "task_success": False, "script": "s"}
    feats = [{"name": "Peak_Position", "stats": {"mean": 1.0}}]
    cubes = {
        "clean": {"status": "success", "extracted_features": feats, "dynamic_analysis_records": [rec_ok]},
        "two_of_three": {"status": "success", "extracted_features": feats, "dynamic_analysis_records": [rec_ok, rec_ok, rec_bad]},
        "partial": {"status": "partial", "extracted_features": feats, "dynamic_analysis_records": [rec_ok]},
        "no_records": {"status": "success", "extracted_features": feats, "dynamic_analysis_records": []},
        "salvaged_only": {"status": "success", "extracted_features": feats, "dynamic_analysis_records": [rec_bad]},
        "failed": {"status": "error", "extracted_features": [], "dynamic_analysis_records": [], "error": "boom"},
        "verbatim_replay": {"status": "success", "extracted_features": feats, "dynamic_analysis_records": [rec_ok],
                            "script_reuse": {"verbatim": True, "n_replayed": 1}},
    }
    rows = {k: build_series_row(i, f"/d/{k}.npy", v, "anchor" if i == 0 else "follower", f"/o/{k}")
            for i, (k, v) in enumerate(cubes.items())}
    for k, row in rows.items():
        assert is_verdict_record(row["unit_verdict"]), k
    assert rows["verbatim_replay"]["unit_verdict"]["decided_by"] == "replay_gate"
    # every combination the series driver can hand to a reader, assembled by the
    # agent's own _compile_series_results (the response's rows and its stamp)
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    import itertools
    agent = object.__new__(HyperspectralAnalysisAgent)
    agent.output_dir = Path("/o")
    agent._validate_scientific_claims = lambda claims: claims
    agent._scout_summary = lambda scout: None
    names = list(rows)
    for k in range(1, 4):
        for combo in itertools.combinations(names, k):
            state = {"series_results": [copy.deepcopy(rows[n]) for n in combo],
                     "synthesis_result": {"detailed_analysis": "d", "scientific_claims": []}}
            final = agent._compile_series_results(state, None)
            assert all(is_verdict_record(r.get("unit_verdict")) for r in final["individual_results"]), combo
            _check(final, f"hs series rows {combo}", hs_rows=True)
            # the response's own stamp says who decided; a series that is not "success" has none to decide
            assert final["verdict"]["decided_by"] == ("qc_gate" if final["status"] == "success" else "none"), combo
