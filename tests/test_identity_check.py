"""#711: a passing replay gate verifies the NUMBERS, not WHAT was measured.

Two checks, both inside ``select_recipe``'s judge so a candidate that fails
them falls through to the next regime's recipe: is the new data the regime's
STATE (``DriftMonitor`` seeded with the regime's own units' data — model-
free, the live loop's change signal), and did the recipe find what the
regime's units found (``identity_check``: the names it assigned, normalised
and without database ids; its STRONG positions matched to the units' by
nearest neighbour, never by index, with a floor). Beyond the regime a replay
is ``poor`` (not verified); against one reference unit a difference is a
flag; the record says ``interpretation_checked`` only when both checks ran
against the regime's units and passed — never for a run its own verifier
approved (the verifier reviews the fit, not the claims) nor for a follower
verified by its recipe. The swarm board posts a replay's claims as verified
only then; its recipe (a script that ran) stays verified by the gate. The
behaviour this commit changes is exactly: a replay that fits but is not the
regime's state or found a different thing (good → poor), and a replay's
CLAIMS on the board (verified → provisional unless the interpretation was
checked).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_series_verdict_path as curve  # noqa: E402
import test_regime_choice as rc  # noqa: E402

from scilink.agents.exp_agents import _replay  # noqa: E402
from scilink.agents.exp_agents._verification_record import (  # noqa: E402
    analysis_verdict, final_verdict_record, unit_verdict_for)
from scilink.agents.meta_agent import board as board_mod  # noqa: E402


def test_identity_features_and_reference_read_what_a_recipe_found():
    # nested components: positions with strengths; count is not identity; amplitudes and widths are not positions
    f = _replay.identity_features({"peak_1": {"center": 144.0, "amplitude": 1.0, "fwhm": 12.0}, "baseline": 0.1,
                                   "peak_2": {"position": 610.0, "height": 0.2}, "peak_3": {"center": 300.0, "amplitude": 0.02}})
    assert f["positions"] == [(144.0, 1.0), (610.0, 0.2), (300.0, 0.02)] and f["names"] == {}
    assert _replay.strong_positions(f) == [144.0, 610.0]                   # the 0.02 one is noise-level (< 10 % of the strongest)
    # flat layouts: peak1_center / peak1_amplitude form a component; peak_height / activation_energy are not positions
    f = _replay.identity_features({"peak1_center": 25.3, "peak1_amplitude": 1.0, "peak2_center": 48.0, "peak2_amplitude": 0.3,
                                   "peak_height": 5.0, "activation_energy": 0.9, "mu_shift": 0.0})
    assert sorted(f["positions"]) == [(25.3, 1.0), (48.0, 0.3)]
    # names: normalised, formatting variants equal, database ids ignored
    xrd = {"identified_phase": "O2 Ti", "space_group": "I 41/a m d :2", "database_id": "2310710", "figure_of_merit": 0.69,
           "fitted_zero_shift": 0.0, "fitted_lattice_scale": 1.022, "strongest_peak_2theta": 25.31}
    fx = _replay.identity_features(xrd)
    assert fx["names"] == {"identified_phase": "o2ti", "space_group": "i41amd2"} and "database_id" not in fx["names"]
    assert _replay.identity_features({"space_group": "I41/amd"})["names"]["space_group"] == "i41amd"
    ref = _replay.identity_reference([fx, {**fx, "names": {"identified_phase": "o2ti", "space_group": "i41amd"}}], x_range=50)
    assert ref["names"]["space_group"] == {"values": ["i41amd", "i41amd2"], "n": 2} and ref["floor"] == 0.5
    # a near-zero position gets the floor, not a near-zero tolerance
    near = _replay.identity_reference([{"names": {}, "positions": [(0.01, 1.0)]}, {"names": {}, "positions": [(0.02, 1.0)]}], x_range=10)
    chk = _replay.identity_check({"names": {}, "positions": [(0.08, 1.0)]}, near)
    assert chk["within"]                                                   # within the 0.1 floor


def test_the_reviewers_auto_detect_cases():
    """Anchors that found 5, 5, 6, 6 and 9 peaks from noise: the same phase
    with other noise peaks is within; the other phase is not; an extra
    impurity peak below the first does not shift anything."""
    units = [rc.auto_detect_parameters(rc.ANATASE, shift=0.3 * i, seed=i, n_noise=n) for i, n in enumerate((5, 5, 6, 6, 9))]
    ref = _replay.identity_reference([_replay.identity_features(u) for u in units], x_range=700)
    assert ref["n_units"] == 5 and [c["n_units"] for c in ref["clusters"] if c["n_units"] == 5].__len__() == 4   # the four anatase bands
    same = _replay.identity_features(rc.auto_detect_parameters(rc.ANATASE, shift=0.5, seed=77, n_noise=8))
    assert _replay.identity_check(same, ref)["within"]
    impurity = {**rc.auto_detect_parameters(rc.ANATASE, shift=0.5, seed=78, n_noise=3), "peak_0": {"center": 120.0, "amplitude": 0.04}}
    assert _replay.identity_check(_replay.identity_features(impurity), ref)["within"]        # weak: not identity
    other = _replay.identity_features(rc.auto_detect_parameters(rc.RUTILE, seed=79, n_noise=6))
    chk = _replay.identity_check(other, ref)
    assert not chk["within"] and chk["spread_known"] and any(d.get("missing") for d in chk["drifted"]) \
        and any(d.get("value") is not None and d["name"] == "position" for d in chk["drifted"])
    # a NEW strong feature is drift; a strong feature gone is drift
    extra = {**rc.auto_detect_parameters(rc.ANATASE, seed=80), "peak_9": {"center": 450.0, "amplitude": 0.8}}
    assert not _replay.identity_check(_replay.identity_features(extra), ref)["within"]
    gone = {k: v for k, v in rc.auto_detect_parameters(rc.ANATASE, seed=81).items() if abs(v["center"] - 516.0) > 5}
    assert not _replay.identity_check(_replay.identity_features(gone), ref)["within"]
    # one reference unit: a spread is not known
    one = _replay.identity_reference([_replay.identity_features(units[0])], x_range=700)
    assert not _replay.identity_check(other, one)["spread_known"] and not _replay.identity_check(other, one)["within"]


def test_a_replay_is_held_to_the_regimes_state_and_to_what_it_found(tmp_path, monkeypatch):
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    # HIGH data, HIGH recipe finding the rutile set: verified, interpretation checked
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=1),
                                  extra_params={"HIGH": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=90, n_noise=6),
                                                         "space_group": "P42/mnm"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["state_distance"] < _replay.SAME_STATE_BAR
    assert rv["identity"]["within"] and rv["identity"]["spread_known"] and rv["identity"]["compared"] == 2
    uv = unit_verdict_for({**res, "success": True})
    assert uv["verified"] and uv["decided_by"] == "replay_gate" and uv["interpretation_checked"] is True
    # the anatase recipe FORCED (named file) on rutile data: not the LOW regime's state → poor, falls nowhere,
    # not verified — however well it "fits" and whatever it identified
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=2),
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=91), "space_group": "P 42/m n m"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "poor" and rv["state_distance"] > _replay.SAME_STATE_BAR and "STATE:" in rv["message"]
    assert rv["identity"]["drifted"] and "IDENTITY:" in rv["message"] and res.get("quality_warning")
    uv = unit_verdict_for({**res, "success": True})
    assert not uv["verified"] and uv["interpretation_checked"] is False
    # the data IS the regime's state but the recipe found a different thing (a wrong phase label): poor
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=3),
                                  extra_params={"HIGH": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=92),
                                                         "space_group": "F m -3 m"},
                                                "LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=93), "space_group": "P42/mnm"}})
    rv = res["reuse_validity"]
    assert [s for s, _ in ex.calls] == ["HIGH", "LOW"]                   # HIGH judged poor on identity, LOW tried
    assert rv["verdict"] == "poor" and rv["regime_choice"]["chosen_regime"] == "high"      # the kept: first executed
    assert rv["identity"]["drifted"][0]["name"] == "space_group" and "space_group = 'fm3m'" in rv["message"]
    # the board's copy, a recipe file with no run behind it: nothing to compare, unchecked
    copy = tmp_path / "board" / "recipe.py"
    copy.parent.mkdir()
    copy.write_text("ONE")
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"ONE": 0.99}, prior=copy, data=rc.spectrum(rc.ANATASE),
                                  extra_params={"ONE": rc.auto_detect_parameters(rc.ANATASE, seed=94)})
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["identity"]["checked"] is False
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
    # a single-run prior (one unit, no spread): a different finding is a flag, the verdict stays
    single = tmp_path / "single"
    (single / "scripts").mkdir(parents=True)
    (single / "spectrum_0000").mkdir()
    np.save(single / "spectrum_0000" / "data.npy", rc.spectrum(rc.ANATASE, seed=5))
    (single / "scripts" / "fitting_script.py").write_text("ONE")
    (single / "series_fit_results.json").write_text(json.dumps({"results": [
        {"index": 0, "name": "s", "success": True, "parameters": rc.auto_detect_parameters(rc.ANATASE, seed=6)}]}))
    res, ex, _, _ = curve._replay(tmp_path / "f", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, seed=7),
                                  extra_params={"ONE": {**rc.auto_detect_parameters(rc.ANATASE, seed=95), "peak_9": {"center": 450.0, "amplitude": 0.9}}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["identity"].get("flagged") is True and not rv["identity"]["spread_known"]
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is False


def test_the_record_and_the_board_tell_numbers_from_interpretation(tmp_path):
    checked = {"checked": True, "within": True, "spread_known": True, "drifted": [], "compared": 2}
    rv_checked = {"reused": True, "verdict": "good", "source": "p", "r_squared": 0.99, "threshold": 0.95,
                  "identity": checked, "state_distance": 0.02}
    final = {"status": "success", "reuse_validity": rv_checked}
    final["verdict"] = final_verdict_record(final)
    assert final["verdict"]["decided_by"] == "replay_gate" and final["verdict"]["interpretation_checked"] is True
    unchecked = {"status": "success", "reuse_validity": {**rv_checked, "identity": {"checked": False}}}
    unchecked["verdict"] = final_verdict_record(unchecked)
    assert unchecked["verdict"]["verified"] and unchecked["verdict"]["interpretation_checked"] is False
    # a state check that ran but no identity reference: not checked either (both are needed)
    half = {"status": "success", "reuse_validity": {**rv_checked, "identity": {"checked": False}, "state_distance": 0.01}}
    assert final_verdict_record(half)["interpretation_checked"] is False
    assert analysis_verdict(unchecked) == {"verified": True, "reason": "locked-script reuse passed the replay gate",
                                           "decided_by": "replay_gate", "interpretation_checked": False}
    # a run its own verifier approved: the verifier reviews the fit, NOT the claims — not checked
    own = {"status": "success", "quality_history": {"approved": True, "final_r2": 0.98, "threshold": 0.95, "verification_iterations": [{}]}}
    own["verdict"] = final_verdict_record(own)
    assert own["verdict"]["decided_by"] == "qc_gate" and own["verdict"]["interpretation_checked"] is False
    # a series: followers verified by their recipe are not checked either
    series = {"status": "success", "individual_results": [
        {"success": True, "unit_verdict": {"verified": True, "reason": "ok", "decided_by": "qc_gate", "interpretation_checked": False}},
        {"success": True, "unit_verdict": {"verified": True, "reason": "ok", "decided_by": "recipe", "interpretation_checked": False}}]}
    assert final_verdict_record(series)["interpretation_checked"] is False
    # a cube replay: checked when every required map was held to the anchor's statistics
    cube = {"status": "success", "script_reuse": {"verbatim": True}, "dynamic_analysis_records": [
        {"target": "t", "task_success": True, "script": "s", "locked_replay": True, "identity_checked": True}]}
    cube["verdict"] = final_verdict_record(cube)
    assert cube["verdict"]["decided_by"] == "replay_gate" and cube["verdict"]["interpretation_checked"] is True
    cube["dynamic_analysis_records"][0]["identity_checked"] = False
    assert final_verdict_record(cube)["interpretation_checked"] is False
    # the board: a replay's CLAIM is provisional unless the interpretation was checked; its recipe is verified;
    # a run its own gate approved posts its claims verified, as before (its claims are not a replay's)
    out = tmp_path / "run"
    (out / "scripts").mkdir(parents=True)
    (out / "scripts" / "fitting_script.py").write_text("S")
    entry = {"index": 1, "label": "replay", "mode": "analysis", "status": "success"}
    row = {"analysis_id": "r1", "status": "success", "output_directory": str(out), "agent_name": "CurveFittingAgent"}
    recs = board_mod.records_for(entry, {"key_findings": ["[r1] the low-temperature phase"], "analyses": [{**row, **analysis_verdict(unchecked)}]})
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "provisional"), ("recipe", "verified")]
    assert "interpretation is not verified" in recs[0]["evidence"]["gate"]
    recs = board_mod.records_for(entry, {"key_findings": ["[r1] the high-temperature phase"], "analyses": [{**row, **analysis_verdict(final)}]})
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "verified"), ("recipe", "verified")]
    recs = board_mod.records_for(entry, {"key_findings": ["[r1] anatase"], "analyses": [{**row, **analysis_verdict(own)}]})
    assert recs[0]["status"] == "verified"
