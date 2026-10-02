"""#711: a passing replay gate verifies the NUMBERS, not WHAT was measured.

Two checks, both inside ``select_recipe``'s judge so a candidate that fails
them falls through to the next regime's recipe: is the new data the regime's
STATE (``DriftMonitor`` seeded with the regime's own units' data — model-
free, the live loop's change signal), and did the recipe find what the
regime's units found (``identity_check``: the names it assigned, normalised
and without database ids; its STRONG positions matched to the units' by
nearest neighbour, never by index, with a floor). The checks WITHHOLD
CERTIFICATION and decide nothing: a difference is a flag on the record and
a caveat in the message, the verdict stays the gate's, and the record says
``interpretation_checked`` only when both checks ran against the regime's
units and agreed — never for a run its own verifier approved (the verifier
reviews the fit, not the claims) nor for a follower verified by its recipe
(round 3 of the review: three rounds moved a deciding identity check's
false positives from one threshold to the next; a deterministic check asked
an interpretive question can withhold a certificate, not issue a verdict). The swarm board posts a replay's claims as verified
only then; its recipe (a script that ran) stays verified by the gate. The
behaviour this commit changes is exactly: a replay's CLAIMS on the board
(verified → provisional unless the interpretation was checked); no verdict
changes.
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
    analysis_verdict, final_verdict_record, interpretation_checked_by, unit_verdict_for)
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
    assert fx["names"] == {"identified_phase": "o2 ti", "space_group": "i 41 a m d"} and "database_id" not in fx["names"]
    # names match as one string without spaces (a group under its settings and spellings) or as token sets
    # (a qualifier in parentheses does not erase the phase: "TiO2 (anatase)" against "TiO2 (rutile)" differs)
    m = _replay.names_match
    assert m(_replay._norm_label("I41/amd"), _replay._norm_label("I 41/a m d :2")) and m(_replay._norm_label("R -3 m :H"), "r-3m")
    # a bar is part of its digit's token (P-1 is not P1, R-3c is not R3c); a Greek letter is spelled out (alpha is not gamma)
    assert not m(_replay._norm_label("R -3 m :H"), "r 3 m") and not m(_replay._norm_label("P-1"), "p1") and not m(_replay._norm_label("R-3c"), "r3c")
    assert _replay._norm_label("α-Fe2O3") == "alpha fe2o3" and not m(_replay._norm_label("α-Fe2O3"), _replay._norm_label("γ-Fe2O3"))
    assert m(_replay._norm_label("Pm−3m"), _replay._norm_label("P m -3 m"))                    # a minus sign is a bar
    assert m(_replay._norm_label("anatase (TiO2)"), "anatase") and m("tio2 anatase", "anatase")
    assert not m(_replay._norm_label("TiO2 (anatase)"), _replay._norm_label("TiO2 (rutile)")) and not m("hematite fe2o3", "maghemite fe2o3")
    assert not m("141", "i41amd")                                               # a number vs a symbol: still a table's job
    ref = _replay.identity_reference([fx, {**fx, "names": {"identified_phase": "o2 ti", "space_group": "p42 mnm"}}], x_range=50)
    assert ref["names"]["space_group"] == {"values": ["i 41 a m d", "p42 mnm"], "n": 2} and ref["floor"] == 0.5
    chk = _replay.identity_check({"names": {"space_group": "p 42 m n m"}, "positions": []}, ref)
    assert chk["checked"] and chk["within"]                                     # spelling: the same group
    chk = _replay.identity_check({"names": {"identified_phase": "tio2 rutile"}, "positions": []},
                                 _replay.identity_reference([{"names": {"identified_phase": "tio2 anatase"}, "positions": []}] * 3, x_range=50))
    assert chk["checked"] and not chk["within"]                                 # the phase, not the formula
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
    # a flat XPS / EPR layout the reader finds no identity in: not checked, and the record says why
    flat = _replay.identity_features({"C1s_binding_energy": 284.8, "resonance_field": 3350.0, "g_factor": 2.003})
    chk = _replay.identity_check(flat, ref)
    assert chk["checked"] is False and "no names and no strong positions" in chk["reason"]
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
    # the anatase recipe FORCED (named file) on rutile data fits at R² 0.99999: the gate's verdict stands (good,
    # the numbers), both checks flag it, and it is NOT certified — its claims are provisional on the board
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.99999}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=2),
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=91), "space_group": "P 42/m n m"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and not res.get("quality_warning")
    assert rv["identity"]["drifted"] and rv["identity"]["spread_known"] and rv["identity"]["flagged"] is True
    assert "IDENTITY:" in rv["message"] and "flagged" in rv["message"]
    assert rv["state_distance"] > _replay.SAME_STATE_BAR and rv["state_flag"] is True and "STATE:" in rv["message"]
    uv = unit_verdict_for({**res, "success": True})
    assert uv["verified"] and uv["decided_by"] == "replay_gate" and uv["interpretation_checked"] is False
    # the data IS the regime's state but the recipe found a different thing (a wrong phase label): the gate
    # decides (good), the first candidate is KEPT (no fall-through on a check), the name drift is flagged
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=3),
                                  extra_params={"HIGH": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=92),
                                                         "space_group": "F m -3 m"},
                                                "LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=93), "space_group": "P42/mnm"}})
    rv = res["reuse_validity"]
    assert [s for s, _ in ex.calls] == ["HIGH"] and rv["verdict"] == "good" and rv["regime_choice"]["chosen_regime"] == "high"
    assert rv["identity"]["drifted"][0]["name"] == "space_group" and "space_group = 'f m -3 m'" in rv["message"]
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
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
    fresh = {"status": "success", "dynamic_analysis_records": [{"target": "t", "task_success": True, "script": "s",
                                                                 "quality_history": {"approved": True}}]}
    assert final_verdict_record(fresh)["decided_by"] == "qc_gate" and final_verdict_record(fresh)["interpretation_checked"] is False
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
    assert "interpretation is not certified" in recs[0]["evidence"]["gate"]
    recs = board_mod.records_for(entry, {"key_findings": ["[r1] the high-temperature phase"], "analyses": [{**row, **analysis_verdict(final)}]})
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "verified"), ("recipe", "verified")]
    recs = board_mod.records_for(entry, {"key_findings": ["[r1] anatase"], "analyses": [{**row, **analysis_verdict(own)}]})
    assert recs[0]["status"] == "verified"


def test_the_checks_withhold_certification_and_never_decide(tmp_path, monkeypatch):
    """Round 2: a monitor seeded from one or a few curves calls a same-phase
    thermal shift of a few cm⁻¹ "not the same state" (a 2 cm⁻¹ shift against
    a single-run prior: 4 of 5 poor at R² 0.99). Round 3: identity then took
    over as the false-positive source (a same-phase shift beyond a multi-unit
    regime was poor from +12 cm⁻¹, every peak "new" and "missing"), and a
    deciding check let the fall-through adopt a far regime. So neither
    decides: the gate's verdict stands everywhere, the checks withhold
    ``interpretation_checked`` and flag."""
    # a single-run prior (one unit): the same phase shifted 4 cm-1 (peak sigma 6) is far by the monitor
    single = tmp_path / "single"
    (single / "scripts").mkdir(parents=True)
    (single / "spectrum_0000").mkdir()
    np.save(single / "spectrum_0000" / "data.npy", rc.spectrum(rc.ANATASE, seed=5))
    (single / "scripts" / "fitting_script.py").write_text("ONE")
    (single / "series_fit_results.json").write_text(json.dumps({"results": [
        {"index": 0, "name": "s", "success": True, "parameters": rc.auto_detect_parameters(rc.ANATASE, seed=6)}]}))
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, shift=4.0, seed=7),
                                  extra_params={"ONE": rc.auto_detect_parameters(rc.ANATASE, shift=4.0, seed=95)})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["state_distance"] > _replay.SAME_STATE_BAR and rv["state_flag"] is True
    assert "STATE:" in rv["message"] and "flagged" in rv["message"] and not res.get("quality_warning")
    uv = unit_verdict_for({**res, "success": True})
    assert uv["verified"] and uv["interpretation_checked"] is False
    # a same-phase shift of 14 cm-1 beyond a three-unit regime (round 3's case): good, flagged on identity, unchecked
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.ANATASE, shift=14.0, seed=8),
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.ANATASE, shift=14.0, seed=96), "space_group": "I41/amd"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and res["script"] == "LOW" and not res.get("quality_warning")
    assert rv["identity"].get("flagged") is True and unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
    # round 3's finding 1: the nearest regime's identity flickering must not hand the choice to a far regime —
    # the fall-through runs on the GATE only; the nearest (distance ~0) is kept although its identity flags
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=3),
                                  extra_params={"HIGH": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=92), "peak_9": {"center": 300.0, "amplitude": 0.9}}})
    rv = res["reuse_validity"]
    assert [s for s, _ in ex.calls] == ["HIGH"] and rv["regime_choice"]["chosen_regime"] == "high" and rv["verdict"] == "good"
    assert rv["identity"].get("flagged") is True and rv["state_distance"] < _replay.SAME_STATE_BAR
    # the gate failing on the nearest still falls through, as on main: the next whose gate passes is kept
    # (good), with the state flagged — a far regime adopted by the GATE is said, not decided against
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.999, "HIGH": 0.70}, prior=prior,
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=13),
                                  extra_params={"HIGH": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=54),
                                                "LOW": rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=55)})
    assert [s for s, _ in ex.calls] == ["HIGH", "LOW"] and res["script"] == "LOW"
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["state_flag"] is True
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
    # a STRICT replay (the live loop's fast clock): the same record, the same verdict — nothing decides anywhere
    (prior / "scripts" / "spectrum_0000.py").write_text("LOW")
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.99}, prior=prior / "scripts" / "spectrum_0000.py",
                                  data=rc.spectrum(rc.RUTILE, shift=0.5, seed=2), strict=True,
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.RUTILE, shift=0.5, seed=97), "space_group": "P42/mnm"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["identity"].get("flagged") is True and rv["state_flag"] is True and not res.get("quality_warning")


def test_a_series_anchored_by_a_replay_rests_on_the_replay_gate(tmp_path):
    """Review of 59a56cee: a series whose anchor replayed a prior recipe
    posted its claims as verified (the series stamp said ``qc_gate``) while
    the same replay as a single run was provisional. The anchor's decider is
    the series'; followers that are replays of the series' own anchor (a
    hyperspectral series) are the mechanics, not a prior replay."""
    replay = {"verified": True, "reason": "locked-script reuse passed the replay gate", "decided_by": "replay_gate",
              "interpretation_checked": False}
    recipe = {"verified": True, "reason": "replayed the locked recipe", "decided_by": "recipe", "interpretation_checked": False}
    own = {"verified": True, "reason": "approved by its own gate", "decided_by": "qc_gate", "interpretation_checked": False}
    series = {"status": "success", "individual_results": [
        {"success": True, "role": "anchor", "unit_verdict": replay}, {"success": True, "unit_verdict": recipe}]}
    v = final_verdict_record(series)
    assert v["verified"] and v["decided_by"] == "replay_gate" and v["interpretation_checked"] is False
    row = {"analysis_id": "s1", "status": "success", "agent_name": "CurveFittingAgent", "series": True, **analysis_verdict({**series, "verdict": v})}
    recs = board_mod.records_for({"index": 1, "label": "series", "mode": "analysis", "status": "success"},
                                 {"key_findings": ["[s1] the series shows anatase"], "analyses": [row]})
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "provisional")]
    # an anchor verified by its own gate with replayed followers: the series' own mechanics, qc_gate
    hs_like = {"status": "success", "individual_results": [
        {"success": True, "role": "anchor", "unit_verdict": own}, {"success": True, "role": "follower", "unit_verdict": replay}]}
    assert final_verdict_record(hs_like)["decided_by"] == "qc_gate"
    # no roles (an older result): the first unit is the anchor
    assert final_verdict_record({"status": "success", "individual_results": [
        {"success": True, "unit_verdict": replay}, {"success": True, "unit_verdict": recipe}]})["decided_by"] == "replay_gate"


def test_certification_needs_the_tight_bar_and_certifies_one_unit_references_and_followers(tmp_path, monkeypatch):
    """Round 4: (A) a fixed-position recipe cannot report an impurity or a low
    mixture, so the state distance is the only check that sees them — they
    sit at 0.08–0.20, clean replays under ~0.07 — and certification needs the
    tighter CERTIFY_STATE_BAR while the flag bar stays. (B) a single-run
    prior and a series' followers could never be certified (a spread was
    required; followers carried no checks); both certify on clean evidence
    now — a one-unit reference under the tight bar with identity within, and
    the cheap checks on each follower against its regime's anchor."""
    assert _replay.CERTIFY_STATE_BAR < _replay.SAME_STATE_BAR
    # (A) the record: a reuse at 0.15 — under the flag bar, above the certification bar — is good, unflagged, uncertified
    rv = {"reused": True, "verdict": "good", "identity": {"checked": True, "within": True, "spread_known": True, "drifted": []},
          "state_distance": 0.15}
    assert interpretation_checked_by(rv) is False and interpretation_checked_by({**rv, "state_distance": 0.05}) is True
    # ...through the real path: a fixed recipe (the regime's set reported as is) on anatase with a 15 % impurity line
    prior = rc.prior_two_regime_run(tmp_path, names=True)
    impure = rc.spectrum(rc.ANATASE, seed=4)
    impure[:, 1] += 0.15 * np.exp(-0.5 * ((impure[:, 0] - 301.0) / 6.0) ** 2)
    res, ex, _, _ = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.99, "HIGH": 0.99}, prior=prior, data=impure,
                                  extra_params={"LOW": {**rc.auto_detect_parameters(rc.ANATASE, seed=40), "space_group": "I41/amd"}})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["identity"]["within"] and not rv.get("state_flag")
    assert _replay.CERTIFY_STATE_BAR < rv["state_distance"] <= _replay.SAME_STATE_BAR, rv["state_distance"]
    assert "above the certification bar" in rv["message"]
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
    # (B) a single-run prior reused on the same kind of data: one reference unit, distance ~0, identity within → certified
    single = tmp_path / "single"
    (single / "scripts").mkdir(parents=True)
    (single / "spectrum_0000").mkdir()
    np.save(single / "spectrum_0000" / "data.npy", rc.spectrum(rc.ANATASE, seed=5))
    (single / "scripts" / "fitting_script.py").write_text("ONE")
    (single / "series_fit_results.json").write_text(json.dumps({"results": [
        {"index": 0, "name": "s", "success": True, "parameters": rc.auto_detect_parameters(rc.ANATASE, seed=6)}]}))
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, seed=7),
                                  extra_params={"ONE": rc.auto_detect_parameters(rc.ANATASE, seed=95, n_noise=3)})
    rv = res["reuse_validity"]
    assert rv["verdict"] == "good" and rv["state_distance"] < _replay.CERTIFY_STATE_BAR and not rv["identity"]["spread_known"]
    assert unit_verdict_for({**res, "success": True})["interpretation_checked"] is True
    # ...and a different finding against that one unit still withholds (flagged, uncertified)
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"ONE": 0.99}, prior=single, data=rc.spectrum(rc.ANATASE, seed=7),
                                  extra_params={"ONE": {**rc.auto_detect_parameters(rc.ANATASE, seed=95), "peak_9": {"center": 450.0, "amplitude": 0.9}}})
    assert res["reuse_validity"]["identity"]["flagged"] is True and unit_verdict_for({**res, "success": True})["interpretation_checked"] is False
    # (B) followers: the real series path checks each follower against its regime's anchor and certifies the ones that agree
    names6 = [f"spectrum_{i:04d}" for i in range(6)]
    state, _ = curve.run_series(tmp_path / "lock", monkeypatch, names=names6, regimes=[[0, 1, 2], [3, 4, 5]],
                                anchors={"spectrum_0000": curve.OK, "spectrum_0003": {**curve.OK, "script": "M3"}},
                                follower_r2={n: 0.97 for n in names6})
    results = curve.compile_results(tmp_path / "lock", state)
    followers = [u for u in results["individual_results"] if u.get("role") != "anchor"]
    assert followers and all(isinstance(u.get("regime_checks"), dict) for u in followers)
    assert all(isinstance(u["regime_checks"]["state_distance"], (int, float)) for u in followers)
    assert all(u["unit_verdict"]["decided_by"] == "recipe" and u["unit_verdict"]["interpretation_checked"] is True for u in followers)
    # a series whose anchor is a certified replay and whose followers are certified posts a VERIFIED claim
    replay = {"verified": True, "reason": "locked-script reuse passed the replay gate", "decided_by": "replay_gate", "interpretation_checked": True}
    follower = {"verified": True, "reason": "replayed the locked recipe", "decided_by": "recipe", "interpretation_checked": True}
    series = {"status": "success", "individual_results": [{"success": True, "role": "anchor", "unit_verdict": replay},
                                                          {"success": True, "unit_verdict": follower}]}
    v = final_verdict_record(series)
    assert v["decided_by"] == "replay_gate" and v["interpretation_checked"] is True
    row = {"analysis_id": "s1", "status": "success", "agent_name": "CurveFittingAgent", "series": True, **analysis_verdict({**series, "verdict": v})}
    recs = board_mod.records_for({"index": 1, "label": "series", "mode": "analysis", "status": "success"},
                                 {"key_findings": ["[s1] the series shows anatase"], "analyses": [row]})
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "verified")]
