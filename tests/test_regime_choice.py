"""#710: which regime a new measurement belongs to is read from the DATA.

A method-level recipe (XRD auto-detect) fits any pattern, so the replay gate
alone cannot tell regimes apart and "the first that passes" was always the
first in lock order. Each regime's recipe now records its anchor's curve on
the drift monitor's grid when it is locked; on a reuse a ``DriftMonitor``
(``live/drift.py``, model-free) seeded with each regime's own units' data
measures how much of the new curve that regime cannot describe, the nearest
regime is tried first (``select_recipe(strategy="nearest_first")``), and the
judge confirms. The choice and its evidence are on
``reuse_validity.regime_choice``; two regimes the data cannot tell apart are
flagged ambiguous, in ``source`` and ``message`` too. An older run with no
data and no stamp keeps lock order — the behaviour this commit changes is
exactly: a multi-regime prior whose regimes' data (or stamps) are there.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_series_verdict_path as curve  # noqa: E402

from scilink.agents.exp_agents import _replay  # noqa: E402
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (  # noqa: E402
    _drift_state, _prior_curve_fit_candidates, _regime_references)

X = np.linspace(100, 800, 1400)
ANATASE = [(144.0, 1.0), (397.0, 0.12), (516.0, 0.18), (639.0, 0.16)]
RUTILE = [(143.0, 0.10), (235.0, 0.15), (447.0, 0.55), (612.0, 1.0)]


def spectrum(peaks, *, shift=0.0, seed=0, noise=0.01, width=6.0):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(X)
    for c, a in peaks:
        y += a * np.exp(-0.5 * ((X - c - shift) / width) ** 2)
    return np.c_[X, y + noise * rng.standard_normal(X.size)]


def auto_detect_parameters(peaks, *, shift=0.0, seed=0, n_noise=0):
    """What an auto-detect recipe reports on one unit: the phase's peaks plus
    ``n_noise`` weak peaks it found in the noise, in detection order — so the
    SAME peak sits at a different index on different units."""
    rng = np.random.default_rng(100 + seed)
    found = [(c + shift + rng.normal(0, 0.3), a * rng.uniform(0.9, 1.1)) for c, a in peaks]
    found += [(float(rng.uniform(120, 780)), float(rng.uniform(0.01, 0.03))) for _ in range(n_noise)]
    rng.shuffle(found)
    return {f"peak_{i + 1}": {"center": float(c), "amplitude": float(a), "fwhm": 12.0} for i, (c, a) in enumerate(found)}


def prior_two_regime_run(tmp_path, *, data=True, stamps=True, noise_peaks=(5, 5, 6, 6, 9, 7), names=False):
    """A prior series run: three LOW (anatase) units, three HIGH (rutile)
    units, each with its data file, auto-detect parameters and regime; the
    regimes' recipes with the anchors' drift states."""
    run = tmp_path / "prior"
    (run / "scripts").mkdir(parents=True)
    rows, curves = [], {}
    for i in range(6):
        low = i < 3
        peaks, shift = (ANATASE, 0.4 * i) if low else (RUTILE, 0.4 * (i - 3))
        xy = spectrum(peaks, shift=shift, seed=i)
        curves[i] = xy
        if data:
            (run / f"spectrum_{i:04d}").mkdir()
            np.save(run / f"spectrum_{i:04d}" / "data.npy", xy)
        params = auto_detect_parameters(peaks, shift=shift, seed=i, n_noise=noise_peaks[i])
        if names:
            params["space_group"] = "I 41/a m d :2" if low else "P 42/m n m"
            params["database_id"] = str(1000 + i)
        rows.append({"index": i, "name": f"spectrum_{i:04d}", "success": True, "regime": "low" if low else "high",
                     "parameters": params, "fit_quality": {"r_squared": 0.98}})
    (run / "series_fit_results.json").write_text(json.dumps({"results": rows}))
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "low": {"unit": "spectrum_0000", "index": 0, "regime": "low", "script": "LOW", "verdict": {"verified": True},
                **({"drift_state": _drift_state(curves[0])} if stamps else {})},
        "high": {"unit": "spectrum_0003", "index": 3, "regime": "high", "script": "HIGH", "verdict": {"verified": True},
                 **({"drift_state": _drift_state(curves[3])} if stamps else {})}}}))
    return run


def test_select_recipe_nearest_first_orders_by_distance_and_flags_ambiguity():
    ran = []

    def run(n, script, source):
        ran.append(script)
        return {"success": True, "script": script}
    good = lambda n, r: _replay.replay_verdict("good", score=0.99)  # noqa: E731
    out = _replay.select_recipe([("A", "a"), ("B", "b"), ("C", "c")], run, good,
                                strategy="nearest_first", distances=[0.6, 0.05, None])
    assert out["order"] == [2, 1, 3] and out["chosen"] == 2 and ran == ["B"]       # nearest first, unknown last
    assert out["ambiguous"] is False and out["margin"] == 0.55
    ran.clear()
    out = _replay.select_recipe([("A", "a"), ("B", "b")], run, good, strategy="nearest_first", distances=[0.30, 0.50])
    assert out["chosen"] == 1 and out["ambiguous"] is True                        # within the 2x ratio
    out = _replay.select_recipe([("A", "a"), ("B", "b")], run, good, strategy="nearest_first", distances=[0.03, 0.08])
    assert out["ambiguous"] is True                                               # both under the material bar
    out = _replay.select_recipe([("A", "a"), ("B", "b")], run, good, strategy="nearest_first", distances=[0.05, 0.70])
    assert out["ambiguous"] is False
    with pytest.raises(ValueError):
        _replay.select_recipe([("A", "a")], run, good, strategy="nearest_first", distances=[0.1, 0.2])
    # the judge sees which candidate it judges, so a candidate that fits but is not its regime's falls through
    ran.clear()
    out = _replay.select_recipe([("A", "a"), ("B", "b")], run,
                                lambda n, r: _replay.replay_verdict("poor" if n == 1 else "good", score=0.99))
    assert out["chosen"] == 2 and ran == ["A", "B"] and [t["verdict"] for t in out["tried"]] == ["poor", "good"]
    # lock order is unchanged without distances
    ran.clear()
    out = _replay.select_recipe([("A", "a"), ("B", "b")], run, good)
    assert out["order"] == [1, 2] and out["chosen"] == 1 and out["ambiguous"] is False and out["margin"] is None


def test_the_lock_records_the_anchors_curve_and_a_reuse_matches_the_regime_by_its_data(tmp_path, monkeypatch):
    # the real series path: each regime's recipe carries its anchor's drift state at lock time
    names6 = [f"spectrum_{i:04d}" for i in range(6)]
    state, _ = curve.run_series(tmp_path / "lock", monkeypatch, names=names6, regimes=[[0, 1, 2], [3, 4, 5]],
                                anchors={"spectrum_0000": curve.OK, "spectrum_0003": {**curve.OK, "script": "M3"}},
                                follower_r2={n: 0.97 for n in names6})
    results = curve.compile_results(tmp_path / "lock", state)
    for r in results["locked_recipes"].values():
        assert isinstance(r.get("drift_state"), dict) and r["drift_state"].get("x") and r["drift_state"].get("seed")
    from scilink.agents.exp_agents._verification_record import series_recipes
    assert all(isinstance(r["drift_state"], dict) for r in series_recipes(results))
    # a reuse: the regimes' references come from the units' data; the new spectrum is a rutile one
    prior = prior_two_regime_run(tmp_path)
    cands = _prior_curve_fit_candidates({"prior_analysis_paths": [str(prior)]})
    assert [c["regime"] for c in cands] == ["low", "high"] and all(c["drift_state"] for c in cands)
    refs = _regime_references({"prior_analysis_paths": [str(prior)]}, cands)
    assert [r["n_curves"] for r in refs] == [3, 3] and all(r["identity"]["n_units"] == 3 for r in refs)
    # both recipes "fit" (an auto-detect recipe fits any pattern): HIGH is tried first and only, although
    # LOW is first in lock order, because the data is the HIGH regime's
    res, ex, corrections, item = curve._replay(tmp_path / "a", monkeypatch, {"LOW": 0.999, "HIGH": 0.999},
                                               prior=prior, data=spectrum(RUTILE, shift=0.5, seed=11),
                                               extra_params={"HIGH": auto_detect_parameters(RUTILE, shift=0.5, seed=50, n_noise=7),
                                                             "LOW": auto_detect_parameters(ANATASE, seed=51, n_noise=5)})
    assert res["script"] == "HIGH" and [s for s, _ in ex.calls] == ["HIGH"] and corrections == []
    rc = res["reuse_validity"]["regime_choice"]
    assert rc["by"] == "state_distance" and rc["chosen_regime"] == "high" and rc["ambiguous"] is False
    assert [r["regime"] for r in rc["ranking"]] == ["high", "low"] and rc["ranking"][0]["distance"] < rc["ranking"][1]["distance"]
    assert rc["ranking"][1]["distance"] > _replay.SAME_STATE_BAR          # a different phase: far
    assert res["reuse_validity"]["verdict"] == "good" and res["reuse_validity"]["recipes_tried"] == 1
    assert "ambiguous" not in res["reuse_validity"]["source"]
    # an anatase spectrum goes to LOW
    res, ex, _, _ = curve._replay(tmp_path / "b", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=prior,
                                  data=spectrum(ANATASE, shift=0.6, seed=12),
                                  extra_params={"LOW": auto_detect_parameters(ANATASE, shift=0.6, seed=52, n_noise=8),
                                                "HIGH": auto_detect_parameters(RUTILE, seed=53)})
    assert res["script"] == "LOW" and res["reuse_validity"]["regime_choice"]["chosen_regime"] == "low"
    # the nearest regime's recipe failing the GATE falls through to the next; that one is not the regime's
    # state, so it is poor too, and the kept result is the nearest (first executed)
    res, ex, _, _ = curve._replay(tmp_path / "c", monkeypatch, {"LOW": 0.999, "HIGH": 0.70}, prior=prior,
                                  data=spectrum(RUTILE, shift=0.5, seed=13),
                                  extra_params={"HIGH": auto_detect_parameters(RUTILE, shift=0.5, seed=54),
                                                "LOW": auto_detect_parameters(RUTILE, shift=0.5, seed=55)})
    assert [s for s, _ in ex.calls] == ["HIGH", "LOW"] and res["script"] == "HIGH"
    assert res["reuse_validity"]["verdict"] == "poor" and res["reuse_validity"]["recipes_tried"] == 2
    # an older run (no data files, no stamps): lock order, as before this commit
    old = prior_two_regime_run(tmp_path / "old", data=False, stamps=False)
    res, ex, _, _ = curve._replay(tmp_path / "d", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=old,
                                  data=spectrum(RUTILE, seed=14), extra_params={"LOW": auto_detect_parameters(RUTILE, seed=56)})
    assert res["reuse_validity"]["regime_choice"]["by"] == "lock_order"
    # the stamps alone (data files pruned) still rank
    stamped = prior_two_regime_run(tmp_path / "st", data=False, stamps=True)
    res, ex, _, _ = curve._replay(tmp_path / "e", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=stamped,
                                  data=spectrum(RUTILE, shift=0.5, seed=15),
                                  extra_params={"HIGH": auto_detect_parameters(RUTILE, shift=0.5, seed=57)})
    assert res["reuse_validity"]["regime_choice"]["by"] == "state_distance" and res["script"] == "HIGH"


def test_two_regimes_the_data_cannot_tell_apart_are_said_to_be_ambiguous(tmp_path, monkeypatch):
    """The reviewer's benchmark cases: regimes that differ by a small shift,
    texture or background are told apart by the monitor; regimes whose data
    is the same are reported ambiguous — in regime_choice, source and message."""
    run = tmp_path / "prior"
    (run / "scripts").mkdir(parents=True)
    rows = []
    for i in range(6):
        xy = spectrum(ANATASE, shift=0.0 if i < 3 else 2.5, seed=i)          # the same phase, a 2.5 cm-1 shift
        (run / f"spectrum_{i:04d}").mkdir()
        np.save(run / f"spectrum_{i:04d}" / "data.npy", xy)
        rows.append({"index": i, "name": f"spectrum_{i:04d}", "success": True, "regime": "a" if i < 3 else "b",
                     "parameters": auto_detect_parameters(ANATASE, shift=0.0 if i < 3 else 2.5, seed=i, n_noise=4)})
    (run / "series_fit_results.json").write_text(json.dumps({"results": rows}))
    (run / "analysis_results.json").write_text(json.dumps({"locked_recipes": {
        "a": {"unit": "spectrum_0000", "index": 0, "regime": "a", "script": "LOW", "verdict": {"verified": True}},
        "b": {"unit": "spectrum_0003", "index": 3, "regime": "b", "script": "HIGH", "verdict": {"verified": True}}}}))
    # a shifted spectrum: the shifted regime is nearer
    res, ex, _, _ = curve._replay(tmp_path / "s", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=run,
                                  data=spectrum(ANATASE, shift=2.6, seed=21),
                                  extra_params={"HIGH": auto_detect_parameters(ANATASE, shift=2.6, seed=58),
                                                "LOW": auto_detect_parameters(ANATASE, shift=2.6, seed=59)})
    rc = res["reuse_validity"]["regime_choice"]
    assert rc["chosen_regime"] == "b" and rc["ranking"][0]["distance"] < rc["ranking"][1]["distance"]
    # identical regimes: ambiguous, and said where the attribution is read
    for i in range(3, 6):
        np.save(run / f"spectrum_{i:04d}" / "data.npy", spectrum(ANATASE, shift=0.0, seed=i))
    res, ex, _, _ = curve._replay(tmp_path / "t", monkeypatch, {"LOW": 0.999, "HIGH": 0.999}, prior=run,
                                  data=spectrum(ANATASE, seed=22),
                                  extra_params={"HIGH": auto_detect_parameters(ANATASE, seed=60), "LOW": auto_detect_parameters(ANATASE, seed=61)})
    rv = res["reuse_validity"]
    assert rv["regime_choice"]["ambiguous"] is True and "does not tell regimes" in rv["regime_choice"]["note"]
    assert "[regime choice ambiguous:" in rv["source"] and "REGIME CHOICE AMBIGUOUS" in rv["message"]


def test_the_repair_fallback_repairs_the_nearest_and_a_strict_failure_records_the_choice(tmp_path, monkeypatch):
    prior = prior_two_regime_run(tmp_path)
    data = spectrum(RUTILE, shift=0.5, seed=31)
    # both raise verbatim: the ladder repairs the NEAREST (HIGH), not candidates[0]
    res, ex, corrections, item = curve._replay(tmp_path / "r", monkeypatch, {"REPAIRED": 0.97}, prior=prior, data=data,
                                               repaired="REPAIRED",
                                               extra_params={"REPAIRED": auto_detect_parameters(RUTILE, shift=0.5, seed=62)})
    assert res["script"] == "REPAIRED" and len(corrections) == 1
    assert [s for s, wd in ex.calls if wd.name.startswith("recipe_")] == ["HIGH", "LOW"]      # nearest first, verbatim
    assert ex.calls[-1][0] in ("HIGH", "REPAIRED") and res["reuse_validity"]["regime_choice"]["chosen_regime"] == "high"
    # strict: both raise, failed, and the choice is on the record
    res, ex, _, _ = curve._replay(tmp_path / "f", monkeypatch, {}, prior=prior, data=data, strict=True)
    assert res["reuse_validity"]["verdict"] == "failed" and res["reuse_validity"]["regime_choice"]["by"] == "state_distance"
    assert res["reuse_validity"]["regime_choice"]["chosen_regime"] is None
