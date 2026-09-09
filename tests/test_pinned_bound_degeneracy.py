"""#592 — a parameter pinned at its bound is a degeneracy, independent of R².
The validator reads the script's reported bounds; the fit loop routes a
pinned fit through the correction ladder (which relaxes exactly those
bounds); a fit that still pins is stamped and the series flags it."""
import json
import logging
from pathlib import Path

import numpy as np

from scilink.skills._shared.curve_fitting_tools import validate_bound_pinning, describe_pinned, PINNED_BOUND_FIX
from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc

PARAMS = {"peak_1": {"amplitude": 0.31, "center": 445.0, "sigma": 30.0},
          "peak_2": {"amplitude": 0.45, "amplitude_err": 0.0, "center": 560.0, "sigma": 60.0},
          "baseline": {"intercept": 0.43, "slope": 0.0}}
BOUNDS = {"peak_1": {"amplitude": [0, None], "center": [400, 445.0], "sigma": [5, 80]},
          "peak_2": {"amplitude": [0, 0.45], "center": [500, 620], "sigma": [10, 120]},
          "baseline": {"intercept": [None, None]}}


def test_validator_names_the_pinned_parameters_only():
    pins = validate_bound_pinning(PARAMS, BOUNDS)
    assert [(p["component"], p["parameter"], p["side"]) for p in pins] == [
        ("peak_1", "center", "upper"), ("peak_2", "amplitude", "upper")]
    assert "peak_2.amplitude = 0.45 at its upper bound 0.45" in describe_pinned(pins)
    # 1 % of the span: 0.446 pins, 0.44 does not
    assert validate_bound_pinning({"p": {"a": 0.446}}, {"p": {"a": [0, 0.45]}})
    assert not validate_bound_pinning({"p": {"a": 0.44}}, {"p": {"a": [0, 0.45]}})
    # an open-sided bound uses the bound's own magnitude for the tolerance
    assert validate_bound_pinning({"p": {"c": 99.5}}, {"p": {"c": [None, 100]}})
    # a zero lower bound is a physical floor (vanished component, zero slope): never reported
    assert not validate_bound_pinning({"p": {"amplitude": 0.0}}, {"p": {"amplitude": [0, 5]}})
    assert not validate_bound_pinning({"b": {"slope": 0.0}}, {"b": {"slope": [0, 1]}})
    # a non-zero floor is a window edge: reported
    assert validate_bound_pinning({"p": {"sigma": 5.0}}, {"p": {"sigma": [5, 80]}})[0]["side"] == "lower"
    # fraction-like parameters legitimately sit at 0 or 1 (a pseudo-Voigt at eta=1 is a Lorentzian)
    assert not validate_bound_pinning({"p": {"eta": 1.0, "fraction": 0.0}}, {"p": {"eta": [0, 1], "fraction": [0, 1]}})
    # nothing reported → nothing to check; garbage never raises
    assert validate_bound_pinning(PARAMS, None) == [] and validate_bound_pinning(PARAMS, {"peak_2": "x"}) == []
    assert validate_bound_pinning({"p": {"a": "nan"}}, {"p": {"a": [0, 1]}}) == []


def _controller(tmp_path, runs, corrections):
    c = cc.UnifiedSeriesProcessingController.__new__(cc.UnifiedSeriesProcessingController)
    c.logger = logging.getLogger("t592"); c.output_dir = Path(tmp_path); c.executor = object()
    c._extract_extra_operands = lambda state, p: None
    c._extra_operand_block = lambda state: ""
    c._compute_statistics = lambda cd: {"n": len(cd)}
    c._should_escalate_timeout_model = lambda *a, **k: False
    c._correct_script = lambda state, script, err: (corrections.append(err) or (script + "\n# relaxed", "widened the bound"))
    return c


def _run(stdout):
    return {"status": "success", "stdout": stdout, "visualization_path": "viz.png",
            "visualization_bytes": b"", "exec": {}}


def _stdout(params, bounds):
    return "FIT_RESULTS_JSON:" + json.dumps({"model_type": "2G+lin", "parameters": params, "bounds": bounds,
                                             "fit_quality": {"r_squared": 0.963}})


def test_a_pinned_locked_script_is_corrected_and_the_relaxed_fit_is_kept(tmp_path, monkeypatch):
    corrections = []
    relaxed_params = {**PARAMS, "peak_2": {**PARAMS["peak_2"], "amplitude": 1.21}, "baseline": {"intercept": 0.01, "slope": 0.0}}
    relaxed_bounds = {**BOUNDS, "peak_1": {"amplitude": [0, None], "center": [400, 460], "sigma": [5, 80]},
                      "peak_2": {**BOUNDS["peak_2"], "amplitude": [0, 2.8]}}
    runs = iter([_run(_stdout(PARAMS, BOUNDS)), _run(_stdout(relaxed_params, relaxed_bounds))])
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: next(runs))
    c = _controller(tmp_path, runs, corrections)
    data = np.column_stack([np.linspace(400, 700, 50), np.ones(50)])
    res = c._fit_single_spectrum({}, data, "s7.csv", "s7", 7, base_script="ub = [inf, 445.0, 0.45]")
    assert res["success"] and "pinned_at_bound" not in res and "pinned_at_bound" not in res["fit_quality"]
    assert res["parameters"]["peak_2"]["amplitude"] == 1.21 and res["script"].endswith("# relaxed")
    assert res["bounds"] == relaxed_bounds
    # the correction was asked for exactly the bound relaxation
    assert len(corrections) == 1 and "peak_2.amplitude = 0.45 at its upper bound 0.45" in corrections[0]
    assert PINNED_BOUND_FIX in corrections[0]
    assert res["script_errors"][0]["error"].startswith("DEGENERATE FIT")


def test_a_fit_that_still_pins_after_the_ladder_is_stamped(tmp_path, monkeypatch):
    corrections = []
    monkeypatch.setattr(cc, "stage_and_run_adaptive", lambda *a, **k: _run(_stdout(PARAMS, BOUNDS)))
    c = _controller(tmp_path, None, corrections)
    c.MAX_ATTEMPTS = 3
    data = np.column_stack([np.linspace(400, 700, 50), np.ones(50)])
    res = c._fit_single_spectrum({}, data, "s8.csv", "s8", 8, base_script="ub = [inf, 445.0, 0.45]")
    assert res["success"] and len(corrections) == 1                      # one relaxation, then kept and stamped
    assert [p["parameter"] for p in res["pinned_at_bound"]] == ["center", "amplitude"]
    assert res["fit_quality"]["pinned_at_bound"] == res["pinned_at_bound"]
    assert res["quality_warning"].startswith("Degenerate fit:")
    assert res["script_errors"][-1]["kind"] == "pinned_bound"


def test_series_flags_a_pinned_fit_regardless_of_r2(tmp_path):
    c = cc.UnifiedSeriesProcessingController.__new__(cc.UnifiedSeriesProcessingController)
    c.logger = logging.getLogger("t592"); c.r2_threshold = 0.95; c.outlier_sigma = 2.0
    def r(i, r2, pins=None):
        fq = {"r_squared": r2}
        if pins: fq["pinned_at_bound"] = pins
        return {"index": i, "name": f"s{i}", "success": True, "fit_quality": fq}
    pins = [{"component": "peak_2", "parameter": "amplitude", "value": 0.45, "bound": 0.45, "side": "upper"}]
    results = [r(0, 0.99), r(1, 0.985), r(2, 0.99), r(3, 0.963, pins), r(4, 0.988)]
    flagged = c._detect_outliers(results)
    assert [(f["index"], f["reason"]) for f in flagged] == [(3, "pinned_at_bound")]
    assert "R² is not evidence here" in flagged[0]["recommendation"] and "peak_2.amplitude" in flagged[0]["recommendation"]


def test_codegen_contract_carries_the_rule_and_the_bounds_field():
    from scilink.agents.exp_agents import instruct
    assert "Bounds are data-relative, never constants read off this spectrum" in instruct.FITTING_SCRIPT_INSTRUCTIONS
    assert '"bounds": {{"peak_1"' in instruct.FITTING_SCRIPT_INSTRUCTIONS
    assert "PINNED AT ITS BOUND" in instruct.FITTING_SCRIPT_CORRECTION_INSTRUCTIONS


def test_single_regime_plan_does_not_block_locked_script_reuse():
    blocked = cc.UnifiedSeriesProcessingController._reuse_blocked_by_regimes
    assert blocked({"series_analysis_plan": {"regimes": [{"spectrum_indices": [0, 1, 2]}]}, "regime_configs": {0: {}, 1: {}, 2: {}}}) is False
    assert blocked({"series_analysis_plan": {"regimes": [{"spectrum_indices": [0, 1]}, {"spectrum_indices": [2]}]}}) is True
    assert blocked({}) is False
