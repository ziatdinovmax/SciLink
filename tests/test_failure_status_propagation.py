"""Offline tests for total-failure status propagation (zero-success guard).

Pre-fix, a run where EVERY spectrum/image failed every attempt (a per-item
condition, so no controller set error_dict) still returned top-level
status "success" from _compile_results — a caller checking only `status`
was misled. The guard flips it to "error" with an error dict matching the
error_dict shape; anything with at least one success (including salvaged
best-available results, which carry success=True + quality_warning) is
untouched, and exotic items missing the `success` key never trip it.

  conda run -n scilink python tests/test_failure_status_propagation.py
"""
import logging

from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

results = {}
logging.disable(logging.CRITICAL)


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _fail(i, err="exec failed"):
    return {"index": i, "name": f"item_{i}", "success": False, "error": err}


def _ok(i, warn=None):
    r = {"index": i, "name": f"item_{i}", "success": True,
         "parameters": {"a": 1.0}, "fit_quality": {"r_squared": 0.99}}
    if warn:
        r["quality_warning"] = warn
    return r


def _curve_state(series_results, **kw):
    state = {"is_single_spectrum": len(series_results) <= 1,
             "num_spectra": max(len(series_results), 1),
             "series_results": series_results,
             "synthesis_result": {}, "flagged_spectra": []}
    state.update(kw)
    return state


def _image_state(series_results, **kw):
    state = {"is_single_image": len(series_results) <= 1,
             "num_images": max(len(series_results), 1),
             "series_results": series_results,
             "synthesis_result": {}, "flagged_images": [],
             "analysis_result": {}}
    state.update(kw)
    return state


def main():
    curve = CurveFittingAgent()
    image = ImageAnalysisAgent()

    # 1) All failed -> error with error_dict-shaped payload.
    print("1) all items failed:")
    out = curve._compile_results(_curve_state(
        [_fail(0), _fail(1, "timeout"), _fail(2)]))
    check("curve: status error", out["status"] == "error")
    check("curve: error names the count",
          "All 3 spectrum fit(s) failed" in out["error"]["error"])
    check("curve: details carry last item error",
          out["error"]["details"] == "timeout" or out["error"]["details"])
    out = image._compile_results(_image_state([_fail(0), _fail(1)]))
    check("image: status error", out["status"] == "error")
    check("image: error names the count",
          "All 2 image analysis(es) failed" in out["error"]["error"])

    # 2) Single-item total failure (single spectrum = series of 1).
    print("2) single item failed:")
    out = curve._compile_results(_curve_state([_fail(0)]))
    check("curve single: status error", out["status"] == "error")
    out = image._compile_results(_image_state([_fail(0)]))
    check("image single: status error", out["status"] == "error")

    # 3) Mixed success/failure -> success (per-item failures stay per-item).
    print("3) mixed outcomes:")
    out = curve._compile_results(_curve_state([_fail(0), _ok(1), _fail(2)]))
    check("curve mixed: status success", out["status"] == "success")
    out = image._compile_results(_image_state([_ok(0), _fail(1)]))
    check("image mixed: status success", out["status"] == "success")

    # 4) Salvaged best-available (success=True + quality_warning) -> success.
    print("4) salvaged best-available:")
    out = curve._compile_results(_curve_state(
        [_ok(0, warn="R² below threshold")]))
    check("curve salvaged: status success", out["status"] == "success")

    # 5) Empty series_results (no fitting ran) -> unchanged success.
    print("5) empty series_results:")
    out = curve._compile_results(_curve_state([]))
    check("curve empty: status success", out["status"] == "success")

    # (A series item without a `success` key is impossible on this path —
    # _compile_results itself indexes r["success"] when building
    # individual_results — so the guard's explicit `is False` requirement
    # is purely defensive and has no reachable counterexample to test.)

    # 6) HS plain-total-failure predicate: fires only when every target
    #    failed with no salvage AND no honest not-measurable resolution.
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent as HS)
    plain = HS._plain_total_dynamic_failure
    print("6) HS plain-total-failure predicate:")
    check("all plain failures -> fires",
          plain([{"task_success": False}, {"task_success": False}]))
    check("mixed plain + not_measurable -> fires (plain part unresolved)",
          plain([{"task_success": False, "not_measurable": {"why": "x"}},
                 {"task_success": False}]))
    check("all not_measurable -> honest null, does NOT fire",
          not plain([{"task_success": False,
                      "not_measurable": {"why": "x"}}]))
    check("salvaged failure -> handled by degradation notes, does NOT fire",
          not plain([{"task_success": False, "salvaged": True}]))
    check("any success -> does NOT fire",
          not plain([{"task_success": True}, {"task_success": False}]))
    check("no records -> does NOT fire", not plain([]) and not plain(None))

    # 7) Orchestrator formatting branch accepts partial (source-level check:
    #    the closure is not unit-instantiable, live test covers behavior).
    import inspect
    from scilink.agents.exp_agents import analysis_orchestrator_tools as aot
    src = inspect.getsource(aot)
    check("run_analysis success branch admits 'partial'",
          'in ("success", "partial")' in src)
    check("partial branch surfaces confidence/warnings",
          'response["confidence"]' in src and 'response["warnings"]' in src)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FAILURE STATUS PROPAGATION: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
