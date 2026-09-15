"""Offline tests for series-plan revision index inheritance.

A plan-validation (or refinement) revision that returns regimes without
spectrum_indices used to collapse a multi-regime plan: the missing-index
fallback assigned every spectrum to regime 1 and dropped the rest as empty.
Observed live (ibuprofen in-situ XRD, 2026-07-16): the validator revised for
air-scatter handling, renamed the regimes slightly, omitted the indices, and
a 2-regime plan silently became 1 regime. The fix inherits omitted indices
from the plan being revised (by name, else by position when the regime count
is unchanged) and shows the indices to the validator in the first place.

  conda run -n scilink python tests/test_series_plan_revision.py
"""
import logging
import types

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    CurveFittingPlanningController,
)

results = {}
logging.basicConfig(level=logging.WARNING)


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _ctrl():
    return types.SimpleNamespace(
        logger=logging.getLogger("revision-test"),
        _extract_series_plan=None,  # unused; unbound call below
    )


def _extract(state, result):
    CurveFittingPlanningController._extract_series_plan(_ctrl(), state, result)


def _state(prior_regimes=None, num=38):
    state = {"is_single_spectrum": False, "num_spectra": num}
    if prior_regimes is not None:
        state["series_analysis_plan"] = {"regimes": prior_regimes}
    return state


def _plan(*regimes):
    return {"series_analysis_plan": {"regimes": list(regimes)}}


def main():
    # 1) The live collapse, replayed: revision renames regimes (name match
    #    fails), omits indices, count unchanged -> positional inheritance.
    state = _state([
        {"name": "Dihydrate phase (below transition)",
         "spectrum_indices": list(range(0, 12)), "physical_model": "PV"},
        {"name": "Dehydrated phase (above transition)",
         "spectrum_indices": list(range(12, 38)), "physical_model": "PV"},
    ])
    _extract(state, _plan(
        {"name": "Dihydrate phase (below transition, T < ~47 C)",
         "physical_model": "PV revised"},
        {"name": "Dehydrated phase (above transition, T > ~47 C)",
         "physical_model": "PV revised"},
    ))
    plan = state["series_analysis_plan"]
    print("1) live collapse scenario (renamed regimes, no indices):")
    check("plan keeps 2 regimes", plan is not None and len(plan["regimes"]) == 2)
    check("regime 1 indices inherited positionally",
          plan["regimes"][0]["spectrum_indices"] == list(range(0, 12)))
    check("regime 2 indices inherited positionally",
          plan["regimes"][1]["spectrum_indices"] == list(range(12, 38)))
    check("revised models kept",
          plan["regimes"][0]["physical_model"] == "PV revised")

    # 2) Name match wins over position (regimes reordered in the revision).
    state = _state([
        {"name": "A", "spectrum_indices": [0, 1, 2]},
        {"name": "B", "spectrum_indices": [3, 4, 5]},
    ], num=6)
    _extract(state, _plan({"name": "B"}, {"name": "A"}))
    plan = state["series_analysis_plan"]
    print("2) reordered revision, names stable:")
    check("B keeps its own indices despite position 0",
          plan["regimes"][0]["name"] == "B"
          and plan["regimes"][0]["spectrum_indices"] == [3, 4, 5])
    check("A keeps its own indices despite position 1",
          plan["regimes"][1]["spectrum_indices"] == [0, 1, 2])

    # 3) Explicit indices from the revision always win over inheritance.
    state = _state([
        {"name": "A", "spectrum_indices": [0, 1, 2]},
        {"name": "B", "spectrum_indices": [3, 4, 5]},
    ], num=6)
    _extract(state, _plan(
        {"name": "A", "spectrum_indices": [0, 1]},
        {"name": "B", "spectrum_indices": [2, 3, 4, 5]},
    ))
    plan = state["series_analysis_plan"]
    print("3) explicit revised indices:")
    check("revised assignment respected",
          plan["regimes"][0]["spectrum_indices"] == [0, 1]
          and plan["regimes"][1]["spectrum_indices"] == [2, 3, 4, 5])

    # 4) Partial omission: only the omitting regime inherits.
    state = _state([
        {"name": "A", "spectrum_indices": [0, 1, 2]},
        {"name": "B", "spectrum_indices": [3, 4, 5]},
    ], num=6)
    _extract(state, _plan(
        {"name": "A", "spectrum_indices": [0, 1, 2, 3]},
        {"name": "B"},
    ))
    plan = state["series_analysis_plan"]
    print("4) partial omission:")
    check("explicit regime kept, omitted regime inherited",
          plan["regimes"][0]["spectrum_indices"] == [0, 1, 2, 3]
          and plan["regimes"][1]["spectrum_indices"] == [3, 4, 5])

    # 5) No prior plan (initial planning): behavior unchanged — missing
    #    indices fall back to regime 1, empty second regime is dropped.
    state = _state(num=6)
    _extract(state, _plan({"name": "A"}, {"name": "B"}))
    plan = state["series_analysis_plan"]
    print("5) initial planning without indices (pre-existing fallback):")
    check("collapses to 1 regime as before",
          plan is not None and len(plan["regimes"]) == 1
          and plan["regimes"][0]["spectrum_indices"] == [0, 1, 2, 3, 4, 5])

    # 6) Count changed AND names changed: no safe match -> fallback as before
    #    (a deliberate restructure is not second-guessed).
    state = _state([
        {"name": "A", "spectrum_indices": [0, 1, 2]},
        {"name": "B", "spectrum_indices": [3, 4, 5]},
    ], num=6)
    _extract(state, _plan({"name": "X"}, {"name": "Y"}, {"name": "Z"}))
    plan = state["series_analysis_plan"]
    print("6) restructured revision (3 new names, no indices):")
    check("no positional guess across counts; fallback applies",
          plan is not None and len(plan["regimes"]) == 1
          and plan["regimes"][0]["name"] == "X")

    # 7) Compact index formatting for the validation prompt.
    fmt = CurveFittingPlanningController._format_spectrum_indices
    print("7) index formatting:")
    check("ranges", fmt(list(range(0, 12))) == "0-11")
    check("mixed", fmt([0, 1, 2, 5, 7, 8]) == "0-2, 5, 7-8")
    check("single", fmt([4]) == "4")
    check("empty", fmt([]) == "")

    # 8) Validation-prompt regime section carries the indices + semantics.
    section = CurveFittingPlanningController._build_regime_section(
        {"regimes": [
            {"name": "A", "spectrum_indices": list(range(0, 12)),
             "physical_model": "PV", "parameters_to_extract": ["center"]},
            {"name": "B", "spectrum_indices": list(range(12, 38)),
             "physical_model": "PV", "parameters_to_extract": ["center"]},
        ]})
    print("8) validation regime section:")
    check("shows indices", "spectra=[0-11]" in section
          and "spectra=[12-37]" in section)
    check("states omission semantics", "inherits" in section)
    check("empty plan -> empty section",
          CurveFittingPlanningController._build_regime_section(None) == "")

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"SERIES PLAN REVISION: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
