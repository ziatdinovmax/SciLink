"""Offline tests for the judgment-calibration principles (post-rerun
hardening): salvage-judge localization rubric + footprint evidence, the
footprint-adaptive measurability gate, planner required-outputs scope
calibration, and the curve verifier's claim-anchored residual check.

Motivated by two live failures on one localized-emitter cube: a false null
judged from field-mean statistics (a 0.2%-area emitter is invisible in any
field mean AND inside a fixed top-5% bright mean), and a doublet claim whose
own residual panel contradicted it.

  conda run -n scilink python tests/test_judgment_calibration.py
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def extra_absence_validator_checks():
    """Section 5 (frozen-shape absence contract) — run via main2()."""
    from scilink.skills._shared.curve_fitting_tools import (
        validate_absent_component_contract as V, ABSENT_COMPONENT_FIX)
    print("5) absence-as-value validator:")
    check("compliant absent component passes",
          V({"c": {"c_absent": True, "area": 0.5, "area_err": 0.4,
                   "center": None}}) == [])
    bad = V({"c": {"c_absent": True, "area": None, "center": None}})
    check("empty amplitude on flagged component -> violation with fix cue",
          bool(bad) and "MEASURED frozen-shape amplitude" in bad[0])
    check("sibling-flag form detected",
          bool(V({"c": {"area": float("nan")}, "c_absent": True})))
    check("non-absent components untouched",
          V({"p": {"area": None, "center": 1052.0}}) == [])
    check("junk input never raises", V(None) == [] and V("x") == [])
    check("fix prescription is form-agnostic",
          "FROZEN" in ABSENT_COMPONENT_FIX and "amplitude" in ABSENT_COMPONENT_FIX)


def main():
    from scilink.agents.exp_agents.instruct import (
        SPECTROSCOPY_SALVAGE_JUDGE_INSTRUCTIONS as S,
        SPECTROSCOPY_REFINEMENT_INSTRUCTIONS as R,
    )
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        _footprint_evidence, build_code_generation_prompt)

    print("1) salvage judge:")
    check("localized-signals principle + spatial evidence slot",
          "LOCALIZED SIGNALS" in S and "{spatial_evidence}" in S)
    check("verdict must name the region statistic",
          "NAME the region statistic" in S)

    h = w = 100
    blob = np.zeros((h, w)); blob[40:46, 30:36] = 1.0
    ev = _footprint_evidence({"final_abundance_maps":
                              np.stack([np.random.rand(h, w) * .05 + 1, blob],
                                       axis=-1)})
    check("compact component tagged LOCALIZED with footprint fraction",
          "LOCALIZED" in ev and "0.36%" in ev and "Component 2" in ev)
    check("frame-scale component NOT tagged localized",
          "Component 1" in ev and "Component 1" in
          [l for l in ev.splitlines() if "LOCALIZED" not in l][0])
    check("no decomposition -> honest placeholder",
          _footprint_evidence({}) == "(no decomposition ran)")
    check("garbage maps never raise",
          isinstance(_footprint_evidence({"final_abundance_maps": "x"}), str))

    print("2) measurability gate (codegen contract):")
    p = build_code_generation_prompt("t", 96, 96, 90, "nm", 400.0, 900.0, "raw")
    check("bright-region test is footprint-adaptive (no fixed top-5%)",
          "AT THE SIGNAL'S OWN" in p and "walk the bright fraction DOWN" in p
          and "top 1-5%" not in p)

    print("3) planner demand calibration:")
    check("scope principle present",
          "Demand outputs at the scope the evidence supports" in R)
    check("weak/localized -> region-integrated required output",
          "REGION-INTEGRATED" in R and "best-effort" in R)
    check("per-pixel phrasing does not override physics",
          "does not override physics" in R)

    print("4) curve verifier claim anchoring:")
    from scilink.agents.exp_agents.instruct import (
        FITTING_INTERPRETATION_STAGE1 as T)
    check("claim-anchored residual check in Stage 1",
          "CLAIM-ANCHORED RESIDUAL CHECK" in T
          and "fit artifact" in T and "no resolved splitting" in T)
    check("R2 explicitly disqualified as arbiter", "cannot arbitrate" in T)

    extra_absence_validator_checks()

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"JUDGMENT CALIBRATION: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()

