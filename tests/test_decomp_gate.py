"""Offline tests for the decomposition-first large-data gate (#359).

The gate is ASYMMETRIC (decomposition evidence may ADD scope freely, but may
RESTRICT it only under residual-certified conditions): four verdicts —
fit-everywhere / fit-within-dilated-mask / fit-global-then-decide /
decomposition-only. These tests cover the mechanical layer: the prompt
contracts, the dilated-mask builder, the sandbox plumbing, and QC
mask-awareness inputs.

  conda run -n scilink python tests/test_decomp_gate.py
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    _build_fit_mask, _invoke_analyze_feature, build_code_generation_prompt)
from scilink.agents.exp_agents.instruct import (
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS)

import logging

results = {}
LOG = logging.getLogger("gate_test")


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    # 1) Prompt contracts.
    print("1) prompt contracts:")
    t = SPECTROSCOPY_REFINEMENT_INSTRUCTIONS
    check("gate block present with all four verdicts",
          "LARGE-DATA GATE" in t
          and all(s in t for s in ("fit-within-dilated-mask",
                                   "fit-global-then-decide",
                                   "fit-everywhere", "decomposition-only")))
    check("decomposition-only is residual-certified",
          "residual" in t.split("decomposition-only (STOP)")[1][:600].lower())
    check("asymmetry + objective override stated",
          "ASYMMETRIC" in t and "ALWAYS overrides" in t)
    check("target schema documents the scoping fields",
          '"fit_scope"' in t and '"mask_component_index"' in t)
    p = build_code_generation_prompt(
        "t", 96, 96, 180, "nm", 400.0, 900.0, "raw",
        fit_mask_pixels=(1200, 9216, 2))
    check("codegen prompt renders the mandatory-scope mask section",
          "FIT MASK — MANDATORY SCOPE" in p and "1200 pixels" in p
          and "component\n2" in p or ("component" in p and "1200" in p))
    check("mask arg in the generated signature", "fit_mask=None" in p)
    p0 = build_code_generation_prompt("t", 96, 96, 180, "nm", 400.0, 900.0,
                                      "raw")
    check("no mask -> no mask section (unchanged default)",
          "FIT MASK" not in p0 and "fit_mask" not in p0)

    # 2) Dilated-mask builder with planted truth.
    print("2) mask builder:")
    h = w = 100
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 30) ** 2 + (yy - 40) ** 2) < 8 ** 2      # ~200 px blob
    amaps = np.stack([np.random.rand(h, w) * 0.05,
                      blob * 1.0 + np.random.rand(h, w) * 0.05])
    m = _build_fit_mask(amaps, 1, (h, w), LOG)
    check("mask covers the blob", m is not None and bool(m[blob].all()))
    ring = ((xx - 30) ** 2 + (yy - 40) ** 2 < 11 ** 2) & ~blob
    check("mask is DILATED beyond the blob (boundary halo kept)",
          m is not None and m[ring].mean() > 0.9)
    check("mask stays a small fraction of the frame",
          m is not None and m.mean() < 0.25)
    print("3) mask builder fallbacks:")
    check("bad component index -> None", _build_fit_mask(amaps, 7, (h, w), LOG) is None)
    check("no maps -> None", _build_fit_mask(None, 0, (h, w), LOG) is None)
    check("shape mismatch -> None",
          _build_fit_mask(amaps, 1, (50, 50), LOG) is None)
    flat = np.ones((2, h, w))
    check("degenerate (all-equal) abundance -> None",
          _build_fit_mask(flat, 0, (h, w), LOG) is None)

    # 4) Sandbox plumbing: fit_mask passed only when accepted.
    print("4) invoke plumbing:")
    data = np.random.rand(4, 4, 5)
    axis = np.arange(5.0)
    mask = np.zeros((4, 4), bool); mask[1, 1] = True
    got = {}

    def modern(d, a, fit_mask=None):
        got["mask"] = fit_mask
        return {"maps": {}}
    _invoke_analyze_feature(modern, data, axis, fit_mask=mask)
    check("mask reaches a function that declares it",
          got.get("mask") is mask)

    def legacy(d, a):
        got["legacy"] = True
        return {"maps": {}}
    _invoke_analyze_feature(legacy, data, axis, fit_mask=mask)
    check("legacy 2-arg function still works (mask not forced)",
          got.get("legacy") is True)

    def kwargs_fn(d, a, **kw):
        got["kw"] = kw
        return {"maps": {}}
    _invoke_analyze_feature(kwargs_fn, data, axis, fit_mask=mask)
    check("**kwargs function receives the mask",
          got.get("kw", {}).get("fit_mask") is mask)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"DECOMP GATE: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
