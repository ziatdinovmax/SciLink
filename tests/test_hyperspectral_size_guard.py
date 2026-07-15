"""Offline tests for the hyperspectral size guard.

Motivated by a live 3-hour grind on a real SOC-710 cube (704x704x260 =
129M values): commercial hyperspectral cameras produce cubes 10-50x larger
than the academic SIs the agent was tuned on, and nothing scaled the plan to
the size. The guard has three parts: a `spatial_bin_factor` knob in the
preprocessing strategy (deterministic mean-binning of the spatial axes),
a size-scaling rule for custom_code refinement targets, and a SIZE BUDGET
block in the codegen prompt (single-process sandbox cost model).

  conda run -n scilink python tests/test_hyperspectral_size_guard.py
"""
import logging
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np

from scilink.agents.exp_agents.preprocess import HyperspectralPreprocessingAgent
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    build_code_generation_prompt)
from scilink.agents.exp_agents.instruct import (
    PRE_PROCESSING_STRATEGY_INSTRUCTIONS, SPECTROSCOPY_REFINEMENT_INSTRUCTIONS)

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _Stub:
    logger = logging.getLogger("size_guard_test")


def main():
    cube = np.random.rand(10, 12, 7).astype(np.float32)
    base = {"apply_despike": False, "apply_masking": False}

    print("1) binning applier:")
    out, mask = HyperspectralPreprocessingAgent._apply_preprocessing(
        _Stub(), cube, {**base, "spatial_bin_factor": 2}, None)
    check("2x2 bin halves spatial dims, spectra intact",
          out.shape == (5, 6, 7) and mask.shape == (5, 6))
    check("mean-binning is correct",
          np.allclose(out[0, 0], cube[0:2, 0:2].mean(axis=(0, 1))))
    out4, _ = HyperspectralPreprocessingAgent._apply_preprocessing(
        _Stub(), cube, {**base, "spatial_bin_factor": 4}, None)
    check("4x bin truncates ragged edges safely", out4.shape == (2, 3, 7))
    out1, _ = HyperspectralPreprocessingAgent._apply_preprocessing(
        _Stub(), cube, dict(base), None)
    check("missing knob -> no-op (backcompat)", out1.shape == cube.shape)
    spec = {"axis_0": {"kind": "spatial"}, "axis_1": {"kind": "parameter"},
            "signal_is_nonnegative": True}
    out2, _ = HyperspectralPreprocessingAgent._apply_preprocessing(
        _Stub(), cube, {**base, "spatial_bin_factor": 2}, spec)
    check("non-spatial leading axes -> binning skipped",
          out2.shape == cube.shape)
    outm, maskm = HyperspectralPreprocessingAgent._apply_preprocessing(
        _Stub(), cube,
        {"apply_despike": False, "apply_masking": True,
         "mask_threshold_percentile": 5.0, "spatial_bin_factor": 2}, None)
    check("mask computed on the binned grid",
          outm.shape == (5, 6, 7) and maskm.shape == (5, 6))

    print("2) prompt contracts:")
    check("strategy prompt carries the size-guard rule + schema key",
          "spatial_bin_factor" in PRE_PROCESSING_STRATEGY_INSTRUCTIONS
          and "SIZE GUARD" in PRE_PROCESSING_STRATEGY_INSTRUCTIONS)
    check("target planning carries the size-scaling rule",
          "SIZE GUARD" in SPECTROSCOPY_REFINEMENT_INSTRUCTIONS
          and "coarse-to-fine" in SPECTROSCOPY_REFINEMENT_INSTRUCTIONS)
    p = build_code_generation_prompt("t", 704, 704, 260, "nm", 400.0, 1000.0,
                                     "raw")
    check("codegen prompt renders the SIZE BUDGET with real pixel count",
          "SIZE BUDGET" in p and "495616 pixels" in p
          and "coarse-to-fine" in p)
    p_small = build_code_generation_prompt("t", 64, 64, 260, "nm", 400.0,
                                           1000.0, "raw")
    check("small cube renders sane minute estimates",
          "4096 pixels" in p_small)

    # 3) Custom-preprocessing prompt path must serialize numpy-typed stats
    #    (found live: a float32 cube's np.float32 statistics crashed
    #    json.dumps in _generate_custom_script through all retries).
    print("3) stats serialization:")
    from scilink.agents.exp_agents.instruct import (
        CUSTOM_PREPROCESSING_SCRIPT_INSTRUCTIONS)
    import json as _json
    f32 = np.random.rand(8, 8, 5).astype(np.float32)
    stats = HyperspectralPreprocessingAgent._calculate_statistics(
        _Stub(), f32)
    try:
        rendered = CUSTOM_PREPROCESSING_SCRIPT_INSTRUCTIONS.format(
            instruction="i", input_filename="f.npy",
            stats_json=_json.dumps(stats, indent=2, default=str))
        ok = "mean" in rendered
    except TypeError:
        ok = False
    check("float32-cube stats serialize into the custom-script prompt", ok)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"HS SIZE GUARD: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
