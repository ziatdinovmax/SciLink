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
    # Component-LAST (H, W, n) — the package convention
    # (run_spectral_unmixing output / reconstruct_cube contract).
    amaps = np.stack([np.random.rand(h, w) * 0.05,
                      blob * 1.0 + np.random.rand(h, w) * 0.05], axis=-1)
    # Index contract is the 1-based "Component N" plot label: the blob
    # lives in the second map -> "Component 2".
    m = _build_fit_mask(amaps, 2, (h, w), LOG)
    check("mask covers the blob", m is not None and bool(m[blob].all()))
    ring = ((xx - 30) ** 2 + (yy - 40) ** 2 < 11 ** 2) & ~blob
    check("mask is DILATED beyond the blob (boundary halo kept)",
          m is not None and m[ring].mean() > 0.9)
    check("mask stays a small fraction of the frame",
          m is not None and m.mean() < 0.25)
    print("3) mask builder fallbacks:")
    check("bad component index -> None", _build_fit_mask(amaps, 7, (h, w), LOG) is None)
    check("no maps -> None", _build_fit_mask(None, 0, (h, w), LOG) is None)
    check("non-integer shape mismatch -> None",
          _build_fit_mask(amaps, 2, (150, 130), LOG) is None)
    check("raw smaller than the abundance grid -> None",
          _build_fit_mask(amaps, 2, (50, 50), LOG) is None)

    # Binned-decomposition upsampling: abundance maps at (h, w) from a
    # spatial_bin_factor=2 preprocessing, raw cube at (2h, 2w). The mask
    # must come back at RAW scale, still covering the (scaled) blob.
    print("3b) binned-decomposition mask upsampling:")
    m2 = _build_fit_mask(amaps, 2, (2 * h, 2 * w), LOG)
    check("integer-factor mismatch upsamples instead of bailing",
          m2 is not None and m2.shape == (2 * h, 2 * w))
    blob_up = np.repeat(np.repeat(blob, 2, axis=0), 2, axis=1)
    check("upsampled mask covers the blob at raw scale",
          m2 is not None and bool(m2[blob_up].all()))
    check("upsampled mask keeps the dilation halo (strictly wider than "
          "the blob)", m2 is not None and m2.sum() > blob_up.sum())
    check("upsampled mask stays a small fraction of the frame",
          m2 is not None and m2.mean() < 0.25)
    m1 = _build_fit_mask(amaps, 2, (h, w), LOG)
    check("same-scale coverage fraction preserved by upsampling",
          m1 is not None and m2 is not None
          and abs(m1.mean() - m2.mean()) < 1e-9)
    m3 = _build_fit_mask(amaps, 2, (3 * h, 3 * w), LOG)
    check("factor-3 binning also upsamples",
          m3 is not None and m3.shape == (3 * h, 3 * w))
    flat = np.ones((h, w, 2))
    check("degenerate (all-equal) abundance -> None",
          _build_fit_mask(flat, 1, (h, w), LOG) is None)
    amaps_cf = np.moveaxis(amaps, -1, 0)          # (n, H, W) legacy stack
    m_cf = _build_fit_mask(amaps_cf, 2, (h, w), LOG)
    check("component-FIRST stack accepted defensively (same mask)",
          m_cf is not None and np.array_equal(m_cf, m))

    print("3c) component-index convention (1-based plot labels):")
    m_lbl = _build_fit_mask(amaps, 2, (h, w), LOG)
    check("'Component 2' label selects the second map (blob found)",
          m_lbl is not None and bool(m_lbl[blob].all()))
    m_first = _build_fit_mask(amaps, 1, (h, w), LOG)
    check("'Component 1' selects the flat first map -> None (no footprint)",
          m_first is None)
    m_zero = _build_fit_mask(amaps, 0, (h, w), LOG)
    check("0 tolerated as the first component (0-based habit)",
          (m_zero is None) == (m_first is None))
    check("one past the last component -> None",
          _build_fit_mask(amaps, 3, (h, w), LOG) is None)

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

    # 5) Honest-null contract (#358 follow-up): prompt + judged declaration.
    print("5) measurability gate / honest null:")
    p = build_code_generation_prompt("t", 96, 96, 180, "nm", 400.0, 900.0,
                                     "raw")
    check("codegen prompt carries the MEASURABILITY GATE contract",
          "MEASURABILITY GATE" in p and "not_measurable" in p
          and "evidence" in p)
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        RunDynamicAnalysisController)

    class _Resp:
        def __init__(self, obj): self._o = obj
        text = ""

    class _Model:
        def __init__(self, verdict): self.verdict = verdict
        def generate_content(self, prompt, generation_config=None,
                             safety_settings=None):
            if isinstance(self.verdict, Exception):
                raise self.verdict
            return _Resp(self.verdict)

    def _parse(resp):
        return resp._o, None

    class _Ctx:
        session = {"flux_table": "band | flux\n400-500 | 100.0"}
        mean_spec_bytes = None

    def _mk(verdict):
        c = object.__new__(RunDynamicAnalysisController)
        c.model = _Model(verdict)
        c.logger = LOG
        c.generation_config = None
        c.safety_settings = None
        c._parse_llm_response = _parse
        return c

    nm = {"feature": "two peaks", "evidence": "prominence 0.08 vs sigma 0.4",
          "description": "flat field"}
    ok, _ = _mk({"defensible": True, "critique": "confirmed"})\
        ._judge_not_measurable(nm, _Ctx())
    check("defensible null accepted", ok is True)
    ok, crit = _mk({"defensible": False, "critique": "flux table shows a band"})\
        ._judge_not_measurable(nm, _Ctx())
    check("indefensible null rejected with critique",
          ok is False and "band" in crit)
    ok, crit = _mk(RuntimeError("api down"))._judge_not_measurable(nm, _Ctx())
    check("judge crash fails CLOSED (reject, retry)",
          ok is False and "unavailable" in crit)
    ok, _ = _mk({"something": "else"})._judge_not_measurable(nm, _Ctx())
    check("unusable verdict fails CLOSED", ok is False)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"DECOMP GATE: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
