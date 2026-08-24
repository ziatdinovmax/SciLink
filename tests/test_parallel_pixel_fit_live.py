"""Live validation of the fit_per_pixel harness (#356).

One scenario, the case the size guard steps aside for: a large cube
(400x400x140 ~ 160k pixels) whose objective legitimately requires NATIVE
single-pixel resolution — a 2-pixel-wide defect stripe with a shifted
emission line, unresolvable after binning. The agent must produce per-pixel
peak-position maps in minutes, and the generated script must CALL the vetted
fit_per_pixel tool rather than hand-rolling a serial loop (verified from the
saved script, not the result dict).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_parallel_pixel_fit_live.py
"""
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(11)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)

    # --- planted-truth cube: smooth center field + 2px defect stripe -------
    h = w = 400
    wl = np.linspace(450, 850, 140)
    yy, xx = np.mgrid[0:h, 0:w]
    centers = 610.0 + 15.0 * np.sin(2 * np.pi * xx / w) \
        + 10.0 * (yy / h - 0.5)
    stripe = (np.abs(yy - (0.31 * h + 0.05 * xx)) < 1.0)   # ~2 px wide, tilted
    centers = centers + stripe * 12.0
    spec = 4.0 * np.exp(
        -0.5 * ((wl[None, None, :] - centers[..., None]) / 16.0) ** 2)
    cube = (5.0 + spec + RNG.normal(0, 0.35, (h, w, wl.size))
            ).astype(np.float32)

    d = tempfile.mkdtemp(prefix="ppf_live_")
    p = os.path.join(d, "emission_map.npy")
    np.save(p, cube)
    json.dump({"technique": "hyperspectral emission map",
               "energy_range": {"start": 450.0, "end": 850.0, "units": "nm"}},
              open(p.replace(".npy", ".json"), "w"))

    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    out = os.path.join(d, "out")
    ag = HyperspectralAnalysisAgent(api_key=None, model_name=MODEL,
                                    output_dir=out,
                                    enable_human_feedback=False)
    print(f"### ppf_live — {d} ({h}x{w}x{wl.size} = {h*w} pixels)")
    t0 = time.monotonic()
    res = ag.analyze(
        p, system_info=p.replace(".npy", ".json"),
        objective=(
            "Map the emission peak position at every pixel at NATIVE "
            "single-pixel resolution. Do NOT spatially bin or subsample: "
            "the sample is suspected to contain a defect line only 1-2 "
            "pixels wide with a shifted emission wavelength, and it must "
            "be resolved. Report Peak_Position (nm) per pixel with its "
            "uncertainty."))
    dt = time.monotonic() - t0
    print(f"[ppf_live] wall: {dt/60:.1f} min | status: {res.get('status')}")

    check("run completed", res.get("status") in ("success", "partial"))
    check("bounded: full-frame native-res map in < 30 min", dt < 1800)

    # Verify tool adoption from the SAVED scripts, not the result dict.
    scripts = []
    for pat in ("**/*.py",):
        scripts += glob.glob(os.path.join(out, pat), recursive=True)
    blob = ""
    for s in scripts:
        try:
            blob += open(s).read()
        except OSError:
            pass
    called = "fit_per_pixel(" in blob
    check("generated script CALLS fit_per_pixel (no hand-rolled loop)",
          called)
    if not called:
        print("    (scripts found:", scripts, ")")

    feats = res.get("extracted_features") or {}
    ftxt = json.dumps(feats, default=str).lower()
    check("per-pixel peak-position output produced",
          "peak" in ftxt or "position" in ftxt or "center" in ftxt)
    txt = (res.get("detailed_analysis") or "").lower()
    check("defect stripe resolved in the interpretation",
          any(k in txt for k in ("stripe", "line", "defect", "narrow",
                                 "linear feature", "diagonal")))

    print(f"\noutput dir: {out}")
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"PARALLEL PIXEL FIT LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
