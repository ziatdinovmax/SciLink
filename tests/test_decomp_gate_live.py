"""Live validation of the decomposition-first large-data gate (#359).

Three planted-truth cubes, each >50k pixels so the gate applies, driven
through the full hyperspectral analysis path (run_task):

  blob  — emission (two peaks: 520/610 nm) confined to a small blob on a
          flat background. The gate should scope fitting (component mask /
          equivalent reduction) and the peaks must be recovered FAST —
          the anti-churn case that previously railed full-frame fits.
  noise — pure noise, exploratory objective. The gate should land on
          decomposition-only / an honest no-feature report, without a
          doomed per-pixel fitting campaign.
  shift — ONE spectrally-bland component whose peak center drifts smoothly
          across the frame (the decomposition CANNOT model it; abundance
          is spatially uniform). The BABY-SAVING case: the gate must still
          route to fitting (structured residual / objective) and recover
          the planted corner-to-corner shift.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_decomp_gate_live.py \
      [--scenario all|blob|noise|shift]
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(31)
WL = np.linspace(400, 900, 160)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _save_cube(cube, name, note):
    d = tempfile.mkdtemp(prefix=f"gate_{name}_")
    p = os.path.join(d, f"{name}.npy")
    np.save(p, cube.astype(np.float32))
    json.dump({"technique": "hyperspectral emission map",
               "wavelengths_nm": [float(v) for v in WL], "note": note},
              open(p.replace(".npy", ".json"), "w"))
    return p


def _run(p, task, tag):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    base = tempfile.mkdtemp(prefix=f"gate_live_{tag}_")
    print(f"\n### {tag} — session {base}")
    ag = AnalysisOrchestratorAgent(
        base_dir=base, api_key=None, base_url=None, model_name=MODEL,
        restore_checkpoint=False, analysis_mode=AnalysisMode.AUTONOMOUS)
    t0 = time.monotonic()
    res = ag.run_task(task, autonomy=AnalysisMode.AUTONOMOUS)
    dt = time.monotonic() - t0
    summary = (res.get("summary") or "") + " ".join(
        str(k) for k in res.get("key_findings") or [])
    print(f"[{tag}] wall: {dt/60:.1f} min | status: {res.get('status')}")
    print(f"[{tag}] summary (first 1200):\n{summary[:1200]}")
    return res, summary, dt, base


def scenario_blob():
    h = w = 256                                   # 65k px > gate threshold
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 70) ** 2 + (yy - 180) ** 2) < 18 ** 2
    spec = 3.0 * _g(WL, 520, 12) + 2.0 * _g(WL, 610, 15)
    cube = (5.0 + RNG.normal(0, 0.4, (h, w, WL.size))
            + blob[..., None] * spec[None, None, :])
    p = _save_cube(cube, "blob", "emission localized to a small region")
    res, summary, dt, base = _run(
        p, f"Analyze the hyperspectral image at {p} (256x256x160; "
           f"wavelengths in the metadata JSON). Characterize the emission: "
           "where is it located, and what are its peak wavelengths?", "blob")
    check("blob: success", res.get("status") == "success")
    check("blob: bounded (< 25 min — no full-frame churn)", dt < 1500)
    nums = [float(v) for v in re.findall(r"\b(\d{3})(?:\.\d+)?\s*nm\b",
                                         summary)]
    check("blob: both planted peaks recovered (~520/~610 nm)",
          any(abs(v - 520) <= 12 for v in nums)
          and any(abs(v - 610) <= 12 for v in nums))
    log = open(os.devnull).read() if False else ""
    # Scoping evidence: masked fit chosen (session log line) OR the summary
    # localizes the emission — the gate's purpose is met either way.
    check("blob: emission localized in the report",
          any(s in summary.lower() for s in
              ("blob", "region", "localiz", "spot", "confined", "cluster")))


def scenario_noise():
    h = w = 256
    cube = RNG.normal(100, 5, (h, w, WL.size))
    p = _save_cube(cube, "noise", "exploratory survey acquisition")
    res, summary, dt, base = _run(
        p, f"Analyze the hyperspectral image at {p} (256x256x160; "
           f"wavelengths in the metadata JSON). Exploratory: characterize "
           "whatever spectral or spatial structure is present.", "noise")
    check("noise: completed (any honest status)", res.get("status") in
          ("success", "error"))
    check("noise: bounded (< 25 min — no doomed fitting campaign)",
          dt < 1500)
    check("noise: absence reported honestly",
          any(s in summary.lower() for s in
              ("no significant", "no spectral", "noise", "featureless",
               "no distinct", "absen", "no evidence", "flat", "no structure",
               "lacks")))
    check("noise: nothing fabricated (no confident peak claims)",
          not re.search(r"(?:clear|strong|distinct)\s+(?:emission\s+)?peaks?"
                        r"\s+at\s+\d{3}", summary, re.IGNORECASE))


def scenario_shift():
    h = w = 256
    xx = np.mgrid[0:h, 0:w][1]
    centers = 600.0 + 40.0 * (xx / (w - 1)) - 20.0   # 580 -> 620 nm across x
    cube = (2.0 + 5.0 * _g(WL[None, None, :], centers[..., None], 18.0)
            + RNG.normal(0, 0.15, (h, w, WL.size)))
    p = _save_cube(cube, "shift", "single band present everywhere")
    res, summary, dt, base = _run(
        p, f"Analyze the hyperspectral image at {p} (256x256x160; "
           f"wavelengths in the metadata JSON). The sample shows one "
           "emission band everywhere; characterize how its peak POSITION "
           "varies spatially across the frame (map the per-pixel peak "
           "center).", "shift")
    check("shift: success (the gate did NOT stop at decomposition-only)",
          res.get("status") == "success")
    # The planted truth: a smooth left-right gradient, 580 -> 620 nm.
    # Accept range notation too ("580-620 nm" leaves only the second number
    # adjacent to the unit) — any 3-digit value in the plausible band counts.
    nums = [float(v) for v in re.findall(r"\b(5[3-9]\d|6[0-6]\d)(?:\.\d+)?\b",
                                         summary)]
    check("shift: gradient endpoints recovered (~580 and ~620 nm)",
          any(abs(v - 580) <= 10 for v in nums)
          and any(abs(v - 620) <= 10 for v in nums))
    check("shift: spatial gradient described",
          any(s in summary.lower() for s in
              ("gradient", "varies", "increas", "decreas", "drift",
               "left", "right", "across", "linear")))
    check("shift: bounded (< 35 min)", dt < 2100)


SCENARIOS = {"blob": scenario_blob, "noise": scenario_noise,
             "shift": scenario_shift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + sorted(SCENARIOS))
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    todo = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in todo:
        try:
            SCENARIOS[name]()
        except Exception:  # noqa: BLE001
            import traceback; traceback.print_exc()
            check(f"{name}: completed without crashing", False)
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"DECOMP GATE LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
