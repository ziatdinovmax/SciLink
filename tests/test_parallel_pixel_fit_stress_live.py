"""Live stress matrix for the fit_per_pixel harness (#356).

  mask_comp — localized emission blob on a large frame: the decomposition
      gate should RESTRICT scope, and the per-pixel fit should compose with
      it (mask path) rather than paying for the whole frame. Checks honest
      localization + peak wavelength.
  multipeak — two bands (fixed 520 nm + drifting ~610 nm) with a per-pixel
      ratio demanded at native resolution: exercises a multi-component
      declarative spec through the tool.
  noise_demand — pure noise, two per-pixel peak maps DEMANDED at native
      resolution. Now that full-frame fitting is cheap, the trap is fitting
      the noise and returning plausible maps; the measurability gate must
      still win with a judged honest null.
  fanout_thread — a fit_per_pixel-scale branch running CONCURRENTLY with a
      curve-series branch inside a meta fan-out (branches are threads): the
      loky-from-thread path in its real production context.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_parallel_pixel_fit_stress_live.py \
      [--scenario all|mask_comp|multipeak|noise_demand|fanout_thread]
"""
import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(17)
WL = np.linspace(450, 850, 140)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _meta(path):
    json.dump({"technique": "hyperspectral emission map",
               "energy_range": {"start": 450.0, "end": 850.0, "units": "nm"}},
              open(path.replace(".npy", ".json"), "w"))


def _scripts_blob(out_dir):
    blob = ""
    for s in glob.glob(os.path.join(out_dir, "**", "*.py"), recursive=True):
        try:
            blob += open(s).read()
        except OSError:
            pass
    return blob


def _hs_agent(out_dir):
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    return HyperspectralAnalysisAgent(api_key=None, model_name=MODEL,
                                      output_dir=out_dir,
                                      enable_human_feedback=False)


def scenario_mask_comp():
    h = w = 350
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 90) ** 2 + (yy - 250) ** 2) < 18 ** 2
    spec = 4.0 * np.exp(-0.5 * ((WL - 620) / 15) ** 2)
    cube = (5.0 + blob[..., None] * spec[None, None, :]
            + RNG.normal(0, 0.35, (h, w, WL.size))).astype(np.float32)
    d = tempfile.mkdtemp(prefix="ppf_mask_")
    p = os.path.join(d, "emission.npy")
    np.save(p, cube); _meta(p)
    out = os.path.join(d, "out")
    print(f"\n### mask_comp — {d} ({h}x{w} = {h*w} px, blob ~1000 px)")
    t0 = time.monotonic()
    res = _hs_agent(out).analyze(
        p, system_info=p.replace(".npy", ".json"),
        objective="Characterize the localized emission in this map: where "
                  "does the sample emit, and what is the per-pixel peak "
                  "wavelength within the emitting region?")
    dt = time.monotonic() - t0
    print(f"[mask_comp] wall: {dt/60:.1f} min | status: {res.get('status')}")
    check("mask_comp: completed", res.get("status") in ("success", "partial"))
    check("mask_comp: bounded (< 25 min)", dt < 1500)
    txt = ((res.get("detailed_analysis") or "") + json.dumps(
        res.get("extracted_features") or {}, default=str)).lower()
    check("mask_comp: peak wavelength ~620 nm reported",
          any(abs(float(v) - 620) <= 8 for v in
              re.findall(r"\b(6[0-3]\d)(?:\.\d+)?\b", txt)))
    check("mask_comp: emission localized (region/blob/spot language + no "
          "claim of frame-wide emission)",
          any(k in txt for k in ("localized", "region", "blob", "spot",
                                 "confined", "small area", "circular")))
    blob_txt = _scripts_blob(out)
    scoped = ("mask=" in blob_txt and "fit_per_pixel(" in blob_txt) or \
             ("fit_per_pixel(" not in blob_txt)
    print(f"    (script used fit_per_pixel: {'fit_per_pixel(' in blob_txt}, "
          f"mask-scoped: {'mask=' in blob_txt})")
    check("mask_comp: cost honest — masked tool call OR no full-frame "
          "tool loop", scoped or dt < 600)


def scenario_multipeak():
    h = w = 300
    centers2 = 595.0 + 30.0 * (np.arange(w) / (w - 1))     # drifts 595->625
    s1 = 3.0 * np.exp(-0.5 * ((WL[None, :] - 520.0) / 12) ** 2)
    s2 = 4.0 * np.exp(-0.5 * ((WL[None, :] - centers2[:, None]) / 14) ** 2)
    cube = (5.0 + np.broadcast_to(s1[None], (h, w, WL.size))
            + np.broadcast_to(s2[None], (h, w, WL.size))
            + RNG.normal(0, 0.3, (h, w, WL.size))).astype(np.float32)
    d = tempfile.mkdtemp(prefix="ppf_multi_")
    p = os.path.join(d, "twoband.npy")
    np.save(p, cube); _meta(p)
    out = os.path.join(d, "out")
    print(f"\n### multipeak — {d} ({h*w} px)")
    t0 = time.monotonic()
    res = _hs_agent(out).analyze(
        p, system_info=p.replace(".npy", ".json"),
        objective="Two emission bands are present (near 520 nm and near "
                  "610 nm). Map BOTH per-pixel peak positions and the "
                  "per-pixel intensity ratio of the two bands at NATIVE "
                  "single-pixel resolution — do not spatially bin.")
    dt = time.monotonic() - t0
    print(f"[multipeak] wall: {dt/60:.1f} min | status: {res.get('status')}")
    check("multipeak: completed", res.get("status") in ("success", "partial"))
    check("multipeak: bounded (< 30 min)", dt < 1800)
    txt = ((res.get("detailed_analysis") or "") + json.dumps(
        res.get("extracted_features") or {}, default=str)).lower()
    vals = [float(v) for v in re.findall(r"\b(\d{3})(?:\.\d+)?\s*nm\b", txt)]
    check("multipeak: band 1 (~520 nm) recovered",
          any(abs(v - 520) <= 8 for v in vals))
    check("multipeak: band 2 (~595-625 nm drift) recovered",
          any(590 <= v <= 630 for v in vals))
    blob_txt = _scripts_blob(out)
    check("multipeak: script used fit_per_pixel for the native-res demand",
          "fit_per_pixel(" in blob_txt)
    two_comp = blob_txt.count("gaussian") + blob_txt.count("lorentzian") \
        + blob_txt.count("voigt") >= 2 if "fit_per_pixel(" in blob_txt else False
    check("multipeak: declarative spec carries two peak components",
          two_comp)


def scenario_noise_demand():
    h = w = 300
    cube = RNG.normal(100, 5, (h, w, WL.size)).astype(np.float32)
    d = tempfile.mkdtemp(prefix="ppf_noise_")
    p = os.path.join(d, "uv.npy")
    np.save(p, cube); _meta(p)
    out = os.path.join(d, "out")
    print(f"\n### noise_demand — {d} ({h*w} px of pure noise)")
    t0 = time.monotonic()
    res = _hs_agent(out).analyze(
        p, system_info=p.replace(".npy", ".json"),
        objective="Extract PER-PIXEL maps of the two UV-excited emission "
                  "peaks at NATIVE resolution (do not bin): report "
                  "Peak1_Position and Peak2_Position at every pixel.")
    dt = time.monotonic() - t0
    print(f"[noise_demand] wall: {dt/60:.1f} min | status: "
          f"{res.get('status')}")
    check("noise_demand: bounded (< 25 min)", dt < 1500)
    txt = ((res.get("detailed_analysis") or "") + " ".join(
        str(w_) for w_ in res.get("warnings") or [])).lower()
    honest = any(k in txt for k in (
        "not measurable", "no measurable", "no emission", "no peak",
        "noise", "absen", "no significant", "no detect", "flat"))
    check("noise_demand: honest null (no fabricated per-pixel peak maps "
          "presented as findings)", honest)
    fabricated = re.search(
        r"peak\s*[12][^.;]{0,40}(?:at|near|=|:)\s*~?\d{3}(?:\.\d+)?\s*nm"
        r"(?![^.;]*(?:not|absen|no |noise|expected|request))", txt)
    check("noise_demand: no confident absent-peak positions", not fabricated)


def scenario_fanout_thread():
    d = tempfile.mkdtemp(prefix="ppf_fan_")
    subject = "ONE synthetic pellet studied in one experiment"
    # Branch 1: native-res cube demanding the parallel tool.
    h = w = 300
    centers = 600 + 40 * (np.arange(w) / (w - 1))
    spec = 4.0 * np.exp(-0.5 * ((WL[None, :] - centers[:, None]) / 16) ** 2)
    cube = (5.0 + np.broadcast_to(spec[None], (h, w, WL.size))
            + RNG.normal(0, 0.3, (h, w, WL.size))).astype(np.float32)
    cp = os.path.join(d, "emission.npy")
    np.save(cp, cube); _meta(cp)
    # Branch 2: planted temperature series (transition at 85 C).
    x = np.linspace(2500, 4000, 800)
    for T in range(30, 165, 10):
        f = 1.0 / (1.0 + np.exp(-(T - 85.0) / 4.0))
        y = ((1 - f) * np.exp(-0.5 * ((x - 3400) / 50) ** 2)
             + f * np.exp(-0.5 * ((x - 2900) / 50) ** 2)
             + RNG.normal(0, 0.004, x.size))
        fp = os.path.join(d, f"vib_{T:03d}C.txt")
        np.savetxt(fp, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T)},
                  open(fp.replace(".txt", ".json"), "w"))

    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="ppf_fan_sess_")
    print(f"\n### fanout_thread — session {base}")
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    t0 = time.monotonic()
    out = json.loads(ag._run_fanout([
        {"data_path": cp, "label": "emission map",
         "task": f"Analyze the hyperspectral image at {cp} (300x300x140, "
                 f"wavelengths in the metadata JSON) of {subject}. Map the "
                 "per-pixel emission peak position at NATIVE single-pixel "
                 "resolution (do not bin) — the position drifts across the "
                 "field and the gradient must be resolved."},
        {"data_path": d, "pattern": "vib_*C.txt", "label": "vibrational series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'vib_*C.txt'. Temperature series on {subject}. "
                 "Characterize the transition and report its temperature."},
    ]))
    dt = time.monotonic() - t0
    print(f"[fanout_thread] wall: {dt/60:.1f} min | "
          + json.dumps({k: out.get(k) for k in
                        ("status", "branches_run", "branches_with_output",
                         "branches_timed_out")}))
    check("fanout_thread: fan-out completed", out.get("status") == "success")
    check("fanout_thread: both branches productive (no loky/thread deadlock)",
          out.get("branches_with_output") == 2)
    check("fanout_thread: no branch timed out",
          not out.get("branches_timed_out"))
    check("fanout_thread: bounded (< 35 min)", dt < 2100)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    em = [e for e in fan if e.get("label") == "emission map"]
    em_txt = ((em[0].get("summary") or "") + " ".join(
        str(k) for k in em[0].get("key_findings") or [])).lower() if em else ""
    check("fanout_thread: gradient/drift resolved in the cube branch",
          any(k in em_txt for k in ("gradient", "drift", "600", "640",
                                    "increas", "varies", "shift")))
    vib = [e for e in fan if e.get("label") == "vibrational series"]
    vib_txt = ((vib[0].get("summary") or "") + " ".join(
        str(k) for k in vib[0].get("key_findings") or [])).lower() if vib else ""
    check("fanout_thread: series branch recovered the ~85 C transition",
          any(abs(float(v) - 85) <= 6 for v in
              re.findall(r"\b(\d{2,3})(?:\.\d+)?\s*(?:°?\s*c|deg)", vib_txt)))
    # Cost honesty inside the fan-out session tree: the cube branch must
    # either call the parallel tool OR use a vectorized estimator — what it
    # may NOT do is hand-roll a per-pixel iterative fitting loop. (A single
    # dominant peak position is legitimately solvable by vectorized argmax +
    # parabolic refinement, which the size-budget rules rank FIRST; the tool
    # is for unavoidable iterative fits.)
    blob_txt = ""
    for s in glob.glob(os.path.join(base, "**", "*.py"), recursive=True):
        if "emission" in s:      # cube branch only: the curve-series branch
            try:                 # legitimately runs per-file fit loops
                blob_txt += open(s).read()
            except OSError:
                pass
    called_tool = "fit_per_pixel(" in blob_txt
    hand_rolled_loop = re.search(
        r"for\s+\w+\s+in\s+range\([^)]*\)\s*:(?:(?!\ndef\s)[\s\S]){0,2000}?"
        r"(?:curve_fit|lmfit|\.fit\()", blob_txt)
    print(f"    (tool called: {called_tool}, "
          f"hand-rolled fit loop: {bool(hand_rolled_loop)})")
    check("fanout_thread: cube branch cost-honest (parallel tool or "
          "vectorized — no per-pixel iterative loop)",
          called_tool or not hand_rolled_loop)


SCENARIOS = {"mask_comp": scenario_mask_comp,
             "multipeak": scenario_multipeak,
             "noise_demand": scenario_noise_demand,
             "fanout_thread": scenario_fanout_thread}


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
    print(f"PPF STRESS LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
