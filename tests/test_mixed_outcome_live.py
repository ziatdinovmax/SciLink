"""Live stress: mixed outcomes inside one analysis and across one fusion.

  mixed_targets — ONE cube carrying a strong real band (560 nm, everywhere)
      while the task ALSO demands two absent peaks (700/800 nm). The run
      must map the real band AND resolve the absent ones to a judged null —
      positive and null determinations coexisting in one analysis.
  mixed_fusion — a 4-branch fan-out: two planted-truth temperature series
      (85/88 C), a localized real emission cube, and a pure-noise cube with
      demanded peaks (a judged null branch). Fusion must integrate the
      trends and the localized finding while reporting the null branch
      honestly — no fabricated cross-links to a branch whose result is an
      absence.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_mixed_outcome_live.py \
      [--scenario all|mixed_targets|mixed_fusion]
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
RNG = np.random.default_rng(41)
WL = np.linspace(400, 900, 160)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def scenario_mixed_targets():
    h = w = 256
    cube = (5.0 + 3.0 * _g(WL, 560, 14)[None, None, :]
            + RNG.normal(0, 0.4, (h, w, WL.size)))
    d = tempfile.mkdtemp(prefix="mixed_t_")
    p = os.path.join(d, "mixed.npy")
    np.save(p, cube.astype(np.float32))
    json.dump({"technique": "hyperspectral emission map",
               "wavelengths_nm": [float(v) for v in WL]},
              open(p.replace(".npy", ".json"), "w"))
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    base = tempfile.mkdtemp(prefix="mixedtargets_")
    print(f"\n### mixed_targets — session {base}")
    ag = AnalysisOrchestratorAgent(
        base_dir=base, api_key=None, base_url=None, model_name=MODEL,
        restore_checkpoint=False, analysis_mode=AnalysisMode.AUTONOMOUS)
    t0 = time.monotonic()
    res = ag.run_task(
        f"Analyze the hyperspectral image at {p} (256x256x160; wavelengths "
        f"in the metadata JSON). Two requests: (1) map the PER-PIXEL peak "
        "position of the emission band near 560 nm; (2) also map the "
        "per-pixel positions of the two additional peaks expected near "
        "700 nm and 800 nm.", autonomy=AnalysisMode.AUTONOMOUS)
    dt = time.monotonic() - t0
    summary = (res.get("summary") or "") + " ".join(
        str(k) for k in res.get("key_findings") or [])
    print(f"[mixed_targets] wall: {dt/60:.1f} min | status: "
          f"{res.get('status')}")
    print(f"[mixed_targets] summary (first 1500):\n{summary[:1500]}")
    check("mixed_targets: success", res.get("status") == "success")
    lo = summary.lower()
    check("mixed_targets: real 560 band mapped",
          any(abs(float(v) - 560) <= 12 for v in
              re.findall(r"\b(5[3-8]\d)(?:\.\d+)?\b", summary)))
    check("mixed_targets: absent 700/800 peaks resolved as null/absent",
          any(s in lo for s in ("not measurable", "no measurable", "absent",
                                "not detect", "no peak", "not present",
                                "no evidence", "no emission")))
    check("mixed_targets: no fabricated 700/800 values",
          not re.search(r"(?:peak|band)[^.;]{0,30}(?:at|near|=|:)\s*~?"
                        r"(?:69\d|7[01]\d|79\d|80\d)(?:\.\d+)?\s*nm(?![^.;]*"
                        r"(?:not|absen|no |expected|request))",
                        summary, re.IGNORECASE))
    check("mixed_targets: bounded (< 25 min)", dt < 1500)


def scenario_mixed_fusion():
    d = tempfile.mkdtemp(prefix="mixed_f_")
    subject = ("ONE synthetic benchmark pellet studied in one experiment "
               "by multiple techniques")
    # Two planted-truth temperature series.
    for stem, t0c, span, centers in (
            ("mA_vib", 85.0, (2500, 4000), (3400, 2900)),
            ("mB_xrd", 88.0, (5, 40), (11.0, 11.9))):
        x = np.linspace(*span, 800)
        for T in range(30, 165, 10):
            f = _sig(T, t0c, 4.0)
            y = ((1 - f) * _g(x, centers[0], (span[1] - span[0]) / 30)
                 + f * _g(x, centers[1], (span[1] - span[0]) / 30)
                 + RNG.normal(0, 0.004, x.size))
            fp = os.path.join(d, f"{stem}_{T:03d}C.txt")
            np.savetxt(fp, np.column_stack([x, y]))
            json.dump({"temperature_C": float(T)},
                      open(fp.replace(".txt", ".json"), "w"))
    # Localized REAL emission cube.
    h = w = 192
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((xx - 60) ** 2 + (yy - 120) ** 2) < 14 ** 2
    spec = 4.0 * _g(WL, 520, 12) + 3.0 * _g(WL, 610, 15)
    cube = (5.0 + RNG.normal(0, 0.4, (h, w, WL.size))
            + blob[..., None] * spec[None, None, :])
    cp = os.path.join(d, "emission_map.npy")
    np.save(cp, cube.astype(np.float32))
    json.dump({"technique": "hyperspectral emission map",
               "wavelengths_nm": [float(v) for v in WL],
               "note": "same pellet, same session"},
              open(cp.replace(".npy", ".json"), "w"))
    # Pure-noise cube with demanded peaks -> judged null branch.
    ncube = RNG.normal(100, 5, (h, w, WL.size)).astype(np.float32)
    npth = os.path.join(d, "uv_map.npy")
    np.save(npth, ncube)
    json.dump({"technique": "hyperspectral UV-emission map",
               "wavelengths_nm": [float(v) for v in WL],
               "note": "same pellet, same session"},
              open(npth.replace(".npy", ".json"), "w"))

    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="mixedfusion_")
    print(f"\n### mixed_fusion — session {base}")
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    t0 = time.monotonic()
    out = json.loads(ag._run_fanout([
        {"data_path": d, "pattern": "mA_vib_*C.txt",
         "label": "vibrational series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'mA_vib_*C.txt'. Temperature series on {subject}. "
                 "Characterize the transition and report its temperature."},
        {"data_path": d, "pattern": "mB_xrd_*C.txt",
         "label": "diffraction series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'mB_xrd_*C.txt'. Temperature series on {subject}. "
                 "Characterize the structural transition and report its "
                 "temperature."},
        {"data_path": cp, "label": "emission map",
         "task": f"Analyze the hyperspectral image at {cp} (192x192x160; "
                 f"wavelengths in metadata JSON) of {subject}. Characterize "
                 "the emission: location and peak wavelengths."},
        {"data_path": npth, "label": "UV emission map",
         "task": f"Analyze the hyperspectral image at {npth} (192x192x160; "
                 f"wavelengths in metadata JSON) of {subject}. Extract "
                 "PER-PIXEL maps of the two UV-excited emission peaks: "
                 "report Peak1_Position and Peak2_Position."},
    ], branch_time_budget_s=1500))
    print("FANOUT:", json.dumps({k: out.get(k) for k in (
        "status", "branches_run", "branches_with_output", "mesh",
        "branches_timed_out")}, indent=2))
    check("mixed_fusion: 4 branches ran", out.get("branches_run") == 4)
    check("mixed_fusion: all four productive (null branch included — an "
          "honest determination IS output)",
          out.get("branches_with_output") == 4)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    uv = [e for e in fan if e.get("label") == "UV emission map"]
    uv_txt = ((uv[0].get("summary") or "") + " ".join(
        str(k) for k in uv[0].get("key_findings") or [])).lower() if uv else ""
    check("mixed_fusion: null branch reports the absence honestly",
          any(s in uv_txt for s in ("not measurable", "no measurable",
                                    "no emission", "no peak", "noise",
                                    "absen", "flat", "no significant")))
    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Integrate the transition trends and the emission "
                    "findings; state plainly which measurements found "
                    "nothing."))
    dt = time.monotonic() - t0
    print(f"[mixed_fusion] total wall: {dt/60:.1f} min")
    check("mixed_fusion: fusion success + gated",
          fout.get("status") == "success"
          and fout.get("complementarity_gated"))
    comp = fout.get("computed_reconciliation") or {}
    check("mixed_fusion: computed + audited",
          comp.get("status") == "success"
          and (comp.get("verification") or {}).get("verdict")
          in ("accept", "refine"))
    vals = []

    def _walk(v):
        if isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (int, float)) and v is not None:
            vals.append(float(v))
    _walk((comp.get("results") or {}).get("quantities") or {})
    check("mixed_fusion: planted transitions recovered (85/88 C)",
          any(abs(v - 85) <= 6 for v in vals)
          and any(abs(v - 88) <= 6 for v in vals))
    blob_txt = (str(fout.get("detailed_analysis", "")) + " ".join(
        str(c) for c in fout.get("caveats") or [])).lower()
    check("mixed_fusion: the null branch's absence stated in the synthesis",
          any(s in blob_txt for s in ("no measurable", "not measurable",
                                      "no emission", "found nothing",
                                      "no uv", "absen", "null")))
    check("mixed_fusion: no fabricated link to the null branch",
          not re.search(r"uv[^.;]{0,80}(?:correlat|coincid|agrees with|"
                        r"matches the)", blob_txt))
    print("\nNARRATIVE (first 1200):\n",
          str(fout.get("detailed_analysis", ""))[:1200])


SCENARIOS = {"mixed_targets": scenario_mixed_targets,
             "mixed_fusion": scenario_mixed_fusion}


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
    print(f"MIXED OUTCOME LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
