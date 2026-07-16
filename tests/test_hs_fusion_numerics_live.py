"""Live acceptance for hyperspectral features reaching the quantitative
fusion: a 2-branch fan-out (emission cube + planted temperature series) on
ONE synthetic sample. Pre-fix, the cube branch never registered numeric
tables, so computed reconciliation needed the OTHER branches alone to cross
its >=2-numerics guard; here it can only run if the cube contributes.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_hs_fusion_numerics_live.py
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(37)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)

    d = tempfile.mkdtemp(prefix="hsfn_live_")
    subject = "ONE synthetic phosphor pellet studied in one session"

    # Branch 1: emission cube, band centered at 620 nm (the planted truth
    # the reconciliation should be able to quote from the cube's table).
    wl = np.linspace(450, 850, 140)
    h = w = 96
    spec = 4.0 * np.exp(-0.5 * ((wl - 620.0) / 16.0) ** 2)
    cube = (5.0 + spec[None, None, :]
            + RNG.normal(0, 0.3, (h, w, wl.size))).astype(np.float32)
    cp = os.path.join(d, "emission.npy")
    np.save(cp, cube)
    json.dump({"technique": "hyperspectral emission map",
               "energy_range": {"start": 450.0, "end": 850.0, "units": "nm"},
               "note": "same pellet, same session"},
              open(cp.replace(".npy", ".json"), "w"))

    # Branch 2: temperature series with a transition at 85 C.
    x = np.linspace(500, 700, 600)
    for T in range(30, 165, 10):
        f = 1.0 / (1.0 + np.exp(-(T - 85.0) / 4.0))
        y = ((1 - f) * np.exp(-0.5 * ((x - 620) / 8) ** 2)
             + f * np.exp(-0.5 * ((x - 640) / 8) ** 2)
             + RNG.normal(0, 0.004, x.size))
        fp = os.path.join(d, f"pl_{T:03d}C.txt")
        np.savetxt(fp, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T)},
                  open(fp.replace(".txt", ".json"), "w"))

    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="hsfn_sess_")
    print(f"session: {base}")
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    t0 = time.monotonic()
    out = json.loads(ag._run_fanout([
        {"data_path": cp, "label": "emission map",
         "task": f"Analyze the hyperspectral image at {cp} (96x96x140, "
                 f"wavelengths in the metadata JSON) of {subject}. "
                 "Characterize the emission band: its center wavelength "
                 "and width."},
        {"data_path": d, "pattern": "pl_*C.txt", "label": "PL series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'pl_*C.txt'. Temperature series on {subject}. "
                 "Characterize the transition and report its temperature."},
    ]))
    check("fan-out succeeded", out.get("status") == "success"
          and out.get("branches_with_output") == 2)

    # The cube branch must now carry a feature table into the ledger.
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    em = [e for e in fan if e.get("label") == "emission map"]
    tables = (em[0].get("feature_tables") or []) if em else []
    check("cube branch registered a feature table (the fixed gap)",
          any(str(t).endswith("features.csv") for t in tables))

    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Relate the emission band to the temperature-driven "
                    "transition; reconcile numerically where possible."))
    dt = time.monotonic() - t0
    print(f"wall: {dt/60:.1f} min")
    check("fusion gated + succeeded",
          fout.get("status") == "success" and fout.get("complementarity_gated"))
    comp = fout.get("computed_reconciliation") or {}
    check("computed reconciliation RAN (>=2 numeric branches incl. the cube)",
          comp.get("status") == "success")
    check("audit verdict recorded",
          (comp.get("verification") or {}).get("verdict") in ("accept",
                                                              "refine"))
    blob = json.dumps(comp.get("results") or {}, default=str)
    check("cube-side quantity present in the computed numerics "
          "(620 nm band reachable)",
          any(s in blob for s in ("619", "620", "621", "Peak_Position")))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"HS FUSION NUMERICS LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
