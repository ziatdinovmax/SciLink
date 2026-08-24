"""Live validation of incremental fusion + the fusion->re-analysis edge.

Synthetic planted-truth study, full real pipeline, three acts:

  1. Fan-out two series (transitions planted at 85 / 88 C) and fuse —
     baseline gated + computed.
  2. A NEW dataset arrives (thermal curve, endotherm planted at 92 C):
     analyzed via a direct delegation WITH the data_path stamp, then the
     ENLARGED set is fused — the mixed set must re-gate live and stay
     gated + computed, with the newcomer's marker entering the computed
     quantities.
  3. Fusion-guided RE-ANALYSIS: the first series is re-delegated citing
     the fusion in context_from — provenance must be stamped mechanically,
     the additive-only note must reach the child, the branch must still
     report its OWN transition (~85), and the next fusion must discount it.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_fusion_incremental_live.py
"""
import json
import os
import re
import sys
import tempfile

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(11)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _series(d, stem, t0, span, centers, technique):
    x = np.linspace(*span, 900)
    for T in range(30, 165, 10):
        f = _sig(T, t0, 4.0)
        y = ((1 - f) * _g(x, centers[0], (span[1] - span[0]) / 30)
             + f * _g(x, centers[1], (span[1] - span[0]) / 30)
             + RNG.normal(0, 0.004, x.size))
        p = os.path.join(d, f"{stem}_{T:03d}C.txt")
        np.savetxt(p, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T), "technique": technique},
                  open(p.replace(".txt", ".json"), "w"))
    return f"{stem}_*C.txt"


def _quant_values(fused):
    vals = []

    def _walk(v):
        if isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (int, float)) and v is not None:
            vals.append(float(v))
    _walk((fused.get("computed_reconciliation") or {}).get(
        "results", {}).get("quantities") or {})
    return vals


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix="fusion_incr_live_")
    print("session:", base)
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)

    d = tempfile.mkdtemp(prefix="incr_data_")
    subject = ("ONE synthetic benchmark pellet heated in-situ in a single "
               "experiment, measured by multiple techniques")
    pat_a = _series(d, "tA_vib", 85.0, (2500, 4000), (3400, 2900), "IR-like")
    pat_b = _series(d, "tB_xrd", 88.0, (5, 40), (11.0, 11.9), "XRD-like")
    # The late-arriving dataset: thermal curve with an endotherm at 92 C.
    T = np.arange(25, 200, 0.25)
    y = (-0.4 - 2.5 * _g(T, 92.0, 6.0) + 0.0012 * (T - 25)
         + RNG.normal(0, 0.01, T.size))
    late = os.path.join(d, "thermal_curve.txt")
    np.savetxt(late, np.column_stack([T, y]),
               header="temperature_C heat_flow_mW_mg")

    # --- Act 1: fan-out + baseline fusion ---
    out = json.loads(ag._run_fanout([
        {"data_path": d, "pattern": pat_a, "label": "vibrational series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'{pat_a}'. Temperature series on {subject}. Characterize "
                 "the transition and report its temperature."},
        {"data_path": d, "pattern": pat_b, "label": "diffraction series",
         "task": f"Analyze the series in {d} — ONLY files matching "
                 f"'{pat_b}'. Temperature series on {subject}. Characterize "
                 "the structural transition and report its temperature."},
    ]))
    check("act1: fan-out ran", out.get("branches_run") == 2)
    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    f1 = json.loads(ag._fuse_delegations(idxs))
    check("act1: baseline fusion gated + computed",
          f1.get("complementarity_gated")
          and (f1.get("computed_reconciliation") or {}).get("status")
          == "success")
    fus1_idx = [e for e in ag._delegation_ledger
                if e.get("mode") == "fusion"][-1]["index"]
    if f1.get("suggested_followups"):
        print("act1 re-analysis suggestions:", f1["suggested_followups"])

    # --- Act 2: late dataset via direct delegation, then incremental fusion ---
    print("\n### Act 2: late-arriving dataset ###")
    ag._delegate(
        "analysis",
        f"Analyze the thermal-analysis curve at {late} (columns: "
        f"temperature_C, heat_flow_mW_mg). One measurement of {subject}. "
        "Characterize the endothermic feature and report its temperature.",
        None, None, "thermal curve", data_path=late)
    new_entry = ag._delegation_ledger[-1]
    check("act2: delegation succeeded + data_path stamped",
          new_entry.get("status") == "success"
          and new_entry.get("data_path") == late)
    f2 = json.loads(ag._fuse_delegations(idxs + [new_entry["index"]]))
    check("act2: ENLARGED fusion re-gated live and stayed gated",
          f2.get("complementarity_gated")
          and not f2.get("complementarity_warning"))
    check("act2: computed reconciliation on the enlarged set",
          (f2.get("computed_reconciliation") or {}).get("status") == "success")
    vals = _quant_values(f2)
    hits = {t0: min((abs(v - t0) for v in vals), default=99)
            for t0 in (85.0, 88.0, 92.0)}
    print(f"act2 planted-marker recovery: {hits}")
    check("act2: newcomer's 92 C marker entered the computation",
          hits[92.0] <= 6.0)
    check("act2: original markers still resolved",
          hits[85.0] <= 6.0 and hits[88.0] <= 6.0)

    # --- Act 3: fusion-guided re-analysis of the vibrational series ---
    print("\n### Act 3: fusion-guided re-analysis ###")
    ag._delegate(
        "analysis",
        f"Re-analyze the series in {d} — ONLY files matching '{pat_a}' "
        f"(temperature series on {subject}). A prior cross-dataset fusion "
        "suggests checking whether a two-component description fits better "
        "and whether the transition temperature is robust to the model "
        "choice. Report the transition temperature your own data supports.",
        None, [fus1_idx], "vibrational series re-analysis",
        data_path=os.path.join(d, pat_a))
    re_entry = ag._delegation_ledger[-1]
    check("act3: fusion_feedback provenance stamped",
          re_entry.get("informed_via") == "fusion_feedback"
          and set(re_entry.get("informed_by") or [])
          == {"vibrational series", "diffraction series"})
    check("act3: additive-only note in the task sent",
          "ADDITIVE-ONLY" in (re_entry.get("task") or ""))
    summary = (re_entry.get("summary") or "") + " ".join(
        str(k) for k in re_entry.get("key_findings") or [])
    own = [float(v) for v in
           re.findall(r"\b(\d{2,3}(?:\.\d+)?)\s*°?\s*C\b", summary)]
    check("act3: re-analysis reports its OWN transition (~85 C)",
          any(abs(v - 85.0) <= 6.0 for v in own))
    f3 = json.loads(ag._fuse_delegations(
        [idxs[1], re_entry["index"]]))
    check("act3: fusion succeeded", f3.get("status") == "success")
    check("act3: re-analyzed branch discounted",
          any("re-analyzed with feedback" in str(c)
              for c in f3.get("caveats") or []))
    check("act3: independence map records the feedback provenance",
          bool((f3.get("independence") or {}).get(
              "vibrational series re-analysis")))

    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"FUSION INCREMENTAL LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
