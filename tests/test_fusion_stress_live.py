"""Stress scenario for the full #296 + mesh stack in ONE fan-out call.

Six inputs, four survivors, planted ground truth everywhere:
  - techA IR-like series (T0=85, 14 pts)          <- steered (multi-payload)
  - techB XRD-like series (T0=88, 27 pts, finer grid)
  - techC transport-like series (T0=84, 9 pts, coarser grid)
  - techD framework series (T0=130 — a genuine OUTLIER technique)
  - an UNRELATED acoustics dataset (different project — gate must prune)
  - an exact duplicate of the techA branch (mechanical dedup must drop it)

Stresses at once: gate partitioning with join_type, true-duplicate dedup,
4-way concurrency with mixed sampling, multi-companion steering payloads,
independent wiring, 4-table computed reconciliation that must report a
3-technique cluster (~84-88) PLUS a separated outlier (~130) without
averaging them, the audit pass, independence discount, and a SECOND fusion
over a subset in the same session (fusion dir 02).

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_fusion_stress_live.py
"""
import json
import os
import sys
import tempfile

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(7)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _series(d, stem, t0, temps, span, centers, technique):
    x = np.linspace(*span, 900)
    c_lo, c_hi = centers
    for T in temps:
        f = _sig(T, t0, 4.0)
        y = ((1 - f) * _g(x, c_lo, (span[1] - span[0]) / 30)
             + f * _g(x, c_hi, (span[1] - span[0]) / 30)
             + RNG.normal(0, 0.004, x.size))
        p = os.path.join(d, f"{stem}_{T:03.0f}C.txt")
        np.savetxt(p, np.column_stack([x, y]))
        json.dump({"temperature_C": float(T), "technique": technique},
                  open(p.replace(".txt", ".json"), "w"))
    return f"{stem}_*C.txt"


def main():
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    import scilink.agents.meta_agent.fanout as fo
    base = tempfile.mkdtemp(prefix="fusion_stress_")
    print("session dir:", base)
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    prompts = []
    real = fo._llm_json

    def spy(orch, prompt, extra_parts=None):
        if "complementary measurements of ONE system" in prompt:
            prompts.append(prompt)
        return real(orch, prompt, extra_parts=extra_parts)
    fo._llm_json = spy

    d = tempfile.mkdtemp(prefix="stress_data_")
    subject = ("ONE synthetic benchmark pellet heated in-situ in a single "
               "experiment, measured simultaneously by four techniques")
    pat_a = _series(d, "sA_ir", 85.0, range(30, 165, 10), (2500, 4000),
                    (3400, 2900), "IR-like")
    pat_b = _series(d, "sB_xrd", 88.0, range(30, 165, 5), (5, 40),
                    (11.0, 11.9), "XRD-like")
    pat_c = _series(d, "sC_tr", 84.0, range(30, 165, 15), (0, 100),
                    (30, 70), "transport-like")
    pat_d = _series(d, "sD_fw", 130.0, range(30, 165, 10), (800, 1800),
                    (1100, 1500), "framework-Raman-like")
    # Unrelated dataset from a different project.
    xa = np.linspace(0, 1, 800)
    ap = os.path.join(d, "acoustics_run.txt")
    np.savetxt(ap, np.column_stack([xa, RNG.normal(0, 1, xa.size)]))

    def b(pat, label, extra=""):
        return {"data_path": d, "pattern": pat, "label": label,
                "task": f"Analyze the measurement series in {d} — ONLY files "
                        f"matching '{pat}'. They are a temperature series on "
                        f"{subject}. Track how the signal evolves and "
                        f"characterize any transition it reveals.{extra}"}

    branches = [
        dict(b(pat_a, "IR series"), steer=True),
        b(pat_b, "XRD series"),
        b(pat_c, "transport series"),
        b(pat_d, "framework series"),
        dict(b(pat_a, "IR series")),   # exact duplicate -> mechanical dedup
        {"data_path": ap, "label": "acoustics dataset",
         "task": "Characterize the periodicity of this acoustics recording "
                 "from a DIFFERENT project (unrelated instrument and sample; "
                 "included by mistake)."},
    ]
    out = json.loads(ag._run_fanout(branches))
    print("FANOUT:", json.dumps({k: out.get(k) for k in (
        "status", "join_axis", "join_type", "mesh", "branches_run",
        "branches_with_output")}, indent=2))
    check("gate + dedup pruned to the 4 complementary branches",
          out.get("branches_run") == 4)
    labels = sorted(r.get("label") for r in out.get("results", []))
    check("the acoustics dataset was pruned, duplicate dropped",
          "acoustics dataset" not in labels and labels.count("IR series") == 1)
    check("independent wiring on the shared-axis set",
          out.get("mesh") == "independent")
    check("all 4 branches productive",
          out.get("branches_with_output") == 4)

    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    steered = [e for e in fan if e.get("informed_by")]
    check("steered branch got MULTIPLE companion payloads",
          len(steered) == 1 and len(steered[0]["informed_by"]) == 3)
    task = steered[0].get("task", "") if steered else ""
    import re
    hints = [float(v) for v in re.findall(
        r"sharpest change near\s+control ≈ (\d+(?:\.\d+)?)", task)]
    print("steering hints received:", hints)
    check("hints locate the three companions' changes",
          len(hints) == 3
          and sum(1 for h in hints if 78 <= h <= 95) == 2
          and sum(1 for h in hints if 122 <= h <= 138) == 1)

    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Which techniques agree on the transition location, and "
                    "which do not?"))
    check("4-way fusion success", fout.get("status") == "success")
    comp = fout.get("computed_reconciliation") or {}
    ver = comp.get("verification") or {}
    print("computed:", comp.get("status"), "| audit:", ver.get("verdict"),
          "| attempts:", comp.get("attempts"))
    q = (comp.get("results") or {}).get("quantities") or {}
    print("QUANTITIES:", json.dumps(q, default=str)[:1600])
    vals = []

    def _walk(v):
        if isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (int, float)) and v is not None:
            vals.append(float(v))
    _walk(q)
    cluster = [v for v in vals if 78.0 <= v <= 95.0]
    outlier = [v for v in vals if 122.0 <= v <= 138.0]
    print(f"cluster-range values: {sorted(set(cluster))[:8]}")
    print(f"outlier-range values: {sorted(set(outlier))[:4]}")
    check("computed output resolves the ~85 C cluster (>=2 markers)",
          len(set(round(v) for v in cluster)) >= 2)
    check("computed output keeps the 130 C outlier separate",
          len(outlier) >= 1)
    check("no fake all-four consensus (no lone mean swallowing the outlier)",
          not any(100.0 <= v <= 120.0 and "mean" in k.lower()
                  and "cluster" not in k.lower() for k, v in q.items()
                  if isinstance(v, (int, float))))
    blob = (str(fout.get("detailed_analysis", "")) + " ".join(
        str(c) for c in fout.get("caveats") or [])).lower()
    check("outlier separation stated in the synthesis", "130" in blob)
    check("independence discount recorded for the steered branch",
          (fout.get("independence") or {}).get("IR series")
          and any("steered at launch" in str(c)
                  for c in fout.get("caveats") or []))
    check("audit ran", ver.get("verdict") in ("accept", "refine"))

    # Second fusion over a subset in the same session (dir 02, no collision).
    f2 = json.loads(ag._fuse_delegations(
        idxs[:2], focus="Compare just these two techniques."))
    check("second fusion in one session succeeded",
          f2.get("status") == "success")
    check("second fusion got its own directory",
          f2.get("report_path") and "/02/" in f2["report_path"]
          and os.path.exists(f2["report_path"]))

    print("\nreport 1:", fout.get("report_html_path"))
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"FUSION STRESS LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
