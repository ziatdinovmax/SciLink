"""Gap-matrix live legs, ORCHESTRATOR mode (one session, sequential run_task).

  A. reuse      — run_analysis with prior_analysis_paths + reuse_locked_script
  B. best-of-N  — run_analysis with n_candidates=2
  C. series     — run_analysis over a directory of 5 VT-NMR spectra with
                  series_metadata
  D. autopilot human-feedback — run_task(autonomy="AUTOPILOT") with a forced
                  below-threshold fit; the patched "human" answers the
                  poor-fit prompt with "threshold 0.9"
"""

import builtins
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent
P4 = OUT.parent / "_qc_phase4_live_runs"

STRONG = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
WEAK = BENCH / "raman_chris" / "weak_metadata" / "Raman_3_data_hold"
NMR = BENCH / "NMR_staged" / "02_medium_JCESR_MgOTf2_TEP"

PICKS = [("JCESR_MgOTf2_G2_TEP__exp5600.npy", 243.2),
         ("JCESR_MgOTf2_G2_TEP__exp5400.npy", 258.2),
         ("JCESR_MgOTf2_G2_TEP__exp200.npy", 278.2),
         ("JCESR_MgOTf2_G2_TEP__exp1200.npy", 293.5),
         ("JCESR_MgOTf2_G2_TEP__exp6300.npy", 313.2)]


def main():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )

    session = OUT / "orch_gap_session"
    orch = AnalysisOrchestratorAgent(
        base_dir=str(session), model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    summary = {}

    # --- A. reuse ---
    out = orch.run_task(
        f"Analyze the Raman spectrum at {STRONG / 'data1.csv'} (metadata: "
        f"{STRONG / 'metadata.json'}). A prior analysis of this exact series "
        f"exists at {P4 / 'curve_raman'} — call run_analysis with "
        f"prior_analysis_paths set to that directory and reuse_locked_script="
        f"True so the locked fitting script is reused verbatim, then report "
        f"the reuse validity verdict."
    )
    summary["A_reuse"] = out.get("status")
    print("\n[A reuse] status:", out.get("status"),
          "| summary head:", str(out.get("summary"))[:300], flush=True)

    # --- B. best-of-N ---
    out = orch.run_task(
        f"Analyze the Raman spectrum at {WEAK / 'data1.csv'} (metadata: "
        f"{WEAK / 'metadata.json'}). Use best-of-N fitting: call run_analysis "
        f"with n_candidates=2."
    )
    summary["B_best_of_n"] = out.get("status")
    print("\n[B best-of-N] status:", out.get("status"), flush=True)

    # --- C. series (directory + series_metadata) ---
    sdir = OUT / "orch_series_data"
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir()
    for f, _t in PICKS:
        shutil.copy(NMR / "data" / f, sdir / f)
    series_meta = {"variable": "temperature",
                   "values": [t for _, t in PICKS], "unit": "K"}
    out = orch.run_task(
        f"Analyze the 31P VT-NMR temperature series in the directory {sdir} "
        f"(campaign metadata: {NMR / 'metadata.json'}). The five spectra "
        f"correspond, in alphabetical filename order exp1200, exp200, exp5400, "
        f"exp5600, exp6300, to temperatures 293.5, 278.2, 258.2, 243.2, 313.2 K. "
        f"Call run_analysis ONCE over the whole directory passing "
        f"series_metadata='{json.dumps(series_meta)}' so it runs as a series, "
        f"and track peak position/width vs temperature."
    )
    summary["C_series"] = out.get("status")
    print("\n[C series] status:", out.get("status"), flush=True)

    # --- D. AUTOPILOT human-feedback ---
    prompts_seen = []
    real_input = builtins.input

    def fake_input(prompt=""):
        prompts_seen.append(str(prompt))
        ans = "threshold 0.9" if "Your input" in str(prompt) else ""
        print(f"\n[fake-human] prompt≈{str(prompt)[:60]!r} -> {ans!r}", flush=True)
        return ans

    builtins.input = fake_input
    try:
        out = orch.run_task(
            f"Analyze the Raman spectrum at {WEAK / 'data1.csv'} (metadata: "
            f"{WEAK / 'metadata.json'}) with a very strict acceptance "
            f"threshold: call run_analysis with r2_threshold=0.999 and "
            f"max_verification_iterations=2.",
            autonomy="AUTOPILOT",
        )
    finally:
        builtins.input = real_input
    summary["D_autopilot_hf"] = out.get("status")
    print("\n[D autopilot] status:", out.get("status"),
          "| prompts seen:", len(prompts_seen),
          "| poor-fit prompt fired:",
          any("Your input" in p for p in prompts_seen), flush=True)

    # underlying records
    print("\n=== records ===")
    for r in orch.analysis_results:
        fr = r.get("full_result") or {}
        print({"agent": r.get("agent_name"), "status": fr.get("status"),
               "reuse": (fr.get("reuse_validity") or {}).get("verdict"),
               "qh": bool(fr.get("quality_history")),
               "n_series": len(fr.get("individual_results") or fr.get("series_fit_results") or [])})
    (OUT / "result_orch_gap.json").write_text(json.dumps(summary, indent=1))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE ORCH GAP MATRIX START\n{'='*70}", flush=True)
    try:
        main()
        status = "done"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE ORCH GAP MATRIX DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
