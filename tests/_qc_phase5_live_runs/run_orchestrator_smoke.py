"""Orchestrator-mode smoke over the refactored agents (#327 phases 4-5).

Drives a REAL AnalysisOrchestratorAgent through its programmatic run_task
surface (the meta-agent contract; wraps one autonomous chat turn), so the
LLM routes through the run_analysis tool into the engine-backed agents:

  1. curve  — Raman strong-metadata (CurveFittingAgent on CodegenQCEngine)
  2. image  — TEM AuNP (ImageAnalysisAgent on CodegenQCEngine)

Verifies the run_task return contract AND the underlying analysis records
(status, quality_history from the engine path).
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent


def main():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )

    session = OUT / "orch_smoke_session"
    orch = AnalysisOrchestratorAgent(
        base_dir=str(session),
        model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )

    case = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
    out1 = orch.run_task(
        f"Analyze the Raman spectrum at {case / 'data1.csv'} "
        f"(metadata: {case / 'metadata.json'}). Fit the peaks and report "
        f"the main peak positions."
    )
    print("\n=== TASK 1 (curve) run_task summary ===")
    print(json.dumps({k: v for k, v in out1.items()
                      if k in ("status", "summary", "key_findings", "warnings")},
                     indent=1, default=str)[:1500])

    out2 = orch.run_task(
        f"Now analyze the TEM image at {BENCH / 'TEM' / '(easy) AuNP.tif'} "
        f"(metadata: {BENCH / 'TEM' / '(easy) AuNP.json'}): measure the Au "
        f"nanoparticle size distribution."
    )
    print("\n=== TASK 2 (image) run_task summary ===")
    print(json.dumps({k: v for k, v in out2.items()
                      if k in ("status", "summary", "key_findings", "warnings")},
                     indent=1, default=str)[:1500])

    # Underlying records: the engine-backed agents' quality history
    print("\n=== underlying analysis records ===")
    recs = []
    for r in orch.analysis_results:
        fr = r.get("full_result") or {}
        qh = fr.get("quality_history") or {}
        entry = {
            "agent": r.get("agent_name"),
            "status": fr.get("status"),
            "approved": qh.get("approved"),
            "final_r2": qh.get("final_r2"),
            "final_score": qh.get("final_score"),
            "n_iters": len(qh.get("verification_iterations", []) or []),
        }
        recs.append(entry)
        print(entry)
    (OUT / "result_orch_smoke.json").write_text(json.dumps({
        "task1": {k: str(v)[:2000] for k, v in out1.items()},
        "task2": {k: str(v)[:2000] for k, v in out2.items()},
        "records": recs,
    }, indent=1, default=str))


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE ORCHESTRATOR SMOKE START\n{'='*70}", flush=True)
    try:
        main()
        status = "done"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE ORCHESTRATOR SMOKE DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
