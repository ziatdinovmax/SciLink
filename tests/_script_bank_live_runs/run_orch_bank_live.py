"""Orchestrator-surface bank test (#346 step 2 scenario test).

Drives a REAL AnalysisOrchestratorAgent via run_task (the meta-agent
contract) with the script bank enabled: retrieval + banking must fire inside
the orchestrator-spawned CurveFittingAgent exactly as in direct-API runs.

Task: a Raman spectrum that the bank's Raman records should match (exemplar
offered), then verify the bank gained/updated a record from the run.

Usage: python run_orch_bank_live.py   (needs AWS_BEARER_TOKEN_BEDROCK)
"""

import json
import os
from pathlib import Path

OUT = Path(__file__).parent
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_HOME"] = str(OUT / "scilink_home")
os.environ["SCILINK_MEMORY"] = "0"
os.environ["SCILINK_SCRIPT_BANK"] = "1"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"


def main():
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )
    from scilink.skills._shared import _script_bank as sb

    before = {r["id"]: r["stats"].copy() for r in sb.list_records("curve_fitting")}

    session = OUT / "orch_bank_session"
    orch = AnalysisOrchestratorAgent(
        base_dir=str(session),
        model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    case = BENCH / "raman_chris" / "strong_metadata" / "Raman_4_data_hold"
    out = orch.run_task(
        f"Analyze the Raman spectrum at {case / 'data1.csv'} "
        f"(metadata: {case / 'metadata.json'}). Fit the peaks and report "
        f"the main peak positions."
    )
    print("\n=== run_task summary ===")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("status", "summary", "key_findings", "warnings")},
                     indent=1, default=str)[:1200], flush=True)

    after = sb.list_records("curve_fitting")
    print("\n=== BANK DELTA ===")
    for r in after:
        b = before.get(r["id"])
        if b is None:
            print(f"NEW record {r['id']}: n_successes={r['stats']['n_successes']}")
        elif b != r["stats"]:
            print(f"UPDATED {r['id']}: {b} -> {r['stats']}")
    print("=== END ===", flush=True)


if __name__ == "__main__":
    main()
