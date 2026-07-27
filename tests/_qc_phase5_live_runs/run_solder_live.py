"""Phase-5 live gate, leg 2 — the ladder/salvage exerciser (hyper_solder).

Same In/Pb/Sn solder case that climbed the full retry ladder [0,1,1,2,2]
and salvaged (status "partial") in the phase-2 pre-port run: the K-edges of
In and Sn are nearly collinear at this energy resolution, so required
outputs keep failing the combined review and the annealed retry + salvage
judge machinery gets a real workout through the ported engine loop.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
OUT = Path(__file__).parent
PHASE2 = OUT.parent / "_qc_phase2_live_runs"

sys.path.insert(0, str(OUT))
from run_phase5_live import _dump  # noqa: E402


def run_hyper_solder():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    xdir = BENCH / "xray_hyper_data"
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_solder"),
        enable_human_feedback=False, executor_timeout=900)
    res = agent.analyze(
        str(xdir / "coupon_radiography_CuSquareSolders.npy"),
        system_info=str(PHASE2 / "metadata_cusolder_fixed.json"),
        objective=("Locate the In, Pb, and Sn solder regions in the Cu coupon "
                   "using their K-edges and map where each element is present."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
    )
    _dump("hyper_solder", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE HYPER_SOLDER (phase-5 engine) START\n{'='*70}", flush=True)
    try:
        status = run_hyper_solder().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE HYPER_SOLDER DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
