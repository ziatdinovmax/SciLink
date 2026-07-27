"""Generalization live test: low-loss plasmon EELS cube (FT:IO nanocrystals).

Different HS regime from the X-ray legs: the agent's home-turf EELS path
(decomposition + refinement), eV axis, no matching registered tool. Checks
that the review-hardening changes generalize: flux table in eV, the
tool-naming mandate stays silent (nothing matches LSPR peak mapping), and
reviews don't over-trigger on a normal case.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
EX = Path.home() / "Code" / "SciLink" / "examples" / "eels_plasmons_demo"
OUT = Path(__file__).parent

import sys
sys.path.insert(0, str(OUT))
from run_phase5_live import _dump  # noqa: E402


def run_plasmon():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    meta = json.load(open(EX / "datacube.json"))
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_plasmon"),
        enable_human_feedback=False, executor_timeout=900,
    )
    res = agent.analyze(
        str(EX / "datacube.npy"),
        system_info=meta,
        objective=("Map the localized surface plasmon resonance (LSPR) peak "
                   "energy (in eV) at each pixel across the nanocrystal "
                   "array, and map the plasmon intensity."),
    )
    _dump("hyper_plasmon", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE PLASMON EELS START\n{'='*70}", flush=True)
    try:
        status = run_plasmon().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE PLASMON EELS DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
