"""Live test: sputtered-Au prior-vs-evidence case (review-history fix).

Reproduces the observed infinite-reject: the metadata calls the sample
"sputtered" (implying sub-um thickness) while the measured K-edge keeps
giving tens of um. Pre-fix, the memoryless combined review rejected the
magnitude identically on all 5 attempts. With attempt-history injection +
the prior-vs-evidence principle, the reviewer should either accept the
convergent measurement (stating the discrepancy) or the run should resolve
honestly without burning the full ladder on the same objection.
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

sys.path.insert(0, str(OUT))
from run_phase5_live import _dump  # noqa: E402


def run_ausputter():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    xdir = BENCH / "xray_hyper_data"
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_ausputter_v4"),
        enable_human_feedback=False, executor_timeout=900,
    )
    res = agent.analyze(
        str(xdir / "coupon_radiography_AuSputter.npy"),
        system_info=str(OUT / "metadata_ausputter.json"),
        objective=("Locate the sputtered Au and measure its thickness from "
                   "the Au K-edge jump at 80.8 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="baseline_I0",
    )
    _dump("hyper_ausputter_v4", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE AU-SPUTTER (prior-vs-evidence) START\n{'='*70}", flush=True)
    try:
        status = run_ausputter().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE AU-SPUTTER DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
