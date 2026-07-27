"""Gap 2 (standalone): curve series live post-refactor — VT 31P NMR 5-frame.

Exercises the curve series layer over the engine: anchor full QC, non-anchor
locked-script application, outlier/consistency, trend synthesis. Same case
as the phase-2 pre-refactor baseline (exchange-regime-aware Voigt lock).
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


def run_curve_series():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    ddir = BENCH / "NMR_staged" / "02_medium_JCESR_MgOTf2_TEP" / "data"
    picks = [  # (file, temperature_K) ascending
        ("JCESR_MgOTf2_G2_TEP__exp5600.npy", 243.2),
        ("JCESR_MgOTf2_G2_TEP__exp5400.npy", 258.2),
        ("JCESR_MgOTf2_G2_TEP__exp200.npy", 278.2),
        ("JCESR_MgOTf2_G2_TEP__exp1200.npy", 293.5),
        ("JCESR_MgOTf2_G2_TEP__exp6300.npy", 313.2),
    ]
    campaign_meta = json.load(open(
        BENCH / "NMR_staged" / "02_medium_JCESR_MgOTf2_TEP" / "metadata.json"))
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "curve_series"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        [str(ddir / f) for f, _ in picks],
        system_info=campaign_meta,
        series_metadata={"variable": "temperature",
                         "values": [t for _, t in picks], "unit": "K"},
        objective=("Track how the 31P NMR spectrum of the mixed G2/TEP "
                   "Mg-electrolyte evolves with temperature (313 K to 243 K): "
                   "peak positions, widths, and any new species or exchange "
                   "behavior relevant to competitive solvation."),
    )
    _dump("curve_series", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE CURVE SERIES START\n{'='*70}", flush=True)
    try:
        status = run_curve_series().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE CURVE SERIES DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
