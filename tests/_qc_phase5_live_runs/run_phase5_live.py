"""Phase-5 live gate — HS dynamic-analysis loop on the shared engine.

Same Au-coupon K-edge case as the phase-4 matrix (and phases 2-3), so the
result is directly comparable: expect ~40 um vs ~39 um truth, per-target
dynamic_analysis_records with quality_history, literature threading, and
no T=2 staging (level-0 success).
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


def _dump(name, result):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "visualization_bytes"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, bytes):
            return "<bytes>"
        return o
    path = OUT / f"result_{name}.json"
    path.write_text(json.dumps(clean(result), indent=2, default=str))
    print(f"\n[{name}] status={result.get('status')} -> {path}")


def run_hyper():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    xdir = BENCH / "xray_hyper_data"
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_au"),
        enable_human_feedback=False, executor_timeout=900,
    )
    res = agent.analyze(
        str(xdir / "coupon_radiography_Au.npy"),
        system_info=str(PHASE2 / "metadata_aucoupon_fixed.json"),
        objective=("Locate the Au coupon and measure its thickness from the "
                   "K-edge jump at 80.7 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
        literature_file=str(PHASE2 / "lit_xray_kedge.md"),
    )
    _dump("hyper", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE HYPER (phase-5 engine) START\n{'='*70}", flush=True)
    try:
        status = run_hyper().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE HYPER DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
