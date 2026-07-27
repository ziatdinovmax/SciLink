"""Phase-2 live validation batch (issue #327) — bedrock opus-4-8.

Runs, sequentially:
  1. curve smoke  — real Raman spectrum (benchmarking_for_paper2)
  2. image smoke  — real TEM AuNP image
  3. hyperspectral — Au coupon K-edge cube + Flat I0 companion +
     literature_file (validates HS-1 records/staging surface + HS-2
     literature + feature surfacing end to end)

Folds in the deferred phase-1 smoke (curve/image exercise the routed gate +
shared history builders under a real LLM).

Usage: python run_phase2_live.py   (needs AWS_BEARER_TOKEN_BEDROCK)
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
RESULTS = {}


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


def run_curve():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
    agent = CurveFittingAgent(
        model_name=MODEL, output_dir=str(OUT / "curve_raman"),
        enable_human_feedback=False, use_literature=False,
    )
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"))
    _dump("curve", res)
    return res


def run_image():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "image_aunp"),
        enable_human_feedback=False, use_literature=False,
        analysis_depth="auto",
    )
    res = agent.analyze(
        str(BENCH / "TEM" / "(easy) AuNP.tif"),
        system_info=str(BENCH / "TEM" / "(easy) AuNP.json"),
        objective="Measure the size distribution of the Au nanoparticles.",
    )
    _dump("image", res)
    return res


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
        system_info=str(xdir / "metadata_aucoupon.json"),
        objective=("Locate the Au coupon and measure its thickness from the "
                   "K-edge jump at 80.7 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
        literature_file=str(OUT / "lit_xray_kedge.md"),
    )
    _dump("hyper", res)
    return res


if __name__ == "__main__":
    for name, fn in [("curve", run_curve), ("image", run_image),
                     ("hyper", run_hyper)]:
        t0 = time.time()
        print(f"\n{'='*70}\nLIVE {name.upper()} START\n{'='*70}", flush=True)
        try:
            RESULTS[name] = fn().get("status")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            RESULTS[name] = f"EXCEPTION: {e}"
        print(f"LIVE {name.upper()} DONE in {time.time()-t0:.0f}s "
              f"-> {RESULTS[name]}", flush=True)
    print("\nSUMMARY:", json.dumps(RESULTS, indent=2))
