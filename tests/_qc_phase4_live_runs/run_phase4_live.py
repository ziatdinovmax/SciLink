"""Phase-4 live matrix (issue #327) — CodegenQCEngine extraction, bedrock opus-4-8.

Runs, sequentially:
  1. curve        — Raman strong-metadata (happy path: engine accept/threshold return)
  2. curve_weak   — Raman weak-metadata (exercises verification loop + annealing
                    escalation + refit THROUGH the engine; hit level 1 in phase-2)
  3. image        — TEM AuNP (verdict-score loop + post-verification accept)
  4. image_series — 2-frame AFM KCl (anchor + non-anchor + locked-script path
                    through the engine wrapper)
  5. hyper        — Au coupon K-edge (modality untouched by phase 4; cross-import
                    sanity + reads the same shared modules)

Usage: python run_phase4_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
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


def run_curve_weak():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / "raman_chris" / "weak_metadata" / "Raman_3_data_hold"
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "curve_weak"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"))
    _dump("curve_weak", res)
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


def run_image_series():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "image_series"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        [str(BENCH / "AFM" / "(hard) fmica 2mM KCl.tif"),
         str(BENCH / "AFM" / "(hard) fmica 10mM KCl.tif")],
        system_info={
            "material_type": "mica surface imaged in KCl solution",
            "microscopy_type": "AFM height map",
            "notes": ("Two images of the same mica/KCl system at different "
                      "KCl concentrations; sub-nm height contrast, atomic-scale "
                      "lattice possibly visible."),
        },
        series_metadata={"variable": "KCl concentration",
                         "values": [2, 10], "unit": "mM"},
        objective=("Characterize how the observed surface structure/adsorbate "
                   "pattern on mica changes between 2 mM and 10 mM KCl."),
    )
    _dump("image_series", res)
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
        system_info=str(PHASE2 / "metadata_aucoupon_fixed.json"),
        objective=("Locate the Au coupon and measure its thickness from the "
                   "K-edge jump at 80.7 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
        literature_file=str(PHASE2 / "lit_xray_kedge.md"),
    )
    _dump("hyper", res)
    return res


ALL = {
    "curve": run_curve,
    "curve_weak": run_curve_weak,
    "image": run_image,
    "image_series": run_image_series,
    "hyper": run_hyper,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    results = {}
    for name in names:
        t0 = time.time()
        print(f"\n{'='*70}\nLIVE {name.upper()} START\n{'='*70}", flush=True)
        try:
            results[name] = ALL[name]().get("status")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = f"EXCEPTION: {e}"
        print(f"LIVE {name.upper()} DONE in {time.time()-t0:.0f}s "
              f"-> {results[name]}", flush=True)
    print("\nSUMMARY:", json.dumps(results, indent=2))
