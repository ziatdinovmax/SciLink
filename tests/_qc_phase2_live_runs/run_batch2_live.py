"""Phase-2 live batch 2 (issue #327) — broader coverage incl. SERIES.

  1. curve-weak    — Raman, weak metadata (cold-start robustness)
  2. curve-series  — 5-spectrum VT 31P NMR series 243–313 K (layer-2: lock,
                     per-item QC, outlier detection, trend codegen, synthesis)
  3. image-medium  — (medium) c98RhuA TEM
  4. image-series  — AFM mica in KCl, 2-image concentration series (2, 10 mM)
  5. hyper-pb      — Pb coupon thickness via K-edge (filename truth: 100 um)
  6. hyper-solder  — In/Pb/Sn solders in Cu coupon: multi-element localization
                     (no literature — tests the plain path)

Usage: python run_batch2_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
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
from run_phase2_live import _dump  # noqa: E402


def run_curve_weak():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / "raman_chris" / "weak_metadata" / "Raman_3_data_hold"
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "curve_weak"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"))
    _dump("curve_weak", res)
    return res


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


def run_image_medium():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "image_medium"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        str(BENCH / "TEM" / "(medium) c98RhuA.tif"),
        system_info=str(BENCH / "TEM" / "(medium) c98RhuA.json"),
    )
    _dump("image_medium", res)
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


def run_hyper_pb():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    xdir = BENCH / "xray_hyper_data"
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_pb"),
        enable_human_feedback=False, executor_timeout=900)
    res = agent.analyze(
        str(xdir / "coupon_radiography_Pb100um.npy"),
        system_info=str(OUT / "metadata_pbcoupon.json"),
        objective=("Locate the Pb coupon and measure its thickness from the "
                   "K-edge jump at 88.0 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
        literature_file=str(OUT / "lit_xray_kedge_pb.md"),
    )
    _dump("hyper_pb", res)
    return res


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
        system_info=str(OUT / "metadata_cusolder_fixed.json"),
        objective=("Locate the In, Pb, and Sn solder regions in the Cu coupon "
                   "using their K-edges and map where each element is present."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
    )
    _dump("hyper_solder", res)
    return res


ALL = {
    "curve_weak": run_curve_weak,
    "curve_series": run_curve_series,
    "image_medium": run_image_medium,
    "image_series": run_image_series,
    "hyper_pb": run_hyper_pb,
    "hyper_solder": run_hyper_solder,
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
