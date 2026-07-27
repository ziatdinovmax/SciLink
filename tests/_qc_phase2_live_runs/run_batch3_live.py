"""Live batch 3 — new benchmark families (benchmarking_for_paper2).

  1. image_llto   — STEM HAADF of LLTO/LSAT (atomic_stem skill + lattice tool)
  2. image_hbn    — (hard) hBN SPM
  3. image_om     — (easy) vaterite optical microscopy
  4. curve_nmr_hard — solid-state 23Na MAS of NLTO (known hard regime)
  5. curve_xrd_series — 6-frame in-situ dehydration XRD spanning the known
     dihydrate→anhydrous transition (~43-48 C): the anchor model SHOULD fail
     post-transition → outlier flagging + adaptive refit under a real regime
     change. Frames pre-cropped to 5-45 deg (documented low-angle limitation,
     xrd_exp/FINDINGS.md #1).

Usage: python run_batch3_live.py [names...]
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


def _image(name, path, meta, objective=None):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / name),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(str(path), system_info=str(meta), objective=objective)
    _dump(name, res)
    return res


def run_image_llto():
    return _image("image_llto", BENCH / "TEM" / "(easy) LLTO.tif",
                  BENCH / "TEM" / "(easy) LLTO.json",
                  objective=("Characterize the atomic lattice: measure the "
                             "lattice constant(s) and identify any structural "
                             "features of the LLTO film vs the LSAT substrate."))


def run_image_hbn():
    return _image("image_hbn", BENCH / "AFM" / "(hard) hBN SPM.tif",
                  BENCH / "AFM" / "(hard) hBN SPM.json")


def run_image_om():
    return _image("image_om", BENCH / "OM" / "(easy) vaterite.tif",
                  BENCH / "OM" / "(easy) vaterite.json",
                  objective="Count the vaterite particles and measure their size distribution.")


def run_curve_nmr_hard():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    ddir = BENCH / "NMR_staged" / "04_hard_ESRA_NLTO_23Na"
    meta = json.load(open(ddir / "metadata.json"))
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "curve_nmr_hard"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        str(ddir / "data" / "20250407_1p9mm_RT_NLTO_NM_23Na__exp1.npy"),
        system_info=meta,
        objective=("Fit the 23Na MAS spectrum of the NLTO synthesis product: "
                   "resolve the distinct sodium environments (quadrupolar "
                   "lineshapes expected) and flag any byproduct signals."),
    )
    _dump("curve_nmr_hard", res)
    return res


def run_curve_xrd_series():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    frames = [("0001_0028-7C.csv", 28.7), ("0004_0035-9C.csv", 35.9),
              ("0006_0040-9C.csv", 40.9), ("0009_0048-3C.csv", 48.3),
              ("0012_0055-8C.csv", 55.8), ("0016_0065-7C.csv", 65.7)]
    ddir = OUT / "xrd_cropped"
    agent = CurveFittingAgent(model_name=MODEL, output_dir=str(OUT / "curve_xrd_series"),
                              enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        [str(ddir / f) for f, _ in frames],
        system_info={
            "technique": "In-situ powder XRD (Cu K-alpha, 1.5406 A)",
            "sample": "sodium ibuprofen dihydrate, heated under N2",
            "details": ("Temperature series during dehydration; 2theta 5-45 deg "
                        "(low-angle air scatter below 5 deg cropped), intensity cps."),
        },
        series_metadata={"variable": "temperature",
                         "values": [t for _, t in frames], "unit": "C"},
        objective=("Track the powder pattern through the dehydration: identify "
                   "whether and where a phase transition occurs across the "
                   "series and characterize the pattern changes."),
    )
    _dump("curve_xrd_series", res)
    return res


ALL = {
    "image_llto": run_image_llto,
    "image_hbn": run_image_hbn,
    "image_om": run_image_om,
    "curve_nmr_hard": run_curve_nmr_hard,
    "curve_xrd_series": run_curve_xrd_series,
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
