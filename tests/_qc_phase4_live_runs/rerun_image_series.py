"""Phase-4 live matrix — image_series re-run with the REAL metadata sidecars.

The first run passed a hand-written system_info dict with no field-of-view,
so the AFM skill's FOV mandate killed the 10 mM frame (codegen scripts raised
on the missing FOV). This re-run feeds the benchmark's actual per-image
sidecars (2 mM: 15.03 nm FOV, z in nm; 10 mM: 22.04x22.09 nm FOV, z in pm) to
exercise the engine's non-anchor SUCCESS branch live.
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
from run_phase4_live import _dump  # noqa: E402


def run_image_series_v2():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    meta_2mm = json.load(open(BENCH / "AFM" / "(hard) fmica 2mM KCl.json"))
    meta_10mm = json.load(open(BENCH / "AFM" / "(hard) fmica 10mM KCl.json"))
    agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "image_series_v2"),
                               enable_human_feedback=False, use_literature=False)
    res = agent.analyze(
        [str(BENCH / "AFM" / "(hard) fmica 2mM KCl.tif"),
         str(BENCH / "AFM" / "(hard) fmica 10mM KCl.tif")],
        system_info={
            "material_type": "mica surface imaged in KCl solution",
            "microscopy_type": "AFM height map",
            "notes": ("Two images of the same mica/KCl system at different "
                      "KCl concentrations; sub-nm height contrast, atomic-scale "
                      "lattice possibly visible. Per-image acquisition metadata "
                      "below — note the two frames have DIFFERENT fields of view "
                      "and different z-unit calibrations."),
            "per_image_metadata": {
                "(hard) fmica 2mM KCl": meta_2mm,
                "(hard) fmica 10mM KCl": meta_10mm,
            },
        },
        series_metadata={"variable": "KCl concentration",
                         "values": [2, 10], "unit": "mM"},
        objective=("Characterize how the observed surface structure/adsorbate "
                   "pattern on mica changes between 2 mM and 10 mM KCl."),
    )
    _dump("image_series_v2", res)
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"\n{'='*70}\nLIVE IMAGE_SERIES_V2 START\n{'='*70}", flush=True)
    try:
        status = run_image_series_v2().get("status")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"EXCEPTION: {e}"
    print(f"LIVE IMAGE_SERIES_V2 DONE in {time.time()-t0:.0f}s -> {status}", flush=True)
