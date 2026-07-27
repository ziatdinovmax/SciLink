"""Cross-modality script-bank generalization matrix (#346 step 2) — bedrock opus-4-8.

Uses the bank grown by the previous live runs (Raman curve records, AuNP
image record, Au-coupon hyperspectral record, XRD record if miss_xrd banked).

  1. image_hit    — (medium) AuNP self-assembly TEM: dry-run 0.705 vs the AuNP
                    record; exemplar must be offered and ADAPTED (different
                    particle arrangement).
  2. image_adv    — (hard) fmica 2mM KCl AFM: dry-run 0.575 — a WRONG-MODALITY
                    exemplar (TEM watershed sizing offered for an AFM height
                    map). Adversarial: locked plan + adapt framing +
                    verification must prevail; watch for particle-sizing
                    nonsense on mica.
  3. hyper_hit    — Au coupon cube re-run: exemplar (K-edge thickness script)
                    must be offered on the thickness target.
  4. hyper_miss   — low-loss plasmon EELS cube (eV axis): axis-units hard
                    filter vs the keV record -> NO exemplar line.
  5. precedence   — Raman_1 with prior_analysis_paths + reuse_locked_script:
                    explicit prior wins -> NO exemplar line, reuse verdict
                    good, bank dedup update.

Usage: python run_general_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
"""

import json
import os
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_HOME"] = str(OUT / "scilink_home")
os.environ["SCILINK_MEMORY"] = "0"
os.environ["SCILINK_SCRIPT_BANK"] = "1"

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
PHASE2 = OUT.parent / "_qc_phase2_live_runs"
PLASMON = Path.home() / "Code" / "SciLink" / "examples" / "eels_plasmons_demo"


def _dump(name, result):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "visualization_bytes"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return "<bytes>" if isinstance(o, bytes) else o
    (OUT / f"result_{name}.json").write_text(
        json.dumps(clean(result), indent=2, default=str))
    print(f"\n[{name}] status={result.get('status')} "
          f"banked={result.get('banked_scripts')}", flush=True)


def run_image_hit():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "image_hit"),
        enable_human_feedback=False, use_literature=False, analysis_depth="auto",
    )
    res = agent.analyze(
        str(BENCH / "TEM" / "(medium) AuNP self-assembly.tif"),
        system_info=str(BENCH / "TEM" / "(medium) AuNP self-assembly.json"),
        objective="Characterize the nanoparticle arrangement and size distribution.",
    )
    _dump("image_hit", res)
    return res


def run_image_adv():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    agent = ImageAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "image_adv"),
        enable_human_feedback=False, use_literature=False, analysis_depth="auto",
    )
    res = agent.analyze(
        str(BENCH / "AFM" / "(hard) fmica 2mM KCl.tif"),
        system_info=str(BENCH / "AFM" / "(hard) fmica 2mM KCl.json"),
        objective="Characterize the surface structure/adsorbate pattern on mica.",
    )
    _dump("image_adv", res)
    return res


def run_hyper_hit():
    # Cross-SAMPLE adapt: the banked Au K-edge thickness script (80.7 keV,
    # attenuation('Au')) is offered for a Bi coupon (dry-run 0.882, fp 0.998
    # — same instrument signature); the adapted script must switch to the Bi
    # K-edge at 90.5 keV and attenuation('Bi').
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    xdir = BENCH / "xray_hyper_data"
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_hit"),
        enable_human_feedback=False, executor_timeout=900,
    )
    res = agent.analyze(
        str(xdir / "coupon_radiography_Bi.npy"),
        system_info=str(OUT / "metadata_bicoupon.json"),
        objective=("Locate the Bi coupon and measure its thickness from the "
                   "Bi K-edge jump at 90.5 keV."),
        auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
        auxiliary_label="baseline_I0",
        literature_file=str(PHASE2 / "lit_xray_kedge.md"),
    )
    _dump("hyper_hit", res)
    return res


def run_hyper_miss():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    meta = json.load(open(PLASMON / "datacube.json"))
    agent = HyperspectralAnalysisAgent(
        model_name=MODEL, output_dir=str(OUT / "hyper_miss"),
        enable_human_feedback=False, executor_timeout=900,
    )
    res = agent.analyze(str(PLASMON / "datacube.npy"), system_info=meta)
    _dump("hyper_miss", res)
    return res


def run_precedence():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
    agent = CurveFittingAgent(
        model_name=MODEL, output_dir=str(OUT / "precedence"),
        enable_human_feedback=False, use_literature=False,
    )
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"),
                        prior_analysis_paths=[str(OUT / "adapt_self")],
                        reuse_locked_script=True)
    _dump("precedence", res)
    print("reuse verdict:", (res.get("reuse_validity") or {}).get("verdict"),
          flush=True)
    return res


ALL = {
    "image_hit": run_image_hit,
    "image_adv": run_image_adv,
    "hyper_hit": run_hyper_hit,
    "hyper_miss": run_hyper_miss,
    "precedence": run_precedence,
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
