"""Script-bank write-hook live matrix (issue #346 step 1) — bedrock opus-4-8.

Runs, sequentially:
  1. curve        — Raman strong-metadata; expect one bank record created with
                    a peak-census fingerprint + Raman context.
  2. curve_reuse  — same data, fresh agent, reuse_locked_script=True on run 1's
                    output; identical script => dedup path: action=updated,
                    n_successes=2, two sessions.
  3. image        — TEM AuNP; image fingerprint (shape/pixel size/texture).
  4. hyper        — Au coupon K-edge cube; hyperspectral fingerprint
                    (axis 20-160 keV, band means, edge peaks).

Bank enabled via SCILINK_SCRIPT_BANK=1 with the memory master switch OFF —
exercises the override path live (staging stays inert, bank writes).

Usage: python run_bank_live.py [names...]   (needs AWS_BEARER_TOKEN_BEDROCK)
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
os.environ["SCILINK_MEMORY"] = "0"        # master off ...
os.environ["SCILINK_SCRIPT_BANK"] = "1"   # ... bank on via override

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code" / "benchmarking_for_paper2"
PHASE2 = OUT.parent / "_qc_phase2_live_runs"


def _show_bank(tag):
    from scilink.skills._shared import _script_bank as sb
    print(f"\n----- BANK STATE after {tag} -----")
    for rec in sb.list_records():
        fp = rec.get("data_fingerprint") or {}
        peaks = (fp.get("peaks") or {}).get("count")
        print(json.dumps({
            "id": rec["id"], "domain": rec["domain"],
            "n_successes": rec["stats"]["n_successes"],
            "sessions": rec["sessions"],
            "fp_kind": fp.get("kind"), "fp_peaks": peaks,
            "fp_x_range": fp.get("x_range") or (fp.get("axis") or {}).get("units"),
            "fp_snr": fp.get("snr"),
            "context_keys": list((rec.get("measurement_context") or {}).keys())[:8],
            "metric": rec.get("outcome", {}).get("metric"),
            "best_metric": rec.get("outcome", {}).get("best_metric"),
            "script_chars": len(rec.get("working_script") or ""),
        }))
    print("----- END BANK STATE -----\n", flush=True)


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
    print(f"\n[{name}] status={result.get('status')} "
          f"banked={result.get('banked_scripts')} -> {path}", flush=True)


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
    _show_bank("curve")
    return res


def run_curve_reuse():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    case = BENCH / "raman_chris" / "strong_metadata" / "Raman_1_data_hold"
    agent = CurveFittingAgent(
        model_name=MODEL, output_dir=str(OUT / "curve_reuse"),
        enable_human_feedback=False, use_literature=False,
    )
    res = agent.analyze(str(case / "data1.csv"),
                        system_info=str(case / "metadata.json"),
                        prior_analysis_paths=[str(OUT / "curve_raman")],
                        reuse_locked_script=True)
    _dump("curve_reuse", res)
    _show_bank("curve_reuse")
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
    _show_bank("image")
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
        auxiliary_label="baseline_I0",
        literature_file=str(PHASE2 / "lit_xray_kedge.md"),
    )
    _dump("hyper", res)
    _show_bank("hyper")
    return res


ALL = {
    "curve": run_curve,
    "curve_reuse": run_curve_reuse,
    "image": run_image,
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
