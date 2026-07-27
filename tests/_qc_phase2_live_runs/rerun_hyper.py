import os, sys, json
sys.path.insert(0, "tests/_qc_phase2_live_runs")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
from pathlib import Path
from run_phase2_live import MODEL, BENCH, OUT, _dump
from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent

xdir = BENCH / "xray_hyper_data"
agent = HyperspectralAnalysisAgent(
    model_name=MODEL, output_dir=str(OUT / "hyper_au"),
    enable_human_feedback=False, executor_timeout=900,
)
res = agent.analyze(
    str(xdir / "coupon_radiography_Au.npy"),
    system_info=str(OUT / "metadata_aucoupon_fixed.json"),
    objective=("Locate the Au coupon and measure its thickness from the "
               "K-edge jump at 80.7 keV."),
    auxiliary_data=str(xdir / "coupon_radiography_Flat.npy"),
    auxiliary_label="I0 flat-field baseline cube (no sample) on the same grid",
    literature_file=str(OUT / "lit_xray_kedge.md"),
)
_dump("hyper", res)
