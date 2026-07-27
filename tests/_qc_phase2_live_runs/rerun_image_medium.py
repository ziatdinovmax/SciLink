import os, sys
sys.path.insert(0, "tests/_qc_phase2_live_runs")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
from run_phase2_live import MODEL, BENCH, OUT, _dump
from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
agent = ImageAnalysisAgent(model_name=MODEL, output_dir=str(OUT / "image_medium"),
                           enable_human_feedback=False, use_literature=False)
res = agent.analyze(
    str(BENCH / "TEM" / "(medium) triangular AuNP.tif"),
    system_info=str(BENCH / "TEM" / "(medium) triangular AuNP.json"),
)
_dump("image_medium", res)
