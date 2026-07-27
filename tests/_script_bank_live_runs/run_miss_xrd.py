"""True-miss live case: XRD pattern vs a Raman-only bank -> no exemplar offered."""
import json, os
from pathlib import Path

OUT = Path(__file__).parent
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
os.environ["SCILINK_HOME"] = str(OUT / "scilink_home")
os.environ["SCILINK_MEMORY"] = "0"
os.environ["SCILINK_SCRIPT_BANK"] = "1"

from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

agent = CurveFittingAgent(
    model_name="bedrock/us.anthropic.claude-opus-4-8",
    output_dir=str(OUT / "miss_xrd"),
    enable_human_feedback=False, use_literature=False,
)
res = agent.analyze(
    str(Path.home() / "Code/benchmarking_for_paper2/xrd_exp/IBD_Insitu_Dehydration_01_0001_0028-7C.txt"),
    system_info={"technique": "powder XRD", "description":
                 "In-situ synchrotron XRD during dehydration, frame 1 (28.7 C); x is 2-theta in degrees."},
)
print(f"\n[miss_xrd] status={res.get('status')} "
      f"r2={(res.get('fit_quality') or {}).get('r_squared')} "
      f"banked={res.get('banked_scripts')}", flush=True)
