import json, logging, os, sys
sys.path.insert(0, "tests/_qc_phase2_live_runs")
os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
logging.basicConfig(level=logging.INFO)
from run_reentry_live2 import test_c_orchestrator_routing
print("C_rerun ->", test_c_orchestrator_routing())
