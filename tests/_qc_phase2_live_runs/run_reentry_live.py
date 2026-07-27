"""Phase-3 live validation: reenter_interpretation on real batch-2 records.

Exercises the SynthesisReEntryController (human producer) via the orchestrator
tool on: (1) the VT-NMR series record with a domain-expert critique, then a
COMPOSING second critique; (2) the triangular-AuNP image record (feature
grounding via extracted_features). refine_interpretation's literature leg
needs FUTUREHOUSE_API_KEY — deferred.
"""

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
logging.basicConfig(level=logging.INFO)

OUT = Path(__file__).parent
MODEL = "bedrock/us.anthropic.claude-opus-4-8"

from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel  # noqa: E402
from scilink.agents.exp_agents.analysis_orchestrator_tools import (  # noqa: E402
    AnalysisOrchestratorTools,
)

model = LiteLLMGenerativeModel(model=MODEL)

records = [
    {"analysis_id": "nmr_vt_series_001",
     "full_result": json.load(open(OUT / "result_curve_series.json")),
     "output_directory": str(OUT / "curve_series")},
    {"analysis_id": "aunp_triangular_001",
     "full_result": json.load(open(OUT / "result_image_medium.json")),
     "output_directory": str(OUT / "image_medium")},
]

orch = SimpleNamespace(analysis_results=records, model=model,
                       futurehouse_api_key=None, current_metadata=None,
                       base_dir=OUT)
tools = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
tools.orch = orch
tools.functions_map = {}
tools.openai_schemas = []
tools._register_all_tools()
reenter = tools.functions_map["reenter_interpretation"]

print("\n=== 1. curve series: expert critique ===")
r1 = json.loads(reenter(
    critique=("Domain expert review: state the coalescence temperature the FWHM "
              "trend implies explicitly, and tie the two-line slow-exchange regime "
              "at 243 K to the two competing solvation environments (G2 vs TEP "
              "coordination of Mg2+) rather than leaving the assignment generic."),
    analysis_id="nmr_vt_series_001"))
print("status:", r1["status"], "| summary:", r1.get("revision_summary", "")[:200])

print("\n=== 2. curve series: composing second critique ===")
r2 = json.loads(reenter(
    critique=("Also add the practical implication: which temperature window is "
              "safe for quantitative single-line integration."),
    analysis_id="nmr_vt_series_001"))
print("status:", r2["status"], "| summary:", r2.get("revision_summary", "")[:200])

print("\n=== 3. image: critique grounded in extracted_features ===")
r3 = json.loads(reenter(
    critique=("The particles are triangular nanoplates: equivalent-circle diameter "
              "systematically understates their edge length. Re-express the size "
              "statistics as triangle edge length (assuming equilateral geometry, "
              "edge = diameter * 1.35 approx) and note the 3 edge-truncated "
              "particles were excluded."),
    analysis_id="aunp_triangular_001"))
print("status:", r3["status"], "| summary:", r3.get("revision_summary", "")[:200])

out = {
    "curve_revisions": records[0].get("interpretation_revisions"),
    "image_revisions": records[1].get("interpretation_revisions"),
}
(OUT / "result_reentry_live.json").write_text(json.dumps(out, indent=2, default=str))
print("\nsaved -> result_reentry_live.json")
print("curve revisions:", len(out["curve_revisions"] or []),
      "| image revisions:", len(out["image_revisions"] or []))
