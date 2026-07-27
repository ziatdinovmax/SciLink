"""Phase-3 live validation of refine_interpretation's LITERATURE leg (#323).

Runs the full feature-conditioned pipeline (feature block → LLM query builder
→ FutureHouse/CROW search → Tier-A revision → claims revision → append-only
storage) on two real batch-2 records:

  1. image  — triangular AuNP (extracted_features path = the phase-3
              modality extension; was curve-only v1 before)
  2. curve  — weak-metadata Raman of an UNKNOWN compound (the canonical
              use case: identify the material from its fitted bands)

Requires FUTUREHOUSE_API_KEY + bedrock env. Each search may take ~20 min.
"""

import json
import logging
import os
import time
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

records = [
    {"analysis_id": "aunp_triangular_lit",
     "full_result": json.load(open(OUT / "result_image_medium.json")),
     "output_directory": str(OUT / "image_medium")},
    {"analysis_id": "raman_unknown_lit",
     "full_result": json.load(open(OUT / "result_curve_weak.json")),
     "output_directory": str(OUT / "curve_weak")},
]

orch = SimpleNamespace(
    analysis_results=records,
    model=LiteLLMGenerativeModel(model=MODEL),
    futurehouse_api_key=os.environ["FUTUREHOUSE_API_KEY"],
    current_metadata=None,
    base_dir=OUT,
)
tools = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
tools.orch = orch
tools.functions_map = {}
tools.openai_schemas = []
tools._register_all_tools()
refine = tools.functions_map["refine_interpretation"]

runs = [
    ("aunp_triangular_lit",
     "reported syntheses and size ranges of triangular gold nanoplates and "
     "what their edge-length distribution implies"),
    ("raman_unknown_lit", None),  # unconstrained — pure feature-conditioned ID
]

summary = {}
for rid, focus in runs:
    t0 = time.time()
    print(f"\n{'='*70}\nREFINE-LIT {rid} START (focus={bool(focus)})\n{'='*70}",
          flush=True)
    try:
        out = json.loads(refine(analysis_id=rid, focus=focus))
        summary[rid] = out.get("status")
        print(f"status: {out.get('status')} in {time.time()-t0:.0f}s")
        print("query:", out.get("query", "")[:160])
        print("lit file:", out.get("literature_file"))
        print("claims revised:", out.get("claims_revised"))
        print("preview:", out.get("revised_interpretation_preview", "")[:400])
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        summary[rid] = f"EXCEPTION: {e}"

rev_dump = {r["analysis_id"]: r.get("interpretation_revisions") for r in records}
(OUT / "result_refine_lit_live.json").write_text(
    json.dumps(rev_dump, indent=2, default=str))
print("\nSUMMARY:", json.dumps(summary, indent=2))
