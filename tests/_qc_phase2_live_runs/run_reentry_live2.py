"""Phase-3 live batch 2 — no-FutureHouse coverage.

A. refine_interpretation graceful no-key degradation (no LLM call).
B. Agent-level reenter_interpretation on the DEGRADED solder record with a
   verifier-source CritiquePayload + structured hints (honesty preservation).
C. Orchestrator ROUTING: a real AnalysisOrchestratorAgent chat turn must
   discover the new reenter_interpretation tool from its schema and call it
   (not re-run the analysis) when the user asks for a revision.
D. Full HS Au-coupon rerun — revalidates the MOVED reflection pair
   (RunSelfReflection/ApplyReflectionUpdates now in base_controllers) inside
   the default hyperspectral synthesis pipeline.
"""

import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
os.environ["UNSAFE_EXECUTION_OK"] = "true"
logging.basicConfig(level=logging.INFO)

OUT = Path(__file__).parent
MODEL = "bedrock/us.anthropic.claude-opus-4-8"
SUMMARY = {}


def test_a_no_key_degradation():
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        AnalysisOrchestratorTools,
    )

    class _NoCallModel:
        def generate_content(self, *a, **k):
            raise AssertionError("no-key path must not call the LLM")

    orch = SimpleNamespace(analysis_results=[{"analysis_id": "x", "full_result":
                          {"detailed_analysis": "t", "model_type": "m"}}],
                           model=_NoCallModel(), futurehouse_api_key=None,
                           current_metadata=None, base_dir=OUT)
    tools = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    tools.orch = orch
    tools.functions_map = {}
    tools.openai_schemas = []
    tools._register_all_tools()
    out = json.loads(tools.functions_map["refine_interpretation"]())
    assert out["status"] == "error" and "Key" in out["message"], out
    return f"graceful error: {out['message'][:60]}"


def test_b_agent_reentry_degraded_record():
    from scilink.agents.exp_agents._critique import CritiquePayload
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel

    class _Concrete(BaseAnalysisAgent):
        def analyze(self, data, system_info=None, **kwargs):
            raise NotImplementedError

    agent = _Concrete.__new__(_Concrete)
    agent.logger = logging.getLogger("reentry_live_b")
    agent.model = LiteLLMGenerativeModel(model=MODEL)
    agent.generation_config = None
    agent.safety_settings = None
    agent._stored_analysis_images = []
    agent._stored_analysis_metadata = {}

    result = json.load(open(OUT / "result_hyper_solder.json"))
    payload = CritiquePayload(
        source="verifier",
        critique=(
            "QC review: the Sn K-edge map never passed verification and the In "
            "map is contaminated by In/Sn edge cross-talk (edges 1.26 keV apart "
            "at 0.2 keV resolution). The final interpretation must not overstate "
            "In localization: fold the approximate/low-confidence caveats into "
            "the narrative itself, clearly attribute which conclusions rest on "
            "the robust Pb edge vs the marginal In/Sn pair, and state what "
            "measurement or analysis change would resolve In vs Sn."
        ),
        hints={"in_kedge_keV": 27.94, "sn_kedge_keV": 29.20,
               "detector_resolution_keV": 0.2,
               "suggested_fix": "joint two-edge constrained fit over 26-31 keV"},
    )
    out = agent.reenter_interpretation(result, payload)
    assert out["status"] == "success", out
    rev = out["revision"]
    (OUT / "result_reentry_solder.json").write_text(
        json.dumps(rev, indent=2, default=str))
    text = rev["revised_analysis"].lower()
    honest = any(w in text for w in ("low confidence", "approximate", "cannot",
                                     "uncertain", "caution", "tentative"))
    return (f"revised {len(rev['revised_analysis'])} chars | honesty language "
            f"preserved: {honest} | summary: {rev['revision_summary'][:120]}")


def test_c_orchestrator_routing():
    import tempfile
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisMode,
        AnalysisOrchestratorAgent,
    )

    session = Path(tempfile.mkdtemp(prefix="reentry_route_"))
    orch = AnalysisOrchestratorAgent(
        base_dir=str(session),
        model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    record = {
        "analysis_id": "nmr_vt_series_route",
        "timestamp": "2026-07-09T00:00:00",
        "data_path": "vt_series.npy",
        "agent_id": "curve_fitting",
        "agent_name": "CurveFittingAgent",
        "status": "success",
        "output_directory": str(OUT / "curve_series"),
        "literature_file": None,
        "full_result": json.load(open(OUT / "result_curve_series.json")),
        "novelty_assessment": None,
    }
    orch.analysis_results.append(record)

    reply = orch.chat(
        "Without re-running anything: the interpretation of the VT-NMR series "
        "analysis should explicitly state the coalescence temperature window "
        "implied by the FWHM trend, and say which temperatures permit "
        "quantitative single-line integration. Please revise it accordingly."
    )
    revs = record.get("interpretation_revisions") or []
    routed = len(revs) > 0
    return (f"routed_to_reentry_tool: {routed} | revisions: {len(revs)} | "
            f"reply preview: {str(reply)[:160]}")


def test_d_hs_pipeline_moved_reflection():
    import subprocess
    import sys
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", str(OUT / "rerun_hyper.py")],
        capture_output=True, text=True, timeout=3600,
        cwd="/Users/maxim.ziatdinov/Code/SciLink",
    )
    log = proc.stdout + proc.stderr
    (OUT / "hyper2.log").write_text(log)
    res = json.load(open(OUT / "result_hyper.json"))
    reflection_ran = "SELF-REFLECTION: REVIEWING ANALYSIS" in log
    thick = next((f for f in res.get("extracted_features", []) or []
                  if "Thickness" in str(f.get("name"))), {})
    return (f"status: {res.get('status')} in {time.time()-t0:.0f}s | moved "
            f"reflection ran: {reflection_ran} | thickness stats: "
            f"{thick.get('stats')}")


if __name__ == "__main__":
    for name, fn in [("A_no_key", test_a_no_key_degradation),
                     ("B_agent_reentry_degraded", test_b_agent_reentry_degraded_record),
                     ("C_orchestrator_routing", test_c_orchestrator_routing),
                     ("D_hs_moved_reflection", test_d_hs_pipeline_moved_reflection)]:
        print(f"\n{'='*70}\n{name} START\n{'='*70}", flush=True)
        try:
            SUMMARY[name] = fn()
            print(f"{name} -> {SUMMARY[name]}", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            SUMMARY[name] = f"EXCEPTION: {e}"
    print("\nSUMMARY:", json.dumps(SUMMARY, indent=2))
