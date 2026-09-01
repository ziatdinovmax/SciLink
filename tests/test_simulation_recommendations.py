"""Offline tests for #45: scale-neutral simulation recommendations.

  python -m pytest tests/test_simulation_recommendations.py -v
"""
import json
import os
import types

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest

from scilink.agents.exp_agents.recommendation_agent import RecommendationAgent

RECS = [
    {"description": "3x3x3 Si supercell with a C substitutional defect",
     "research_goal": "Compute the defect formation energy",
     "suggested_scale": "periodic_dft",
     "scientific_interest": "Tests the Z-contrast assignment",
     "priority": 1},
    {"description": "10x10x10 anatase TiO2 supercell, 2% O vacancies",
     "research_goal": "MD from 300-800 K for domain stability",
     "suggested_scale": "molecular_dynamics",
     "scientific_interest": "Thermal stability of the observed pattern",
     "priority": 2},
    {"description": "Isolated CO2 + amine cluster",
     "research_goal": "Binding energetics of the capture complex",
     "suggested_scale": "molecular_qc",
     "scientific_interest": "x", "priority": 3},
    {"description": "Graphene on Ni(111) interface",
     "research_goal": "Interfacial adhesion energy",
     "suggested_scale": "machine_learning_potentials",  # NOT a scale (#429)
     "scientific_interest": "xx", "priority": 5},
    {"description": "missing goal is fine (optional key)",
     "scientific_interest": "y", "priority": 4},
    {"description": "bad priority", "scientific_interest": "z",
     "priority": "high"},                          # skipped entirely
]


def _stub_agent(payload):
    ag = RecommendationAgent.__new__(RecommendationAgent)
    import logging
    ag.logger = logging.getLogger("test_recs")
    ag.generation_config = None
    ag.safety_settings = None

    class _Model:
        def generate_content(self, contents, generation_config=None,
                             safety_settings=None):
            return types.SimpleNamespace(text=json.dumps(payload))

    ag.model = _Model()
    return ag


def test_validation_scale_and_priority():
    ag = _stub_agent({"detailed_reasoning_for_recommendations": "r",
                      "structure_recommendations": RECS})
    out = ag.generate_simulation_recommendations_from_text("analysis", "ctx")
    recs = out["recommendations"]
    assert [r["priority"] for r in recs] == [1, 2, 3, 4, 5]  # sorted; bad dropped
    assert recs[0]["suggested_scale"] == "periodic_dft"
    assert recs[1]["suggested_scale"] == "molecular_dynamics"
    assert recs[2]["suggested_scale"] == "molecular_qc"
    assert "suggested_scale" not in recs[4]   # MLIP is not a scale -> dropped
    assert "research_goal" in recs[0]
    # A rec without a research_goal falls back to its description, so the
    # simulate-mode handoff never carries an empty goal.
    assert recs[3]["research_goal"] == recs[3]["description"]


def test_backcompat_method_aliases_new_path():
    ag = _stub_agent({"detailed_reasoning_for_recommendations": "r",
                      "structure_recommendations": RECS[:1]})
    out = ag.generate_dft_recommendations_from_text("analysis", "ctx")
    assert out["recommendations"][0]["suggested_scale"] == "periodic_dft"


# ------------------------------------------------------------ tool surface

@pytest.fixture()
def orch(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    return AnalysisOrchestratorAgent(api_key="sk-dummy",
                                     base_dir=str(tmp_path / "an"))


def test_schema_has_new_name_only_map_has_both(orch):
    names = {s["function"]["name"] for s in orch.tools.openai_schemas}
    assert "recommend_simulations" in names
    assert "recommend_dft_structures" not in names          # schema: new only
    assert "recommend_dft_structures" in orch.tools.functions_map  # alias
    assert "recommend_simulations" in orch.tools.functions_map


def test_tool_runs_and_stores_under_both_keys(orch, tmp_path, monkeypatch):
    import scilink.agents.exp_agents.analysis_orchestrator_tools as aot

    class _FakeRecAgent:
        def __init__(self, **kw):
            pass

        def generate_simulation_recommendations_from_text(self, **kw):
            return {"analysis_summary_or_reasoning": "because",
                    "recommendations": RECS[:2]}

    monkeypatch.setattr(aot, "RecommendationAgent", _FakeRecAgent)
    out_dir = tmp_path / "res"
    out_dir.mkdir()
    orch.analysis_results.append({
        "analysis_id": "demo_001",
        "output_directory": str(out_dir),
        "full_result": {"detailed_analysis": "We saw a substitutional defect "
                                             "and thermal domain motion."},
    })

    res = json.loads(orch.tools.execute_tool("recommend_simulations",
                                             analysis_index=-1))
    assert res["status"] == "success" and res["count"] == 2
    assert res["recommendations"][0]["suggested_scale"] == "periodic_dft"
    assert res["recommendations"][1]["research_goal"].startswith("MD from")

    rec = orch.analysis_results[-1]
    assert rec["simulation_recommendations"] == RECS[:2]
    assert rec["dft_recommendations"] == RECS[:2]           # back-compat key
    sidecar = out_dir / "simulation_recommendations" / \
        "simulation_recommendations.json"
    assert sidecar.exists()

    # The old tool name still dispatches (programmatic back-compat).
    res2 = json.loads(orch.tools.execute_tool("recommend_dft_structures",
                                              analysis_index=-1))
    assert res2["status"] == "success"


def test_run_dft_workflow_refuses_md_scale_recommendation(orch, tmp_path):
    out_dir = tmp_path / "res2"
    out_dir.mkdir()
    orch.analysis_results.append({
        "analysis_id": "demo_002",
        "output_directory": str(out_dir),
        "simulation_recommendations": RECS[:3],
    })
    res = json.loads(orch.tools.execute_tool(
        "run_dft_workflow", analysis_id="demo_002", recommendation_index=1))
    assert res["status"] == "error"
    assert "molecular_dynamics" in res["message"]
    assert "simulate mode" in res["message"]
    assert res["recommendation"]["suggested_scale"] == "molecular_dynamics"
    assert res["recommendation"]["research_goal"].startswith("MD from")


def test_run_dft_workflow_passes_qc_and_unscored_scales(orch, tmp_path,
                                                        monkeypatch):
    import sys
    stub = types.ModuleType("scilink.agents.sim_agents.simulation_pipeline")
    calls = []

    def run_complete_workflow(description, **kw):
        calls.append(description)
        return {"final_status": "success"}

    stub.run_complete_workflow = run_complete_workflow
    monkeypatch.setitem(
        sys.modules, "scilink.agents.sim_agents.simulation_pipeline", stub)

    out_dir = tmp_path / "res3"
    out_dir.mkdir()
    orch.analysis_results.append({
        "analysis_id": "demo_003",
        "output_directory": str(out_dir),
        "simulation_recommendations": [RECS[0], RECS[2], RECS[4]],
    })
    # periodic_dft, molecular_qc and a rec with no suggested_scale all pass.
    for idx in (0, 1, 2):
        res = json.loads(orch.tools.execute_tool(
            "run_dft_workflow", analysis_id="demo_003",
            recommendation_index=idx))
        assert res["status"] == "success", res
    assert len(calls) == 3


def test_system_prompt_mentions_new_tool(orch):
    sys_prompt = orch.messages[0]["content"]
    assert "recommend_simulations" in sys_prompt
    assert "recommend_dft_structures" not in sys_prompt


def test_valid_scales_match_the_simulate_router():
    """Drift guard: the recommendation agent's scale vocabulary is a mirror
    of the router's (duplicated so exp_agents imports without the sim
    stack). If this fails, update RecommendationAgent.VALID_SCALES and the
    TEXT_ONLY_SIMULATION_RECOMMENDATION_INSTRUCTIONS scale text together."""
    from scilink.agents.sim_agents.simulation_router import (
        DEFAULT_SCALE_DESCRIPTIONS)
    assert set(RecommendationAgent.VALID_SCALES) == set(
        DEFAULT_SCALE_DESCRIPTIONS)
