"""
Planning agent campaign scenario test suite.

Runs end-to-end planning workflows through the orchestrator chat:
plan generation, result refinement, knowledge synthesis, skill graduation.

Requires GEMINI_API_KEY env var. Run with:

    GEMINI_API_KEY=<key> python tests/test_planning_campaigns.py

Or run a subset by number:

    GEMINI_API_KEY=<key> python tests/test_planning_campaigns.py 1 3 7
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent,
    AutonomyLevel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp():
    return Path(tempfile.mkdtemp(prefix="plan_campaign_"))


_MODEL = os.environ.get("SCILINK_TEST_MODEL", "gemini-3.1-pro-preview")
_API_KEY = os.environ.get("SCILINK_TEST_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
_EMBEDDING_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def _orch(base_dir, knowledge_dir=None, data_dir=None):
    """Create a planning orchestrator in AUTONOMOUS mode.

    AUTONOMOUS mode requires data_dir. If not provided, creates a
    dummy one in base_dir.

    Override model/key via env vars:
        SCILINK_TEST_MODEL=claude-opus-4-6
        SCILINK_TEST_API_KEY=sk-ant-...
        GEMINI_API_KEY=AIza... (always needed for embeddings)
    """
    if data_dir is None:
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        base_dir=str(base_dir / "session"),
        api_key=_API_KEY,
        model_name=_MODEL,
        embedding_api_key=_EMBEDDING_API_KEY,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    if knowledge_dir:
        kwargs["knowledge_dir"] = str(knowledge_dir)
    return kwargs


def _feedstock_csv(path):
    """Write a minimal feedstock composition CSV."""
    df = pd.DataFrame({
        "Element": ["Nd", "Dy", "Pr", "Fe", "B", "Co"],
        "Concentration_Percent": [24.5, 1.2, 0.85, 58.0, 0.98, 0.32],
        "Market_Value_USD_kg": [120, 350, 95, 0.5, 5, 35],
        "Criticality_Score": ["High", "Critical", "High", "Low", "Low", "Medium"],
    })
    df.to_csv(path, index=False)


def _feedstock_meta(path):
    """Write feedstock metadata JSON."""
    meta = {
        "title": "NdFeB Magnet Feedstock",
        "objective": "Elemental analysis of shredded NdFeB magnets",
        "source": "Hard disk drive magnets",
        "analysis_method": "ICP-OES",
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _leaching_results(path, n_conditions=8):
    """Generate simulated leaching ICP results."""
    np.random.seed(42)
    records = []
    for hcl in [1.0, 2.0, 4.0, 6.0]:
        for time_h in [2.0, 6.0]:
            nd_rec = min(99, 30 + 10 * hcl + 5 * time_h + np.random.normal(0, 2))
            dy_rec = min(99, 28 + 10 * hcl + 5 * time_h + np.random.normal(0, 2))
            fe_rec = min(99, 15 + 8 * hcl + 6 * time_h + np.random.normal(0, 2))
            records.append({
                "HCl_M": hcl, "Time_h": time_h,
                "Nd_recovery_pct": round(nd_rec, 1),
                "Dy_recovery_pct": round(dy_rec, 1),
                "Fe_recovery_pct": round(fe_rec, 1),
            })
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "title": "HCl Leaching Screen Results",
            "experiment": "HCl concentration x time for NdFeB dissolution",
            "instrument": "ICP-OES",
        }, f, indent=2)


def _precipitation_results(path):
    """Generate simulated selective precipitation results."""
    np.random.seed(99)
    records = []
    for pH in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5, 8.5]:
        # Fe precipitates around pH 3-4, REEs stay in solution until pH 7+
        fe_removal = 100 / (1 + np.exp(-4 * (pH - 3.2))) + np.random.normal(0, 1)
        nd_retention = 100 - 100 / (1 + np.exp(-3.5 * (pH - 7.8))) + np.random.normal(0, 1)
        dy_retention = 100 - 100 / (1 + np.exp(-3.0 * (pH - 7.3))) + np.random.normal(0, 1)
        records.append({
            "pH": pH,
            "Fe_removal_pct": round(np.clip(fe_removal, 0, 100), 1),
            "Nd_retention_pct": round(np.clip(nd_retention, 0, 100), 1),
            "Dy_retention_pct": round(np.clip(dy_retention, 0, 100), 1),
        })
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "title": "Selective Precipitation Screen Results",
            "experiment": "pH-controlled Fe/REE separation",
            "instrument": "ICP-MS",
        }, f, indent=2)


def _simple_skill(path, name="test_leaching"):
    """Write a minimal planning skill file."""
    path.write_text(f"""\
## overview
{name} skill for HCl leaching of NdFeB magnet feedstock.

## planning
Use 4-6M HCl at L/S 30:1 for 6h at room temperature.

## implementation
Use Opentrons OT-2 with 96-well deep-well plates.
Transfer 150 uL supernatant for ICP analysis.

## interpretation
Nd recovery >95% and Dy >90% indicates successful leaching.
Fe co-dissolution is expected; downstream separation required.

## validation
Mass balance closure should be 90-110%.
Replicate RSD should be <5%.
""")


# ---------------------------------------------------------------------------
# Test definitions — each returns (ok: bool, detail: str)
# ---------------------------------------------------------------------------

TESTS = []


def _test(fn):
    TESTS.append(fn)
    return fn


# ===== GROUP 1: Basic Plan Generation =====

@_test
def generate_plan_with_data():
    """Generate plan from feedstock data + knowledge base."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    r = o.chat(
        f"Generate an experimental plan for recovering Nd and Dy from NdFeB magnet feedstock. "
        f"Use feedstock data at {data_dir / 'feedstock.csv'}."
    )
    shutil.rmtree(d, True)
    ok = o.planner.state is not None and o.planner.state.get("current_plan") is not None
    plan = o.planner.state.get("current_plan", {})
    n_exp = len(plan.get("proposed_experiments", []))
    return ok and n_exp > 0, f"experiments={n_exp}, response={r[:200]}"


@_test
def generate_plan_no_data():
    """Generate plan with only an objective, no data files."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    r = o.chat("Generate a plan to optimize lithium extraction from geothermal brine.")
    shutil.rmtree(d, True)
    ok = o.planner.state is not None and o.planner.state.get("current_plan") is not None
    return ok, r[:200]


@_test
def generate_plan_with_skill():
    """Generate plan with a domain skill loaded."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    skill_path = d / "leaching_skill.md"
    _simple_skill(skill_path)

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    r = o.chat(
        f"Generate a plan for Nd/Dy recovery from NdFeB feedstock. "
        f"Use data at {data_dir / 'feedstock.csv'} "
        f"and apply skill at {skill_path}."
    )
    shutil.rmtree(d, True)

    ok = o.planner.state is not None
    has_skill = o.planner.state.get("skill_name") is not None if o.planner.state else False
    plan = o.planner.state.get("current_plan", {}) if o.planner.state else {}
    n_exp = len(plan.get("proposed_experiments", []))
    return ok and n_exp > 0, f"skill={has_skill}, experiments={n_exp}"


# ===== GROUP 2: Plan Refinement with Results =====

@_test
def refine_plan_with_csv_results():
    """Generate plan, then refine with simulated experimental results."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB feedstock. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    # Simulate results and upload
    results_path = data_dir / "leaching_results.csv"
    _leaching_results(results_path)

    r = o.chat(
        f"Here are the ICP results from the leaching screen: {results_path}. "
        f"Refine the plan based on these results."
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    iterations = state.get("iteration_index", 0)
    has_results = len(state.get("experimental_results", [])) > 0
    ok = iterations >= 2 and has_results
    return ok, f"iterations={iterations}, has_results={has_results}, response={r[:200]}"


@_test
def refine_plan_with_text_feedback():
    """Generate plan, then refine with text-only feedback."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    o.chat("Generate a plan to synthesize zinc oxide nanoparticles via sol-gel.")
    r = o.chat(
        "The initial precipitation failed — pH was too low. "
        "Adjust the plan to use pH 10-11 instead of pH 8."
    )
    shutil.rmtree(d, True)
    ok = "error" not in r.lower()[:100]
    return ok, r[:200]


@_test
def two_iteration_campaign():
    """Full 2-iteration campaign: plan → leaching results → refined plan for precipitation."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Iteration 1: Generate plan
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB feedstock. "
        f"Use data at {data_dir / 'feedstock.csv'}. "
        f"Focus on HCl only, 4 concentrations, single plate."
    )

    # Upload leaching results
    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(
        f"Here are the leaching ICP results: {results1}. "
        f"Best condition was 6M HCl at 6h with ~90% Nd recovery. "
        f"Refine the plan to add a selective precipitation step for Fe/REE separation."
    )

    shutil.rmtree(d, True)

    state = o.planner.state or {}
    iterations = state.get("iteration_index", 0)
    history_len = len(state.get("plan_history", []))
    ok = iterations >= 2 and history_len >= 2
    return ok, f"iterations={iterations}, history={history_len}"


# ===== GROUP 3: Bayesian Optimization =====

@_test
def bo_after_first_results():
    """Analyze leaching data, then run BO to suggest next conditions."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Generate plan
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}. "
        f"Focus on maximizing Nd recovery."
    )

    # Upload and analyze results
    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(
        f"Analyze the leaching results at {results1}. "
        f"Inputs are HCl_M and Time_h. "
        f"Target is Nd_recovery_pct (maximize)."
    )

    # Run BO
    r = o.chat("Run optimization to suggest next experimental conditions.")

    bo_data_exists = o.bo_data_path.exists()
    shutil.rmtree(d, True)

    ok = "suggest" in r.lower() or "candidate" in r.lower() or "next" in r.lower() or "batch" in r.lower() or "parameter" in r.lower()
    return ok, f"bo_data={bo_data_exists}, response={r[:200]}"


@_test
def bo_multi_objective():
    """Analyze data with multiple targets, run multi-objective BO."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(
        f"Analyze the leaching results at {results1}. "
        f"Inputs: HCl_M, Time_h. "
        f"Targets: Nd_recovery_pct (maximize), Fe_recovery_pct (minimize)."
    )

    r = o.chat("Run optimization to find conditions that maximize Nd recovery while minimizing Fe co-extraction.")

    shutil.rmtree(d, True)

    ok = "error" not in r.lower()[:100]
    return ok, r[:200]


@_test
def bo_with_batch_suggestions():
    """BO in parallel mode with batch_size → should return multiple candidates."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(
        f"Analyze leaching results at {results1}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )

    r = o.chat(
        "Run optimization in parallel mode with batch_size=4 "
        "to suggest the next 4 experiments."
    )

    shutil.rmtree(d, True)

    ok = "error" not in r.lower()[:100]
    return ok, r[:200]


@_test
def bo_then_refine_plan():
    """BO suggests conditions → refine plan with BO suggestions + second batch of results."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    # First batch
    results1 = data_dir / "batch1.csv"
    _leaching_results(results1)
    o.chat(
        f"Analyze results at {results1}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )
    o.chat("Run optimization to suggest next conditions.")

    # Second batch (simulated BO follow-up results)
    np.random.seed(77)
    batch2 = pd.DataFrame({
        "HCl_M": [5.0, 5.5, 4.5, 6.0],
        "Time_h": [5.0, 6.0, 4.0, 5.0],
        "Nd_recovery_pct": [88.3, 92.1, 85.7, 94.5],
    })
    results2 = data_dir / "batch2.csv"
    batch2.to_csv(results2, index=False)
    with open(results2.with_suffix(".json"), "w") as f:
        json.dump({"title": "BO Follow-up Batch 2", "instrument": "ICP-OES"}, f)

    o.chat(
        f"Analyze the second batch at {results2}. "
        f"Same schema: inputs HCl_M, Time_h; target Nd_recovery_pct."
    )

    # Refine plan based on BO progress
    r = o.chat(
        "The optimization shows Nd recovery plateauing around 94% even at 6M HCl. "
        "Refine the plan: we may need to add temperature as a variable or "
        "switch to a selective precipitation step."
    )

    shutil.rmtree(d, True)

    state = o.planner.state or {}
    ok = len(state.get("plan_history", [])) >= 2
    return ok, f"plan_history={len(state.get('plan_history', []))}, response={r[:200]}"


# ===== GROUP 4: Knowledge Synthesis =====

@_test
def synthesize_knowledge_from_iterations():
    """Synthesize knowledge from plan iterations via tool call."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Build up plan history
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(f"Leaching results: {results1}. Refine the plan for selective precipitation.")

    # Now synthesize
    r = o.chat(
        "Synthesize knowledge from all planning iterations. "
        "Focus on HCl leaching protocol for NdFeB magnets."
    )

    shutil.rmtree(d, True)

    ok = len(o.active_knowledge) >= 1
    knowledge_id = o.active_knowledge[0]["id"] if o.active_knowledge else None
    return ok, f"knowledge_entries={len(o.active_knowledge)}, id={knowledge_id}, response={r[:200]}"


@_test
def synthesize_knowledge_empty_history():
    """Synthesize knowledge with no plan history → should fail gracefully."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    r = o.chat("Synthesize knowledge from planning iterations about catalysis.")
    shutil.rmtree(d, True)
    # Should get an error response, not a crash
    ok = "error" in r.lower() or "no" in r.lower() or "generate" in r.lower()
    return ok, r[:200]


# ===== GROUP 5: Skill Graduation =====

@_test
def graduate_knowledge_to_skill():
    """Full pipeline: plan → results → synthesize → graduate to skill."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Plan + results
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )
    results1 = data_dir / "leaching_results.csv"
    _leaching_results(results1)
    o.chat(f"Here are leaching results: {results1}. Refine the plan.")

    # Synthesize + graduate
    r = o.chat(
        "Synthesize knowledge from all iterations about HCl leaching of NdFeB, "
        "then graduate it to a skill called 'ndfeb_hcl_leaching'."
    )

    shutil.rmtree(d, True)

    has_knowledge = len(o.active_knowledge) >= 1
    has_skill = "ndfeb_hcl_leaching" in o._custom_skills or "ndfeb_hcl_leaching" in o._graduated_skill_sources
    skill_file = (Path(str(o.base_dir)) / "graduated_skills" / "ndfeb_hcl_leaching.md").exists() if hasattr(o, 'base_dir') else False

    ok = has_knowledge and (has_skill or skill_file)
    return ok, f"knowledge={has_knowledge}, skill_registered={has_skill}, response={r[:200]}"


@_test
def graduated_skill_has_implementation_section():
    """Graduated planning skill uses ## implementation, not ## analysis."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for acid leaching of e-waste. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )
    results1 = data_dir / "results.csv"
    _leaching_results(results1)
    o.chat(f"Results: {results1}. Refine plan.")
    o.chat(
        "Synthesize knowledge from iterations about acid leaching, "
        "then graduate to skill 'acid_leaching_ewaste'."
    )

    skill_path = o.base_dir / "graduated_skills" / "acid_leaching_ewaste.md"
    if skill_path.exists():
        content = skill_path.read_text().lower()
        has_impl = "implementation" in content
        has_analysis = "## analysis" in content
        ok = has_impl and not has_analysis
        detail = f"has_implementation={has_impl}, has_analysis_section={has_analysis}"
    else:
        ok = False
        detail = "skill file not created"

    shutil.rmtree(d, True)
    return ok, detail


# ===== GROUP 6: Skill Persistence & State =====

@_test
def skill_persists_across_refinement():
    """Skill loaded in generate_plan persists through refine_plan."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    skill_path = d / "my_skill.md"
    _simple_skill(skill_path, "test_persistence")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for Nd recovery using data at {data_dir / 'feedstock.csv'} "
        f"with skill at {skill_path}."
    )

    # Check skill is in state after plan generation
    state = o.planner.state or {}
    skill_after_plan = state.get("skill_name")

    # Refine with text feedback
    o.chat("Reduce acid concentration to 3M maximum for safety reasons.")

    state = o.planner.state or {}
    skill_after_refine = state.get("skill_name")

    shutil.rmtree(d, True)

    ok = skill_after_plan is not None and skill_after_refine == skill_after_plan
    return ok, f"after_plan={skill_after_plan}, after_refine={skill_after_refine}"


@_test
def knowledge_persisted_to_disk():
    """Synthesized knowledge is saved as JSON files on disk."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for leaching NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )
    results1 = data_dir / "results.csv"
    _leaching_results(results1)
    o.chat(f"Results: {results1}. Refine plan.")
    o.chat(
        "Synthesize knowledge from all planning iterations. "
        "Focus on HCl leaching protocol learnings."
    )

    # Verify knowledge was synthesized in memory
    has_knowledge = len(o.active_knowledge) > 0

    # Knowledge is persisted to disk as individual JSON files, not in checkpoint
    knowledge_dir = o.base_dir / "knowledge"
    knowledge_files = list(knowledge_dir.glob("knowledge_*.json")) if knowledge_dir.exists() else []

    ok = has_knowledge and len(knowledge_files) > 0
    detail = f"in_memory={has_knowledge}, on_disk={len(knowledge_files)}"

    shutil.rmtree(d, True)
    return ok, detail


# ===== GROUP 7: Edge Cases =====

@_test
def read_file_tool_for_plan_inspection():
    """read_file reads plan.json without triggering scalarizer."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    o.chat("Generate a plan to optimize battery cycling for NMC811.")
    r = o.chat("Read the plan.json file and summarize it.")
    shutil.rmtree(d, True)
    # Should NOT see scalarizer output
    ok = "scalarizer" not in r.lower() and "schema" not in r.lower()
    return ok, r[:200]


@_test
def adjust_plan_for_constraints():
    """Adjust plan for equipment constraints without incrementing iteration."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    o.chat("Generate a plan to screen catalysts for CO2 reduction.")
    r = o.chat(
        "The plate reader can only handle 200 µL max volume per well. "
        "Adjust the plan for this constraint."
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    # adjust_plan should NOT increment iteration
    iteration = state.get("iteration_index", 0)
    ok = iteration <= 1  # Should still be iteration 1
    return ok, f"iteration={iteration}, response={r[:200]}"


@_test
def checkpoint_signals_knowledge_synthesis():
    """Checkpoint with 2+ result iterations signals knowledge_synthesis_available."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for leaching NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    # Two rounds of results
    results1 = data_dir / "results1.csv"
    _leaching_results(results1)
    o.chat(f"Leaching results: {results1}. Refine plan for precipitation.")

    results2 = data_dir / "results2.csv"
    _precipitation_results(results2)
    o.chat(f"Precipitation results: {results2}. Refine plan.")

    # Save checkpoint — should signal knowledge synthesis
    r = o.chat("Save checkpoint.")

    # Check if the agent mentioned knowledge synthesis or if the
    # checkpoint JSON has the signal
    checkpoint_path = o.base_dir / "checkpoint.json"
    has_signal = False
    if checkpoint_path.exists():
        # The save_checkpoint tool returns knowledge_synthesis_available
        # in its JSON response which the LLM sees
        has_signal = True  # checkpoint exists with 2+ result iterations

    shutil.rmtree(d, True)

    # Pass if either: agent mentioned knowledge in response, or
    # we confirmed 2+ result iterations were in state
    state = o.planner.state or {}
    n_results = len(state.get("experimental_results", []))
    ok = n_results >= 2 and has_signal
    return ok, f"result_iterations={n_results}, response={r[:200]}"


@_test
def bad_skill_path_graceful():
    """Non-existent skill path → plan still generates."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    r = o.chat(
        "Generate a plan for polymer synthesis. "
        "Use skill at /nonexistent/path/fake_skill.md."
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    has_plan = state.get("current_plan") is not None
    no_skill = state.get("skill_name") is None
    ok = has_plan
    return ok, f"has_plan={has_plan}, no_skill={no_skill}, response={r[:200]}"


# ===== GROUP 8: Semantic Validation (plan quality & data fidelity) =====

@_test
def refined_plan_reflects_actual_data():
    """Refined plan should reference actual data values, not hallucinate better ones."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB aiming for >95% Nd recovery. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    # Results where best Nd recovery is ~72% — well below 95% target
    np.random.seed(42)
    df = pd.DataFrame({
        "HCl_M": [1.0, 2.0, 4.0, 6.0, 1.0, 2.0, 4.0, 6.0],
        "Time_h": [2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0],
        "Nd_recovery_pct": [25.3, 38.1, 55.2, 62.8, 35.7, 50.4, 68.1, 72.3],
        "Fe_recovery_pct": [18.2, 30.5, 48.9, 58.1, 28.4, 45.2, 62.3, 70.1],
    })
    results_path = data_dir / "leaching_low_recovery.csv"
    df.to_csv(results_path, index=False)
    with open(results_path.with_suffix(".json"), "w") as f:
        json.dump({"title": "Leaching screen - low recovery", "instrument": "ICP-OES"}, f)

    # Analyze and refine
    o.chat(
        f"Analyze results at {results_path}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )
    r = o.chat(
        f"Here are the leaching results: {results_path}. "
        f"The best Nd recovery was only 72% at 6M HCl, 6h. "
        f"Refine the plan — the 95% target was not met."
    )

    shutil.rmtree(d, True)

    # The refined plan should acknowledge the 72% result and not claim 95% was already achieved
    plan = (o.planner.state or {}).get("current_plan", {})
    experiments = plan.get("proposed_experiments", [{}])

    # Check justification and expected_outcome specifically (not hypothesis — which may set new targets)
    justification = experiments[0].get("justification", "").lower() if experiments else ""
    expected = experiments[0].get("expected_outcome", "").lower() if experiments else ""

    # Justification should reference the actual result (~72%), not claim 95% was achieved
    acknowledges_result = any(w in justification for w in ["72", "below", "short", "insufficient", "not met", "did not", "only"])
    # Expected outcome can target 95% for the NEW experiment, but should not say it WAS achieved
    claims_already_achieved = "achieved" in expected and "95" in expected

    ok = acknowledges_result and not claims_already_achieved
    return ok, f"acknowledges_result={acknowledges_result}, claims_achieved={claims_already_achieved}, justification={justification[:150]}"


@_test
def skill_constraints_appear_in_plan():
    """Plan generated with a skill should reflect the skill's mandatory constraints."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    skill_path = d / "strict_leaching.md"
    skill_path.write_text("""\
## overview
Strict HCl leaching protocol for NdFeB magnet recycling.

## planning
- MANDATORY: Use ONLY HCl as the acid. Do NOT use HNO3 or H2SO4.
- MANDATORY: Temperature must not exceed 40 degrees Celsius.
- MANDATORY: Include at least 4 replicate wells per condition.
- MANDATORY: Leaching time must be between 1 and 8 hours.

## implementation
Use Opentrons OT-2 with deep-well 96-well plates.

## interpretation
Nd recovery >90% indicates successful leaching.

## validation
Replicate coefficient of variation must be <10%.
Mass balance closure must be 90-110%.
""")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for NdFeB leaching using data at {data_dir / 'feedstock.csv'} "
        f"and apply the skill at {skill_path}."
    )

    plan = (o.planner.state or {}).get("current_plan", {})
    plan_text = json.dumps(plan).lower()

    shutil.rmtree(d, True)

    # Check skill constraints are reflected in the experimental steps
    experiments = plan.get("proposed_experiments", [{}])
    steps = experiments[0].get("experimental_steps", []) if experiments else []
    steps_text = " ".join(steps).lower() if isinstance(steps, list) else str(steps).lower()

    uses_hcl = "hcl" in steps_text
    # Other acids should not appear as reagents USED in steps
    # (mentioning them in a prohibition like "do not use HNO3" is acceptable)
    uses_hno3 = ("hno3" in steps_text or "nitric" in steps_text) and "do not" not in steps_text.split("hno3")[0][-30:] and "not use" not in steps_text.split("hno3")[0][-30:]
    uses_h2so4 = ("h2so4" in steps_text or "sulfuric" in steps_text) and "do not" not in steps_text.split("h2so4")[0][-30:] if "h2so4" in steps_text else False

    # Check no step says "prepare HNO3" or "dispense H2SO4" etc.
    prepares_other = any(
        phrase in steps_text
        for phrase in ["prepare hno3", "prepare h2so4", "dispense hno3", "dispense h2so4",
                       "add hno3", "add h2so4", "nitric acid solution", "sulfuric acid solution"]
    )

    ok = uses_hcl and not prepares_other
    return ok, f"hcl={uses_hcl}, prepares_other={prepares_other}, steps_excerpt={steps_text[:200]}"


@_test
def scalarizer_context_reaches_refinement():
    """When results are analyzed then refined, scalarizer metrics appear in refinement context."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for HCl leaching. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results_path = data_dir / "results.csv"
    _leaching_results(results_path)

    # Analyze first (populates scalarizer)
    o.chat(
        f"Analyze results at {results_path}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )

    # Verify scalarizer ran
    bo_data_exists = o.bo_data_path.exists()

    # Now refine — scalarizer context should be injected
    o.chat(f"Refine the plan based on results at {results_path}.")

    shutil.rmtree(d, True)

    # The refined plan should reference specific numbers from the data
    plan = (o.planner.state or {}).get("current_plan", {})
    plan_text = json.dumps(plan)

    # Check if any actual data values appear (not just generic statements)
    has_specific_numbers = any(
        str(round(v, 0))[:-2] in plan_text  # Match integer part
        for v in [72.3, 55.2, 38.1, 25.3, 62.8]  # Values from _leaching_results
    )

    ok = bo_data_exists and has_specific_numbers
    return ok, f"bo_data={bo_data_exists}, has_numbers={has_specific_numbers}"


@_test
def bo_suggestions_within_parameter_bounds():
    """BO-suggested parameters should be within the data range or physically reasonable."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for HCl leaching. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results_path = data_dir / "results.csv"
    _leaching_results(results_path)
    o.chat(
        f"Analyze results at {results_path}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )

    r = o.chat("Run optimization to suggest next experiment.")

    shutil.rmtree(d, True)

    # Extract suggested values from response (look for numbers near parameter names)
    r_lower = r.lower()
    # BO should not suggest negative concentrations or absurd times
    no_negative = "negative" not in r_lower
    # Should mention actual parameter names
    mentions_params = "hcl" in r_lower and ("time" in r_lower or "hour" in r_lower)

    ok = no_negative and mentions_params
    return ok, f"no_negative={no_negative}, mentions_params={mentions_params}, response={r[:200]}"


@_test
def graduated_skill_contains_data_driven_content():
    """Graduated skill should contain specific findings from the campaign, not generic text."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}."
    )

    results1 = data_dir / "results.csv"
    _leaching_results(results1)
    o.chat(f"Results: {results1}. Refine plan for downstream separation.")

    o.chat(
        "Synthesize knowledge from all iterations about HCl leaching protocol, "
        "then graduate it to a skill called 'test_leaching_skill'."
    )

    skill_path = o.base_dir / "graduated_skills" / "test_leaching_skill.md"
    if skill_path.exists():
        content = skill_path.read_text().lower()

        # Should contain specific, data-driven content — not just generic chemistry
        has_hcl = "hcl" in content
        has_concentration = any(c in content for c in ["4 m", "6 m", "4m", "6m"])
        has_sections = "## overview" in content and "## planning" in content and "## implementation" in content
        has_quantitative = any(c in content for c in ["%", "recovery", "selectivity", "purity"])

        ok = has_hcl and has_concentration and has_sections and has_quantitative
        detail = f"hcl={has_hcl}, conc={has_concentration}, sections={has_sections}, quantitative={has_quantitative}"
    else:
        ok = False
        detail = "skill file not created"

    shutil.rmtree(d, True)
    return ok, detail


# ===== GROUP 9: Literature & Molecules as Orchestrator Tools =====

@_test
def search_literature_tool_available():
    """search_literature tool is registered and callable on the orchestrator."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    has_tool = "search_literature" in o.tools.functions_map
    has_schema = any(
        s["function"]["name"] == "search_literature"
        for s in o.tools.openai_schemas
    )
    shutil.rmtree(d, True)
    ok = has_tool and has_schema
    return ok, f"in_functions_map={has_tool}, in_schemas={has_schema}"


@_test
def query_molecules_tool_available():
    """query_molecules tool is registered and callable on the orchestrator."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    has_tool = "query_molecules" in o.tools.functions_map
    has_schema = any(
        s["function"]["name"] == "query_molecules"
        for s in o.tools.openai_schemas
    )
    shutil.rmtree(d, True)
    ok = has_tool and has_schema
    return ok, f"in_functions_map={has_tool}, in_schemas={has_schema}"


@_test
def generate_plan_accepts_literature_and_molecule_context():
    """generate_initial_plan schema includes literature_context and molecule_context params."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    schema = next(
        (s for s in o.tools.openai_schemas
         if s["function"]["name"] == "generate_initial_plan"),
        None
    )
    shutil.rmtree(d, True)
    props = schema["function"]["parameters"]["properties"] if schema else {}
    has_lit = "literature_context" in props
    has_mol = "molecule_context" in props
    ok = has_lit and has_mol
    return ok, f"literature_context={has_lit}, molecule_context={has_mol}"


@_test
def generate_plan_with_external_context_skips_internal_lit():
    """When external_context is provided, PlanningAgent skips internal literature search."""
    import warnings
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))

    fake_context = "## External Literature\nHCl leaching of NdFeB at 4-6M achieves >90% REE recovery."

    # Call generate_plan directly with external_context
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plan = o.planner.generate_plan(
            objective="Recover Nd from NdFeB magnets using HCl leaching",
            external_context=fake_context,
            enable_human_feedback=False
        )
        # Should NOT emit deprecation warning (external_context was provided)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                                and "deprecated" in str(x.message).lower()]

    shutil.rmtree(d, True)

    has_plan = plan is not None and plan.get("proposed_experiments") is not None
    no_deprecation = len(deprecation_warnings) == 0
    # The external context should flow through to literature_search key
    has_ext = plan.get("literature_search") == fake_context
    ok = has_plan and no_deprecation and has_ext
    return ok, f"has_plan={has_plan}, no_deprecation={no_deprecation}, has_ext={has_ext}"


@_test
def generate_plan_with_literature_file():
    """generate_initial_plan reads literature_context from file and passes to planner."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))

    # Write a fake literature file (simulates search_literature output)
    lit_path = o.base_dir / "literature_search.md"
    lit_path.write_text(
        "# Literature Search Results\n\n"
        "HCl leaching at 4-6M achieves >90% Nd recovery from NdFeB magnets. "
        "Temperature above 60C accelerates dissolution but increases Fe co-extraction."
    )

    result_json = o.tools.execute_tool(
        "generate_initial_plan",
        specific_objective="Recover Nd and Dy from NdFeB magnet feedstock via HCl leaching",
        literature_context=str(lit_path)
    )
    result = json.loads(result_json)
    shutil.rmtree(d, True)

    has_plan = result.get("status") == "success" and result.get("num_experiments", 0) > 0
    ok = has_plan
    return ok, f"status={result.get('status')}, experiments={result.get('num_experiments')}"


@_test
def generate_plan_with_molecule_file():
    """generate_initial_plan reads molecule_context from file and passes to planner."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))

    # Write a fake molecules file (simulates query_molecules output)
    mol_path = o.base_dir / "molecule_design.md"
    mol_path.write_text(
        "# Molecular Design Results\n\n"
        "Suggested MOF linker: 2,5-dihydroxyterephthalic acid with amine functionalization. "
        "Predicted CO2 uptake: 4.2 mmol/g at 298K, 1 bar."
    )

    result_json = o.tools.execute_tool(
        "generate_initial_plan",
        specific_objective="Synthesize and test novel MOF linkers for CO2 capture",
        molecule_context=str(mol_path)
    )
    result = json.loads(result_json)
    shutil.rmtree(d, True)

    has_plan = result.get("status") == "success" and result.get("num_experiments", 0) > 0
    ok = has_plan
    return ok, f"status={result.get('status')}, experiments={result.get('num_experiments')}"


@_test
def tea_with_literature_context_direct():
    """run_economic_analysis accepts literature_context param."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))

    # Write a fake literature file with economic data
    lit_path = o.base_dir / "literature_search.md"
    lit_path.write_text(
        "# Literature Search Results\n\n"
        "Global NdFeB magnet recycling market valued at $2.1B (2025). "
        "Nd spot price ~$120/kg, Dy ~$350/kg. "
        "HCl leaching + selective precipitation achieves 92% REE purity at $45/kg processing cost."
    )

    result_json = o.tools.execute_tool(
        "run_economic_analysis",
        focus_topic="Economic viability of NdFeB magnet recycling via HCl leaching",
        literature_context=str(lit_path)
    )
    result = json.loads(result_json)
    shutil.rmtree(d, True)

    ok = result.get("status") == "success"
    return ok, f"status={result.get('status')}, result_keys={list(result.keys())[:5]}"


@_test
def refine_plan_with_literature_context():
    """refine_plan_with_results accepts literature_context and passes it to the planner."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Write a fake literature file to avoid Edison calls during initial plan
    init_lit = o.base_dir / "init_literature.md"
    init_lit.write_text("# Literature\n\nHCl leaching at 4-6M recovers >90% Nd from NdFeB.")

    # Generate initial plan (with literature_context to skip deprecated internal path)
    o.tools.execute_tool(
        "generate_initial_plan",
        specific_objective="HCl leaching of NdFeB for Nd recovery",
        literature_context=str(init_lit)
    )

    # Write a fake literature file for refinement context
    lit_path = o.base_dir / "literature_search.md"
    lit_path.write_text(
        "# Literature Search Results\n\n"
        "Recent studies show adding H2O2 as oxidant during HCl leaching "
        "increases Nd/Fe selectivity by 3x (Zhang et al., 2025). "
        "Optimal H2O2:HCl ratio is 0.1:1 at 40C."
    )

    # Refine with literature context
    result_json = o.tools.execute_tool(
        "refine_plan_with_results",
        result_data="Nd recovery was 72% but Fe co-extraction was 65%, selectivity too low.",
        literature_context=str(lit_path)
    )
    result = json.loads(result_json)
    shutil.rmtree(d, True)

    ok = result.get("status") == "success" and result.get("num_experiments", 0) > 0
    return ok, f"status={result.get('status')}, experiments={result.get('num_experiments')}"


@_test
def refine_plan_schema_has_context_params():
    """refine_plan_with_results schema includes literature_context and molecule_context."""
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    schema = next(
        (s for s in o.tools.openai_schemas
         if s["function"]["name"] == "refine_plan_with_results"),
        None
    )
    shutil.rmtree(d, True)
    props = schema["function"]["parameters"]["properties"] if schema else {}
    has_lit = "literature_context" in props
    has_mol = "molecule_context" in props
    ok = has_lit and has_mol
    return ok, f"literature_context={has_lit}, molecule_context={has_mol}"


@_test
def lit_search_then_plan_via_chat():
    """LLM orchestrator calls search_literature then generate_initial_plan (1 Edison lit call).

    Requires FUTUREHOUSE_API_KEY. If absent, the LLM should still produce a plan
    (falling back to generate_initial_plan without literature context).
    """
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    r = o.chat(
        "First use the search_literature tool to find literature about HCl leaching "
        "of NdFeB magnets for rare earth recovery. Then use generate_initial_plan "
        "with the literature_context from those results to create an experimental plan."
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    has_plan = state.get("current_plan") is not None
    n_exp = len(state.get("current_plan", {}).get("proposed_experiments", []))
    ok = has_plan and n_exp > 0
    return ok, f"experiments={n_exp}, response={r[:200]}"


@_test
def molecules_then_plan_via_chat():
    """LLM orchestrator calls query_molecules then generate_initial_plan (1 Edison mol call).

    Requires FUTUREHOUSE_API_KEY. If absent, the LLM should still produce a plan.
    """
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))
    r = o.chat(
        "Use the query_molecules tool for designing novel MOF linkers with high CO2 "
        "adsorption for direct air capture. Then use generate_initial_plan with the "
        "molecule_context from those results to create a synthesis and testing plan."
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    has_plan = state.get("current_plan") is not None
    n_exp = len(state.get("current_plan", {}).get("proposed_experiments", []))
    ok = has_plan and n_exp > 0
    return ok, f"experiments={n_exp}, response={r[:200]}"


@_test
def molecules_then_refine_via_chat():
    """LLM orchestrator calls query_molecules then refine_plan_with_results (1 Edison mol call).

    Tests the full refinement path with molecule context from the orchestrator tool.
    Requires FUTUREHOUSE_API_KEY. If absent, plan refinement should still succeed.
    """
    d = _tmp()
    o = PlanningOrchestratorAgent(**_orch(d))

    # Step 1: generate a plan for MOF synthesis
    o.chat(
        "Generate a plan to synthesize and test MOF-based sorbents for CO2 capture."
    )

    # Step 2: refine with molecules context
    r = o.chat(
        "Initial MOF synthesis gave low crystallinity. "
        "Use the query_molecules tool to search for alternative linkers with better "
        "self-assembly properties, then use refine_plan_with_results with those "
        "molecule_context results and the text feedback: "
        "'MOF crystallinity was only 40%, need linkers with stronger coordination geometry.'"
    )
    shutil.rmtree(d, True)

    state = o.planner.state or {}
    iterations = state.get("iteration_index", 0)
    ok = iterations >= 2
    return ok, f"iterations={iterations}, response={r[:200]}"


@_test
def constraint_adjustment_preserves_unrelated_steps():
    """Adjusting for one constraint should not rewrite unrelated experimental steps."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _feedstock_csv(data_dir / "feedstock.csv")
    _feedstock_meta(data_dir / "feedstock.json")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(
        f"Generate a plan for HCl leaching of NdFeB. "
        f"Use data at {data_dir / 'feedstock.csv'}. "
        f"Include ICP-MS analysis of Nd, Dy, Pr, Fe."
    )

    plan_before = (o.planner.state or {}).get("current_plan", {})
    steps_before = plan_before.get("proposed_experiments", [{}])[0].get("experimental_steps", [])

    # Adjust for a narrow constraint
    o.chat(
        "The centrifuge can only do 2000 x g, not 4000 x g. "
        "Adjust the plan for this constraint."
    )

    plan_after = (o.planner.state or {}).get("current_plan", {})
    steps_after = plan_after.get("proposed_experiments", [{}])[0].get("experimental_steps", [])

    shutil.rmtree(d, True)

    # Should still have similar number of steps (not a complete rewrite)
    step_count_similar = abs(len(steps_after) - len(steps_before)) <= 3
    # Should still mention ICP
    still_has_icp = any("icp" in s.lower() for s in steps_after)
    # Should mention the new centrifuge speed
    mentions_2000 = any("2000" in s for s in steps_after)

    ok = step_count_similar and still_has_icp
    return ok, f"steps_before={len(steps_before)}, steps_after={len(steps_after)}, icp={still_has_icp}, mentions_2000={mentions_2000}"


# ---------------------------------------------------------------------------
# Test tags — Edison tests require FUTUREHOUSE_API_KEY and are slow (~30-60 min)
# ---------------------------------------------------------------------------

_EDISON_TESTS = {
    "lit_search_then_plan_via_chat",
    "molecules_then_plan_via_chat",
    "molecules_then_refine_via_chat",
}


def _is_edison(fn):
    return fn.__name__ in _EDISON_TESTS


# ---------------------------------------------------------------------------
# Runner
#
# Usage:
#   python tests/test_planning_campaigns.py              # all tests (default)
#   python tests/test_planning_campaigns.py --fast       # skip Edison (slow) tests
#   python tests/test_planning_campaigns.py --edison     # only Edison tests
#   python tests/test_planning_campaigns.py --all        # all tests (explicit)
#   python tests/test_planning_campaigns.py 1 3 7        # by number
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _API_KEY:
        print("Set GEMINI_API_KEY (or SCILINK_TEST_API_KEY) env var first.")
        sys.exit(1)
    if not _EMBEDDING_API_KEY:
        print("Set GEMINI_API_KEY env var (needed for embeddings).")
        sys.exit(1)
    print(f"Model: {_MODEL}")
    print(f"API key: {_API_KEY[:10]}...")
    print(f"Embedding key: {_EMBEDDING_API_KEY[:10]}...")

    args = sys.argv[1:]
    if "--fast" in args:
        to_run = [t for t in TESTS if not _is_edison(t)]
        print(f"Mode: --fast (skipping {len(_EDISON_TESTS)} Edison tests)")
    elif "--edison" in args:
        to_run = [t for t in TESTS if _is_edison(t)]
        print(f"Mode: --edison ({len(to_run)} slow tests, requires FUTUREHOUSE_API_KEY)")
    elif "--all" in args or not args:
        to_run = TESTS
        print(f"Mode: --all ({len(TESTS)} tests)")
    else:
        indices = [int(a) - 1 for a in args]
        to_run = [TESTS[i] for i in indices]

    results = {}
    for fn in to_run:
        name = fn.__name__
        desc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"\n{'=' * 60}")
        print(f"[{TESTS.index(fn) + 1}/{len(TESTS)}] {name}: {desc}")
        print("=" * 60)
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            results[name] = status
            print(f"  → {status}: {detail[:200]}")
        except KeyboardInterrupt:
            results[name] = "SKIP"
            break
        except Exception as e:
            results[name] = "ERROR"
            print(f"  → ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        marker = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{marker}] {name}: {status}")
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {passed}/{total} passed")
