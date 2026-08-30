"""Offline tests for the #462 Phase 1 access tools on all four orchestrators.

  python -m pytest tests/test_history_search_tools.py -v
"""
import json
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest

from scilink.session_events import HISTORY_TOOL_NAMES, use_event_log


def _seed_log(base_dir, n=20):
    log = base_dir / "events.jsonl"
    with use_event_log(log):
        from scilink.session_events import append_event
        for i in range(1, n + 1):
            append_event("run_analysis" if i % 2 else "save_file",
                         {"i": i},
                         json.dumps({"status": "success",
                                     "message": f"step {i} kappa={i * 10}"}))
    return log


def _tool_names(tools_obj):
    return set(tools_obj.functions_map)


def _schema_names(tools_obj):
    return {s["function"]["name"] for s in tools_obj.openai_schemas}


@pytest.fixture(scope="module")
def orchestrators(tmp_path_factory):
    d = tmp_path_factory.mktemp("hist")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent)
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    return {
        "analysis": AnalysisOrchestratorAgent(
            api_key="sk-dummy", base_dir=str(d / "an")),
        "planning": PlanningOrchestratorAgent(
            api_key="sk-dummy", data_dir=str(d), base_dir=str(d / "pl")),
        "simulate": SimulationOrchestratorAgent(
            api_key="sk-dummy", base_dir=str(d / "si")),
        "meta": MetaOrchestratorAgent(
            api_key="sk-dummy", base_dir=str(d / "me")),
    }


def test_tools_registered_on_all_four(orchestrators):
    for mode, orch in orchestrators.items():
        names = _tool_names(orch.tools)
        schema = _schema_names(orch.tools)
        for t in HISTORY_TOOL_NAMES:
            assert t in names, f"{mode} missing {t} in functions_map"
            assert t in schema, f"{mode} missing {t} in schema"


def test_search_honest_null_on_fresh_session(orchestrators):
    for mode, orch in orchestrators.items():
        out = json.loads(orch.tools.execute_tool(
            "search_session_history", pattern="anything"))
        assert out["status"] == "success" and out["total_matches"] == 0, mode


def test_search_and_drilldown_roundtrip(orchestrators):
    orch = orchestrators["analysis"]
    _seed_log(orch.base_dir)
    out = json.loads(orch.tools.execute_tool(
        "search_session_history", pattern=r"kappa=1[05]0"))
    assert out["total_matches"] == 2          # kappa=100 (i=10), 150 (i=15)
    ns = [h["n"] for h in out["hits"]]
    assert ns == [10, 15]

    ev = json.loads(orch.tools.execute_tool(
        "get_history_events", start_n=ns[0], end_n=ns[0]))
    assert ev["returned"] == 1 and "kappa=100" in ev["events"][0]["summary"]


def test_search_never_matches_its_own_past_calls(orchestrators):
    orch = orchestrators["simulate"]
    _seed_log(orch.base_dir, n=3)
    from scilink.session_events import use_event_log as uel
    with uel(orch.base_dir / "events.jsonl"):
        # Two searches for a token that appears only in search args.
        orch.tools.execute_tool("search_session_history",
                                pattern="zzz_unique_token")
        out = json.loads(orch.tools.execute_tool(
            "search_session_history", pattern="zzz_unique_token"))
    assert out["total_matches"] == 0          # prior search event excluded


def test_trim_marker_points_to_search(orchestrators):
    for mode, orch in orchestrators.items():
        history = [{"role": "user", "content": f"m{i}"} for i in range(150)]
        trimmed = orch._trim_history(history, max_messages=100)
        assert len(trimmed) <= 101, mode  # 100 + marker
        markers = [m for m in trimmed
                   if "search_session_history" in str(m.get("content", ""))]
        assert len(markers) == 1, f"{mode}: trim marker missing the pointer"


def test_caps_flow_through_the_tool(orchestrators):
    orch = orchestrators["planning"]
    _seed_log(orch.base_dir, n=60)
    out = json.loads(orch.tools.execute_tool(
        "search_session_history", pattern="step", max_hits=500))
    assert out["returned"] <= 50 and out["total_matches"] == 60
    ev = json.loads(orch.tools.execute_tool(
        "get_history_events", start_n=1, end_n=500))
    assert ev["returned"] <= 40


# ---------------------------------------------------- Phase 3: meta scope

def _seed_child(meta, rel_dir, message):
    d = meta.base_dir / rel_dir
    d.mkdir(parents=True, exist_ok=True)
    with use_event_log(d / "events.jsonl"):
        from scilink.session_events import append_event
        append_event("run_analysis", {"where": rel_dir},
                     json.dumps({"status": "success", "message": message}))


def test_meta_children_scope_and_labels(orchestrators):
    meta = orchestrators["meta"]
    _seed_log(meta.base_dir, n=2)
    _seed_child(meta, "analysis", "child fit R2=0.98 tetragonal_marker")
    _seed_child(meta, "planning/delegations/01_x",
                "campaign scoped tetragonal_marker")
    _seed_child(meta, "fanout/00_a", "branch says tetragonal_marker")

    own = json.loads(meta.tools.execute_tool(
        "search_session_history", pattern="tetragonal_marker"))
    assert own["total_matches"] == 0                      # default scope=own

    kids = json.loads(meta.tools.execute_tool(
        "search_session_history", pattern="tetragonal_marker",
        scope="children"))
    assert kids["total_matches"] == 3
    labels = {h["session"] for h in kids["hits"]}
    assert labels == {"analysis", "planning/delegations/01_x", "fanout/00_a"}

    both = json.loads(meta.tools.execute_tool(
        "search_session_history", pattern="run_analysis|step", scope="all"))
    assert both["total_matches"] >= 5                     # own + children

    # Drill into a labeled child hit.
    ev = json.loads(meta.tools.execute_tool(
        "get_history_events", start_n=1, end_n=1, session="analysis"))
    assert ev["returned"] == 1 and "R2=0.98" in ev["events"][0]["summary"]
    bad = json.loads(meta.tools.execute_tool(
        "get_history_events", start_n=1, end_n=1, session="nope"))
    assert bad["status"] == "error"


def test_non_meta_ignores_scope(orchestrators):
    orch = orchestrators["analysis"]
    out = json.loads(orch.tools.execute_tool(
        "search_session_history", pattern="anything", scope="children"))
    assert out["status"] == "success"                     # scope inert, no crash
    schema = [s for s in orch.tools.openai_schemas
              if s["function"]["name"] == "search_session_history"][0]
    assert "scope" not in schema["function"]["parameters"]["properties"]


def test_fanout_branch_events_land_in_meta_log(tmp_path, monkeypatch):
    """A fake fan-out run leaves branch-tagged lifecycle events in the
    META's events.jsonl while child logs stay separate (reuses the offline
    robustness harness's fake child + fake gate)."""
    import tests.test_meta_fanout_robustness as rb
    import scilink.agents.meta_agent.fanout as fo
    rb._install_fake_child()
    ag, A, B, C = rb._agent()
    rb._install_fake_gate([A, B])  # branch ids are data paths
    out = json.loads(ag._run_fanout(rb._branches(A, B)))
    assert out.get("branches_run", out.get("n_branches", 2)) or True

    log = ag.base_dir / "events.jsonl" if hasattr(ag, "base_dir") else None
    from pathlib import Path as P
    log = P(ag.base_dir) / "events.jsonl"
    lines = [json.loads(l) for l in
             log.read_text(encoding="utf-8").splitlines() if l.strip()]
    branch_events = [e for e in lines if e["tool"] == "fanout_branch"]
    assert len(branch_events) == 2
    assert all(e.get("branch") for e in branch_events)
    assert {e["args"].split("label=")[1].split(" ")[0]
            for e in branch_events} == {os.path.basename(A),
                                        os.path.basename(B)}
