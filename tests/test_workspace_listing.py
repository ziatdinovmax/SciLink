"""list_workspace_files must expose the delegation tree and the
deliverables index.

Live failure this pins: under a meta session every artifact lives in
delegations/<task-slug>/, invisible to the flat listing — an agent
probed six guessed paths for a companion whose real filename it could
not know, and only found it by reading deliverables.json as a last
resort. The listing now answers both questions directly: what exists
(one-level delegation contents) and what things are called
(title -> path index from the registry).
"""

import json
from pathlib import Path
from types import SimpleNamespace

from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.user_interface import record_deliverable


def make_tools(tmp_path):
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        planner=SimpleNamespace(state={}, model=None),
        base_dir=tmp_path,
        _active_output_subdir=None,
        lit_agent=None,
        bo_data_path=tmp_path / "optimization_data.csv",
        active_scalarizer_script=None,
    )
    cap = {}
    tools._register_tool = (
        lambda func, name, description, parameters, required=None:
        cap.update({name: func}))
    ot.OrchestratorTools._register_all_tools(tools)
    return cap


def test_listing_surfaces_delegations_and_registry(tmp_path):
    d01 = tmp_path / "delegations" / "01_brainstorm_a_chemical_dynamic"
    d01.mkdir(parents=True)
    doc = d01 / "cdoc_platform_engineering_companion.md"
    doc.write_text("# Companion\n")
    (d01 / "campaign_workflow.png").write_bytes(b"png")
    (d01 / "outputs").mkdir()
    (tmp_path / "chat_history.json").write_text("[]")
    record_deliverable(tmp_path, doc, "MZI Engineering Companion",
                       deliverable=True)

    cap = make_tools(tmp_path)
    out = json.loads(cap["list_workspace_files"]())

    assert out["status"] == "success"
    assert "chat_history.json" in out["files"]

    # the semantic index: title -> real path, starred flag preserved
    idx = out["deliverables_index"]
    assert idx == [{"title": "MZI Engineering Companion",
                    "path": str(doc), "deliverable": True}]
    assert "guessing" in out["hint"]

    # one-level delegation contents, dirs marked with a trailing slash
    assert out["delegation_folders"] == {
        "01_brainstorm_a_chemical_dynamic": [
            "campaign_workflow.png",
            "cdoc_platform_engineering_companion.md",
            "outputs/",
        ]}


def test_vanished_registry_entries_are_dropped(tmp_path):
    d = tmp_path / "delegations" / "02_x"
    d.mkdir(parents=True)
    gone = d / "renamed_away.md"
    gone.write_text("x")
    record_deliverable(tmp_path, gone, "Old name")
    gone.unlink()                      # renamed/deleted since recording

    cap = make_tools(tmp_path)
    out = json.loads(cap["list_workspace_files"]())
    assert "deliverables_index" not in out   # only existing files listed
    assert out["delegation_folders"] == {"02_x": []}


def test_flat_session_payload_unchanged(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    cap = make_tools(tmp_path)
    out = json.loads(cap["list_workspace_files"]())
    assert out["files"] == ["plan.json"]
    assert "deliverables_index" not in out
    assert "delegation_folders" not in out
    assert out["optimization_ready"] is False
