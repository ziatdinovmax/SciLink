"""The three orchestrators' file tools as thin wrappers over the shared
engine (#481): identical windowing / writing behavior per mode, plus each
mode's own extras (planning: deliverable recording + delegation scoping;
analysis: literature-file persistence; sim: whole-read report stems).
Analysis mode gains append / read / edit / rename here.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


# ── fixtures: bare orchestrators, real tool registration ─────────

def _analysis_tools(base):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    t = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=Path(base), model=None, analysis_results=[],
                             futurehouse_api_key=None, current_metadata=None)
    t.functions_map, t.openai_schemas = {}, []
    t._register_all_tools()
    return t


def _sim_tools(base):
    from scilink.agents.sim_agents.simulation_orchestrator_tools import SimulationOrchestratorTools
    t = SimulationOrchestratorTools.__new__(SimulationOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=str(base), model=None, generated_structures=[])
    t.functions_map, t.openai_schemas = {}, []
    t.logger = logging.getLogger("sim-test")
    t._register_all_tools()
    return t


def _planning_tools(base):
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    return OrchestratorTools(SimpleNamespace(base_dir=Path(base), planner=SimpleNamespace()))


def _call(t, name, **kw):
    return json.loads(t.functions_map[name](**kw))


# ── analysis: the new surface ────────────────────────────────────

def test_analysis_registers_the_full_surface(tmp_path):
    t = _analysis_tools(tmp_path)
    names = {s["function"]["name"] for s in t.openai_schemas}
    assert {"save_file", "append_file", "read_file", "edit_file", "rename_file",
            "read_document"} <= names
    desc = next(s["function"]["description"] for s in t.openai_schemas
                if s["function"]["name"] == "read_file")
    assert "do not read it repeatedly" in desc and "tail=true" in desc


def test_analysis_write_read_edit_rename_roundtrip(tmp_path):
    t = _analysis_tools(tmp_path)
    r = _call(t, "save_file", filename="notes.md", content="# Notes\n\nalpha = 1\n",
              subfolder="reports")
    assert r["status"] == "success" and r["created"] is True
    p = tmp_path / "reports" / "notes.md"
    r = _call(t, "append_file", filename="notes.md", content="beta = 2\n", subfolder="reports")
    assert r["status"] == "success" and p.read_text().endswith("beta = 2\n")
    # read: relative to the session dir, whole (short file)
    r = _call(t, "read_file", file_path="reports/notes.md")
    assert r["status"] == "success" and r["truncated"] is False and "alpha = 1" in r["content"]
    # edit: surgical, backed up; a miss names read_file
    r = _call(t, "edit_file", path="reports/notes.md", old_text="alpha = 1", new_text="alpha = 10")
    assert r["status"] == "success" and "alpha = 10" in p.read_text()
    assert (tmp_path / "reports" / "notes.before_edit.md").read_text().startswith("# Notes")
    r = _call(t, "edit_file", path="reports/notes.md", old_text="gamma", new_text="x")
    assert r["status"] == "error" and "read_file" in r["message"]
    # overwrite through save_file keeps a backup
    r = _call(t, "save_file", filename="notes.md", content="rewritten\n", subfolder="reports")
    assert r["overwritten"] is True and Path(r["backup"]).name == "notes.before_overwrite.md"
    assert "alpha = 10" in Path(r["backup"]).read_text()
    # rename: byte-exact, within the directory, path-escapes stripped
    r = _call(t, "rename_file", path="reports/notes.md", new_name="../../final.md")
    assert r["status"] == "success" and (tmp_path / "reports" / "final.md").read_text() == "rewritten\n"
    assert not (tmp_path / "reports" / "notes.md").exists()
    assert _call(t, "rename_file", path="reports/final.md", new_name="final.md")["status"] == "error"


def test_analysis_read_file_windows_and_json(tmp_path):
    t = _analysis_tools(tmp_path)
    log = tmp_path / "run.log"
    log.write_text("".join(f"step {i}\n" for i in range(1, 1001)) + "DONE ok\n")
    head = _call(t, "read_file", file_path=str(log))
    assert head["truncated"] is True and head["shown_lines"] == "1-200"
    tail = _call(t, "read_file", file_path=str(log), tail=True, max_lines=3)
    assert "DONE ok" in tail["content"]
    found = _call(t, "read_file", file_path="run.log", search=r"^DONE")
    assert found["matches"] == 1 and found["match_lines"] == [1001]
    res = tmp_path / "analysis_results.json"
    res.write_text(json.dumps({"peaks": [{"pos": i} for i in range(500)]}))
    j = _call(t, "read_file", file_path="analysis_results.json")
    assert j["status"] == "success" and j["truncated"] is True     # the JSON cap
    assert _call(t, "read_file", file_path="analysis_results.json", search='"pos": 499')["matches"] == 1
    # whole-read stems: a report is returned whole
    rep = tmp_path / "analysis_report.md"
    rep.write_text("".join(f"# S{i}\nbody\n" for i in range(300)))
    assert _call(t, "read_file", file_path="analysis_report.md")["truncated"] is False
    assert _call(t, "read_file", file_path="missing.txt")["status"] == "error"


def test_analysis_read_document_persists_literature_file(tmp_path):
    t = _analysis_tools(tmp_path)
    (tmp_path / "methods.md").write_text("# Methods\n\nUse pseudo-Voigt.\n")
    r = _call(t, "read_document", paths=["methods.md", "nope.pdf"])   # relative resolves
    assert r["status"] == "success" and r["n_documents"] == 1
    assert r["errors"] == ["Not a file: nope.pdf"]
    assert "pseudo-Voigt" in r["text"] and "literature_file" in r["hint"]
    lit = Path(r["file_path"])
    assert lit.parent == tmp_path / "literature" and "pseudo-Voigt" in lit.read_text()
    assert _call(t, "read_document", paths=[])["status"] == "error"
    assert _call(t, "read_document", paths="nope.md")["status"] == "error"


# ── simulation: behavior preserved + backup ───────────────────────

def test_sim_file_tools_on_the_engine(tmp_path):
    t = _sim_tools(tmp_path)
    r = _call(t, "save_file", filename="INCAR_notes.md", content="ENCUT = 400\n", subfolder="reports")
    assert r["status"] == "success"
    r = _call(t, "save_file", filename="INCAR_notes.md", content="ENCUT = 520\n", subfolder="reports")
    assert r["overwritten"] is True and Path(r["backup"]).read_text() == "ENCUT = 400\n"
    r = _call(t, "append_file", filename="INCAR_notes.md", content="ISMEAR = 0\n", subfolder="reports")
    assert (tmp_path / "reports" / "INCAR_notes.md").read_text() == "ENCUT = 520\nISMEAR = 0\n"
    # whole-read stems: report / summary
    rep = tmp_path / "run_summary.md"
    rep.write_text("".join(f"line {i}\n" for i in range(600)))
    assert _call(t, "read_file", file_path="run_summary.md")["truncated"] is False
    other = tmp_path / "OUTCAR"
    other.write_text("".join(f"line {i}\n" for i in range(600)))
    assert _call(t, "read_file", file_path="OUTCAR")["truncated"] is True
    assert _call(t, "read_file", file_path="OUTCAR", search="line 599")["matches"] == 1
    r = _call(t, "read_document", paths=["run_summary.md"])
    assert r["status"] == "success" and "file_path" not in r      # sim persists nothing


# ── planning: extras preserved on top of the engine ──────────────

def test_planning_save_keeps_deliverable_and_gets_backup(tmp_path):
    t = _planning_tools(tmp_path)
    r = json.loads(t.execute_tool("save_file", filename="brief.md", content="# v1\n",
                                  deliverable=True, title="Brief"))
    assert r["status"] == "success" and r["deliverable"] is True and "pdf_refreshed" in r
    from scilink.agents.planning_agents.user_interface import load_deliverables
    assert any(e["path"].endswith("brief.md") and e["deliverable"] for e in load_deliverables(tmp_path))
    r = json.loads(t.execute_tool("save_file", filename="brief.md", content="# v2\n",
                                  deliverable=True, title="Brief"))
    assert r["overwritten"] is True and Path(r["backup"]).read_text() == "# v1\n"
    # the overwrite backup is recorded as a produced (non-deliverable) file
    entries = {Path(e["path"]).name: e for e in load_deliverables(tmp_path)}
    assert entries["brief.before_overwrite.md"]["deliverable"] is False
    assert entries["brief.md"]["deliverable"] is True
    r = json.loads(t.execute_tool("read_file", file_path=str(tmp_path / "brief.md")))
    assert r["content"] == "# v2\n"
