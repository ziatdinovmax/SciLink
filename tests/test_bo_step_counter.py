"""#593 — the BO step index is campaign-level and monotonic. A reset sets
the history aside as a backup; the next step continues the numbering, so
iteration N never overwrites iteration N-1's step_* artifacts. A same-data
re-run replaces the step it repeats, keeping that step's number."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scilink.agents.planning_agents.bo_agent import BOAgent


def _agent(tmp_path):
    a = BOAgent.__new__(BOAgent)
    a.output_dir = Path(tmp_path)
    a.history_file = a.output_dir / "bo_history.json"
    return a


def test_next_step_continues_from_the_live_history(tmp_path):
    a = _agent(tmp_path)
    assert a._next_step_number([]) == 1
    assert a._next_step_number([{"step": 1}, {"step": 2}]) == 3
    assert a._next_step_number([{"step": 4}]) == 5             # gaps are preserved, not re-packed
    assert a._next_step_number([{"data_points": 9}]) == 2      # legacy entry without a step


def test_next_step_continues_from_a_reset_backup(tmp_path):
    a = _agent(tmp_path)
    (tmp_path / "bo_history.json.backup").write_text(json.dumps([{"step": 1}, {"step": 2}]))
    (tmp_path / "bo_history.json.backup.2").write_text(json.dumps([{"step": 3}]))
    (tmp_path / "bo_history.json.backup.3").write_text("not json")
    assert a._next_step_number([]) == 4


def _ctx(history, override=None):
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})
    written = {}
    optimizer = SimpleNamespace(generate_diagnostics=lambda *a, **k: written.setdefault("plot", k.get("save_path")) and {})
    return SimpleNamespace(history=history, df=df, optimizer=optimizer, output_dir="/tmp/bo593",
                           is_moo=False, target_cols=["y"], next_x_batch=np.array([[1.5]]),
                           plot_acq=False, save_acq=False, step_override=override), written


def test_diagnostics_number_the_step_after_the_backup(tmp_path):
    a = _agent(tmp_path)
    (tmp_path / "bo_history.json.backup").write_text(json.dumps([{"step": 1, "data_points": 9}]))
    c, written = _ctx([])
    BOAgent._stage_diagnostics(a, c)
    assert c.step_num == 2 and c.plot_path.endswith("step_2.png") and written["plot"].endswith("step_2.png")


def test_a_same_data_rerun_keeps_the_replaced_step_number(tmp_path):
    a = _agent(tmp_path)
    c, _ = _ctx([{"step": 1}, {"step": 2}], override=3)
    BOAgent._stage_diagnostics(a, c)
    assert c.step_num == 3


def test_reset_sets_the_history_aside_without_clobbering_and_reports_the_next_step(tmp_path):
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    bo = _agent(tmp_path / "bo_artifacts"); bo.output_dir.mkdir()
    orch = SimpleNamespace(base_dir=tmp_path, planner=SimpleNamespace(), bo=bo,
                           active_scalarizer_script="s", expected_input_columns=["x"],
                           expected_target_columns=["y"], analyzed_files={"f": 1},
                           analyzed_files_path=tmp_path / "analyzed_files.json",
                           bo_data_path=tmp_path / "optimization_data.csv")
    t = OrchestratorTools(orch)
    bo.history_file.write_text(json.dumps([{"step": 1, "data_points": 9}]))
    orch.bo_data_path.write_text("x,y\n1,2\n")
    out = json.loads(t.functions_map["reset_analysis_logic"]())
    assert out["next_bo_step"] == 2 and "numbered 2" in out["hint"]
    assert (bo.output_dir / "bo_history.json.backup").exists() and not bo.history_file.exists()
    # a second reset after another iteration keeps BOTH backups
    bo.history_file.write_text(json.dumps([{"step": 2, "data_points": 10}]))
    out = json.loads(t.functions_map["reset_analysis_logic"]())
    assert out["next_bo_step"] == 3
    assert (bo.output_dir / "bo_history.json.backup.2").exists()
    assert json.loads((bo.output_dir / "bo_history.json.backup").read_text())[0]["step"] == 1
