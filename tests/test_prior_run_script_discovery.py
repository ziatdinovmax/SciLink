"""#591 — a prior run's locked fitting scripts are discoverable: read_file
on a directory lists it, examine_data names an analysis output directory
and its scripts, and the curve / image manifests advertise the scripts."""
import json
from pathlib import Path
from types import SimpleNamespace

from scilink.utils.file_io import read_file_content, list_directory


def _prior_run(tmp_path, n=3):
    run = tmp_path / "curve_fit_20260908"
    (run / "scripts").mkdir(parents=True)
    for i in range(n):
        (run / "scripts" / f"spectrum_{i:04d}.py").write_text("print('fit')\n")
    (run / "trend_analysis.py").write_text("print('trend')\n")
    (run / "analysis_results.json").write_text(json.dumps({"status": "success"}))
    (run / "features.csv").write_text("a,b\n1,2\n")
    (run / "spectrum_0000").mkdir()
    return run


def test_read_file_on_a_directory_lists_it_and_names_the_scripts(tmp_path):
    run = _prior_run(tmp_path)
    out = read_file_content(run / "scripts", display_path="curve_fit_20260908/scripts")
    assert out["status"] == "success" and out["is_directory"] is True
    assert [e["name"] for e in out["entries"]] == ["spectrum_0000.py", "spectrum_0001.py", "spectrum_0002.py"]
    assert out["scripts"] == ["spectrum_0000.py", "spectrum_0001.py", "spectrum_0002.py"]
    assert "is a directory, not a file" in out["hint"] and "spectrum_0000.py" in out["hint"]
    # the run dir itself: sub-directories first, scripts/ contents surfaced
    top = read_file_content(run)
    names = [e["name"] for e in top["entries"]]
    assert names[:2] == ["scripts/", "spectrum_0000/"] and "analysis_results.json" in names
    assert top["scripts"] == ["trend_analysis.py", "scripts/spectrum_0000.py", "scripts/spectrum_0001.py", "scripts/spectrum_0002.py"]
    assert next(e for e in top["entries"] if e["name"] == "scripts/")["entries"] == 3
    # a real file still reads as before
    assert read_file_content(run / "features.csv")["status"] == "success"


def test_directory_listing_is_capped(tmp_path):
    d = tmp_path / "many"; d.mkdir()
    for i in range(20):
        (d / f"f{i:03d}.txt").write_text("x")
    out = list_directory(d, max_entries=5)
    assert len(out["entries"]) == 5 and out["truncated"] is True


def _examine(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    t = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=Path(tmp_path), model=None, analysis_results=[],
                             futurehouse_api_key=None, current_metadata=None)
    t.functions_map, t.openai_schemas = {}, []
    t._register_all_tools()
    return t.functions_map["examine_data"]


def test_examine_data_names_an_analysis_output_dir_and_its_scripts(tmp_path):
    run = _prior_run(tmp_path)
    out = json.loads(_examine(tmp_path)(str(run)))
    assert out["analysis_output_dir"] is True and out["analysis_results"] == "analysis_results.json"
    assert out["saved_scripts"] == ["scripts/spectrum_0000.py", "scripts/spectrum_0001.py", "scripts/spectrum_0002.py"]
    assert "prior analysis OUTPUT directory" in out["prior_run_hint"]
    assert "reuse_locked_script=true" in out["prior_run_hint"] and "any one is the reusable fit script" in out["prior_run_hint"]


def test_examine_data_leaves_a_plain_data_dir_alone(tmp_path):
    d = tmp_path / "series"; d.mkdir()
    for i in range(3):
        (d / f"s{i}.csv").write_text("x,y\n1,2\n3,4\n")
    out = json.loads(_examine(tmp_path)(str(d)))
    assert "analysis_output_dir" not in out and "prior_run_hint" not in out


def test_curve_manifest_advertises_the_locked_scripts(tmp_path):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    a = CurveFittingAgent.__new__(CurveFittingAgent)
    a.output_dir = tmp_path; a.logger = __import__("logging").getLogger("t")
    state = {"is_single_spectrum": False,
             "series_results": [{"index": i, "name": f"T={300 + 10 * i}K", "script": "print(1)\n", "success": True}
                                for i in range(3)]}
    saved = a._save_fitting_scripts(state)
    rec = a._fitting_scripts_record(saved, state)
    assert rec["files"] == ["T_300K.py", "T_310K.py", "T_320K.py"] and rec["representative"] == "T_300K.py"
    assert rec["dir"] == str(tmp_path / "scripts")
    assert "3 copies of the same script" in rec["note"] and "reuse_locked_script=true" in rec["note"]
    single = a._fitting_scripts_record(["/x/scripts/fitting_script.py"], {"is_single_spectrum": True})
    assert single["note"].startswith("The fitting script that produced this result.")
