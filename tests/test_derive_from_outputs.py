"""#599 — a lightweight, sandboxed compute step over a completed run's
outputs: inventory → (supplied or generated) script → guard → sandbox →
deterministic product gate → retry with feedback. No planning, no
re-analysis, no writes outside the derive directory."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scilink.executors as executors
from scilink.agents.exp_agents.derive_outputs import (
    build_artifact_inventory, check_products, parse_result_marker, run_derivation, static_guard, RESULT_MARKER)
from scilink.executors import ScriptExecutor


def _prior_run(tmp_path):
    run = tmp_path / "results" / "analysis_image_ImageAnalysis_20260909_000001"
    (run / "scripts").mkdir(parents=True); (run / "_scratch").mkdir()
    np.save(run / "polarization_px.npy", np.arange(12, dtype=float).reshape(3, 4))
    np.save(run / "polarization_py.npy", -np.arange(12, dtype=float).reshape(3, 4))
    (run / "features.csv").write_text("median_P_pm,n_cells\n42.1,12\n")
    (run / "analysis_results.json").write_text(json.dumps({"status": "success", "detailed_analysis": "two domains"}))
    (run / "scripts" / "analysis_script.py").write_text("print('x')\n")
    (run / "_scratch" / "junk.npy").write_bytes(b"0" * 10)
    return run


TABLE_SCRIPT = '''
import os, json, numpy as np, pandas as pd
files = {os.path.basename(f): f for f in _DERIVE["files"]}
px = np.load(files["polarization_px.npy"]); py = np.load(files["polarization_py.npy"])
rows = [{"row": i, "col": j, "Px": float(px[i, j]), "Py": float(py[i, j]), "P": float(np.hypot(px[i, j], py[i, j]))}
        for i in range(px.shape[0]) for j in range(px.shape[1])]
out = os.path.join(_DERIVE["out_dir"], "per_cell_polarization.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("DERIVE_RESULT_JSON:" + json.dumps({"products": [{"path": out, "description": "per-cell Px, Py, |P|"}],
                                          "summary": "12 cells from the two saved component maps"}))
'''


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies); self.calls = []

    def generate_content(self, prompt, **kw):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.replies.pop(0))


def test_inventory_describes_arrays_tables_and_json_and_skips_scratch(tmp_path):
    run = _prior_run(tmp_path)
    text, files = build_artifact_inventory([run])
    assert "polarization_px.npy  —  npy shape=(3, 4) dtype=float64" in text
    assert "features.csv  —  table rows=1 columns=['median_P_pm', 'n_cells']" in text
    assert "analysis_results.json  —  json keys=['status', 'detailed_analysis']" in text
    assert "junk.npy" not in text and all("_scratch" not in f for f in files)
    assert any(f.endswith("scripts/analysis_script.py") for f in files)


def test_guard_marker_and_product_gate(tmp_path):
    assert static_guard("import subprocess\n_DERIVE") and static_guard("x = 1") and static_guard("import os\nos.remove('x')\n_DERIVE")
    assert static_guard("import numpy as np\np = _DERIVE['out_dir']") is None
    assert parse_result_marker("nothing")[0] is None and parse_result_marker(RESULT_MARKER + "[1]")[0] is None
    out = tmp_path / "d"; out.mkdir()
    import time; t0 = time.time() - 5
    (out / "ok.csv").write_text("a\n1\n"); (out / "empty.csv").write_text("")
    (tmp_path / "outside.csv").write_text("a\n")
    problems, clean = check_products({"products": [{"path": str(out / "ok.csv"), "description": "d"},
                                                    {"path": str(out / "empty.csv")}, {"path": str(tmp_path / "outside.csv")},
                                                    {"path": "missing.csv"}]}, out, t0)
    assert [c["path"] for c in clean] == [str((out / "ok.csv").resolve())]
    assert any("empty" in p for p in problems) and any("outside" in p for p in problems) and any("does not exist" in p for p in problems)
    assert check_products({"products": []}, out, t0)[0] == ["the result names no products"]


def test_engine_runs_a_supplied_script_without_calling_the_model(tmp_path):
    run = _prior_run(tmp_path)
    model = FakeModel([])
    out = tmp_path / "derive"
    res = run_derivation(model=model, executor=ScriptExecutor(timeout=60), sources=[str(run)],
                         task="per-cell table", out_dir=out, scratch_dir=out / "_scratch", code=TABLE_SCRIPT)
    assert res["status"] == "success" and res["attempts"] == 1 and model.calls == []
    csv = Path(res["products"][0]["path"])
    assert csv.name == "per_cell_polarization.csv" and csv.read_text().count("\n") == 13
    assert (out / "scripts" / "derive_script.py").is_file() and (out / "derivation_receipt.json").is_file()
    ar = json.loads((out / "analysis_results.json").read_text())
    assert ar["agent_type"] == "derivation" and ar["products"][0]["description"] == "per-cell Px, Py, |P|"
    assert ar["sources"] == [str(run.resolve())]


def test_engine_generates_retries_with_feedback_then_succeeds(tmp_path):
    run = _prior_run(tmp_path)
    bad = "import numpy as np\nprint(_DERIVE['out_dir'])\n"          # runs, prints no marker
    model = FakeModel(["```python\n" + bad + "```", "```python\n" + TABLE_SCRIPT + "```"])
    out = tmp_path / "derive"
    res = run_derivation(model=model, executor=ScriptExecutor(timeout=60), sources=[str(run)],
                         task="per-cell table of Px, Py, |P|", out_dir=out, scratch_dir=out / "_scratch")
    assert res["status"] == "success" and res["attempts"] == 2
    assert "PREVIOUS ATTEMPT FAILED" in model.calls[1] and "did not print the DERIVE_RESULT_JSON" in model.calls[1]
    assert "polarization_px.npy  —  npy shape=(3, 4)" in model.calls[0]     # the inventory reached the model
    assert "do not re-plan" in model.calls[0]


def test_engine_gives_up_after_max_attempts_and_reports_the_last_failure(tmp_path):
    run = _prior_run(tmp_path)
    escape = "import subprocess\n_DERIVE\n"
    model = FakeModel(["```python\n" + escape + "```"] * 2)
    res = run_derivation(model=model, executor=ScriptExecutor(timeout=60), sources=[str(run)],
                         task="t", out_dir=tmp_path / "d", scratch_dir=tmp_path / "s", max_attempts=2)
    assert res["status"] == "error" and "forbidden module" in res["message"] and res["attempts"] == 2


def test_engine_rejects_a_product_written_outside_out_dir(tmp_path):
    run = _prior_run(tmp_path)
    leak = TABLE_SCRIPT.replace('os.path.join(_DERIVE["out_dir"], "per_cell_polarization.csv")',
                                'os.path.join(_DERIVE["sources"][0], "leak.csv")')
    res = run_derivation(model=FakeModel([]), executor=ScriptExecutor(timeout=60), sources=[str(run)],
                         task="t", out_dir=tmp_path / "d", scratch_dir=tmp_path / "s", code=leak, max_attempts=1)
    assert res["status"] == "error" and "outside the output directory" in res["message"]


# ── the orchestrator tool ──────────────────────────────────────────

def _tools(tmp_path, model):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import AnalysisOrchestratorTools
    t = AnalysisOrchestratorTools.__new__(AnalysisOrchestratorTools)
    t.orch = SimpleNamespace(base_dir=Path(tmp_path), results_dir=Path(tmp_path) / "results", model=None,
                             analysis_results=[], _analysis_run_counter=0, futurehouse_api_key=None,
                             current_metadata=None, current_data_path=None, _custom_skills={})
    t.functions_map, t.openai_schemas = {}, []
    t._register_all_tools()
    t._internal_model = lambda: model
    return t


@pytest.fixture(autouse=True)
def _sandbox_ok(monkeypatch):
    monkeypatch.setattr(executors, "_GLOBAL_SANDBOX_APPROVED", True)


def test_tool_derives_from_a_recorded_run_and_records_the_derivation(tmp_path):
    run = _prior_run(tmp_path)
    t = _tools(tmp_path, FakeModel(["```python\n" + TABLE_SCRIPT + "```"]))
    t.orch.analysis_results.append({"analysis_id": "img_001", "status": "success", "output_directory": str(run)})
    out = json.loads(t.functions_map["derive_from_outputs"](task="per-cell CSV of Px, Py, |P|", analysis_id="img_001"))
    assert out["status"] == "success" and out["derived_from"] == "img_001"
    assert out["files_produced"] == [str((Path(out["output_directory"]) / "per_cell_polarization.csv").resolve())]
    assert Path(out["output_directory"]).name.startswith("derive_") and "no analysis was re-run" in out["note"]
    rec = t.orch.analysis_results[-1]
    assert rec["agent_id"] == "derive" and rec["derived_from"] == "img_001" and rec["status"] == "success"
    # the schema is registered with task required and the prompt-facing description says NOT run_analysis
    schema = next(s for s in t.openai_schemas if s["function"]["name"] == "derive_from_outputs")
    assert schema["function"]["parameters"]["required"] == ["task"]
    assert "NO re-analysis" in schema["function"]["description"]


def test_tool_defaults_to_the_latest_completed_run_and_accepts_supplied_code(tmp_path):
    run = _prior_run(tmp_path)
    t = _tools(tmp_path, FakeModel([]))
    t.orch.analysis_results += [{"analysis_id": "failed", "status": "error", "output_directory": str(tmp_path / "nope")},
                                {"analysis_id": "img_002", "status": "success", "output_directory": str(run)}]
    out = json.loads(t.functions_map["derive_from_outputs"](task="table", code=TABLE_SCRIPT))
    assert out["status"] == "success" and out["derived_from"] == "img_002" and out["attempts"] == 1


def test_tool_errors_are_plain(tmp_path):
    t = _tools(tmp_path, FakeModel([]))
    assert "task is required" in json.loads(t.functions_map["derive_from_outputs"]())["message"]
    assert "No completed run" in json.loads(t.functions_map["derive_from_outputs"](task="x"))["message"]
    assert "No analysis with id" in json.loads(t.functions_map["derive_from_outputs"](task="x", analysis_id="zzz"))["message"]
    assert "Path not found" in json.loads(t.functions_map["derive_from_outputs"](task="x", source_paths="/no/such"))["message"]


def test_prompt_routes_reshaping_to_the_new_tool():
    from scilink.agents.exp_agents import analysis_orchestrator as ao
    assert "`derive_from_outputs`" in ao._SYSTEM_PROMPT_BODY_POST
    assert "produce X from what run Y already computed" in ao._SYSTEM_PROMPT_BODY_POST
