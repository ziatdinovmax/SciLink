"""A hyperspectral series: each unit sees its own metadata, and a partial
series says how partial it is (#723, items 2 and 3).

Through the orchestrator's real ``run_analysis`` → hyperspectral series path,
the single-cube pipeline stubbed at the seam tests/test_hs_series.py uses:

- with an explicit ``series_metadata``, each unit's prompts still carry its
  own sidecar fields (an exposure that differs 17x between units), and the
  unit's own value wins over a shared field built from one file;
- a series where one unit of three produced features says so in
  ``run_task``'s warnings, on the board's claim, in ``features.csv`` (every
  unit listed, ``verified`` / ``flag_reason``, the series variable kept beside
  the sidecar fields, a sidecar column that only repeats it dropped), in the
  flag's recommendation (no recipe locked: nothing re-analyses it) and in the
  series synthesis prompt (no locked recipe, no "same verified script").

  conda run -n scilink python -m pytest tests/test_hs_series_visibility.py -q
"""
import contextlib
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa
from scilink.agents.meta_agent import board as board_mod

AXIS = {"technique": "EELS", "sample": "TiOx",
        "energy_range": {"start": 450.0, "end": 550.0, "units": "eV"}}
EXPOSURES = [0.1, 1.7, 0.5]
DOSES = [1, 2, 3]
SCRIPT = "def analyze_feature(data, axis):\n    return {'maps': {'Mean_Map': data.mean(2)}}\n"


def _series_dir(tmp_path, repeat_series_variable=False):
    d = tmp_path / "cubes"
    d.mkdir()
    for i, (exp, dose) in enumerate(zip(EXPOSURES, DOSES)):
        np.save(d / f"cube_{i}.npy", np.full((4, 4, 8), float(i + 1), dtype=np.float32))
        # a bundle-wide condition index and an acquisition field that differs
        # per unit (the live shape); optionally the series variable under
        # another name, as an older sidecar would carry it
        side = {"condition_index": 7 + i, "exposure_s": exp}
        if repeat_series_variable:
            side["dose_mC"] = dose
        (d / f"cube_{i}.json").write_text(json.dumps(side))
    return d


def _orchestrator(tmp_path, monkeypatch, seen, fail=(), salvage=()):
    """The real orchestrator and hyperspectral agent; the single-cube pipeline
    records the system_info each unit receives, and fails the named cubes."""
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent, AnalysisMode
    from test_hs_series import _Calls, _FakeModel
    calls = _Calls()

    def pipeline(self, data_path, system_info, instruction_prompt, reuse_records=None, **kw):
        name = Path(data_path).stem
        seen.append((name, dict(system_info or {})))
        if name in fail:
            return None, {"error": "fit failed", "details": f"nothing measurable on {name}"}
        value = float(np.load(data_path).mean())
        ok = name not in salvage            # a salvaged attempt: features, nothing approved
        rec = {"target": "mean map", "task_success": ok, "required_outputs": ["Mean_Map"],
               "script": SCRIPT, "quality_history": {"approved": ok},
               "locked_replay": bool(reuse_records), "replay_verbatim": True}
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps([rec]))
        return {"detailed_analysis": f"analysis of {name}", "scientific_claims": [],
                "extracted_features": [{"name": "Mean_Map", "units": "a.u.",
                                        "stats": {"min": value - 1, "max": value + 1, "mean": value}}],
                "dynamic_analysis_records": [rec]}, None

    # on the class: the series driver builds one child agent per dataset
    A = hsa.HyperspectralAnalysisAgent
    monkeypatch.setattr(A, "_run_analysis_pipeline", pipeline)
    monkeypatch.setattr(A, "_maybe_bank_scripts", lambda *x, **k: [])
    monkeypatch.setattr(A, "_maybe_stage_t2_solutions", lambda *x, **k: [])
    monkeypatch.setattr(A, "_auto_select_skills", lambda *x, **k: [])
    monkeypatch.setenv("SCILINK_HS_SERIES_POOL", "thread")

    def make_agent(agent_id, out_dir, **kw):
        a = A(api_key="sk-dummy", output_dir=str(out_dir), enable_human_feedback=False, executor_timeout=120)
        a.model = _FakeModel(calls)
        return a
    with contextlib.redirect_stdout(io.StringIO()):
        orch = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                         model_name="claude-opus-4-6", analysis_mode=AnalysisMode.AUTONOMOUS)
    orch.create_agent_for_analysis = make_agent
    # the shared metadata as one file's sidecar once filled it: unit 0's exposure
    orch.current_metadata = {**AXIS, "exposure_s": EXPOSURES[0]}
    return orch, calls


def _table(result):
    with open(Path(result["analyses"][0]["output_directory"]) / "features.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _run(orch, data_dir):
    def chat(prompt):
        out = json.loads(orch.tools.execute_tool(
            "run_analysis", data_path=str(data_dir), agent_id=2, analysis_goal="track the mean",
            series_metadata={"variable": "dose", "unit": "mC",
                             "values": {f"cube_{i}.npy": d for i, d in enumerate(DOSES)}}))
        assert out.get("status") in ("success", "partial"), out
        return "analysed"
    orch.chat = chat
    with contextlib.redirect_stdout(io.StringIO()):
        return orch.run_task("analyse the dose series")


def test_each_unit_sees_its_own_sidecar_fields_with_explicit_series_metadata(tmp_path, monkeypatch):
    seen = []
    orch, _ = _orchestrator(tmp_path, monkeypatch, seen)
    _run(orch, _series_dir(tmp_path))
    by_unit = {name: si for name, si in seen}
    assert {n: by_unit[n].get("exposure_s") for n in by_unit} == {
        f"cube_{i}": e for i, e in enumerate(EXPOSURES)}


def test_a_partial_series_says_how_partial_it_is(tmp_path, monkeypatch):
    """The issue's shape: no unit anchors (two fail, the third commits a
    salvaged attempt), so no recipe locks and nothing can be refit."""
    seen = []
    orch, calls = _orchestrator(tmp_path, monkeypatch, seen, fail=("cube_0", "cube_1"), salvage=("cube_2",))
    result = _run(orch, _series_dir(tmp_path))
    row = result["analyses"][0]
    assert row["series_coverage"]["units"] == 3 and row["series_coverage"]["with_features"] == 1
    assert row["series_coverage"]["failed"] == ["cube_0", "cube_1"]
    assert row["series_coverage"]["unverified"] == ["cube_2"]
    assert any("(series): 1 of 3 units produced features; unverified: cube_2; failed: cube_0, cube_1" in w
               for w in result["warnings"]), result["warnings"]
    # the board's claim carries the coverage
    entry = {"index": 1, "label": "dose", "mode": "analysis", "status": "success"}
    claims = [r for r in board_mod.records_for(entry, {**result, "key_findings": [
        f"[{row['analysis_id']}] the mean rises with dose"]}) if r["kind"] == "claim"]
    assert claims[0]["evidence"]["coverage"] == {"units": 3, "with_features": 1, "n_unverified": 1, "n_failed": 2}
    # the table: every unit, its status, the series variable beside the sidecar
    with open(Path(row["output_directory"]) / "features.csv", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert [t["unit"] for t in table] == ["cube_0", "cube_1", "cube_2"]
    assert [t["verified"] for t in table] == ["False", "False", "False"]
    assert [t["flag_reason"] for t in table] == ["analysis_failed", "analysis_failed", "unverified"]
    assert [float(t["dose"]) for t in table] == DOSES       # the series variable is in the table
    assert [t["condition_index"] for t in table] == ["7", "8", "9"]
    assert [t["exposure_s"] for t in table] == ["0.1", "1.7", "0.5"]
    # no recipe locked: the flag does not promise a re-analysis
    from scilink.agents.exp_agents.controllers.hyperspectral_series import FLAGGED_FILENAME
    flags = json.loads((Path(row["output_directory"]) / FLAGGED_FILENAME).read_text())["flagged_datasets"]
    recs = [f["recommendation"] for f in flags]
    assert len(recs) == 3 and all("not re-analysed" in r and "refit budget allows" not in r for r in recs)
    # the series synthesis is not told a verified script ran on every dataset
    synth = next(p for p in calls.llm if "WHAT A GATE CHECKED" in p)
    assert "NO recipe was locked" in synth and "same verified script" not in synth


def test_with_a_locked_recipe_a_failed_unit_is_still_promised_a_refit(tmp_path, monkeypatch):
    """The other side: the last unit anchors and locks a recipe, so the
    earlier failures CAN be refit and their flag says so."""
    seen = []
    orch, _ = _orchestrator(tmp_path, monkeypatch, seen, fail=("cube_0",))
    result = _run(orch, _series_dir(tmp_path, repeat_series_variable=True))
    out = Path(result["analyses"][0]["output_directory"])
    # a sidecar column already holding the series variable keeps its name, as
    # tables had it before (a planning campaign keyed on it still finds it);
    # the series variable is not added a second time
    table = _table(result)
    assert [float(t["dose_mC"]) for t in table] == DOSES and "dose" not in table[0]
    from scilink.agents.exp_agents.controllers.hyperspectral_series import FLAGGED_FILENAME
    flags = json.loads((out / FLAGGED_FILENAME).read_text())["flagged_datasets"]
    recs = [f["recommendation"] for f in flags if f.get("reason") == "analysis_failed"]
    assert recs and all("refit budget allows" in r for r in recs)
