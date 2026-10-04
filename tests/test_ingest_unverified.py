"""#737: the optimization hand-off does not train on unverified units.

Since #735 every unit of a series is a row of ``features.csv`` with a
``verified`` column; the planning ingestion read every row. Through the real
path — a series result written by the real ``write_feature_table``, the real
``analyze_file`` tool and the real scalarizer's table pass-through (explicit
inputs/targets: no model call) — a salvaged unit is skipped and named with
its reason, a failed one too, and recorded on the campaign;
``include_unverified=True`` keeps the salvaged one as low-confidence data and
records that; a table without the column ingests as before.

  conda run -n scilink python -m pytest tests/test_ingest_unverified.py -q
"""
import contextlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.feature_table import write_feature_table
from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel, PlanningOrchestratorAgent


def _series_table(run_dir: Path) -> Path:
    """A hyperspectral-shaped series result: d1, d3, d4 verified, d2 salvaged
    (values, unverified), d5 failed (no values) — written to features.csv by
    the real writer."""
    run_dir.mkdir(parents=True)
    rows = []
    for i, (name, ok, verified) in enumerate([("d1", True, True), ("d2", True, False), ("d3", True, True),
                                               ("d4", True, True), ("d5", False, False)]):
        r = {"index": i, "name": name, "data_path": None, "success": ok, "verified": verified,
             "extracted_features": ({"Depth_mean": 0.1 * (i + 1)} if ok else {}),
             "unit_verdict": {"verified": verified, "reason": "x"}}
        if ok and not verified:
            r.update(flagged=True, flag_reason="unverified")
        if not ok:
            r.update(flagged=True, flag_reason="analysis_failed", error="fit failed")
        rows.append(r)
    (run_dir / "series_analysis_results.json").write_text(json.dumps({
        "results": rows, "series_metadata": {"variable": "dose", "values": [10, 20, 30, 40, 50], "unit": "mJ"}}))
    return Path(write_feature_table(run_dir))


@pytest.fixture(autouse=True)
def _sandbox_approved(monkeypatch):
    # per test: another module pops this from os.environ without restoring it
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")


def _orch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        return PlanningOrchestratorAgent(base_dir=str(tmp_path / "session"), api_key="sk-dummy",
                                         autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data))


def _ingest(o, path, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return json.loads(o.tools.execute_tool("analyze_file", file_path=str(path), extraction_goal="dose series",
                                               inputs=["dose"], targets=["Depth_mean"], **kw))


def test_unverified_units_are_skipped_named_and_recorded(tmp_path):
    table = _series_table(tmp_path / "run")
    assert list(pd.read_csv(table)["verified"]) == [True, False, True, True, False]
    o = _orch(tmp_path)
    out = _ingest(o, table)
    assert out["status"] == "success" and out["rows_added"] == 3, out
    assert out["rows_skipped_unverified_units"] == [{"unit": "d2", "reason": "unverified"},
                                                    {"unit": "d5", "reason": "analysis_failed"}]
    assert "d2 (unverified)" in out["unverified_warning"] and "include_unverified=True" in out["unverified_warning"]
    ingested = pd.read_csv(o.bo_data_path)
    assert sorted(ingested["dose"].tolist()) == [10.0, 30.0, 40.0]           # the surrogate's data
    rec = o.analyzed_files[str(table.resolve())]
    assert [u["unit"] for u in rec["skipped_unverified"]] == ["d2", "d5"]
    assert json.loads(Path(o.analyzed_files_path).read_text())[str(table.resolve())]["skipped_unverified"]
    # the copy extraction read is named as derived, never as the analysis's own table
    copies = list(Path(o.bo_data_path).parent.glob("ingest/**/*.csv"))
    assert [c.name for c in copies] == ["features.verified_only.csv"]


def test_include_unverified_keeps_the_salvaged_unit_and_says_so(tmp_path):
    table = _series_table(tmp_path / "run")
    o = _orch(tmp_path)
    out = _ingest(o, table, include_unverified=True)
    assert out["status"] == "success" and out["rows_added"] == 4, out           # d5 has no value: still skipped
    assert out["rows_skipped_units"] == ["d5"]
    assert out["included_unverified_units"] == [{"unit": "d2", "reason": "unverified"}]
    assert "UNVERIFIED" in out["unverified_warning"] and "weights them like any other point" in out["unverified_warning"]
    assert [u["unit"] for u in o.analyzed_files[str(table.resolve())]["included_unverified"]] == ["d2"]


def test_a_table_without_the_column_ingests_as_before(tmp_path):
    table = tmp_path / "plain.csv"
    table.write_text("unit,dose,Depth_mean\na,10,0.1\nb,20,0.2\nc,30,0.3\n")
    o = _orch(tmp_path)
    out = _ingest(o, table)
    assert out["status"] == "success" and out["rows_added"] == 3
    assert "rows_skipped_unverified" not in out and "unverified_warning" not in out
    assert "skipped_unverified" not in o.analyzed_files[str(table.resolve())]


def test_a_table_with_only_unverified_rows_is_refused_with_the_reason(tmp_path):
    table = tmp_path / "bad.csv"
    table.write_text("unit,dose,Depth_mean,verified,flag_reason\na,10,0.1,False,unverified\nb,20,0.2,False,unverified\n")
    out = _ingest(_orch(tmp_path), table)
    assert out["status"] == "error" and "Every row of bad.csv is unverified" in out["message"]
    assert "include_unverified=True" in out["hint"]
