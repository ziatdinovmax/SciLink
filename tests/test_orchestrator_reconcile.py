"""Integration test for the orchestrator reconcile_series tool wiring.

Drives the registered orchestrator tool against a minimal fake orchestrator
whose analysis_results point at two synthetic series-analysis output dirs (a
profile-fitting pass and an identification pass), proving the wiring:
lookup by index/id -> extract from output dirs -> reconcile. Technique
vocabulary is kept generic (not XRD) so this also exercises the shared core.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_series_run(out_dir: Path, per_frame, values):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_results.json").write_text(json.dumps(
        {"status": "success", "individual_results": per_frame}))
    (out_dir / "series_fit_results.json").write_text(json.dumps(
        {"series_metadata": {"values": values, "variable": "temperature"}}))


@pytest.fixture()
def two_runs(tmp_path):
    n = 21
    values = [30 + 40 * i / (n - 1) for i in range(n)]
    prof, ids = [], []
    for i in range(n):
        f = i / (n - 1)
        prof.append({"index": i, "parameters": {
            "peak_1": {"center": 11.0, "area": 1000 * (1 - f)},
            "peak_2": {"center": 22.0, "area": 500 * (1 - f)},
            "peak_3": {"center": 12.0, "area": 1000 * f},
            "peak_4": {"center": 20.0, "area": 500 * f}}})
        lab = None if 9 <= i <= 11 else ("PhaseA" if i < 10 else "PhaseB")
        ids.append({"index": i, "parameters": {"identified_phase": lab}})
    mf = tmp_path / "profile_run"
    idr = tmp_path / "id_run"
    _write_series_run(mf, prof, values)
    _write_series_run(idr, ids, values)
    return mf, idr


def _make_tools(tmp_path, mf, idr):
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        AnalysisOrchestratorTools)
    orch = SimpleNamespace(
        _agent_registry={},
        analysis_results=[
            {"analysis_id": "prof-1", "output_directory": str(mf)},
            {"analysis_id": "id-1", "output_directory": str(idr)},
        ],
        results_dir=tmp_path,
    )
    return AnalysisOrchestratorTools(orch)


def test_reconcile_tool_registered(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    assert "reconcile_series" in tools.functions_map


def test_reconcile_by_index(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    out = json.loads(tools.functions_map["reconcile_series"](
        profile_analysis=0, identification_analysis=1))
    assert out["status"] == "success"
    assert out["low_regime_label"] == "PhaseA"
    assert out["high_regime_label"] == "PhaseB"
    assert out["agreement"]["verdict"] == "consistent"
    assert (tmp_path / "reconciled_series.png").exists()


def test_reconcile_by_id(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    out = json.loads(tools.functions_map["reconcile_series"](
        profile_id="prof-1", identification_id="id-1"))
    assert out["status"] == "success"
    assert out["low_regime_label"] == "PhaseA"


def test_reconcile_feeds_figure_but_cache_stays_lean(two_runs, tmp_path):
    # The returned result carries the figure as image_base64 (so the loop can
    # feed it to the LLM), but the persisted cache must NOT — finalize does not
    # need the bytes, and the cache would bloat.
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    out = json.loads(tools.functions_map["reconcile_series"](
        profile_analysis=0, identification_analysis=1))
    assert out.get("image_base64")                         # figure fed to the LLM
    cache = json.loads((tmp_path / "reconciled_series_result.json").read_text())
    assert "image_base64" not in cache                     # cache stays lean


def test_finalize_embeds_interpretation(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    # reconcile first (writes the cached result + skeleton report)
    tools.functions_map["reconcile_series"](profile_analysis=0, identification_analysis=1)
    assert (tmp_path / "reconciled_series_result.json").exists()
    # then finalize with the orchestrator LLM's synthesis
    out = json.loads(tools.functions_map["finalize_reconcile_report"](
        interpretation="PhaseA converts to PhaseB near the mid-series; "
                       "both estimates corroborate the transition."))
    assert out["status"] == "success"
    html = Path(out["report"]).read_text()
    assert "Interpretation" in html
    assert "PhaseA converts to PhaseB" in html                 # narrative embedded
    assert "PhaseA" in html and "PhaseB" in html               # computed labels stay


def test_finalize_without_reconcile_errors(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    out = json.loads(tools.functions_map["finalize_reconcile_report"](
        interpretation="nothing to finalize"))
    assert out["status"] == "error"
    assert "reconcile" in out["message"].lower()


def test_reconcile_errors_gracefully(two_runs, tmp_path):
    mf, idr = two_runs
    tools = _make_tools(tmp_path, mf, idr)
    # swapped: identification dir as the profile-fitting pass -> no fitted peaks
    out = json.loads(tools.functions_map["reconcile_series"](
        profile_analysis=1, identification_analysis=0))
    assert out["status"] == "error"
    assert "peak" in out["message"].lower()
