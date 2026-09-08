"""#535: a continuation on an established campaign repeats the prior
ingestion decision. When analyze_file is called WITHOUT inputs/targets and
the campaign schema is already established (and no script is locked — the
first ingestion was a table pass-through), the call arms the scalarizer's
explicit pass-through (`_schema_requirements` / `column_role_hints`) from the
campaign schema instead of dropping to codegen rediscovery.

Also covers the #534 planning-side amplifier: a row skipped for an empty
target is reported by UNIT NAME.

Drives the REAL analyze_file tool; scalarize() is stubbed to capture its
kwargs and return fixture results (no LLM, no network)."""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import pytest

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)

SEED = ("unit,T,t,Y\n"
        "s1,30.0,10.0,8.56\ns2,30.0,50.0,3.88\ns3,90.0,10.0,22.47\n"
        "s4,90.0,50.0,10.81\ns5,50.0,35.0,48.83\ns6,78.0,18.0,61.1\n")
SEED_METRICS = {"T": [30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
                "t": [10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
                "Y": [8.56, 3.88, 22.47, 10.81, 48.83, 61.1]}
NEXT = "unit,T,t,Y\ns7,65.0,25.0,70.2\n"


@pytest.fixture
def orch(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        o = PlanningOrchestratorAgent(
            base_dir=str(tmp_path / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir))
    o._calls = []

    def fake_scalarize(**kw):
        o._calls.append(kw)
        fx = o._fixture
        return dict(fx) if callable(fx) is False else fx(kw)
    o.scalarizer.scalarize = fake_scalarize
    o._data_dir_path = data_dir
    return o


def _passthrough(metrics, inputs, targets):
    return {"status": "success", "metrics": metrics, "source_script": None,
            "column_roles": {"inputs": inputs, "targets": targets},
            "passthrough": True, "error": None}


def _call(o, path, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = o.tools.execute_tool("analyze_file", file_path=str(path),
                                   extraction_goal="offline fixture", **kw)
    return json.loads(out), buf.getvalue()


def _seed_campaign(o):
    seed = o._data_dir_path / "seed.csv"
    seed.write_text(SEED)
    o._fixture = _passthrough(dict(SEED_METRICS), ["T", "t"], ["Y"])
    res, _ = _call(o, seed, inputs=["T", "t"], targets=["Y"])
    assert res["status"] == "success" and res["rows_added"] == 6
    assert o.expected_input_columns == ["T", "t"]
    assert o.expected_target_columns == ["Y"]
    assert not o.active_scalarizer_script  # pass-through locks nothing
    return seed


def test_first_ingestion_without_schema_arms_nothing(orch):
    """Unchanged: no campaign schema yet -> discovery, no hints."""
    seed = orch._data_dir_path / "seed.csv"
    seed.write_text(SEED)
    orch._fixture = _passthrough(dict(SEED_METRICS), ["T", "t"], ["Y"])
    _call(orch, seed)
    kw = orch._calls[-1]
    assert kw["column_role_hints"] is None
    assert "_schema_requirements" not in (kw["experiment_context"] or {})


def test_continuation_arms_passthrough_from_campaign_schema(orch):
    _seed_campaign(orch)
    nxt = orch._data_dir_path / "next.csv"
    nxt.write_text(NEXT)
    orch._fixture = _passthrough({"T": 65.0, "t": 25.0, "Y": 70.2}, ["T", "t"], ["Y"])
    res, log = _call(orch, nxt)          # NO inputs / targets
    kw = orch._calls[-1]
    assert kw["reuse_script_path"] is None
    assert kw["column_role_hints"] == {"inputs": ["T", "t"], "targets": ["Y"]}
    assert kw["experiment_context"]["_schema_requirements"] == {
        "input_columns": ["T", "t"], "target_columns": ["Y"],
        "optimization_type": "single-objective"}
    assert "Continuation: schema armed from the campaign" in log
    assert res["status"] == "success" and res["rows_added"] == 1
    assert res["data_points_collected"] == 7
    assert orch.expected_input_columns == ["T", "t"]


def test_continuation_sidecar_inputs_not_demanded_from_table(orch):
    _seed_campaign(orch)
    nxt = orch._data_dir_path / "next.csv"
    nxt.write_text("unit,t,Y\ns7,25.0,70.2\n")
    nxt.with_suffix(".json").write_text(json.dumps({"T": 65.0, "note": [1]}))
    orch._fixture = _passthrough({"t": 25.0, "Y": 70.2}, ["t"], ["Y"])
    res, _ = _call(orch, nxt)
    kw = orch._calls[-1]
    assert kw["column_role_hints"] == {"inputs": ["t"], "targets": ["Y"]}
    assert kw["experiment_context"]["_schema_requirements"]["input_columns"] == ["t"]
    assert res["status"] == "success" and res["rows_added"] == 1
    df = pd.read_csv(orch.bo_data_path)
    assert float(df.iloc[-1]["T"]) == 65.0   # merged from the sidecar


def test_continuation_survives_checkpoint_restore(orch, tmp_path):
    """The armed schema is the checkpointed one, so a restored session
    continues deterministically too."""
    _seed_campaign(orch)
    with contextlib.redirect_stdout(io.StringIO()):
        orch.tools.execute_tool("save_checkpoint")
        restored = PlanningOrchestratorAgent(
            base_dir=str(tmp_path / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            data_dir=str(orch._data_dir_path), restore_checkpoint=True)
    assert restored.expected_input_columns == ["T", "t"]
    calls = []
    restored.scalarizer.scalarize = lambda **kw: (
        calls.append(kw) or _passthrough({"T": 65.0, "t": 25.0, "Y": 70.2},
                                         ["T", "t"], ["Y"]))
    nxt = orch._data_dir_path / "next.csv"
    nxt.write_text(NEXT)
    with contextlib.redirect_stdout(io.StringIO()):
        out = json.loads(restored.tools.execute_tool(
            "analyze_file", file_path=str(nxt), extraction_goal="x"))
    assert out["status"] == "success"
    assert calls[-1]["column_role_hints"] == {"inputs": ["T", "t"], "targets": ["Y"]}


def test_locked_script_still_wins_over_campaign_arming(orch):
    seed = orch._data_dir_path / "seed.csv"
    seed.write_text(SEED)
    script = orch._data_dir_path / "proc_seed.py"
    script.write_text("# fixture\n")
    orch._fixture = {**_passthrough(dict(SEED_METRICS), ["T", "t"], ["Y"]),
                     "passthrough": False, "source_script": str(script)}
    # Codegen-shaped fixture: list-of-dicts rows.
    orch._fixture["metrics"] = [dict(T=a, t=b, Y=c) for a, b, c in
                                zip(SEED_METRICS["T"], SEED_METRICS["t"], SEED_METRICS["Y"])]
    res, _ = _call(orch, seed, inputs=["T", "t"], targets=["Y"])
    assert res["status"] == "success"
    assert orch.active_scalarizer_script == str(script)
    nxt = orch._data_dir_path / "next.csv"
    nxt.write_text(NEXT)
    orch._fixture["metrics"] = [dict(T=65.0, t=25.0, Y=70.2)]
    res, log = _call(orch, nxt)
    kw = orch._calls[-1]
    assert kw["reuse_script_path"] == str(script)
    assert kw["column_role_hints"] is None       # Consistency Mode, untouched
    assert "Continuation: schema armed" not in log


def test_force_regenerate_with_explicit_schema_arms_it(orch):
    """A locked script + force_regenerate=True generates a NEW script, which
    must see the caller's explicit schema (previously suppressed by the
    locked script's mere existence)."""
    seed = orch._data_dir_path / "seed.csv"
    seed.write_text(SEED)
    script = orch._data_dir_path / "proc_seed.py"
    script.write_text("# fixture\n")
    rows = [dict(T=a, t=b, Y=c) for a, b, c in
            zip(SEED_METRICS["T"], SEED_METRICS["t"], SEED_METRICS["Y"])]
    orch._fixture = {"status": "success", "metrics": rows,
                     "source_script": str(script),
                     "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
                     "error": None}
    _call(orch, seed, inputs=["T", "t"], targets=["Y"])
    _call(orch, seed, inputs=["T", "t"], targets=["Y"], force_regenerate=True)
    kw = orch._calls[-1]
    assert kw["reuse_script_path"] is None
    assert kw["column_role_hints"] == {"inputs": ["T", "t"], "targets": ["Y"]}


def test_skipped_rows_are_named_by_unit_from_passthrough_units(orch):
    """#534 amplifier: a unit whose target cell is empty is excluded from the
    optimization data — the response must say WHICH unit. The real
    pass-through returns only the requested columns and carries the row
    identities as ``units``."""
    seed = orch._data_dir_path / "seed.csv"
    seed.write_text(SEED)
    metrics = dict(SEED_METRICS)
    metrics["Y"] = [8.56, 3.88, None, 10.81, 48.83, 61.1]
    orch._fixture = {**_passthrough(metrics, ["T", "t"], ["Y"]),
                     "units": ["s1", "s2", "s3", "s4", "s5", "s6"]}
    res, _ = _call(orch, seed, inputs=["T", "t"], targets=["Y"])
    assert res["status"] == "success"
    assert res["rows_added"] == 5
    assert res["rows_skipped_missing"] == 1
    assert res["rows_skipped_units"] == ["s3"]
    assert "s3" in res["warning"] and "optimizer will NOT see" in res["warning"]
    assert "unit" not in pd.read_csv(orch.bo_data_path).columns


def test_skipped_rows_named_from_a_unit_column_after_dedup_slice(orch):
    """A codegen script that emits 'unit' rows; when the same file (same
    hash) yields more rows than before, only the new rows are appended, and
    the labels must follow that slice."""
    f = orch._data_dir_path / "grow.csv"
    f.write_text(SEED)
    rows = [dict(unit=f"s{i+1}", T=a, t=b, Y=c) for i, (a, b, c) in
            enumerate(zip(SEED_METRICS["T"], SEED_METRICS["t"], SEED_METRICS["Y"]))]
    orch._fixture = {"status": "success", "metrics": rows[:3],
                     "source_script": None, "column_roles": {}, "error": None}
    res, _ = _call(orch, f, inputs=["T", "t"], targets=["Y"])
    assert res["rows_added"] == 3
    more = rows + [dict(unit="s7", T=1.0, t=1.0, Y=None)]
    more[4]["Y"] = None
    orch._fixture = {"status": "success", "metrics": more,
                     "source_script": None, "column_roles": {}, "error": None}
    res, _ = _call(orch, f)
    assert res["rows_added"] == 2                      # s4, s6
    assert res["rows_skipped_units"] == ["s5", "s7"]
