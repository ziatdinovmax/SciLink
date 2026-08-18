"""analyze_file must not admit rows whose input/target is missing.

Seen live: a 96-well feature table where 4 adaptively re-fitted units
reported the target under a different column name -> 4 NaN targets were
ingested and run_optimization failed with 'Missing values detected' one
call later, with no way to tell which rows. Now the rows are skipped at
ingest with a count + warning in the response, and BO runs on the rest.
"""
import contextlib
import io
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


NAN = float("nan")
L6 = dict(T=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
          t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
          Y=[8.56, NAN, 22.47, 10.81, 48.83, NAN])   # two units w/o target


def make_orch(tmp, metrics):
    data_dir = Path(tmp) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch = PlanningOrchestratorAgent(
            base_dir=str(Path(tmp) / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir))
    orch.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(metrics), "source_script": None,
        "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
        "passthrough": True, "error": None}
    captured = {}

    def fake_loop(**kw):
        captured.update(kw)
        return {"status": "success", "next_parameters": {"T": 60.0, "t": 25.0},
                "strategy": {}}
    orch.bo.run_optimization_loop = fake_loop
    return orch, captured


def ingest(orch, tmp):
    csv = Path(tmp) / "seed.csv"
    csv.write_text("T,t,Y\n" + "\n".join(
        f"{a},{b},{'' if math.isnan(c) else c}"
        for a, b, c in zip(L6['T'], L6['t'], L6['Y'])) + "\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return json.loads(orch.tools.execute_tool(
            "analyze_file", file_path=str(csv), extraction_goal="fixture",
            inputs=["T", "t"], targets=["Y"])), buf.getvalue()


with tempfile.TemporaryDirectory() as tmp:
    orch, captured = make_orch(tmp, L6)
    out, log = ingest(orch, tmp)
    check("partial_rows_skipped",
          out["status"] == "success" and out["rows_added"] == 4
          and out.get("rows_skipped_missing") == 2
          and "Y" in out.get("warning", ""), str(out))
    import pandas as pd
    df = pd.read_csv(Path(tmp) / "session" / "optimization_data.csv")
    check("dataset_has_no_nans", len(df) == 4 and not df.isna().any().any())
    with contextlib.redirect_stdout(io.StringIO()):
        bo = json.loads(orch.tools.execute_tool("run_optimization"))
    check("bo_runs_on_remaining_rows", bo.get("status") == "success", str(bo)[:100])

with tempfile.TemporaryDirectory() as tmp:
    all_nan = dict(L6, Y=[NAN] * 6)
    orch, _ = make_orch(tmp, all_nan)
    out, _ = ingest(orch, tmp)
    check("all_rows_missing_is_clean_error",
          out["status"] == "error" and "missing values" in out["message"],
          str(out)[:120])
    check("nothing_written",
          not (Path(tmp) / "session" / "optimization_data.csv").exists())

with tempfile.TemporaryDirectory() as tmp:
    clean = dict(L6, Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])
    orch, _ = make_orch(tmp, clean)
    out, _ = ingest(orch, tmp)
    check("clean_table_unchanged",
          out["rows_added"] == 6 and "rows_skipped_missing" not in out
          and "warning" not in out)

print("=" * 50)
print(f"INGEST MISSING VALUES: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
