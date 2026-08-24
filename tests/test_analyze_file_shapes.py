"""Proof harness: every metrics shape through the REAL analyze_file tool.

Monkeypatches scalarizer.scalarize with fixture results (no LLM, no net).
For each case asserts the resulting optimization_data.csv row count and
dtypes — i.e., exactly what changed and what stayed identical after the
pass-through expansion fix in orchestrator_tools.py.
"""
import io
import contextlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def fresh_orch(tmp):
    data_dir = Path(tmp) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch = PlanningOrchestratorAgent(
            base_dir=str(Path(tmp) / "session"),
            api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            data_dir=str(data_dir),
        )
    return orch


CSV_6ROW = (
    "Temperature_C,Time_min,Yield_pct\n"
    "30.0,10.0,8.56\n30.0,50.0,3.88\n90.0,10.0,22.47\n"
    "90.0,50.0,10.81\n50.0,35.0,48.83\n78.0,18.0,61.1\n"
)

L6 = dict(T=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
          t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
          Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])


def run_case(name, metrics, passthrough, inputs, targets,
             source_script=None, expect_rows=None, expect_numeric=True,
             expect_error=False):
    with tempfile.TemporaryDirectory() as tmp:
        orch = fresh_orch(tmp)
        csv = Path(tmp) / "seed.csv"
        csv.write_text(CSV_6ROW)
        res_fixture = {
            "status": "success", "metrics": metrics,
            "source_script": source_script,
            "column_roles": {"inputs": inputs, "targets": targets},
            "error": None,
        }
        if passthrough:
            res_fixture["passthrough"] = True
        orch.scalarizer.scalarize = lambda **kw: dict(res_fixture)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = orch.tools.execute_tool(
                "analyze_file", file_path=str(csv),
                extraction_goal="offline fixture",
                inputs=inputs, targets=targets)
        out_d = json.loads(out)

        if expect_error:
            check(name, out_d.get("status") == "error", f"status={out_d.get('status')}")
            return

        ok = out_d.get("status") in ("success", "warning")
        df = pd.read_csv(orch.bo_data_path) if orch.bo_data_path.exists() else None
        rows = len(df) if df is not None else 0
        numeric = (df is not None and
                   all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns))
        detail = f"(status={out_d.get('status')}, rows={rows}, numeric={numeric})"
        cond = ok and rows == expect_rows and (numeric == expect_numeric)
        check(name, cond, detail)
        # Evaluate preflight facts INSIDE the tempdir's lifetime.
        preflight_ok = (orch.bo_data_path.exists() and rows >= 3)
        return orch, df, preflight_ok


print("1) passthrough multi-row dict-of-lists -> N rows (THE FIX)")
r = run_case("pt_multirow_expands", dict(L6), True,
             ["T", "t"], ["Y"], expect_rows=6)
if r:
    orch, df, preflight_ok = r
    check("pt_multirow_schema_set",
          orch.expected_input_columns == ["T", "t"]
          and orch.expected_target_columns == ["Y"])
    check("pt_multirow_preflight_ok",
          preflight_ok and not orch.active_scalarizer_script)

print("2) passthrough single-row scalars -> 1 row (unchanged)")
run_case("pt_single_row", {"T": 30.0, "t": 10.0, "Y": 8.56}, True,
         ["T", "t"], ["Y"], expect_rows=1)

print("3) codegen dict of scalars, NO passthrough flag -> 1 row (unchanged)")
run_case("codegen_scalar_dict", {"T": 30.0, "t": 10.0, "Y": 8.56}, False,
         ["T", "t"], ["Y"], source_script="/tmp/dummy_script.py",
         expect_rows=1)

print("4) dict-of-lists WITHOUT passthrough flag -> 1 stringified row (old behavior preserved)")
run_case("nonpt_dict_of_lists_unchanged", dict(L6), False,
         ["T", "t"], ["Y"], expect_rows=1, expect_numeric=False)

print("5) list-of-dicts (multi-well) -> N rows (existing branch untouched)")
run_case("list_of_dicts_multiwell",
         [dict(T=a, t=b, Y=c) for a, b, c in zip(L6["T"], L6["t"], L6["Y"])],
         False, ["T", "t"], ["Y"], expect_rows=6)

print("6) passthrough multi-row + scalar sidecar value -> broadcast to N rows")
r = run_case("pt_scalar_broadcast", {**L6, "pH": 7.0}, True,
             ["T", "t", "pH"], ["Y"], expect_rows=6)
if r:
    _, df, _ = r
    check("pt_broadcast_constant", (df["pH"] == 7.0).all())

print("=" * 50)
print(f"ANALYZE_FILE SHAPES: {len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
