"""Offline proof for the `input_bounds` argument on run_optimization.

No LLM, no torch: scalarizer stubbed to a 6-row pass-through, BOAgent's
run_optimization_loop stubbed to capture its kwargs. Verifies:
  1. no-arg call keeps the data-derived box (observed range +/-10%) and
     reports it, with source "data" per column;
  2. caller bounds win over the data-derived box for the columns given,
     other columns keep theirs, and the response reports source "caller";
  3. sticky: a later no-arg call still uses the caller bounds;
  4. malformed / unknown-column entries are ignored with a warning and
     never reach the BO layer;
  5. the override survives a checkpoint save/restore.
"""
import contextlib
import io
import json
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


L6 = dict(T=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
          t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
          Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])
CSV_6ROW = "T,t,Y\n" + "\n".join(
    f"{a},{b},{c}" for a, b, c in zip(L6["T"], L6["t"], L6["Y"])) + "\n"


def make_orch(tmp):
    data_dir = Path(tmp) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch = PlanningOrchestratorAgent(
            base_dir=str(Path(tmp) / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir))
    orch.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(L6), "source_script": None,
        "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
        "passthrough": True, "error": None}
    csv = Path(tmp) / "seed.csv"
    csv.write_text(CSV_6ROW)
    with contextlib.redirect_stdout(buf):
        orch.tools.execute_tool("analyze_file", file_path=str(csv),
                                extraction_goal="fixture",
                                inputs=["T", "t"], targets=["Y"])
    captured = {}

    def fake_loop(**kw):
        captured.clear()
        captured.update(kw)
        return {"status": "success", "next_parameters": {"T": 60.0, "t": 25.0},
                "strategy": {}}
    orch.bo.run_optimization_loop = fake_loop
    return orch, captured


def run(orch, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return json.loads(orch.tools.execute_tool("run_optimization", **kw))


with tempfile.TemporaryDirectory() as tmp:
    orch, captured = make_orch(tmp)
    out = run(orch)
    # data range T 30-90 -> +/-6 ; t 10-50 -> +/-4
    check("data_bounds_default",
          captured.get("input_bounds") == [[24.0, 96.0], [6.0, 54.0]],
          str(captured.get("input_bounds")))
    check("data_source_reported",
          out.get("input_bounds_source") == {"T": "data", "t": "data"}
          and out.get("input_bounds") == {"T": [24.0, 96.0], "t": [6.0, 54.0]})

    out = run(orch, input_bounds={"t": [10, 50]})
    check("caller_bounds_win_for_given_column",
          captured.get("input_bounds") == [[24.0, 96.0], [10.0, 50.0]],
          str(captured.get("input_bounds")))
    check("caller_source_reported",
          out.get("input_bounds_source") == {"T": "data", "t": "caller"})

    out = run(orch)
    check("sticky_across_calls",
          captured.get("input_bounds") == [[24.0, 96.0], [10.0, 50.0]])

    out = run(orch, input_bounds={"T": [40, 80]})
    check("later_call_adds_without_dropping",
          captured.get("input_bounds") == [[40.0, 80.0], [10.0, 50.0]])

    out = run(orch, input_bounds={"t": [50, 10], "nope": [0, 1], "T": "bad"})
    check("malformed_and_unknown_ignored_with_warning",
          captured.get("input_bounds") == [[40.0, 80.0], [10.0, 50.0]]
          and len(out.get("input_bounds_warnings", [])) == 3,
          str(out.get("input_bounds_warnings")))

    # Checkpoint round-trip.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._auto_checkpoint()
        restored = PlanningOrchestratorAgent(
            base_dir=str(Path(tmp) / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            data_dir=str(Path(tmp) / "data"), restore_checkpoint=True)
    check("override_survives_checkpoint",
          restored.input_bounds_override == {"T": (40.0, 80.0), "t": (10.0, 50.0)},
          str(restored.input_bounds_override))

print("=" * 50)
print(f"INPUT_BOUNDS PLUMBING: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
