"""Offline proof for candidate_pool plumbing through run_optimization.

No LLM, no torch: the scalarizer is stubbed to a pass-through fixture
(6-row ingest) and BOAgent.run_optimization_loop is stubbed to capture
its kwargs. Verifies:
  1. no-arg call forwards candidate_pool=None (old behavior);
  2. a valid pool path is resolved + forwarded, and pool provenance from
     the BO result surfaces in the tool response;
  3. a nonexistent pool path returns a clean error before any BO work.
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
    # Ingest 6 rows via the pass-through fixture.
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
    # Capture what reaches the BO layer.
    captured = {}

    def fake_loop(**kw):
        captured.update(kw)
        return {"status": "success", "next_parameters": {"T": 60.0, "t": 25.0},
                "strategy": {}, "candidate_pool":
                    {"provided": 4, "unmeasured": 3}}
    orch.bo.run_optimization_loop = fake_loop
    return orch, captured


with tempfile.TemporaryDirectory() as tmp:
    orch, captured = make_orch(tmp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.execute_tool("run_optimization"))
    check("noarg_forwards_none",
          out["status"] == "success" and captured.get("candidate_pool") is None)

with tempfile.TemporaryDirectory() as tmp:
    orch, captured = make_orch(tmp)
    pool_csv = Path(tmp) / "candidates.csv"
    pool_csv.write_text("T,t\n60,25\n61,26\n62,27\n63,28\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.execute_tool(
            "run_optimization", candidate_pool=str(pool_csv)))
    check("pool_path_forwarded",
          captured.get("candidate_pool") == str(pool_csv))
    check("pool_provenance_in_response",
          out.get("candidate_pool") == {"provided": 4, "unmeasured": 3})

with tempfile.TemporaryDirectory() as tmp:
    orch, captured = make_orch(tmp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.execute_tool(
            "run_optimization", candidate_pool=str(Path(tmp) / "nope.csv")))
    check("bad_pool_path_clean_error",
          out.get("status") == "error" and "candidate_pool" not in captured)

print("=" * 50)
print(f"CANDIDATE_POOL PLUMBING: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
