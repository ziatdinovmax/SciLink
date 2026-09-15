"""Explicit optimization directions through the planning tools.

A pass-through feature table never runs the scalarizer's classification, so
target_directions stayed empty and the optimizer silently MAXIMIZED — a
'loss' target was maximized in an MCP-driven campaign. analyze_file and
run_optimization now take directions={col: maximize|minimize} (sticky,
checkpointed); both responses report the directions in force and warn when
one was assumed.
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


def make(tmp):
    d = Path(tmp) / "data"; d.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        o = PlanningOrchestratorAgent(base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
                                      autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(d))
    o.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(L6), "source_script": None,
        "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},   # no optimization_direction
        "passthrough": True, "error": None}
    cap = {}
    def loop(**kw):
        cap.clear(); cap.update(kw)
        return {"status": "success", "next_parameters": {"T": 60.0, "t": 25.0}, "strategy": {}}
    o.bo.run_optimization_loop = loop
    csv = Path(tmp) / "seed.csv"
    csv.write_text("T,t,Y\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(L6["T"], L6["t"], L6["Y"])) + "\n")
    return o, cap, csv


def call(o, tool, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return json.loads(o.tools.execute_tool(tool, **kw))


with tempfile.TemporaryDirectory() as tmp:
    o, cap, csv = make(tmp)
    a = call(o, "analyze_file", file_path=str(csv), extraction_goal="x", inputs=["T", "t"], targets=["Y"])
    check("ingest_without_direction_warns", "direction_warning" in a and a["target_directions"] == {}, str(a)[:160])
    r = call(o, "run_optimization")
    check("bo_assumes_maximize_and_says_so",
          cap.get("target_directions") in ({}, None) and r["target_directions"] == {"Y": "maximize (assumed)"}
          and any("MAXIMIZE was assumed" in w for w in r.get("warnings", [])), str(r)[:200])
    r = call(o, "run_optimization", directions={"Y": "minimize"})
    check("bo_directions_forwarded", cap.get("target_directions") == {"Y": "minimize"}
          and r["target_directions"] == {"Y": "minimize"} and not r.get("warnings"))
    r = call(o, "run_optimization")
    check("sticky_on_next_call", cap.get("target_directions") == {"Y": "minimize"})
    r = call(o, "run_optimization", directions={"Y": "MAX", "nope": "minimize", "T": "sideways"})
    check("aliases_and_bad_entries",
          cap.get("target_directions") == {"Y": "maximize"} and len(r.get("warnings", [])) == 2, str(r.get("warnings")))
    with contextlib.redirect_stdout(io.StringIO()):
        o._auto_checkpoint()
        o2 = PlanningOrchestratorAgent(base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
                                       autonomy_level=AutonomyLevel.AUTONOMOUS,
                                       data_dir=str(Path(tmp) / "data"), restore_checkpoint=True)
    check("directions_survive_checkpoint", o2.target_directions == {"Y": "maximize"}, str(o2.target_directions))

with tempfile.TemporaryDirectory() as tmp:
    o, cap, csv = make(tmp)
    a = call(o, "analyze_file", file_path=str(csv), extraction_goal="x", inputs=["T", "t"],
             targets=["Y"], directions={"Y": "minimize"})
    check("ingest_directions_captured", a["target_directions"] == {"Y": "minimize"} and "direction_warning" not in a)
    r = call(o, "run_optimization")
    check("ingest_directions_reach_bo", cap.get("target_directions") == {"Y": "minimize"})

print("=" * 50)
print(f"TARGET DIRECTIONS: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
