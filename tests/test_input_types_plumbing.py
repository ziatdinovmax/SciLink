"""Explicit input types through the planning tools + candidate-pool encoding.

A pass-through feature table never runs the scalarizer's input-type
classification, so a numeric-coded categorical input was modelled as a
continuous knob and there was no way to say otherwise. analyze_file /
run_optimization take input_types={col: categorical|continuous} (sticky,
checkpointed); categorical inputs are level-encoded for the surrogate,
the candidate pool is encoded through the SAME maps, and recommendations
are decoded back.
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 'L' is a categorical stored as numeric codes 0/1/2/10 (string sort != numeric sort on purpose)
ROWS = dict(L=[0.0, 1.0, 2.0, 10.0, 1.0, 2.0], t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
            Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])


def make(tmp):
    d = Path(tmp) / "data"; d.mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        o = PlanningOrchestratorAgent(base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
                                      autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(d))
    o.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(ROWS), "source_script": None,
        "column_roles": {"inputs": ["L", "t"], "targets": ["Y"]}, "passthrough": True, "error": None}
    cap = {}
    def loop(**kw):
        cap.clear(); cap.update(kw)
        return {"status": "success", "next_parameters": {"L": 3.0, "t": 25.0}, "strategy": {}}  # encoded index 3
    o.bo.run_optimization_loop = loop
    csv = Path(tmp) / "seed.csv"
    csv.write_text("L,t,Y\n" + "\n".join(f"{a},{b},{c}" for a, b, c in zip(ROWS["L"], ROWS["t"], ROWS["Y"])) + "\n")
    pool = Path(tmp) / "pool.csv"
    pool.write_text("L,t\n0.0,12\n1.0,20\n2.0,30\n10.0,40\n")
    return o, cap, csv, pool


def call(o, tool, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return json.loads(o.tools.execute_tool(tool, **kw))


with tempfile.TemporaryDirectory() as tmp:
    o, cap, csv, pool = make(tmp)
    a = call(o, "analyze_file", file_path=str(csv), extraction_goal="x", inputs=["L", "t"], targets=["Y"])
    r = call(o, "run_optimization", candidate_pool=str(pool))
    check("default_is_continuous", cap.get("cat_dims") is None and r["input_types"] == {"L": "continuous", "t": "continuous"}
          and cap.get("candidate_pool") == str(pool), str(r.get("input_types")))

    r = call(o, "run_optimization", candidate_pool=str(pool), input_types={"L": "categorical"})
    check("categorical_declared_gives_cat_dims", cap.get("cat_dims") == [0] and r["input_types"]["L"] == "categorical")
    enc_pool = cap.get("candidate_pool")
    pdf = pd.read_csv(enc_pool)
    levels = sorted(["0", "1", "2", "10"])          # normalised labels, string order: 0, 1, 10, 2
    check("pool_encoded_through_same_level_map",
          enc_pool != str(pool) and pdf["L"].tolist() == [levels.index("0"), levels.index("1"),
                                                            levels.index("2"), levels.index("10")],
          f"{pdf['L'].tolist()} levels={levels}")
    # the encoded data must use the same indices
    ddf = pd.read_csv(cap["data_path"])
    check("data_encoded_with_same_map", ddf["L"].tolist()[:4] == [levels.index(v) for v in ("0", "1", "2", "10")])
    # recommendation decoded back to the original label (index 3 -> '2' in string order)
    rec = r["recommended_parameters"]
    check("recommendation_decoded_to_label", str(rec["L"]) == levels[3], str(rec))

    r = call(o, "run_optimization", candidate_pool=str(pool))
    check("sticky_on_next_call", cap.get("cat_dims") == [0])

    # a pool level the data has not measured yet is legitimate: it joins the universe
    wide = Path(tmp) / "widepool.csv"; wide.write_text("L,t\n0,12\n7,20\n")
    r = call(o, "run_optimization", candidate_pool=str(wide))
    wdf = pd.read_csv(cap["candidate_pool"])
    check("unmeasured_pool_level_joins_universe",
          r.get("status") == "success" and sorted(wdf["L"].tolist()) == [0.0, 4.0],   # '7' sorts last of 0,1,10,2,7
          f"{r.get('status')} {wdf['L'].tolist()}")

    r = call(o, "run_optimization", candidate_pool=str(pool), input_types={"t": "sideways", "zz": "categorical"})
    check("bad_entries_warned", len([w for w in r.get("warnings", []) if "input_types" in w]) == 2, str(r.get("warnings")))

    with contextlib.redirect_stdout(io.StringIO()):
        o._auto_checkpoint()
        o2 = PlanningOrchestratorAgent(base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
                                       autonomy_level=AutonomyLevel.AUTONOMOUS,
                                       data_dir=str(Path(tmp) / "data"), restore_checkpoint=True)
    check("types_survive_checkpoint", (o2.expected_input_types or {}).get("L") == "categorical")

with tempfile.TemporaryDirectory() as tmp:
    o, cap, csv, pool = make(tmp)
    a = call(o, "analyze_file", file_path=str(csv), extraction_goal="x", inputs=["L", "t"], targets=["Y"],
             input_types={"L": "categorical"})
    r = call(o, "run_optimization")
    check("ingest_types_reach_bo", a["input_types"] == {"L": "categorical"} and cap.get("cat_dims") == [0])

print("=" * 50)
print(f"INPUT TYPES: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
