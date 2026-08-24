"""A planning orchestrator started on a session dir that holds campaign
data from an earlier process — but no checkpoint and no restore request —
must not enter the half-state where the on-disk dedup ledger says the file
is "already analyzed" while no schema exists in memory (run_optimization
then fails with "Schema not established"). It archives the stale files and
starts a consistent, empty campaign; a checkpointed dir is left alone.
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


def make_orch(tmp, restore=False):
    data_dir = Path(tmp) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch = PlanningOrchestratorAgent(
            base_dir=str(Path(tmp) / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir),
            restore_checkpoint=restore)
    orch.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(L6), "source_script": None,
        "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
        "passthrough": True, "error": None}
    orch.bo.run_optimization_loop = lambda **kw: {
        "status": "success", "next_parameters": {"T": 60.0, "t": 25.0},
        "strategy": {}}
    return orch, buf


def ingest(orch, csv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return json.loads(orch.tools.execute_tool(
            "analyze_file", file_path=str(csv), extraction_goal="fixture",
            inputs=["T", "t"], targets=["Y"]))


with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / "seed.csv"
    csv.write_text(CSV_6ROW)
    # Process 1: a campaign that never checkpoints (autonomous MCP-style).
    orch1, _ = make_orch(tmp)
    r1 = ingest(orch1, csv)
    check("p1_ingest", r1.get("rows_added") == 6, str(r1)[:100])
    session = Path(tmp) / "session"
    check("p1_left_data_no_checkpoint",
          (session / "optimization_data.csv").exists()
          and (session / "analyzed_files.json").exists()
          and not (session / "checkpoint.json").exists())

    # Process 2: same session dir, no restore -> must start clean.
    orch2, buf2 = make_orch(tmp)
    archives = list(session.glob("stale_campaign_*"))
    check("stale_files_archived",
          len(archives) == 1
          and (archives[0] / "optimization_data.csv").exists()
          and not (session / "optimization_data.csv").exists()
          and orch2.analyzed_files == {},
          str([p.name for p in session.iterdir()]))
    check("archive_notice_printed", "stale_campaign_" in buf2.getvalue())
    r2 = ingest(orch2, csv)
    check("p2_reingest_not_blocked_by_stale_dedup",
          r2.get("status") == "success" and r2.get("rows_added") == 6,
          str(r2)[:120])
    with contextlib.redirect_stdout(io.StringIO()):
        bo = json.loads(orch2.tools.execute_tool("run_optimization"))
    check("p2_schema_established", bo.get("status") == "success", str(bo)[:100])

with tempfile.TemporaryDirectory() as tmp:
    # A checkpointed dir is NOT archived when restore is requested...
    csv = Path(tmp) / "seed.csv"
    csv.write_text(CSV_6ROW)
    orch1, _ = make_orch(tmp)
    ingest(orch1, csv)
    with contextlib.redirect_stdout(io.StringIO()):
        orch1._auto_checkpoint()
    session = Path(tmp) / "session"
    orch3, _ = make_orch(tmp, restore=True)
    check("checkpointed_dir_restored_not_archived",
          not list(session.glob("stale_campaign_*"))
          and orch3.expected_input_columns == ["T", "t"]
          and (session / "optimization_data.csv").exists())
    # ...nor when a checkpoint exists but restore was not requested
    # (recoverable state; existing behavior kept).
    orch4, _ = make_orch(tmp)
    check("checkpoint_present_no_restore_untouched",
          not list(session.glob("stale_campaign_*")))

with tempfile.TemporaryDirectory() as tmp:
    orch, buf = make_orch(tmp)
    check("fresh_dir_no_archive_no_notice",
          not list((Path(tmp) / "session").glob("stale_campaign_*"))
          and "stale_campaign_" not in buf.getvalue())

print("=" * 50)
print(f"STALE SESSION DIR: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
