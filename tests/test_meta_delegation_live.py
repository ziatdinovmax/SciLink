"""Live delegation test for the meta-agent (ad-hoc; not part of the suite).

Runs a real MetaOrchestratorAgent in AUTONOMOUS mode against claude-opus-4-6
and exercises two planning delegations (the Part 0 non-overwrite regression
guard) and one analysis delegation.

Keys are read from the env or untracked sibling files so the run command
carries no secret:
  - LLM key:       ANTHROPIC_API_KEY  or  tests/.meta_live_key
  - embedding key: GEMINI_API_KEY/GOOGLE_API_KEY  or  tests/.meta_live_embed_key

Run:            python tests/test_meta_delegation_live.py
Planning-only:  python tests/test_meta_delegation_live.py planning-only
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from scilink.agents.meta_agent.meta_orchestrator import (
    MetaOrchestratorAgent, MetaMode,
)

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "tests" / "_meta_live_run"
HERE = Path(__file__).resolve().parent
DATA = REPO.parent / "scilink_examples_data_backup"
GO_TIF = DATA / "GO_cafm.tif"
GO_JSON = DATA / "GO_cafm.json"


def banner(msg):
    print("\n" + "=" * 72, flush=True)
    print(msg, flush=True)
    print("=" * 72, flush=True)


def _resolve(env_vars, keyfile_name):
    """Resolve a credential from the first set env var, else a sibling file."""
    for var in env_vars:
        v = os.environ.get(var)
        if v:
            return v.strip()
    keyfile = HERE / keyfile_name
    if keyfile.exists():
        return keyfile.read_text().strip()
    return None


def main():
    planning_only = "planning-only" in sys.argv[1:]

    api_key = _resolve(["ANTHROPIC_API_KEY"], ".meta_live_key")
    if not api_key:
        print("FAIL: no LLM key (set ANTHROPIC_API_KEY or write tests/.meta_live_key)")
        return 1
    # Embedding key — planning's generate_plan embeds via gemini-embedding-001.
    embed_key = _resolve(["GEMINI_API_KEY", "GOOGLE_API_KEY"], ".meta_live_embed_key")
    if not embed_key:
        print("WARNING: no embedding key — planning generate_plan may fail", flush=True)

    if BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True)

    # The child orchestrators generate and execute Python code. Code execution
    # is gated by SciLink's sandbox consent; this run is authorized via the
    # documented UNSAFE_EXECUTION_OK=true environment override (set on the
    # command line), which executors.require_sandbox_or_approval() honors.
    banner("Building MetaOrchestratorAgent (AUTONOMOUS, claude-opus-4-6)")
    agent = MetaOrchestratorAgent(
        base_dir=str(BASE),
        api_key=api_key,
        model_name="claude-opus-4-6",
        embedding_api_key=embed_key,
        meta_mode=MetaMode.AUTONOMOUS,
    )
    print(f"tools: {sorted(agent.tools.functions_map)}", flush=True)

    tasks = [
        ("planning #1",
         "Delegate to the planning specialist with this task: produce ONLY an "
         "initial experimental campaign plan to maximize the tensile strength "
         "of a 3D-printed PLA part by varying nozzle temperature and infill "
         "density. Do not search literature, do not run techno-economic "
         "analysis, do not iterate — just the initial plan. Then report the "
         "specialist's result back to me."),
        ("planning #2",
         "Now delegate to the planning specialist again with a different task: "
         "produce ONLY an initial experimental campaign plan to minimize the "
         "surface roughness of an electroplated copper film by tuning current "
         "density and bath temperature. No literature search, no TEA, no "
         "iteration. Report the result back."),
        ("analysis #1",
         f"Delegate to the analysis specialist: examine and analyze the "
         f"microscopy image at {GO_TIF}, using the metadata file at {GO_JSON}. "
         f"Report what the analysis found."),
    ]
    if planning_only:
        tasks = tasks[:2]
    n_tasks = len(tasks)

    timings = []
    for label, task in tasks:
        banner(f"TASK: {label}")
        t0 = time.time()
        try:
            reply = agent.chat(task)
            dt = time.time() - t0
            print(f"\n[{label}] done in {dt:.0f}s\n--- meta reply ---\n{reply}\n",
                  flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f"\n[{label}] RAISED after {dt:.0f}s: {e}", flush=True)
        timings.append((label, dt))

    # ── Assertions ──────────────────────────────────────────────────
    banner("ASSERTIONS")
    checks = []

    analysis_dir = BASE / "analysis"
    planning_dir = BASE / "planning"
    ledger = agent._delegation_ledger
    ran_analysis = any(e["mode"] == "analysis" for e in ledger)

    checks.append(("planning child dir exists", planning_dir.is_dir()))

    deleg_dirs = sorted((planning_dir / "delegations").glob("*")) \
        if (planning_dir / "delegations").is_dir() else []
    plan_jsons = sorted((planning_dir / "delegations").glob("*/plan.json")) \
        if (planning_dir / "delegations").is_dir() else []
    checks.append(("planning made >=2 per-delegation subdirs (R1 fix)",
                   len(deleg_dirs) >= 2))
    checks.append(("each planning delegation kept its own plan.json (no overwrite)",
                   len(plan_jsons) >= 2))
    print(f"  planning delegation subdirs: {[d.name for d in deleg_dirs]}", flush=True)
    print(f"  plan.json files: {[str(p.relative_to(BASE)) for p in plan_jsons]}",
          flush=True)

    if ran_analysis:
        checks.append(("analysis child dir exists", analysis_dir.is_dir()))
        results_dirs = sorted((analysis_dir / "results").glob("*")) \
            if (analysis_dir / "results").is_dir() else []
        checks.append(("analysis produced a results/<id> dir", len(results_dirs) >= 1))
        print(f"  analysis results dirs: {[d.name for d in results_dirs]}", flush=True)

    checks.append((f"delegation ledger has {n_tasks} entries", len(ledger) == n_tasks))
    for e in ledger:
        print(f"  ledger[{e['index']}] {e['mode']} status={e['status']}", flush=True)

    state = json.loads(agent._session_state_summary())
    checks.append((f"summarize_session_state reports {n_tasks} delegations",
                   state.get("delegations_total") == n_tasks))
    want_hist = min(2, n_tasks)
    hist = json.loads(agent._delegation_history(limit=2))
    checks.append((f"get_delegation_history(limit=2) returns {want_hist}",
                   len(hist) == want_hist))

    print("", flush=True)
    passed = 0
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        passed += ok

    banner(f"RESULT: {passed}/{len(checks)} checks passed")
    print("timings: " + ", ".join(f"{l}={d:.0f}s" for l, d in timings), flush=True)
    print("\nsession_state_summary:\n" + json.dumps(state, indent=2), flush=True)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
