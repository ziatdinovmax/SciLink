"""Pool-based BO campaigns driven THROUGH THE MCP SERVER (tool layer), not the
Python API — the path an external framework actually uses.

Per campaign: one `scilink serve --mode plan` process on a fresh session dir;
analyze_file(seed csv) locks the schema; then for each step
run_optimization(candidate_pool=<pool csv>, input_bounds=<design box>,
experimental_budget=<remaining>) -> recommended row -> measured from the
pool -> appended through analyze_file. Same inits / pool / recording as
bench.py so results_mcp/ compares directly with results_regress_pool/.

Differences from the direct harness (by construction of the MCP surface):
  * no cat_dims argument — encoded categorical inputs arrive as numeric
    columns and are modelled as continuous (candidate_pool still restricts
    recommendations to real pool rows);
  * the planning orchestrator's scalarizer ingests each CSV (pass-through).

Usage:
  python bench_mcp.py <dataset> <seed> [n_iters]     (client venv: deepagents_bo_demo/.venv)
Env: BENCH_RESULTS (default results_mcp), BENCH_N_INIT, BENCH_N_ITERS,
     SCILINK_BIN / SCILINK_MODEL / credentials via ../deepagents-scilink/.env
"""
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/maxim.ziatdinov/Code/deepagents-scilink")

from pools import Pool  # noqa: E402
from scilink_mcp import server_config, load_env  # noqa: E402
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402

N_INIT = int(os.environ.get("BENCH_N_INIT", "5"))
N_ITERS = int(os.environ.get("BENCH_N_ITERS", "15"))
RES_SUB = os.environ.get("BENCH_RESULTS", "results_mcp")
RUNS_SUB = f"runs_{RES_SUB}"


def _text(r):
    if isinstance(r, list):
        r = r[0]["text"] if r and isinstance(r[0], dict) else (r[0] if r else "{}")
    return json.loads(r)


def _record(pool, measured, extras, dataset, method, seed):
    seq = [int(i) for i in measured]
    oriented_seq = pool.oriented(pool.y[seq])
    best = np.maximum.accumulate(oriented_seq)
    best_nat = (-best).tolist() if pool.direction == "minimize" else best.tolist()
    out = {"dataset": dataset, "method": method, "seed": seed, "n_init": N_INIT,
           "direction": pool.direction, "pool_size": pool.n, "pool_optimum": pool.optimum,
           "picked_indices": seq, "picked_y": pool.y[seq].tolist(), "best_so_far": best_nat,
           **extras}
    d = HERE / RES_SUB / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{method}_seed{seed}.json").write_text(json.dumps(out, indent=1))
    return out


async def run(dataset, seed, n_iters):
    pool = Pool(dataset)
    measured = pool.init_indices(seed, N_INIT)
    run_dir = HERE / RUNS_SUB / f"{dataset}_seed{seed}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "client").mkdir(parents=True)
    session_dir = run_dir / "scilink"
    client_dir = run_dir / "client"

    pool_csv = client_dir / "pool.csv"
    pool.df[pool.input_cols].to_csv(pool_csv, index=False)
    seed_csv = client_dir / "seed.csv"
    pool.measured_df(measured).to_csv(seed_csv, index=False)
    bounds = {c: [float(lo), float(hi)] for c, (lo, hi) in zip(pool.input_cols, pool.bounds)}
    goal = (f"{pool.objective_text} Inputs: {pool.input_cols}. Single target "
            f"'{pool.target_col}', to be {'MAXIMIZED' if pool.direction == 'maximize' else 'MINIMIZED'}.")

    os.environ.update(load_env())
    client = MultiServerMCPClient(server_config(session_dir=str(session_dir), mode="plan"))
    steps_meta, fallbacks, tool_errors = [], 0, []
    t0 = time.time()
    async with client.session("scilink") as s:
        tools = {t.name: t for t in await load_mcp_tools(s)}
        ing = _text(await tools["scilink_analyze_file"].ainvoke({
            "file_path": str(seed_csv), "extraction_goal": goal,
            "inputs": pool.input_cols, "targets": [pool.target_col],
            "directions": {pool.target_col: pool.direction}}))
        if ing.get("status") != "success":
            raise RuntimeError(f"seed ingest failed: {ing}")
        if ing.get("target_directions", {}).get(pool.target_col) != pool.direction:
            raise RuntimeError(f"direction not honoured: {ing.get('target_directions')}")
        for step in range(n_iters):
            res, err = None, None
            for attempt in range(2):
                try:
                    res = _text(await tools["scilink_run_optimization"].ainvoke({
                        "parallel_capable": False, "experimental_budget": n_iters - step,
                        "candidate_pool": str(pool_csv), "input_bounds": bounds}))
                except Exception as exc:  # noqa: BLE001
                    res, err = None, f"{type(exc).__name__}: {exc}"
                    continue
                if res.get("status") == "success":
                    break
                err = f"{res.get('message')} | {res.get('hint', '')}"
                res = None
                await asyncio.sleep(5)
            if res is not None:
                p = res["recommended_parameters"]
                if isinstance(p, list):
                    p = p[0]
                x = [float(p[c]) for c in pool.input_cols]
                idx = pool.snap(x, set(measured))
                strat = res.get("strategy") or {}
                steps_meta.append({"step": step, "proposal": x, "snapped_index": idx,
                                   "acq": (strat.get("acquisition_strategy") or {}).get("type"),
                                   "surrogate": (strat.get("model_config") or {}).get("surrogate"),
                                   "bounds_source": res.get("input_bounds_source"),
                                   "directions": res.get("target_directions"),
                                   "pool": res.get("candidate_pool"), "fallback": False})
            else:
                fallbacks += 1
                tool_errors.append(err)
                rng = np.random.RandomState(4000 + seed * 100 + step)
                remaining = [i for i in range(pool.n) if i not in set(measured)]
                idx = int(rng.choice(remaining))
                steps_meta.append({"step": step, "proposal": None, "snapped_index": idx,
                                   "fallback": True, "error": err})
            measured.append(idx)
            row_csv = client_dir / f"results_round_{step + 1:02d}.csv"
            pool.measured_df([idx]).to_csv(row_csv, index=False)
            ing = _text(await tools["scilink_analyze_file"].ainvoke({
                "file_path": str(row_csv), "extraction_goal": goal,
                "inputs": pool.input_cols, "targets": [pool.target_col]}))
            if ing.get("status") not in ("success", "warning"):
                tool_errors.append(f"ingest step {step}: {ing}")
            best = pool.oriented(pool.y[measured]).max()
            best_nat = -best if pool.direction == "minimize" else best
            print(f"  [mcp {dataset} seed {seed}] step {step + 1}/{n_iters} -> idx {idx}, "
                  f"y={pool.y[idx]:.4g}, best={best_nat:.4g}"
                  f"{' (FALLBACK)' if steps_meta[-1]['fallback'] else ''}", flush=True)
    out = _record(pool, measured, {"model": os.environ.get("SCILINK_MODEL"), "fallbacks": fallbacks,
                                   "tool_errors": tool_errors[:20], "steps": steps_meta,
                                   "elapsed_s": round(time.time() - t0)},
                  dataset, "agent_mcp", seed)
    print(json.dumps({k: out[k] for k in ("dataset", "method", "seed", "fallbacks", "elapsed_s")}
                     | {"final": out["best_so_far"][-1]}))


if __name__ == "__main__":
    ds, sd = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else N_ITERS
    asyncio.run(run(ds, sd, n))
