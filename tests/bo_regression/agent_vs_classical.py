"""Live comparison: the agentic OptimizationAgent vs the saved STANDARD
(non-agentic) classical-BO baseline (ensemble_baseline.json).

The classical baseline is model-free (pure BoTorch), so this is a clean test of
whether the refactored agent still matches/beats textbook BO on identical
problems, init data, and budget — no LLM-model-mismatch confound. Lower
final-best = better (minimization).

The agent runs a real multi-step campaign (run_optimization_loop per step) from
the SAME seeded initial data the classical baseline used, for the same n_iters.
Reference = classical `single_task/log_ei` (matern_2.5 / min_noise_low) — the
textbook default.

Run:
  AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 UNSAFE_EXECUTION_OK=true \
    conda run -n scilink python agent_vs_classical.py <prob1,prob2,...> <seedcount>
"""
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from regression_ensemble import PROBLEMS, ENSEMBLE_PATH
from _benchmarks import generate_initial_data
from scilink.agents.planning_agents import OptimizationAgent

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
REF_COMBO = "single_task/log_ei"  # textbook classical BO


def run_agent_combo(problem, seed):
    func, true_func = problem["func"], problem["true"]
    eval_func = true_func or func
    bounds, cols, n_iters = problem["bounds"], problem["cols"], problem["n_iters"]

    df = generate_initial_data(func, bounds, problem["n_init"], cols, seed=seed)
    true_vals = [eval_func(*row) for row in df[cols].values]
    best_history = []

    tmp = Path(tempfile.mkdtemp(prefix="agent_cmp_"))
    data_path = tmp / "data.csv"
    df.to_csv(data_path, index=False)
    agent = OptimizationAgent(api_key=None, model_name=MODEL, output_dir=str(tmp))

    for step in range(n_iters):
        try:
            res = agent.run_optimization_loop(
                data_path=str(data_path), objective_text="Minimize y",
                input_cols=cols, input_bounds=bounds, target_cols=["y"],
                target_directions={"y": "minimize"}, output_dir=str(tmp),
                batch_size=1, experimental_budget=n_iters - step,
                save_acq=False, plot_acq=False,
            )
        except Exception as exc:
            print(f"      [step {step}] error: {type(exc).__name__}: {exc}")
            best_history.append(float(min(true_vals)))
            continue
        if res.get("status") != "success":
            best_history.append(float(min(true_vals)))
            continue
        p = res["next_parameters"]
        x_new = [p[c] for c in cols]
        y_new = func(*x_new)
        true_vals.append(eval_func(*x_new))
        row = {cols[i]: x_new[i] for i in range(len(cols))}
        row["y"] = y_new
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(data_path, index=False)
        best_history.append(float(min(true_vals)))
    return best_history


def main():
    probs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["branin_2d", "ackley_2d", "critical_cusp"]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ref = json.loads(ENSEMBLE_PATH.read_text())["combos"]
    seeds = list(range(n_seeds))

    print(f"Agent vs classical {REF_COMBO} | seeds {seeds} | model {MODEL}\n")
    rows = []
    for pname in probs:
        problem = PROBLEMS[pname]
        cls_per_seed = ref[f"{pname}/{REF_COMBO}"]["per_seed"]
        agent_finals, cls_finals = [], []
        for s in seeds:
            print(f"  {pname} seed={s} ...")
            h = run_agent_combo(problem, s)
            agent_finals.append(h[-1])
            cls_finals.append(cls_per_seed[s])
            print(f"      agent_final={h[-1]:.4f}  classical_final={cls_per_seed[s]:.4f}")
        a, c = np.mean(agent_finals), np.mean(cls_finals)
        verdict = "AGENT better" if a < c - 1e-9 else ("classical better" if c < a - 1e-9 else "tie")
        rows.append((pname, a, c, verdict))

    print(f"\n{'problem':18s} {'agent_mean':>11s} {'classical_mean':>14s}  verdict (lower=better)")
    for pname, a, c, verdict in rows:
        print(f"{pname:18s} {a:11.4f} {c:14.4f}  {verdict}")


if __name__ == "__main__":
    main()
