"""
BO agent advantage tests — showcase where the LLM-driven BOAgent beats a
fixed-strategy LogEI baseline.

Scenarios:

1. Budget-aware exploration → exploitation on Hartmann-6.
   Sparse 8-point init in 6D; greedy LogEI is well-known to converge slowly
   on this surface, while the agent's high-budget context should pick an
   exploration acquisition (thompson / max_variance) early and only commit
   to exploitation late.

2. Discovery in the presence of biased initial data (Bimodal-1D).
   Initial points are clustered in the *wrong* basin (shallow peak at x=2);
   the deeper peak lives at x=8. No strategy hint is passed — the agent's
   budget context + diagnostics must drive exploration on their own. The
   baseline runs the same data with frozen LogEI, which tends to stay
   trapped near the seed cluster.

3. Budget-phase strategy diversity (Ackley-2D, behavioral assertion only).
   Verifies the agent uses an exploration acquisition (thompson /
   max_variance / ucb) in the early high-budget phase and an exploitation
   acquisition (log_ei) in the late low-budget phase, independent of regret.

4. Constraint-aware batch design (96-well plate).
   A capability test, not a regret race: the agent is asked to design a
   batch of 16 experiments respecting "rows share temperature (≤8 unique
   values), columns share pH (≤12 unique values)". A second call from the
   same data with no constraints serves as the contrast. Conventional BO
   has no story here at all.

Both surfaces use the same ``bo_tools.get_optimizer`` GP backend; the only
difference between baseline and agent is who picks the strategy.

Artifacts (full per-step reasoning) are written under

    tests/_bo_advantage_runs/<test_name>/seed_<n>/

Each seed directory contains:
  - data.csv               — initial + accumulated observations the agent saw
  - bo_history.json        — BOAgent's own per-step log: full strategy config
                             (kernel, noise, surrogate, acquisition + params),
                             rationale, vision-inspection JSON, budget context,
                             acquisition metadata
  - step_*.png / acq_*.png — diagnostic + acquisition surface plots
  - baseline_log.json      — per-step picks made by the fixed LogEI baseline,
                             for apples-to-apples comparison

Requires ANTHROPIC_API_KEY (or SCILINK_TEST_API_KEY + SCILINK_TEST_MODEL).
Run all:

    ANTHROPIC_API_KEY=<key> python tests/test_bo_advantage.py

Or a subset by number:

    ANTHROPIC_API_KEY=<key> python tests/test_bo_advantage.py 1 3
"""

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.planning_agents.bo_tools import get_optimizer
from scilink.agents.planning_agents.bo_agent import BOAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("SCILINK_TEST_MODEL", "claude-opus-4-6")
_API_KEY = (
    os.environ.get("SCILINK_TEST_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY", "")
)

N_SEEDS = 3
_RESULTS_DIR = Path(__file__).parent / "_bo_advantage_runs"

EXPLORE_ACQ = {"thompson", "max_variance", "ucb"}
EXPLOIT_ACQ = {"log_ei", "ei"}

# ---------------------------------------------------------------------------
# Synthetic objectives
# ---------------------------------------------------------------------------

def ackley_2d(x1, x2):
    """Ackley function in 2D. Global minimum f(0,0)=0; many local minima."""
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    r = np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
    return -a * np.exp(-b * r) - np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2))) + a + np.e


def bimodal_1d(x):
    """Two basins on [0, 10]. Shallow peak ≈1 at x=2, deep peak ≈2 at x=8."""
    return float(np.exp(-(x - 2.0) ** 2) + 2.0 * np.exp(-((x - 8.0) ** 2) / 0.5))


_HARTMANN6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = np.array([
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
])
_HARTMANN6_P = 1e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
])


def hartmann_6d(*x):
    """Hartmann-6, canonical BO benchmark. Domain [0,1]^6; global min ≈ -3.32237
    at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)."""
    xa = np.asarray(x, dtype=float)
    inner = np.sum(_HARTMANN6_A * (xa - _HARTMANN6_P) ** 2, axis=1)
    return float(-np.sum(_HARTMANN6_ALPHA * np.exp(-inner)))


def yield_surface_TpH(T, pH):
    """Synthetic reaction-yield surface used in the plate-constraint test.
    Peak yield ≈ 100 at (T=350, pH=4.5)."""
    return float(
        100.0
        * np.exp(-((T - 350.0) / 30.0) ** 2)
        * np.exp(-((pH - 4.5) / 0.5) ** 2)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_dir(test_name, seed, suffix):
    """Create and return tests/_bo_advantage_runs/<test>/seed_<n>/<suffix>/."""
    d = _RESULTS_DIR / test_name / f"seed_{seed}" / suffix
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _initial_sample(observe_fn, bounds, n, seed, col_names, target_name,
                    sample_bounds=None):
    """Draw n uniform samples within ``sample_bounds`` (defaults to bounds).

    ``sample_bounds`` lets us bias the initial data toward one region of the
    space, which is what produces the failure mode for greedy LogEI.
    """
    rng = np.random.default_rng(seed)
    if sample_bounds is None:
        sample_bounds = bounds
    X = np.zeros((n, len(bounds)))
    for i, (lo, hi) in enumerate(sample_bounds):
        X[:, i] = rng.uniform(lo, hi, n)
    y = np.array([observe_fn(*row) for row in X], dtype=float)
    data = {col_names[i]: X[:, i] for i in range(len(bounds))}
    data[target_name] = y
    return pd.DataFrame(data)


def _baseline_step(df, input_cols, target_col, bounds, direction):
    """One step of fixed LogEI + Matern-2.5 baseline. Returns next candidate as a list."""
    X = df[input_cols].values.astype(np.float64)
    y = df[[target_col]].values.astype(np.float64)
    if direction == "minimize":
        y = -y
    optimizer = get_optimizer(is_moo=False, device="cpu")
    optimizer.fit(
        X, y, np.array(bounds, dtype=np.float64),
        {"kernel": "matern_2.5", "noise": "min_noise_low"},
        input_cols,
    )
    cand = optimizer.recommend(n_candidates=1, strategy="log_ei", params={})[0]
    return list(cand)


def _run_baseline(observe_fn, initial_df, input_cols, target_col, bounds,
                  n_iters, direction, artifacts_dir):
    """Run the fixed-strategy baseline. Persists every step to baseline_log.json
    inside ``artifacts_dir``. Returns (best_so_far_traj, final_df)."""
    df = initial_df.copy()
    best_so_far = []
    step_log = []

    for step in range(n_iters):
        cand = _baseline_step(df, input_cols, target_col, bounds, direction)
        new_y = observe_fn(*cand)
        new_row = {input_cols[i]: cand[i] for i in range(len(input_cols))}
        new_row[target_col] = new_y
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        best = float(df[target_col].min() if direction == "minimize" else df[target_col].max())
        best_so_far.append(best)
        step_log.append({
            "step": step + 1,
            "recommendation": {c: float(v) for c, v in zip(input_cols, cand)},
            "observed_y": float(new_y),
            "best_so_far": best,
        })

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(artifacts_dir / "data_final.csv", index=False)
    with open(artifacts_dir / "baseline_log.json", "w") as f:
        json.dump({
            "strategy": {"kernel": "matern_2.5", "noise": "min_noise_low",
                         "acquisition": "log_ei"},
            "direction": direction,
            "n_init": len(initial_df),
            "n_iters": n_iters,
            "steps": step_log,
            "best_so_far": best_so_far,
        }, f, indent=2)

    return best_so_far, df


def _run_agent(observe_fn, initial_df, input_cols, target_col, bounds,
               n_iters, direction, artifacts_dir, strategy_hint=None):
    """Run BOAgent for n_iters steps with shrinking experimental_budget.

    All BOAgent outputs (bo_history.json, step_*.png, acq_*.png) are written
    directly into ``artifacts_dir`` and NOT deleted, so the full per-step
    reasoning (strategy config, rationale, vision inspection, budget context)
    persists for later inspection.

    Returns (best_so_far_traj, final_df, strategies). ``strategies`` is a list
    of dicts ``{kernel, noise, acq, phase, rationale, inspection_status}``.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_path = artifacts_dir / "data.csv"
    df = initial_df.copy()
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL,
                    output_dir=str(artifacts_dir))
    best_so_far = []
    strategies = []
    step_records = []

    for step in range(n_iters):
        remaining = n_iters - step
        result = agent.run_optimization_loop(
            data_path=str(data_path),
            objective_text=f"Optimize {target_col}",
            input_cols=input_cols,
            input_bounds=bounds,
            target_cols=[target_col],
            target_directions={target_col: direction},
            output_dir=str(artifacts_dir),
            batch_size=1,
            experimental_budget=remaining,
            strategy_hint=strategy_hint,
            save_acq=False,
            plot_acq=True,  # vision inspection needs the plot
        )
        if result.get("status") != "success":
            # Persist what we have so far before bailing
            with open(artifacts_dir / "agent_run_log.json", "w") as f:
                json.dump({"failed_at_step": step + 1, "steps": step_records,
                           "last_result": result}, f, indent=2, default=str)
            raise RuntimeError(
                f"Agent step {step + 1} failed: status={result.get('status')}, "
                f"error={result.get('error')}"
            )

        params = result["next_parameters"]
        new_x = [float(params[c]) for c in input_cols]
        new_y = observe_fn(*new_x)
        new_row = {input_cols[i]: new_x[i] for i in range(len(input_cols))}
        new_row[target_col] = new_y
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(data_path, index=False)

        best = float(df[target_col].min() if direction == "minimize" else df[target_col].max())
        best_so_far.append(best)

        strat_full = result.get("strategy") or {}
        mc = strat_full.get("model_config") or {}
        ac = strat_full.get("acquisition_strategy") or {}
        inspection = result.get("inspection") or {}
        budget = result.get("budget") or {}

        strategies.append({
            "kernel": mc.get("kernel"),
            "noise": mc.get("noise"),
            "acq": ac.get("type"),
            "phase": budget.get("budget_phase"),
            "rationale": strat_full.get("rationale"),
            "inspection_status": inspection.get("status"),
        })
        # Full per-step record — keeps everything the agent emitted so the
        # bo_history.json contents are mirrored in a flatter shape for review.
        step_records.append({
            "step": step + 1,
            "experimental_budget_remaining": remaining,
            "recommendation": {c: float(v) for c, v in zip(input_cols, new_x)},
            "observed_y": float(new_y),
            "best_so_far": best,
            "strategy": strat_full,
            "budget": budget,
            "inspection": inspection,
            "plot_path": result.get("plot_path"),
            "acq_plot_path": result.get("acq_plot_path"),
        })

    with open(artifacts_dir / "agent_run_log.json", "w") as f:
        json.dump({
            "strategy_hint": strategy_hint,
            "direction": direction,
            "n_init": len(initial_df),
            "n_iters": n_iters,
            "steps": step_records,
            "best_so_far": best_so_far,
        }, f, indent=2, default=str)

    return best_so_far, df, strategies


def _run_agent_single_batch(initial_df, input_cols, target_col, bounds,
                            batch_size, direction, artifacts_dir,
                            physical_constraints=None):
    """One BOAgent call returning a batch. Used for the constraint-aware test.

    Returns (batch, result). The agent's full strategy + constrained_planning
    metadata is mirrored into agent_batch_log.json inside ``artifacts_dir``,
    in addition to the usual bo_history.json the agent writes itself.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_path = artifacts_dir / "data.csv"
    initial_df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL,
                    output_dir=str(artifacts_dir))
    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text=f"Optimize {target_col}",
        input_cols=input_cols,
        input_bounds=bounds,
        target_cols=[target_col],
        target_directions={target_col: direction},
        output_dir=str(artifacts_dir),
        batch_size=batch_size,
        physical_constraints=physical_constraints,
        save_acq=False,
        plot_acq=False,
    )
    if result.get("status") != "success":
        with open(artifacts_dir / "agent_batch_log.json", "w") as f:
            json.dump({"failed": True, "result": result}, f, indent=2, default=str)
        raise RuntimeError(
            f"Constrained-batch call failed: status={result.get('status')}, "
            f"error={result.get('error')}"
        )

    batch = result["next_parameters"]
    if not isinstance(batch, list):
        batch = [batch]

    with open(artifacts_dir / "agent_batch_log.json", "w") as f:
        json.dump({
            "batch_size": batch_size,
            "physical_constraints": physical_constraints,
            "batch": batch,
            "strategy": result.get("strategy"),
            "constrained_planning": result.get("constrained_planning"),
            "constraint_aware": result.get("constraint_aware", False),
        }, f, indent=2, default=str)

    return batch, result


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = []


def _test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Test 1 — Budget-aware on Ackley-2D
# ---------------------------------------------------------------------------

@_test
def t1_budget_aware_hartmann6():
    """Agent beats fixed LogEI on Hartmann-6 with sparse 8-point init.

    Hartmann-6 is the canonical hard BO benchmark; with only 8 points in 6D,
    greedy LogEI tends to commit to a local basin near its seed cluster,
    while the agent's high-budget context should drive exploration in the
    early steps before exploiting at the end.
    """
    bounds = [[0.0, 1.0]] * 6
    cols = [f"x{i + 1}" for i in range(6)]
    n_iters = 10
    n_init = 8

    per_seed = []
    bl_finals, ag_finals = [], []

    for seed in range(N_SEEDS):
        df0 = _initial_sample(
            hartmann_6d, bounds, n=n_init, seed=seed,
            col_names=cols, target_name="y",
        )

        bl_dir = _seed_dir("t1_budget_aware_hartmann6", seed, "baseline")
        ag_dir = _seed_dir("t1_budget_aware_hartmann6", seed, "agent")
        bl_traj, _ = _run_baseline(
            hartmann_6d, df0, cols, "y", bounds,
            n_iters=n_iters, direction="minimize", artifacts_dir=bl_dir,
        )
        ag_traj, _, strategies = _run_agent(
            hartmann_6d, df0, cols, "y", bounds,
            n_iters=n_iters, direction="minimize", artifacts_dir=ag_dir,
        )

        bl_final = bl_traj[-1]
        ag_final = ag_traj[-1]
        bl_finals.append(bl_final)
        ag_finals.append(ag_final)

        per_seed.append({
            "seed": seed,
            "baseline_final": bl_final,
            "agent_final": ag_final,
            "agent_wins": ag_final < bl_final,
            "phases": [s["phase"] for s in strategies],
            "acqs": [s["acq"] for s in strategies],
        })

    wins = sum(1 for r in per_seed if r["agent_wins"])
    ok = wins >= 2

    detail = (
        f"agent wins {wins}/{N_SEEDS}; "
        f"baseline mean={np.mean(bl_finals):.3f}, agent mean={np.mean(ag_finals):.3f} "
        f"(Hartmann-6 min ≈ -3.322). per_seed={per_seed}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 2 — Escape misleading prior on Bimodal-1D
# ---------------------------------------------------------------------------

@_test
def t2_discover_deeper_basin():
    """Agent discovers the deeper basin on bimodal-1D from biased initial data;
    fixed-strategy LogEI baseline tends to stay trapped near the seed cluster.

    No strategy hint is given — the discovery has to come from the agent's
    own budget context + diagnostics.
    """
    bounds = [[0.0, 10.0]]
    sample_bounds = [[0.0, 4.0]]  # initial cluster lives in the shallow basin
    n_iters = 10
    n_init = 5

    per_seed = []
    for seed in range(N_SEEDS):
        df0 = _initial_sample(
            bimodal_1d, bounds, n=n_init, seed=seed,
            col_names=["x"], target_name="y",
            sample_bounds=sample_bounds,
        )

        bl_dir = _seed_dir("t2_discover_deeper_basin", seed, "baseline")
        ag_dir = _seed_dir("t2_discover_deeper_basin", seed, "agent")
        _, bl_df = _run_baseline(
            bimodal_1d, df0, ["x"], "y", bounds,
            n_iters=n_iters, direction="maximize", artifacts_dir=bl_dir,
        )
        _, ag_df, strategies = _run_agent(
            bimodal_1d, df0, ["x"], "y", bounds,
            n_iters=n_iters, direction="maximize", artifacts_dir=ag_dir,
        )

        bl_best_x = float(bl_df.loc[bl_df["y"].idxmax(), "x"])
        bl_best_y = float(bl_df["y"].max())
        ag_best_x = float(ag_df.loc[ag_df["y"].idxmax(), "x"])
        ag_best_y = float(ag_df["y"].max())

        per_seed.append({
            "seed": seed,
            "baseline_best_x": bl_best_x, "baseline_best_y": bl_best_y,
            "agent_best_x": ag_best_x, "agent_best_y": ag_best_y,
            "agent_found_deep": ag_best_x > 6.0 and ag_best_y > 1.5,
            "baseline_found_deep": bl_best_x > 6.0 and bl_best_y > 1.5,
            "acqs": [s["acq"] for s in strategies],
        })

    ag_finds = sum(1 for r in per_seed if r["agent_found_deep"])
    bl_finds = sum(1 for r in per_seed if r["baseline_found_deep"])
    # Agent should find the deeper basin in ≥ 2/3 seeds AND do strictly better
    # than the baseline across seeds.
    ok = ag_finds >= 2 and ag_finds > bl_finds

    detail = (
        f"agent finds deep basin in {ag_finds}/{N_SEEDS} seeds; "
        f"baseline in {bl_finds}/{N_SEEDS} (true peak at x=8, y≈2). "
        f"per_seed={per_seed}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 3 — Strategy diversity across budget phases
# ---------------------------------------------------------------------------

@_test
def t3_strategy_shifts_with_budget():
    """Agent uses exploration early (high-budget phase) and exploitation late
    (low-budget phase) across a 10-step run."""
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]
    n_iters = 10
    n_init = 5

    early_third = max(2, n_iters // 3)
    late_third = max(2, n_iters // 3)

    per_seed = []
    for seed in range(N_SEEDS):
        df0 = _initial_sample(
            ackley_2d, bounds, n=n_init, seed=seed,
            col_names=["x1", "x2"], target_name="y",
        )
        ag_dir = _seed_dir("t3_strategy_shifts_with_budget", seed, "agent")
        _, _, strategies = _run_agent(
            ackley_2d, df0, ["x1", "x2"], "y", bounds,
            n_iters=n_iters, direction="minimize", artifacts_dir=ag_dir,
        )

        early_acqs = [s["acq"] for s in strategies[:early_third]]
        late_acqs = [s["acq"] for s in strategies[-late_third:]]
        phases = [s["phase"] for s in strategies]

        per_seed.append({
            "seed": seed,
            "phases": phases,
            "early_acqs": early_acqs,
            "late_acqs": late_acqs,
            "early_has_explore": any(a in EXPLORE_ACQ for a in early_acqs),
            "late_has_exploit": any(a in EXPLOIT_ACQ for a in late_acqs),
        })

    early_explore_seeds = sum(1 for r in per_seed if r["early_has_explore"])
    late_exploit_seeds = sum(1 for r in per_seed if r["late_has_exploit"])

    # Need at least one early-explore example overall and exploit dominance in
    # the late phase of the majority of seeds.
    ok = early_explore_seeds >= 1 and late_exploit_seeds >= 2

    detail = (
        f"early_explore in {early_explore_seeds}/{N_SEEDS} seeds, "
        f"late_exploit in {late_exploit_seeds}/{N_SEEDS} seeds. "
        f"per_seed={per_seed}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Test 4 — Constraint-aware batch design (96-well plate)
# ---------------------------------------------------------------------------

@_test
def t4_constraint_aware_plate():
    """Agent's constrained batch planner produces a 96-well-plate-feasible batch
    (≤8 unique temperatures, ≤12 unique pH values); the SAME agent given the
    same data with no constraints emits ~16 distinct (T, pH) combinations.

    Capability test, not regret race — conventional BO cannot honor this
    constraint at all.
    """
    bounds = [[250.0, 400.0], [3.0, 6.0]]
    n_init = 12
    batch_size = 16
    constraint = (
        "96-well plate: rows share temperature (8 unique values), "
        "columns share pH (12 unique values). Design 16 wells that fit on the plate."
    )

    def n_unique(values, tol):
        """Count unique values up to a tolerance — guards against tiny float drift."""
        if not values:
            return 0
        v = np.array(sorted(values))
        clusters = 1
        for i in range(1, len(v)):
            if v[i] - v[i - 1] > tol:
                clusters += 1
        return clusters

    per_seed = []
    for seed in range(N_SEEDS):
        df0 = _initial_sample(
            yield_surface_TpH, bounds, n=n_init, seed=seed,
            col_names=["T", "pH"], target_name="yield",
        )

        con_dir = _seed_dir("t4_constraint_aware_plate", seed, "agent_constrained")
        unc_dir = _seed_dir("t4_constraint_aware_plate", seed, "agent_unconstrained")

        con_batch, con_result = _run_agent_single_batch(
            df0, ["T", "pH"], "yield", bounds,
            batch_size=batch_size, direction="maximize",
            artifacts_dir=con_dir, physical_constraints=constraint,
        )
        unc_batch, _ = _run_agent_single_batch(
            df0, ["T", "pH"], "yield", bounds,
            batch_size=batch_size, direction="maximize",
            artifacts_dir=unc_dir, physical_constraints=None,
        )

        con_T = [float(p["T"]) for p in con_batch]
        con_pH = [float(p["pH"]) for p in con_batch]
        unc_T = [float(p["T"]) for p in unc_batch]
        unc_pH = [float(p["pH"]) for p in unc_batch]

        con_unique_T = n_unique(con_T, tol=0.5)
        con_unique_pH = n_unique(con_pH, tol=0.02)
        unc_unique_T = n_unique(unc_T, tol=0.5)
        unc_unique_pH = n_unique(unc_pH, tol=0.02)

        per_seed.append({
            "seed": seed,
            "constrained_unique_T": con_unique_T,
            "constrained_unique_pH": con_unique_pH,
            "unconstrained_unique_T": unc_unique_T,
            "unconstrained_unique_pH": unc_unique_pH,
            "constraint_aware_flag": bool(con_result.get("constraint_aware", False)),
            "fits_plate": con_unique_T <= 8 and con_unique_pH <= 12,
        })

    fits = sum(1 for r in per_seed if r["fits_plate"])
    ok = fits >= 2  # ≥ 2/3 seeds produce a plate-feasible batch

    detail = (
        f"constrained batch fits plate in {fits}/{N_SEEDS} seeds. "
        f"per_seed={per_seed}"
    )
    return ok, detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main(argv):
    if not _API_KEY:
        print(
            "ERROR: ANTHROPIC_API_KEY (or SCILINK_TEST_API_KEY) not set.",
            file=sys.stderr,
        )
        return 2

    selected = [int(a) - 1 for a in argv if a.isdigit()]
    tests = TESTS if not selected else [TESTS[i] for i in selected]

    _RESULTS_DIR.mkdir(exist_ok=True)
    results = []

    for i, fn in enumerate(tests, 1):
        title = fn.__doc__.splitlines()[0].strip() if fn.__doc__ else fn.__name__
        print(f"\n=== [{i}/{len(tests)}] {fn.__name__} ===")
        print(f"    {title}")
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            print(f"    {status}: {detail}")
            results.append({"name": fn.__name__, "ok": bool(ok), "detail": detail})
        except Exception as e:
            traceback.print_exc()
            results.append({
                "name": fn.__name__,
                "ok": False,
                "detail": f"EXCEPTION: {e}",
            })

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['name']}")
    n_pass = sum(1 for r in results if r["ok"])
    print(f"\n{n_pass}/{len(results)} tests passed.")

    summary_path = _RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": _MODEL,
            "n_seeds": N_SEEDS,
            "results": results,
            "artifact_root": str(_RESULTS_DIR),
            "per_seed_layout": (
                "<test_name>/seed_<n>/{agent,baseline}/  — agent dir contains "
                "bo_history.json (full strategy + rationale + inspection per "
                "step), step_*.png diagnostics, agent_run_log.json; baseline "
                "dir contains baseline_log.json and data_final.csv"
            ),
        }, f, indent=2)
    print(f"Summary: {summary_path}")
    print(f"Per-seed artifacts under: {_RESULTS_DIR}/")

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
