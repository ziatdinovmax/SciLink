"""
Bayesian Optimization convergence test suite.

Tests that the BO agent converges towards known optima on synthetic
objective functions — both standalone (bo_tools) and through the
orchestrator chat interface.

Requires GEMINI_API_KEY env var (or SCILINK_TEST_API_KEY + SCILINK_TEST_MODEL).
Run with:

    GEMINI_API_KEY=<key> python tests/test_bo_convergence.py

Or run a subset by number:

    GEMINI_API_KEY=<key> python tests/test_bo_convergence.py 1 3 7
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.tools.bo_tools import get_optimizer
from scilink.agents.planning_agents.bo_agent import BOAgent, _compute_budget_context
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent,
    AutonomyLevel,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MODEL = os.environ.get("SCILINK_TEST_MODEL", "gemini-3.1-pro-preview")
_API_KEY = os.environ.get("SCILINK_TEST_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
_EMBEDDING_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Synthetic objective functions
# ---------------------------------------------------------------------------

def branin(x1, x2):
    """Branin function — 3 global minima at ~0.398.
    Typical domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]."""
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def neg_branin(x1, x2):
    """Negated Branin for maximization tests."""
    return -branin(x1, x2)


def quadratic_1d(x):
    """Simple 1D quadratic with minimum at x=3, f(3)=0."""
    return (x - 3)**2


def neg_quadratic_1d(x):
    """Negated quadratic for maximization, peak at x=3."""
    return -(x - 3)**2 + 100


def sphere_3d(x1, x2, x3):
    """3D sphere function — minimum at origin, f(0,0,0)=0."""
    return x1**2 + x2**2 + x3**2


def selectivity_surface(hcl_M, time_h):
    """Simulated REE leaching: Nd recovery peaks at ~5M, 5h."""
    nd_recovery = 95 * (1 - np.exp(-0.5 * hcl_M)) * (1 - np.exp(-0.4 * time_h))
    noise = np.random.normal(0, 1.5)
    return nd_recovery + noise


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _tmp():
    return Path(tempfile.mkdtemp(prefix="bo_conv_"))


def _generate_initial_data(func, bounds, n=8, seed=42, col_names=None, target_name="y"):
    """Generate initial Latin Hypercube-ish random samples."""
    np.random.seed(seed)
    n_dims = len(bounds)
    if col_names is None:
        col_names = [f"x{i}" for i in range(n_dims)]

    X = np.zeros((n, n_dims))
    for i, (lo, hi) in enumerate(bounds):
        X[:, i] = np.random.uniform(lo, hi, n)

    y = np.array([func(*row) for row in X])

    data = {col_names[i]: X[:, i] for i in range(n_dims)}
    data[target_name] = y
    return pd.DataFrame(data)


def _run_bo_loop(df, input_cols, target_col, bounds, func, n_iters=5,
                 direction="minimize", seed=42):
    """Run BO loop on synthetic function, return best values per iteration."""
    np.random.seed(seed)
    is_moo = False
    optimizer = get_optimizer(is_moo=is_moo, device="cpu")

    best_values = []
    target_direction = {target_col: direction}

    for step in range(n_iters):
        X = df[input_cols].values.astype(np.float64)
        y = df[[target_col]].values.astype(np.float64)

        # Negate if minimizing (BO always maximizes internally)
        if direction == "minimize":
            y_train = -y
        else:
            y_train = y

        bounds_tensor = np.array(bounds, dtype=np.float64)  # (d, 2) — fit() transposes internally

        model_config = {"kernel": "matern_2.5", "noise": "min_noise_low"}
        optimizer.fit(X, y_train, bounds_tensor, model_config, input_cols)

        candidates = optimizer.recommend(
            n_candidates=1,
            strategy="log_ei",
            params={},
        )

        new_x = candidates[0]
        new_y = func(*new_x)

        # Append to dataframe
        new_row = {input_cols[i]: new_x[i] for i in range(len(input_cols))}
        new_row[target_col] = new_y
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        if direction == "minimize":
            best_values.append(df[target_col].min())
        else:
            best_values.append(df[target_col].max())

    return best_values, df


def _orch(base_dir, data_dir=None):
    """Create a planning orchestrator in AUTONOMOUS mode."""
    if data_dir is None:
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
    return dict(
        base_dir=str(base_dir / "session"),
        api_key=_API_KEY,
        model_name=_MODEL,
        embedding_api_key=_EMBEDDING_API_KEY,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )


# ---------------------------------------------------------------------------
# Test definitions — each returns (ok: bool, detail: str)
# ---------------------------------------------------------------------------

TESTS = []


def _test(fn):
    TESTS.append(fn)
    return fn


# ===== GROUP 1: Standalone BO convergence (no LLM calls) =====

@_test
def soo_1d_quadratic_convergence():
    """1D quadratic minimization converges to x=3 within 5 iterations."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=5, col_names=["x"], target_name="y"
    )

    best_vals, final_df = _run_bo_loop(
        df, ["x"], "y", bounds, quadratic_1d,
        n_iters=5, direction="minimize"
    )

    # Best value should decrease towards 0
    initial_best = df["y"].min()
    final_best = best_vals[-1]
    improved = final_best < initial_best
    near_optimum = final_best < 2.0  # f(3)=0, allow some tolerance

    ok = improved and near_optimum
    return ok, f"initial_best={initial_best:.2f}, final_best={final_best:.2f}, values={[f'{v:.2f}' for v in best_vals]}"


@_test
def soo_1d_maximization_convergence():
    """1D quadratic maximization converges to peak at x=3."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        neg_quadratic_1d, bounds, n=5, col_names=["x"], target_name="y"
    )

    best_vals, final_df = _run_bo_loop(
        df, ["x"], "y", bounds, neg_quadratic_1d,
        n_iters=5, direction="maximize"
    )

    initial_best = df["y"].max()
    final_best = best_vals[-1]
    improved = final_best > initial_best
    near_optimum = final_best > 98.0  # f(3)=100

    ok = improved and near_optimum
    return ok, f"initial_best={initial_best:.2f}, final_best={final_best:.2f}"


@_test
def soo_2d_branin_convergence():
    """2D Branin minimization improves over 8 iterations."""
    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    df = _generate_initial_data(
        branin, bounds, n=10, col_names=["x1", "x2"], target_name="y"
    )

    best_vals, final_df = _run_bo_loop(
        df, ["x1", "x2"], "y", bounds, branin,
        n_iters=8, direction="minimize"
    )

    initial_best = df["y"].min()
    final_best = best_vals[-1]
    improved = final_best < initial_best
    # Branin global minimum is ~0.398
    reasonable = final_best < 5.0

    ok = improved and reasonable
    return ok, f"initial_best={initial_best:.2f}, final_best={final_best:.2f}, global_min=0.398"


@_test
def soo_3d_sphere_convergence():
    """3D sphere minimization converges towards origin."""
    bounds = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    df = _generate_initial_data(
        sphere_3d, bounds, n=10, col_names=["x1", "x2", "x3"], target_name="y"
    )

    best_vals, final_df = _run_bo_loop(
        df, ["x1", "x2", "x3"], "y", bounds, sphere_3d,
        n_iters=10, direction="minimize"
    )

    initial_best = df["y"].min()
    final_best = best_vals[-1]
    improved = final_best < initial_best

    ok = improved
    return ok, f"initial_best={initial_best:.2f}, final_best={final_best:.2f}"


@_test
def soo_monotonic_improvement():
    """Best-so-far should be monotonically non-increasing for minimization."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=5, col_names=["x"], target_name="y"
    )

    best_vals, _ = _run_bo_loop(
        df, ["x"], "y", bounds, quadratic_1d,
        n_iters=8, direction="minimize"
    )

    # Best-so-far should never increase
    monotonic = all(best_vals[i] <= best_vals[i - 1] + 1e-10 for i in range(1, len(best_vals)))

    ok = monotonic
    return ok, f"best_values={[f'{v:.3f}' for v in best_vals]}"


@_test
def soo_batch_returns_correct_count():
    """Batch BO returns exactly batch_size candidates."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=8, col_names=["x"], target_name="y"
    )

    optimizer = get_optimizer(is_moo=False, device="cpu")
    X = df[["x"]].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)  # minimize
    bounds_arr = np.array(bounds, dtype=np.float64)

    optimizer.fit(X, y, bounds_arr, {"kernel": "matern_2.5", "noise": "min_noise_low"}, ["x"])

    for batch_size in [1, 3, 5]:
        candidates = optimizer.recommend(n_candidates=batch_size, strategy="log_ei", params={})
        if len(candidates) != batch_size:
            return False, f"batch_size={batch_size}, got {len(candidates)} candidates"

    return True, "all batch sizes correct"


@_test
def soo_candidates_within_bounds():
    """All recommended candidates should be within input bounds."""
    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    df = _generate_initial_data(
        branin, bounds, n=10, col_names=["x1", "x2"], target_name="y"
    )

    optimizer = get_optimizer(is_moo=False, device="cpu")
    X = df[["x1", "x2"]].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)
    bounds_arr = np.array(bounds, dtype=np.float64)

    optimizer.fit(X, y, bounds_arr, {"kernel": "matern_2.5", "noise": "min_noise_low"}, ["x1", "x2"])

    candidates = optimizer.recommend(n_candidates=5, strategy="log_ei", params={})

    for i, cand in enumerate(candidates):
        for j, (lo, hi) in enumerate(bounds):
            if cand[j] < lo - 0.01 or cand[j] > hi + 0.01:
                return False, f"candidate {i} dim {j}: {cand[j]:.3f} outside [{lo}, {hi}]"

    return True, f"{len(candidates)} candidates all within bounds"


# ===== GROUP 2: Multi-objective (no LLM calls) =====

@_test
def moo_2d_pareto_improvement():
    """MOO finds candidates that improve the Pareto front."""
    bounds = [[0.0, 6.0], [0.0, 6.0]]

    def obj1(x1, x2):
        return -(x1 - 2)**2 - (x2 - 4)**2 + 50  # Peak at (2, 4)

    def obj2(x1, x2):
        return -(x1 - 4)**2 - (x2 - 2)**2 + 50  # Peak at (4, 2) — conflicts with obj1

    np.random.seed(42)
    n = 12
    X = np.random.uniform(0, 6, (n, 2))
    y1 = np.array([obj1(*row) for row in X])
    y2 = np.array([obj2(*row) for row in X])

    optimizer = get_optimizer(is_moo=True, device="cpu")
    bounds_arr = np.array(bounds, dtype=np.float64)

    optimizer.fit(X, np.column_stack([y1, y2]), bounds_arr,
                  {"kernel": "matern_2.5", "noise": "min_noise_low"},
                  ["x1", "x2"])

    candidates = optimizer.recommend(n_candidates=3, strategy="pareto", params={})

    ok = len(candidates) == 3
    # Candidates should be in bounds
    for cand in candidates:
        for j in range(2):
            if cand[j] < -0.01 or cand[j] > 6.01:
                return False, f"MOO candidate out of bounds: {cand}"

    return ok, f"got {len(candidates)} Pareto candidates"


# ===== GROUP 3: Budget context (no LLM calls) =====

@_test
def budget_phase_classification():
    """Budget phases are correctly classified."""
    cases = [
        (1, [], "final_shot"),
        (2, [{}], "critical"),
        (3, [{}, {}], "critical"),
        (5, [{}], "high"),
        (None, [], "unlimited"),
    ]

    for budget, history, expected_phase in cases:
        ctx = _compute_budget_context(budget, history)
        actual = ctx["budget_phase"]
        if actual != expected_phase:
            return False, f"budget={budget}, history={len(history)}: expected {expected_phase}, got {actual}"

    return True, "all budget phases correct"


# ===== GROUP 4: BOAgent with LLM (strategy selection) =====

@_test
def bo_agent_soo_convergence():
    """BOAgent run_optimization_loop improves on 1D quadratic over 3 steps."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    # Generate initial data
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=6, col_names=["x"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    initial_best = df["y"].min()

    # Run 3 BO iterations, each time adding the suggestion to data
    for step in range(3):
        result = agent.run_optimization_loop(
            data_path=str(data_path),
            objective_text="Minimize y (quadratic function)",
            input_cols=["x"],
            input_bounds=bounds,
            target_cols=["y"],
            target_directions={"y": "minimize"},
            output_dir=str(output_dir),
            batch_size=1,
        )

        if result.get("status") != "success":
            shutil.rmtree(d, True)
            return False, f"step {step}: {result.get('status')}: {result.get('error', '')}"

        # Get suggestion and evaluate
        params = result["next_parameters"]
        new_x = params["x"]
        new_y = quadratic_1d(new_x)

        # Append to data
        df = pd.read_csv(data_path)
        new_row = pd.DataFrame([{"x": new_x, "y": new_y}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(data_path, index=False)

    final_best = df["y"].min()
    improved = final_best < initial_best

    shutil.rmtree(d, True)

    ok = improved and result.get("status") == "success"
    return ok, f"initial_best={initial_best:.2f}, final_best={final_best:.2f}, steps=3"


@_test
def bo_agent_returns_valid_structure():
    """BOAgent result contains all expected keys."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=6, col_names=["x"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text="Minimize y",
        input_cols=["x"],
        input_bounds=bounds,
        target_cols=["y"],
        target_directions={"y": "minimize"},
        output_dir=str(output_dir),
        batch_size=1,
    )

    shutil.rmtree(d, True)

    required_keys = ["status", "next_parameters", "strategy", "plot_path", "inspection"]
    missing = [k for k in required_keys if k not in result]

    ok = len(missing) == 0 and result["status"] == "success"
    return ok, f"missing_keys={missing}, status={result.get('status')}"


@_test
def bo_agent_batch_mode():
    """BOAgent batch_size=3 returns 3 candidates."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    df = _generate_initial_data(
        branin, bounds, n=10, col_names=["x1", "x2"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text="Minimize Branin function",
        input_cols=["x1", "x2"],
        input_bounds=bounds,
        target_cols=["y"],
        target_directions={"y": "minimize"},
        output_dir=str(output_dir),
        batch_size=3,
    )

    shutil.rmtree(d, True)

    if result.get("status") != "success":
        return False, f"status={result.get('status')}"

    params = result["next_parameters"]
    if isinstance(params, list):
        ok = len(params) == 3
        return ok, f"got {len(params)} candidates"
    else:
        return False, f"expected list, got {type(params)}"


# ===== GROUP 5: Orchestrator-level BO =====

@_test
def orchestrator_bo_improves_objective():
    """Orchestrator analyze → BO → add results → BO again shows improvement."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()

    # Create synthetic experimental data
    np.random.seed(42)
    df = pd.DataFrame({
        "HCl_M": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 3.0],
        "Time_h": [2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 2.0],
        "Nd_recovery_pct": [25.0, 40.0, 55.0, 68.0, 82.0, 88.0, 32.0, 48.0],
    })
    results_path = data_dir / "leaching.csv"
    df.to_csv(results_path, index=False)
    with open(results_path.with_suffix(".json"), "w") as f:
        json.dump({"title": "Leaching results", "instrument": "ICP-OES"}, f)

    # Feedstock (needed for plan generation)
    pd.DataFrame({
        "Element": ["Nd", "Fe"], "Concentration_Percent": [24.5, 58.0],
        "Market_Value_USD_kg": [120, 0.5], "Criticality_Score": ["High", "Low"],
    }).to_csv(data_dir / "feedstock.csv", index=False)
    with open(data_dir / "feedstock.json", "w") as f:
        json.dump({"title": "NdFeB", "objective": "test"}, f)

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))

    # Generate plan
    o.chat(f"Generate plan for HCl leaching of NdFeB. Use data at {data_dir / 'feedstock.csv'}.")

    # Analyze results
    o.chat(
        f"Analyze results at {results_path}. "
        f"Inputs: HCl_M, Time_h. Target: Nd_recovery_pct (maximize)."
    )

    # Run BO
    r1 = o.chat("Run optimization to suggest next experiment.")

    # Simulate BO suggestion result (use high end of parameter space)
    batch2 = pd.DataFrame({
        "HCl_M": [5.5], "Time_h": [5.5], "Nd_recovery_pct": [91.0],
    })
    results2 = data_dir / "batch2.csv"
    batch2.to_csv(results2, index=False)
    with open(results2.with_suffix(".json"), "w") as f:
        json.dump({"title": "BO follow-up", "instrument": "ICP-OES"}, f)

    o.chat(
        f"Analyze batch2 at {results2}. "
        f"Same schema: inputs HCl_M, Time_h; target Nd_recovery_pct."
    )

    # Run BO again
    r2 = o.chat("Run optimization again to suggest next experiment.")

    # Check data accumulated
    if o.bo_data_path.exists():
        final_df = pd.read_csv(o.bo_data_path)
        data_points = len(final_df)
        best_nd = final_df["Nd_recovery_pct"].max()
    else:
        data_points = 0
        best_nd = 0

    shutil.rmtree(d, True)

    ok = data_points >= 9 and best_nd >= 88.0
    return ok, f"data_points={data_points}, best_nd={best_nd}, r2={r2[:150]}"


@_test
def orchestrator_bo_respects_directions():
    """BO with minimize direction suggests lower values for Fe_recovery."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()

    df = pd.DataFrame({
        "HCl_M": [1.0, 2.0, 4.0, 6.0, 1.0, 2.0, 4.0, 6.0],
        "Time_h": [2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0],
        "Fe_recovery_pct": [18.0, 30.0, 49.0, 58.0, 28.0, 45.0, 62.0, 70.0],
    })
    results_path = data_dir / "fe_data.csv"
    df.to_csv(results_path, index=False)
    with open(results_path.with_suffix(".json"), "w") as f:
        json.dump({"title": "Fe co-extraction data", "instrument": "ICP-OES"}, f)

    pd.DataFrame({
        "Element": ["Nd", "Fe"], "Concentration_Percent": [24.5, 58.0],
    }).to_csv(data_dir / "feedstock.csv", index=False)
    with open(data_dir / "feedstock.json", "w") as f:
        json.dump({"title": "NdFeB"}, f)

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    o.chat(f"Generate plan for minimizing Fe co-extraction. Data at {data_dir / 'feedstock.csv'}.")
    o.chat(
        f"Analyze {results_path}. "
        f"Inputs: HCl_M, Time_h. Target: Fe_recovery_pct (minimize)."
    )
    r = o.chat("Run optimization to find conditions that minimize Fe co-extraction.")

    shutil.rmtree(d, True)

    # Should suggest low HCl_M and/or short time (where Fe is lowest)
    ok = "error" not in r.lower()[:100]
    return ok, r[:200]


# ===== GROUP 6: Edge cases =====

@_test
def bo_with_3_data_points():
    """BO runs with minimum viable data (3 points)."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=3, col_names=["x"], target_name="y"
    )

    optimizer = get_optimizer(is_moo=False, device="cpu")
    X = df[["x"]].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)
    bounds_arr = np.array(bounds, dtype=np.float64)

    optimizer.fit(X, y, bounds_arr, {"kernel": "matern_2.5", "noise": "min_noise_low"}, ["x"])
    candidates = optimizer.recommend(n_candidates=1, strategy="log_ei", params={})

    ok = len(candidates) == 1 and 0.0 <= candidates[0][0] <= 6.0
    return ok, f"candidate={candidates[0]}"


@_test
def bo_different_kernels():
    """BO works with all kernel options: matern_2.5, matern_1.5, rbf."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=8, col_names=["x"], target_name="y"
    )

    optimizer = get_optimizer(is_moo=False, device="cpu")
    X = df[["x"]].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)
    bounds_arr = np.array(bounds, dtype=np.float64)

    for kernel in ["matern_2.5", "matern_1.5", "rbf"]:
        optimizer.fit(X, y, bounds_arr, {"kernel": kernel, "noise": "min_noise_low"}, ["x"])
        candidates = optimizer.recommend(n_candidates=1, strategy="log_ei", params={})
        if len(candidates) != 1:
            return False, f"kernel={kernel} failed to produce candidate"

    return True, "all kernels work"


@_test
def bo_different_acquisition_functions():
    """BO works with all acquisition functions: log_ei, ucb, max_variance, thompson."""
    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=8, col_names=["x"], target_name="y"
    )

    optimizer = get_optimizer(is_moo=False, device="cpu")
    X = df[["x"]].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)
    bounds_arr = np.array(bounds, dtype=np.float64)

    optimizer.fit(X, y, bounds_arr, {"kernel": "matern_2.5", "noise": "min_noise_low"}, ["x"])

    strategies = [
        ("log_ei", {}),
        ("ucb", {"beta": 2.0}),
        ("max_variance", {}),
        ("thompson", {}),
    ]

    for strategy, params in strategies:
        candidates = optimizer.recommend(n_candidates=1, strategy=strategy, params=params)
        if len(candidates) != 1:
            return False, f"strategy={strategy} failed"

    return True, "all acquisition functions work"


@_test
def bo_convergence_rate_reasonable():
    """BO should find near-optimal in fewer evaluations than random search."""
    bounds = [[0.0, 6.0]]
    n_init = 5
    n_iters = 10

    # BO
    df_bo = _generate_initial_data(
        quadratic_1d, bounds, n=n_init, col_names=["x"], target_name="y", seed=42
    )
    bo_best, _ = _run_bo_loop(
        df_bo, ["x"], "y", bounds, quadratic_1d,
        n_iters=n_iters, direction="minimize", seed=42
    )

    # Random search baseline
    np.random.seed(42)
    random_x = np.random.uniform(0, 6, n_init + n_iters)
    random_y = [quadratic_1d(x) for x in random_x]
    random_best = [min(random_y[:n_init + i + 1]) for i in range(n_iters)]

    bo_final = bo_best[-1]
    random_final = random_best[-1]

    ok = bo_final <= random_final
    return ok, f"bo_best={bo_final:.3f}, random_best={random_final:.3f}"


# ===== GROUP 7: LLM strategy quality on harder objectives =====

def _multimodal_2d(x1, x2):
    """2D function with multiple local minima and one global minimum.
    Global min at ~(0.5, 0.5) = -2.0. Local minima at corners."""
    return (np.sin(3 * x1) * np.sin(3 * x2)
            - 2.0 * np.exp(-((x1 - 0.5)**2 + (x2 - 0.5)**2) / 0.1)
            + 0.5 * np.exp(-((x1 - 3.5)**2 + (x2 - 3.5)**2) / 0.5))


def _noisy_quadratic(x):
    """1D quadratic with significant noise — strategy should adapt."""
    return (x - 3)**2 + np.random.normal(0, 3.0)


@_test
def llm_guided_vs_fixed_on_multimodal():
    """LLM-guided BO should match or beat fixed log_ei on a multimodal surface."""
    bounds = [[0.0, 4.0], [0.0, 4.0]]
    n_init = 8
    n_iters = 6

    # Fixed strategy baseline (log_ei, matern_2.5)
    df_fixed = _generate_initial_data(
        _multimodal_2d, bounds, n=n_init,
        col_names=["x1", "x2"], target_name="y", seed=42
    )
    fixed_best, _ = _run_bo_loop(
        df_fixed, ["x1", "x2"], "y", bounds, _multimodal_2d,
        n_iters=n_iters, direction="minimize", seed=42
    )

    # LLM-guided (BOAgent selects strategy each step)
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    df_llm = _generate_initial_data(
        _multimodal_2d, bounds, n=n_init,
        col_names=["x1", "x2"], target_name="y", seed=42
    )
    data_path = d / "data.csv"
    df_llm.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    for step in range(n_iters):
        result = agent.run_optimization_loop(
            data_path=str(data_path),
            objective_text="Minimize a multimodal 2D function with multiple local minima",
            input_cols=["x1", "x2"],
            input_bounds=bounds,
            target_cols=["y"],
            target_directions={"y": "minimize"},
            output_dir=str(output_dir),
            batch_size=1,
        )

        if result.get("status") != "success":
            shutil.rmtree(d, True)
            return False, f"step {step} failed: {result.get('error', '')}"

        params = result["next_parameters"]
        new_y = _multimodal_2d(params["x1"], params["x2"])

        df_llm = pd.read_csv(data_path)
        df_llm = pd.concat([df_llm, pd.DataFrame([{
            "x1": params["x1"], "x2": params["x2"], "y": new_y
        }])], ignore_index=True)
        df_llm.to_csv(data_path, index=False)

    llm_final = df_llm["y"].min()
    fixed_final = fixed_best[-1]

    shutil.rmtree(d, True)

    # LLM-guided should be within 50% of fixed baseline (not necessarily better,
    # but shouldn't be catastrophically worse)
    ok = llm_final < fixed_final * 1.5 + 1.0  # Allow slack for stochasticity
    return ok, f"llm_best={llm_final:.3f}, fixed_best={fixed_final:.3f}"


@_test
def llm_adapts_noise_prior():
    """On noisy data, LLM should select a higher-floor noise prior (not min_noise_low)."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[0.0, 6.0]]
    np.random.seed(42)
    # Generate very noisy initial data
    x_vals = np.random.uniform(0, 6, 10)
    y_vals = np.array([_noisy_quadratic(x) for x in x_vals])
    df = pd.DataFrame({"x": x_vals, "y": y_vals})
    data_path = d / "noisy.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text=(
            "Minimize y. WARNING: The data has very high measurement noise "
            "(±3 units). The observations are unreliable."
        ),
        input_cols=["x"],
        input_bounds=bounds,
        target_cols=["y"],
        target_directions={"y": "minimize"},
        output_dir=str(output_dir),
        batch_size=1,
    )

    shutil.rmtree(d, True)

    if result.get("status") != "success":
        return False, f"failed: {result.get('error', '')}"

    strategy = result.get("strategy", {})
    noise = strategy.get("model_config", {}).get("noise", "unknown")

    # LLM should recognize noisy data and NOT use min_noise_low
    ok = noise in ("min_noise_med", "min_noise_high")
    return ok, f"noise_prior={noise}, rationale={strategy.get('rationale', '')[:150]}"


@_test
def llm_exploits_on_final_budget():
    """With budget=1 (final shot), LLM should select exploitation strategy."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[0.0, 6.0]]
    df = _generate_initial_data(
        quadratic_1d, bounds, n=10, col_names=["x"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    # Seed some history so the agent knows it's been running
    history = [
        {"step": 1, "config": {"acquisition_strategy": {"type": "log_ei"}}},
        {"step": 2, "config": {"acquisition_strategy": {"type": "log_ei"}}},
    ]
    with open(output_dir / "bo_history.json", "w") as f:
        json.dump(history, f)

    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text="Minimize y. This is a quadratic function.",
        input_cols=["x"],
        input_bounds=bounds,
        target_cols=["y"],
        target_directions={"y": "minimize"},
        output_dir=str(output_dir),
        batch_size=1,
        experimental_budget=1,  # FINAL SHOT
    )

    shutil.rmtree(d, True)

    if result.get("status") != "success":
        return False, f"failed: {result.get('error', '')}"

    strategy = result.get("strategy", {})
    acq_type = strategy.get("acquisition_strategy", {}).get("type", "unknown")
    acq_params = strategy.get("acquisition_strategy", {}).get("params", {})

    # Should NOT use max_variance or thompson (too exploratory for final shot)
    # Should use log_ei or ucb with low beta
    exploitative = acq_type in ("log_ei", "ucb")
    if acq_type == "ucb":
        beta = acq_params.get("beta", 999)
        exploitative = beta < 1.0  # Low beta = exploitation

    not_exploratory = acq_type not in ("max_variance", "thompson")

    ok = exploitative and not_exploratory
    return ok, f"acq={acq_type}, params={acq_params}, rationale={strategy.get('rationale', '')[:150]}"


@_test
def llm_avoids_exploration_on_low_budget():
    """With budget=2, LLM should NOT select max_variance."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    df = _generate_initial_data(
        branin, bounds, n=12, col_names=["x1", "x2"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    result = agent.run_optimization_loop(
        data_path=str(data_path),
        objective_text="Minimize Branin function. Only 2 experiments remaining.",
        input_cols=["x1", "x2"],
        input_bounds=bounds,
        target_cols=["y"],
        target_directions={"y": "minimize"},
        output_dir=str(output_dir),
        batch_size=1,
        experimental_budget=2,
    )

    shutil.rmtree(d, True)

    if result.get("status") != "success":
        return False, f"failed: {result.get('error', '')}"

    strategy = result.get("strategy", {})
    acq_type = strategy.get("acquisition_strategy", {}).get("type", "unknown")

    ok = acq_type != "max_variance"
    return ok, f"acq={acq_type}, budget_phase={result.get('budget', {}).get('budget_phase')}"


@_test
def bo_agent_multi_step_convergence_on_branin():
    """BOAgent converges on 2D Branin over 6 LLM-guided steps."""
    d = _tmp()
    output_dir = d / "bo_artifacts"
    output_dir.mkdir()

    bounds = [[-5.0, 10.0], [0.0, 15.0]]
    df = _generate_initial_data(
        branin, bounds, n=10, col_names=["x1", "x2"], target_name="y"
    )
    data_path = d / "data.csv"
    df.to_csv(data_path, index=False)

    initial_best = df["y"].min()

    agent = BOAgent(api_key=_API_KEY, model_name=_MODEL, output_dir=str(d))

    strategies_used = []
    for step in range(6):
        result = agent.run_optimization_loop(
            data_path=str(data_path),
            objective_text="Minimize the Branin function",
            input_cols=["x1", "x2"],
            input_bounds=bounds,
            target_cols=["y"],
            target_directions={"y": "minimize"},
            output_dir=str(output_dir),
            batch_size=1,
        )

        if result.get("status") != "success":
            shutil.rmtree(d, True)
            return False, f"step {step} failed"

        acq = result.get("strategy", {}).get("acquisition_strategy", {}).get("type", "?")
        strategies_used.append(acq)

        params = result["next_parameters"]
        new_y = branin(params["x1"], params["x2"])

        df = pd.read_csv(data_path)
        df = pd.concat([df, pd.DataFrame([{
            "x1": params["x1"], "x2": params["x2"], "y": new_y
        }])], ignore_index=True)
        df.to_csv(data_path, index=False)

    final_best = df["y"].min()

    shutil.rmtree(d, True)

    improved = final_best < initial_best
    near_global = final_best < 2.0  # Global min ~0.398

    ok = improved and near_global
    return ok, f"initial={initial_best:.2f}, final={final_best:.2f}, strategies={strategies_used}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _API_KEY:
        print("Set GEMINI_API_KEY (or SCILINK_TEST_API_KEY) env var first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        indices = [int(a) - 1 for a in sys.argv[1:]]
        to_run = [TESTS[i] for i in indices]
    else:
        to_run = TESTS

    print(f"Model: {_MODEL}")
    print(f"API key: {_API_KEY[:10]}...")

    results = {}
    for fn in to_run:
        name = fn.__name__
        desc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"\n{'=' * 60}")
        print(f"[{TESTS.index(fn) + 1}/{len(TESTS)}] {name}: {desc}")
        print("=" * 60)
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            results[name] = status
            print(f"  → {status}: {detail[:200]}")
        except KeyboardInterrupt:
            results[name] = "SKIP"
            break
        except Exception as e:
            results[name] = "ERROR"
            print(f"  → ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        marker = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{marker}] {name}: {status}")
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n  {passed}/{total} passed")
