"""Offline tests for the three BO-agent robustness/capability fixes:

1. Likelihood noise initialization is well-conditioned (the constraint floor
   is unchanged, but the fit no longer STARTS at a near-singular point —
   regression fixture: the crossed-barrel benchmark init that failed 30/30).
2. A surrogate-fit failure re-enters the strategy stage with the failure in
   context instead of erroring the whole loop.
3. recommend(candidates=...) selects from a discrete design library, and
   cat_dims are optimized over observed levels (optimize_acqf_mixed), not a
   continuous relaxation.

No network, no API key.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scilink.agents.planning_agents.bo_tools import (
    ALLOWED_NOISE_PRIORS,
    build_likelihood,
    get_optimizer,
)

# The exact 5-point crossed-barrel init on which every fit attempt failed
# before the noise-init fix (deterministically, across torch seeds).
_CB_X = np.array([
    [10.0, 200.0, 1.9, 0.70],
    [10.0, 50.0, 2.3, 1.05],
    [12.0, 50.0, 1.7, 0.70],
    [6.0, 200.0, 1.7, 1.05],
    [12.0, 25.0, 2.4, 1.05],
])
_CB_Y = np.array([26.12, 24.40, 3.82, 24.32, 4.36]).reshape(-1, 1)
_CB_BOUNDS = [[6.0, 12.0], [0.0, 200.0], [1.5, 2.5], [0.7, 1.4]]
_CFG = {"surrogate": "single_task", "kernel": "matern_2.5", "noise": "min_noise_low"}


def test_noise_menu_unchanged():
    # The gate-pinned initialization (floor * 2) and the floors themselves are
    # deliberately untouched — the healing happens only in the fit-rescue path,
    # so successful fits stay byte-identical to the frozen classical baseline.
    for key, cfg in ALLOWED_NOISE_PRIORS.items():
        lik = build_likelihood(key)
        floor = cfg["min_noise"]
        assert lik.noise_covar.raw_noise_constraint.lower_bound.item() == pytest.approx(floor)
        assert lik.noise.item() == pytest.approx(floor * 2.0, rel=1e-3)


def test_fit_succeeds_on_failing_benchmark_init():
    for seed in range(5):
        torch.manual_seed(seed)
        opt = get_optimizer(is_moo=False, device="cpu")
        opt.fit(_CB_X, _CB_Y, np.array(_CB_BOUNDS), dict(_CFG), None)
        assert opt.model is not None


def test_fit_rescue_stress_no_failures():
    """Random small crossed-barrel-scaled subsets: the rescue restarts must
    absorb every ModelFittingError (pre-fix the classical benchmark lane hit
    dozens across such subsets)."""
    rng = np.random.RandomState(7)
    lo = np.array([b[0] for b in _CB_BOUNDS])
    hi = np.array([b[1] for b in _CB_BOUNDS])
    for trial in range(20):
        torch.manual_seed(trial)
        n = rng.randint(5, 9)
        X = lo + (hi - lo) * rng.rand(n, 4)
        # Sharp bimodal targets like the toughness data (small-n hostile fits)
        y = np.where(rng.rand(n) > 0.5, 25.0, 4.0).reshape(-1, 1)
        y += rng.randn(n, 1) * 0.2
        opt = get_optimizer(is_moo=False, device="cpu")
        opt.fit(X, y, np.array(_CB_BOUNDS), dict(_CFG), None)
        assert opt.model is not None


def _toy_optimizer(n=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = (-((X[:, 0] - 0.3) ** 2) - (X[:, 1] - 0.7) ** 2).reshape(-1, 1)
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, y, np.array([[0.0, 1.0], [0.0, 1.0]]), dict(_CFG), ["a", "b"])
    return opt


def _grid_pool(k=6):
    g = np.linspace(0.0, 1.0, k)
    return np.array([[a, b] for a in g for b in g])


@pytest.mark.parametrize("strategy", ["log_ei", "ucb", "max_variance", "thompson"])
def test_candidates_pool_selection(strategy):
    torch.manual_seed(0)
    opt = _toy_optimizer()
    pool = _grid_pool()
    picked = opt.recommend(n_candidates=3, strategy=strategy, params={}, candidates=pool)
    assert picked.shape == (3, 2)
    pool_set = {tuple(row) for row in pool}
    rows = [tuple(row) for row in picked]
    assert all(r in pool_set for r in rows)
    assert len(set(rows)) == 3  # distinct picks


def test_candidates_argmax_consistency():
    torch.manual_seed(0)
    opt = _toy_optimizer()
    pool = _grid_pool()
    picked = opt.recommend(n_candidates=1, strategy="log_ei", params={}, candidates=pool)
    scores = opt.evaluate_acquisition(pool)
    assert tuple(picked[0]) == tuple(pool[int(np.argmax(scores))])


def test_candidates_pool_smaller_than_batch_returns_pool():
    torch.manual_seed(0)
    opt = _toy_optimizer()
    pool = _grid_pool()[:2]
    picked = opt.recommend(n_candidates=5, strategy="log_ei", params={}, candidates=pool)
    assert picked.shape == (2, 2)


def test_moo_rejects_candidates():
    opt = get_optimizer(is_moo=True, device="cpu")
    with pytest.raises(ValueError, match="single-objective"):
        opt.recommend(candidates=_grid_pool())


def test_mixed_cat_dims_recommend_snaps_to_levels():
    torch.manual_seed(0)
    rng = np.random.RandomState(1)
    cat = rng.choice([0.0, 1.0, 2.0], size=15)
    x1 = rng.rand(15)
    X = np.column_stack([cat, x1])
    y = (cat - (x1 - 0.5) ** 2).reshape(-1, 1)
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, y, np.array([[0.0, 2.0], [0.0, 1.0]]),
            {"surrogate": "mixed", "kernel": "matern_2.5", "noise": "min_noise_med"},
            ["cat", "x1"], cat_dims=[0])
    picked = opt.recommend(n_candidates=1, strategy="log_ei", params={})
    # The categorical dim must land exactly on an observed level, not between.
    assert picked[0, 0] in {0.0, 1.0, 2.0}


# --------------------------------------------------------------------------- #
#  Agent-level tests (stubbed LLM, full pipeline offline)
# --------------------------------------------------------------------------- #

_STRATEGY_JSON = json.dumps({
    "model_config": {"surrogate": "single_task", "kernel": "matern_2.5",
                     "noise": "min_noise_med", "input_transform": "none"},
    "acquisition_strategy": {"type": "log_ei", "params": {}},
    "rationale": "offline stub",
})
_INSPECTION_JSON = json.dumps({"status": "ok", "reason": "stub", "suggested_adjustments": {}})


class _StubModel:
    """Returns a valid strategy config for text-only calls and a valid
    inspection verdict for image calls; records the strategy prompts."""

    def __init__(self):
        self.strategy_prompts = []

    def generate_content(self, parts, generation_config=None):
        if all(isinstance(p, str) for p in parts):
            self.strategy_prompts.append("\n".join(parts))
            return SimpleNamespace(text=_STRATEGY_JSON)
        return SimpleNamespace(text=_INSPECTION_JSON)


def _make_agent(tmp, stub):
    from scilink.agents.planning_agents import OptimizationAgent
    agent = OptimizationAgent(api_key="test", model_name="claude-opus-4-6",
                              output_dir=str(tmp))
    agent.model = stub
    return agent


def _write_toy_csv(path, n=8, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = -((X[:, 0] - 0.3) ** 2) - (X[:, 1] - 0.7) ** 2
    pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "y": y}).to_csv(path, index=False)


def test_agent_candidate_pool_end_to_end(tmp_path):
    torch.manual_seed(0)
    data = tmp_path / "data.csv"
    _write_toy_csv(data)
    stub = _StubModel()
    agent = _make_agent(tmp_path, stub)
    pool = _grid_pool()
    res = agent.run_optimization_loop(
        data_path=str(data), objective_text="maximize y",
        input_cols=["a", "b"], input_bounds=[[0, 1], [0, 1]],
        target_cols=["y"], output_dir=str(tmp_path),
        save_acq=False, plot_acq=False,
        candidate_pool=pool,
    )
    assert res.get("status") == "success", res
    picked = (res["next_parameters"]["a"], res["next_parameters"]["b"])
    assert picked in {tuple(row) for row in pool}
    assert res["candidate_pool"]["provided"] == len(pool)
    assert res["candidate_pool"]["unmeasured"] == len(pool)


def test_fit_failure_reenters_strategy_with_context(tmp_path, monkeypatch):
    torch.manual_seed(0)
    from scilink.agents.planning_agents import bo_agent as bo_agent_mod

    real_get_optimizer = bo_agent_mod.get_optimizer
    calls = {"n": 0}

    class _FlakyOptimizer:
        def __init__(self, real):
            self._real = real

        def fit(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic fit failure")
            return self._real.fit(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(
        bo_agent_mod, "get_optimizer",
        lambda is_moo, device="cpu": _FlakyOptimizer(real_get_optimizer(is_moo, device)),
    )

    data = tmp_path / "data.csv"
    _write_toy_csv(data)
    stub = _StubModel()
    agent = _make_agent(tmp_path, stub)
    res = agent.run_optimization_loop(
        data_path=str(data), objective_text="maximize y",
        input_cols=["a", "b"], input_bounds=[[0, 1], [0, 1]],
        target_cols=["y"], output_dir=str(tmp_path),
        save_acq=False, plot_acq=False,
    )
    assert res.get("status") == "success", res
    assert calls["n"] == 2  # first fit failed, second succeeded
    assert len(stub.strategy_prompts) == 2
    assert "FAILED ATTEMPT THIS STEP" not in stub.strategy_prompts[0]
    assert "FAILED ATTEMPT THIS STEP" in stub.strategy_prompts[1]
    assert "synthetic fit failure" in stub.strategy_prompts[1]


def test_fit_failure_exhaustion_returns_error(tmp_path, monkeypatch):
    from scilink.agents.planning_agents import bo_agent as bo_agent_mod

    class _AlwaysFailing:
        def fit(self, *args, **kwargs):
            raise RuntimeError("always fails")

    monkeypatch.setattr(bo_agent_mod, "get_optimizer",
                        lambda is_moo, device="cpu": _AlwaysFailing())

    data = tmp_path / "data.csv"
    _write_toy_csv(data)
    stub = _StubModel()
    agent = _make_agent(tmp_path, stub)
    res = agent.run_optimization_loop(
        data_path=str(data), objective_text="maximize y",
        input_cols=["a", "b"], input_bounds=[[0, 1], [0, 1]],
        target_cols=["y"], output_dir=str(tmp_path),
        save_acq=False, plot_acq=False,
    )
    assert "error" in res
    assert "3 strategy attempts" in res["error"]
    assert len(stub.strategy_prompts) == 3
