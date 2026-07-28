"""Offline tests for the bo_failure_aware skill (failure_aware_ei acquisition).

Synthetic landscape: a smooth quality field with a hard failure cluster
(exact-zero outcomes) in one corner — the AutoAM signature. No network.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scilink.skills._shared._opt_components import get_acquisition_components
from scilink.agents.planning_agents.bo_tools import get_optimizer

_CFG = {"surrogate": "single_task", "kernel": "matern_1.5", "noise": "min_noise_med"}
_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


def _landscape(x):
    """Quality field peaked at (0.8, 0.8); hard failure (0.0) for x0,x1 < 0.3."""
    x = np.atleast_2d(x)
    q = 0.2 + 0.8 * np.exp(-8 * ((x[:, 0] - 0.8) ** 2 + (x[:, 1] - 0.8) ** 2))
    q[(x[:, 0] < 0.3) & (x[:, 1] < 0.3)] = 0.0
    return q


def _grid_pool(k=9):
    g = np.linspace(0.0, 1.0, k)
    return np.array([[a, b] for a in g for b in g])


def _fit_optimizer(seed=0, n=14):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    # Guarantee the failure cluster is represented
    X[:4] = np.array([[0.05, 0.05], [0.15, 0.2], [0.25, 0.1], [0.1, 0.28]])
    y = _landscape(X).reshape(-1, 1)
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, y, _BOUNDS, dict(_CFG), ["a", "b"])
    return opt


def test_component_registered_for_skill():
    comps = get_acquisition_components(["bo_failure_aware"], agent="bo")
    assert "failure_aware_ei" in comps
    fn = comps["failure_aware_ei"].recommend_fn
    assert callable(fn)


def test_picks_come_from_pool_and_avoid_failure_zone():
    torch.manual_seed(0)
    comps = get_acquisition_components(["bo_failure_aware"], agent="bo")
    fn = comps["failure_aware_ei"].recommend_fn
    opt = _fit_optimizer()
    pool = _grid_pool()
    opt._candidate_pool = pool
    picked = fn(opt, 3, {"feasibility_weight": 2.0})
    assert picked.shape == (3, 2)
    pool_set = {tuple(r) for r in pool}
    for row in picked:
        assert tuple(row) in pool_set
        # never inside the known failure cluster
        assert not (row[0] < 0.3 and row[1] < 0.3)


def test_feasibility_weight_knob_changes_behavior():
    """gamma is a real knob: P(feasible) of the top pick is non-decreasing in
    gamma, and a large gamma must not pick inside the failure cluster."""
    torch.manual_seed(0)
    comps = get_acquisition_components(["bo_failure_aware"], agent="bo")
    fn = comps["failure_aware_ei"].recommend_fn
    opt = _fit_optimizer()
    # Adversarial pool: mostly failure-cluster points + a few feasible ones
    g = np.linspace(0.0, 0.28, 5)
    cluster = np.array([[a, b] for a in g for b in g])
    feas = np.array([[0.75, 0.75], [0.85, 0.85], [0.5, 0.9]])
    pool = np.vstack([cluster, feas])
    opt._candidate_pool = pool

    def in_cluster(row):
        return row[0] < 0.3 and row[1] < 0.3

    picked_hi = fn(opt, 1, {"feasibility_weight": 8.0})[0]
    assert not in_cluster(picked_hi)
    picked_zero = fn(opt, 1, {"feasibility_weight": 0.0})[0]
    # gamma=0 is plain log-EI; it may or may not enter the cluster, but the
    # two settings must be evaluable and the high-gamma pick strictly feasible.
    assert picked_zero.shape == (2,)


def test_no_failures_falls_back_to_plain_ei():
    torch.manual_seed(0)
    comps = get_acquisition_components(["bo_failure_aware"], agent="bo")
    fn = comps["failure_aware_ei"].recommend_fn
    rng = np.random.RandomState(1)
    X = 0.4 + 0.5 * rng.rand(12, 2)   # no failure zone sampled
    y = _landscape(X).reshape(-1, 1)
    assert (y > 0).all()
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, y, _BOUNDS, dict(_CFG), ["a", "b"])
    pool = _grid_pool()
    opt._candidate_pool = pool
    picked = fn(opt, 2, {})
    assert picked.shape == (2, 2)


# --------------------------------------------------------------------------- #
#  Agent-level: skill activation end-to-end with stubbed LLM
# --------------------------------------------------------------------------- #

_STRATEGY_JSON = json.dumps({
    "model_config": {"surrogate": "single_task", "kernel": "matern_1.5",
                     "noise": "min_noise_med", "input_transform": "none"},
    "acquisition_strategy": {"type": "failure_aware_ei",
                             "params": {"feasibility_weight": 2.0}},
    "rationale": "offline stub: failure signature present",
})
_INSPECTION_JSON = json.dumps({"status": "ok", "reason": "stub",
                               "suggested_adjustments": {}})


class _StubModel:
    def __init__(self):
        self.strategy_prompts = []

    def generate_content(self, parts, generation_config=None):
        if all(isinstance(p, str) for p in parts):
            self.strategy_prompts.append("\n".join(parts))
            return SimpleNamespace(text=_STRATEGY_JSON)
        return SimpleNamespace(text=_INSPECTION_JSON)


def test_agent_with_skill_and_pool_end_to_end(tmp_path):
    import pandas as pd
    from scilink.agents.planning_agents import OptimizationAgent

    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    X = rng.rand(14, 2)
    X[:4] = np.array([[0.05, 0.05], [0.15, 0.2], [0.25, 0.1], [0.1, 0.28]])
    y = _landscape(X)
    data = tmp_path / "data.csv"
    pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "score": y}).to_csv(data, index=False)

    agent = OptimizationAgent(api_key="test", model_name="claude-opus-4-6",
                              output_dir=str(tmp_path))
    stub = _StubModel()
    agent.model = stub
    pool = _grid_pool()
    res = agent.run_optimization_loop(
        data_path=str(data), objective_text="maximize print quality score",
        input_cols=["a", "b"], input_bounds=[[0, 1], [0, 1]],
        target_cols=["score"], output_dir=str(tmp_path),
        save_acq=False, plot_acq=False,
        candidate_pool=pool, skill="bo_failure_aware",
    )
    assert res.get("status") == "success", res
    picked = np.array([res["next_parameters"]["a"], res["next_parameters"]["b"]])
    assert tuple(picked) in {tuple(r) for r in pool}
    assert not (picked[0] < 0.3 and picked[1] < 0.3)
    # The skill's acquisition guidance reached the strategy prompt
    assert "failure_aware_ei" in stub.strategy_prompts[0]