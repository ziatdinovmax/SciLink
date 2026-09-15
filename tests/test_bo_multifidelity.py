"""Offline tests for multi-fidelity as a BASELINE OptimizationAgent capability
(issue #196) — surfaced by the `fidelity_config` data signal, like `mixed` is by
`cat_dims`. Not a skill.

Covers: the core surrogate + mf_kg acquisition build/fit/recommend, and the
data-signal gating in _validate_config + the prompt addendum.
"""
import tempfile

import numpy as np
import pandas as pd

from scilink.agents.planning_agents.bo_tools import get_optimizer, ALLOWED_SURROGATES
from scilink.agents.planning_agents import OptimizationAgent


def test_mf_surrogate_registered():
    assert "single_task_multi_fidelity" in ALLOWED_SURROGATES


def test_mf_build_fit_recommend():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.uniform(-2, 2, 30), rng.uniform(-2, 2, 30),
                         rng.choice([0.0, 1.0], 30)])
    y = np.array([-(np.sin(3 * r[0]) + r[1] ** 2) - 0.4 * (1 - r[2]) for r in X]).reshape(-1, 1)
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, -y, bounds=[(-2, 2), (-2, 2), (0.0, 1.0)],
            model_config={"surrogate": "single_task_multi_fidelity",
                          "kernel": "matern_2.5", "noise": "min_noise_low"},
            fidelity_config={"fidelity_col": 2, "target_fidelity": 1.0})
    rec = opt.recommend(n_candidates=1, strategy="mf_kg")
    assert rec.shape == (1, 3)
    # fidelity snapped to an observed discrete level
    assert round(float(rec[0][2]), 3) in (0.0, 1.0)


def _agent():
    return OptimizationAgent(api_key="test", model_name="claude-opus-4-6",
                             output_dir=tempfile.mkdtemp())


def test_validate_gates_mf_on_fidelity_signal():
    ag = _agent()
    cfg = {"model_config": {"surrogate": "single_task_multi_fidelity"},
           "acquisition_strategy": {"type": "mf_kg"}}
    # No fidelity declared -> MF surrogate + mf_kg both rejected (defaulted).
    out = ag._validate_config({k: dict(v) for k, v in cfg.items()}, fidelity_declared=False)
    assert out["model_config"]["surrogate"] == "single_task"
    assert out["acquisition_strategy"]["type"] == "log_ei"
    # Fidelity declared -> both accepted.
    out2 = ag._validate_config({k: dict(v) for k, v in cfg.items()}, fidelity_declared=True)
    assert out2["model_config"]["surrogate"] == "single_task_multi_fidelity"
    assert out2["acquisition_strategy"]["type"] == "mf_kg"


def test_mf_kg_requires_mf_surrogate():
    ag = _agent()
    # mf_kg with a non-MF surrogate, even with fidelity declared -> rejected.
    out = ag._validate_config(
        {"model_config": {"surrogate": "single_task"},
         "acquisition_strategy": {"type": "mf_kg"}}, fidelity_declared=True)
    assert out["acquisition_strategy"]["type"] == "log_ei"


def test_addendum_gated_on_fidelity():
    ag = _agent()
    df = pd.DataFrame({"x": [1.0, 2], "fid": [0.0, 1.0], "y": [3.0, 4]})
    base = dict(is_moo=False, objective_text="o", target_directions=None,
                target_cols=["y"], batch_size=1, trend_context="No history.", df=df,
                budget_ctx={"budget_guidance": "g", "steps_completed": 0, "budget_phase": "unlimited"},
                input_cols=["x", "fid"], cat_dims=None, physical_constraints=None,
                strategy_hint=None)
    assert "MULTI-FIDELITY" not in ag._build_strategy_prompt(**base)[0]
    assert "MULTI-FIDELITY" in ag._build_strategy_prompt(**base, fidelity_config={"fidelity_col": 1})[0]


def test_fidelity_costs_shift_chosen_fidelity():
    """Declared per-fidelity costs must actually move MFKG's fidelity choice
    (regression: the costs were extracted and plumbed but previously ignored by
    the cost model). Paired: one fitted model, identical seeds, only cost differs."""
    import torch
    rng = np.random.RandomState(123)
    n = 40
    x = rng.uniform(0, 1, n)
    fid = rng.choice([0.0, 1.0], n)
    y = -((x - 0.6) ** 2) - 0.3 * (1 - fid)
    X = np.column_stack([x, fid])
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, y.reshape(-1, 1), bounds=[(0, 1), (0.0, 1.0)],
            model_config={"surrogate": "single_task_multi_fidelity",
                          "kernel": "matern_2.5", "noise": "min_noise_low"},
            fidelity_config={"fidelity_col": 1, "target_fidelity": 1.0})

    def low_fidelity_rate(costs):
        opt.fidelity_config["fidelity_costs"] = costs
        picks = []
        for s in range(4):
            np.random.seed(s)
            torch.manual_seed(s)
            picks.append(round(float(opt.recommend(1, "mf_kg")[0][1]), 3))
        return sum(p == 0.0 for p in picks) / len(picks)

    # High fidelity expensive -> lean to the cheap low fidelity; flip the costs,
    # flip the preference. If costs were ignored these two rates would coincide.
    assert low_fidelity_rate({0: 1, 1: 1000}) > low_fidelity_rate({0: 1000, 1: 1})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all passed")
