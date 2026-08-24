"""Offline tests for skill-contributed optimization engine components (issue
#196): an in-package skill bundle's .py helper can add a custom SURROGATE and a
custom ACQUISITION, discovered like TOOL_SPEC, gated by skill activation.

Proves the contributor rung end-to-end: SAAS surrogate (bo_saas bundle) + PI
acquisition (bo_prob_improvement bundle) are built/dispatched by the engine
without any edit to bo_tools.py.
"""
import numpy as np

from scilink.skills._shared._opt_components import (
    get_surrogate_components, get_acquisition_components, smoke_test_surrogate,
)
from scilink.agents.planning_agents.bo_tools import get_optimizer


def test_discovery_and_gating():
    assert "deep_ensemble" in get_surrogate_components(["bo_deep_ensemble"])
    assert "changeover_ei" in get_acquisition_components(["bo_changeover"])
    assert get_surrogate_components([]) == {}  # nothing without an active skill
    assert get_acquisition_components([]) == {}
    # bundle-gated: deep_ensemble not visible under a different active skill
    assert "deep_ensemble" not in get_surrogate_components(["bo_changeover"])


def test_saas_is_core_not_skill():
    # SAAS is BoTorch -> core baseline toolkit, NOT a skill component.
    from scilink.agents.planning_agents.bo_tools import ALLOWED_SURROGATES
    assert "saas" in ALLOWED_SURROGATES
    assert get_surrogate_components(["bo_saas"]) == {}  # bundle removed


def test_deep_ensemble_surrogate_smoke():
    comp = get_surrogate_components(["bo_deep_ensemble"])["deep_ensemble"]
    smoke_test_surrogate(comp, input_dim=4)  # raises if no usable posterior


def test_engine_builds_skill_surrogate_and_dispatches_skill_acquisition():
    surr = get_surrogate_components(["bo_deep_ensemble"])
    acq = get_acquisition_components(["bo_changeover"])
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, (24, 4))
    y = ((X[:, 0] - 0.5) ** 2 + 0.3 * X[:, 1]).reshape(-1, 1)
    opt = get_optimizer(is_moo=False, device="cpu")
    # surrogate 'deep_ensemble' lives in the bundle .py, not bo_tools.py
    opt.fit(X, -y, bounds=[(0, 1)] * 4,
            model_config={"surrogate": "deep_ensemble", "kernel": "matern_2.5", "noise": "min_noise_low"},
            surrogate_components=surr, acquisition_components=acq)
    # composes with a CORE analytic acquisition (Gaussian posterior) ...
    assert opt.recommend(n_candidates=1, strategy="log_ei").shape == (1, 4)
    # ... and with the skill-shipped acquisition 'changeover_ei'
    rec = opt.recommend(n_candidates=1, strategy="changeover_ei")
    assert rec.shape == (1, 4)
    assert np.all(rec >= -1e-6) and np.all(rec <= 1 + 1e-6)


def test_changeover_penalty_keeps_expensive_dim_near_last_setpoint():
    """A heavy changeover_weight on the expensive dim should keep the next
    recommendation's expensive input near the last experiment, even when the EI
    optimum is far away — the whole point of the acquisition."""
    acq = get_acquisition_components(["bo_changeover"])
    # Optimum (low y) is at x0 ~ 0.9, but the last experiment sits at x0 ~ 0.1.
    grid = np.linspace(0, 1, 25)
    X = np.column_stack([grid, np.full_like(grid, 0.5)])
    X = np.vstack([X, [0.1, 0.5]])              # last row = current setpoint at x0=0.1
    y = ((X[:, 0] - 0.9) ** 2).reshape(-1, 1)   # best near x0=0.9
    opt = get_optimizer(is_moo=False, device="cpu")
    opt.fit(X, -y, bounds=[(0, 1), (0, 1)],
            model_config={"surrogate": "single_task", "kernel": "matern_2.5", "noise": "min_noise_low"},
            acquisition_components=acq)
    far = opt.recommend(1, "changeover_ei", params={"expensive_dims": [0], "changeover_weight": 0.0})[0]
    near = opt.recommend(1, "changeover_ei", params={"expensive_dims": [0], "changeover_weight": 50.0})[0]
    # heavy penalty pulls x0 back toward the last setpoint (0.1); no penalty heads to ~0.9
    assert abs(near[0] - 0.1) < abs(far[0] - 0.1)
    assert near[0] < far[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all passed")
