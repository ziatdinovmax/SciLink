"""SingleObjectiveOptimizer.recommend on an ALL-categorical design (mixed
surrogate, every dim a fixed feature) must return shape (q, d): botorch's
optimize_acqf_mixed squeezes to (d,) in that case, which broke BOAgent's
row iteration for q=1 ("'numpy.float64' object is not iterable") — every
step of an all-categorical campaign in continuous mode fell back to random.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _fit_all_cat(n=5, seed=0):
    from scilink.agents.planning_agents.bo_tools import get_optimizer
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, 4, n), rng.integers(0, 22, n),
                         rng.integers(0, 3, n), rng.integers(0, 15, n)]).astype(float)
    y = rng.uniform(0, 100, n)
    opt = get_optimizer(is_moo=False)
    opt.fit(X, y, bounds=[(0, 3), (0, 21), (0, 2), (0, 14)],
            model_config={"kernel": "matern_2.5", "noise": "min_noise_low",
                          "surrogate": "mixed"},
            feature_names=["a", "b", "c", "d"], cat_dims=[0, 1, 2, 3])
    return opt


def test_all_categorical_q1_is_2d():
    opt = _fit_all_cat()
    r = np.asarray(opt.recommend(n_candidates=1, strategy="log_ei"))
    assert r.shape == (1, 4), r.shape


def test_all_categorical_q3_is_2d():
    opt = _fit_all_cat(n=8, seed=1)
    r = np.asarray(opt.recommend(n_candidates=3, strategy="log_ei"))
    assert r.shape[1] == 4 and r.ndim == 2


def test_mixed_with_continuous_dim_unchanged():
    from scilink.agents.planning_agents.bo_tools import get_optimizer
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 3, 12), rng.uniform(0, 1, 12)]).astype(float)
    y = X[:, 1] + rng.normal(0, 0.05, 12)
    opt = get_optimizer(is_moo=False)
    opt.fit(X, y, bounds=[(0, 2), (0, 1)],
            model_config={"kernel": "matern_2.5", "noise": "min_noise_low",
                          "surrogate": "mixed"}, feature_names=["a", "b"], cat_dims=[0])
    assert np.asarray(opt.recommend(n_candidates=1, strategy="log_ei")).shape == (1, 2)


def test_all_categorical_never_recommends_a_measured_point():
    opt = _fit_all_cat(n=6, seed=2)
    seen = opt.X_train.detach().cpu().numpy()
    for strategy in ("log_ei", "ucb", "max_variance"):
        r = np.asarray(opt.recommend(n_candidates=2, strategy=strategy))
        for row in r:
            assert not np.any(np.all(np.isclose(seen, row), axis=1)), (strategy, row)


def test_all_categorical_many_combos_scored_not_relaxed():
    # 5 seeds over 5 categorical dims -> up to 5^5 = 3125 combinations; the
    # continuous relaxation used to raise (no gradient w.r.t. cat dims).
    from scilink.agents.planning_agents.bo_tools import get_optimizer
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.integers(0, 7, 5), rng.integers(0, 4, 5), rng.integers(0, 12, 5),
                         rng.integers(0, 8, 5), rng.integers(0, 6, 5)]).astype(float)
    # force distinct levels per dim so combos > 512
    X[:, 0] = [0, 1, 2, 3, 4]; X[:, 2] = [0, 1, 2, 3, 4]; X[:, 3] = [0, 1, 2, 3, 4]; X[:, 4] = [0, 1, 2, 3, 4]
    y = rng.uniform(0, 100, 5)
    opt = get_optimizer(is_moo=False)
    opt.fit(X, y, bounds=[(0, 6), (0, 3), (0, 11), (0, 7), (0, 5)],
            model_config={"kernel": "matern_2.5", "noise": "min_noise_low", "surrogate": "mixed"},
            feature_names=list("abcde"), cat_dims=[0, 1, 2, 3, 4])
    for strategy in ("log_ei", "max_variance", "ucb"):
        r = np.asarray(opt.recommend(n_candidates=1, strategy=strategy))
        assert r.shape == (1, 5), (strategy, r.shape)
