"""Failure-aware Expected Improvement.

Experiments that fail outright (failed prints, zero-yield syntheses, crashed
runs) enter the dataset as a sentinel outcome — typically an exact zero or a
repeated identical worst value. A GP treats those as valid smooth field values,
which drags the posterior mean down around failure clusters, attracts
exploration to them, and deflates the improvement baseline.

``failure_aware_ei`` factors the problem: a classifier learns P(feasible | x)
from the sentinel labels, the GP handles quality-where-it-works, and the
acquisition is log-EI (best_f over feasible observations only) plus
``feasibility_weight`` * log P(feasible). It honors a discrete candidate pool
when the optimizer was given one; otherwise it scores a continuous Sobol pool.

Params (all tunable by the strategy LLM — see the bundle markdown):
  - ``failure_value``      : sentinel outcome (internal maximize orientation).
                             Default: auto-detect the exact-repeated worst value.
  - ``min_failures``       : sentinel repeats required to activate weighting
                             (default 3); below it, plain log-EI.
  - ``feasibility_weight`` : gamma >= 0 (default 1.0); higher avoids failure
                             zones harder, 0 recovers plain log-EI.
"""
import numpy as np

from scilink.skills._shared._opt_components import AcquisitionComponent


def _recommend_failure_aware_ei(optimizer, n_candidates, params):
    import torch
    from botorch.acquisition import LogExpectedImprovement
    from botorch.utils.sampling import draw_sobol_samples

    p = params or {}
    gamma = max(0.0, float(p.get("feasibility_weight", 1.0)))
    min_failures = int(p.get("min_failures", 3))

    y = optimizer.y_train.detach().cpu().numpy().ravel()
    X = optimizer.X_train.detach().cpu().numpy()

    fv = p.get("failure_value")
    if fv is not None:
        is_fail = np.isclose(y, float(fv), rtol=0.0, atol=1e-9)
    else:
        is_fail = np.isclose(y, y.min(), rtol=0.0, atol=1e-12)
        if is_fail.sum() < min_failures:
            is_fail = np.zeros(len(y), dtype=bool)
    feasible = ~is_fail
    weighting_active = bool(is_fail.sum() >= min_failures and feasible.any())

    pool = getattr(optimizer, "_candidate_pool", None)
    if pool is None:
        pool = (draw_sobol_samples(bounds=optimizer.bounds, n=4096, q=1)
                .squeeze(1).detach().cpu().numpy())
    pool = np.asarray(pool, dtype=np.float64)
    X_cand = torch.tensor(pool, dtype=torch.double, device=optimizer.device)

    # Improvement baseline over feasible observations only — a failure cluster
    # must not deflate best_f.
    best_f = float(y[feasible].max()) if feasible.any() else float(y.max())
    acq = LogExpectedImprovement(model=optimizer.model, best_f=best_f)
    scores = np.empty(len(X_cand))
    chunk = 256
    with torch.no_grad():
        for s in range(0, len(X_cand), chunk):
            e = min(s + chunk, len(X_cand))
            scores[s:e] = acq(X_cand[s:e].unsqueeze(1)).detach().cpu().numpy()

    if weighting_active and gamma > 0.0:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=0)
        clf.fit(X, feasible.astype(int))
        classes = list(clf.classes_)
        if 1 in classes:
            p_feas = clf.predict_proba(pool)[:, classes.index(1)]
        else:
            p_feas = np.ones(len(pool))
        scores = scores + gamma * np.log(np.clip(p_feas, 1e-6, 1.0))

    top = np.argsort(scores)[::-1][:n_candidates]
    return pool[top]


ACQUISITION_SPEC = AcquisitionComponent(
    key="failure_aware_ei",
    recommend_fn=_recommend_failure_aware_ei,
    agents=["bo"],
    description=("Failure-aware log-EI: weights expected improvement by a "
                 "classifier's P(feasible) learned from sentinel (failed) "
                 "outcomes; best_f uses feasible observations only. Params: "
                 "failure_value (float), min_failures (int), "
                 "feasibility_weight (float)."),
)
