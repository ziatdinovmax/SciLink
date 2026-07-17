"""Changeover-aware Expected Improvement.

In many experimental campaigns the dominant cost between consecutive runs is the
setpoint CHANGEOVER, not the measurement: re-stabilizing a furnace to a new
temperature can take hours; switching solvent or substrate means cleaning and
re-priming the rig. Standard EI ignores this and jumps to the global optimum
regardless of how far it is from the current setpoint, so a campaign spends most
of its wall time on changeovers.

`changeover_ei` discounts Expected Improvement by how far a candidate moves the
*expensive-to-change* inputs from the last experiment's setpoint, trading a
little improvement for far fewer changeovers.

Params:
  - ``expensive_dims``      : indices of the slow/expensive-to-change inputs
                              (e.g. the temperature column). Default: all inputs.
  - ``changeover_weight``   : lambda >= 0; higher penalizes moves harder. Default 1.0.
"""
import numpy as np

from scilink.skills._shared._opt_components import AcquisitionComponent


def _recommend_changeover_ei(optimizer, n_candidates, params):
    import torch
    from botorch.acquisition import ExpectedImprovement
    from botorch.utils.sampling import draw_sobol_samples

    p = params or {}
    lam = float(p.get("changeover_weight", 1.0))
    d = optimizer.input_dim
    dims = list(p.get("expensive_dims") or range(d))
    bounds = optimizer.bounds
    span = (bounds[1] - bounds[0]).clamp_min(1e-9)
    last = optimizer.X_train[-1]  # most recent experiment = current setpoint

    pool = draw_sobol_samples(bounds=bounds, n=4096, q=1).squeeze(1)  # (N, d)
    acq = ExpectedImprovement(model=optimizer.model, best_f=optimizer.y_train.max())
    with torch.no_grad():
        ei = acq(pool.unsqueeze(1)).clamp_min(0.0)                              # (N,)
        move = ((pool[:, dims] - last[dims]).abs() / span[dims]).sum(dim=-1)    # (N,) normalized
        score = ei / (1.0 + lam * move)                                        # EI per unit changeover
    top = torch.topk(score, n_candidates).indices
    return pool[top].detach().cpu().numpy()


ACQUISITION_SPEC = AcquisitionComponent(
    key="changeover_ei",
    recommend_fn=_recommend_changeover_ei,
    agents=["bo"],
    description=("Changeover-aware Expected Improvement: discounts EI by how far a "
                "candidate moves expensive-to-change setpoints from the last "
                "experiment. Params: expensive_dims (list), changeover_weight (float)."),
)
