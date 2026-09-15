"""Deep-ensemble surrogate for Bayesian optimization.

An ensemble of small neural networks, each trained on a bootstrap resample of
the data; their disagreement provides the epistemic uncertainty. It exposes a
Gaussian posterior (the ensemble mean and variance), so the standard acquisition
functions apply unchanged. A non-GP alternative for larger or strongly
non-stationary datasets where a stationary GP kernel fits poorly.
"""
import torch
import torch.nn.functional as F
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

from scilink.skills._shared._opt_components import SurrogateComponent


class _MLP(torch.nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class DeepEnsembleModel(Model):
    """Ensemble of MLPs (deep-ensemble epistemic uncertainty). Each member is
    trained on a bootstrap resample; the posterior is the Gaussian with the
    ensemble mean and variance, so analytic and MC acquisitions both apply."""

    def __init__(self, train_X, train_Y, n_models: int = 5, hidden: int = 64):
        super().__init__()
        X = train_X.to(torch.double)
        Y = train_Y.to(torch.double)
        self.register_buffer("Xmu", X.mean(0, keepdim=True))
        self.register_buffer("Xsd", X.std(0, keepdim=True).clamp_min(1e-6))
        self.register_buffer("Ymu", Y.mean())
        self.register_buffer("Ysd", Y.std().clamp_min(1e-6))
        self.register_buffer("_Xn", (X - self.Xmu) / self.Xsd)
        self.register_buffer("_Yn", (Y - self.Ymu) / self.Ysd)
        self.members = torch.nn.ModuleList(
            [_MLP(X.shape[-1], hidden).to(torch.double) for _ in range(n_models)]
        )

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None, **kw):
        Xn = (X.to(torch.double) - self.Xmu) / self.Xsd
        preds = torch.stack([m(Xn) for m in self.members], dim=0)  # (K, ..., q, 1)
        mean = preds.mean(0)                       # (..., q, 1)
        var = preds.var(0).clamp_min(1e-6)         # (..., q, 1)
        mean = mean * self.Ysd + self.Ymu
        var = var * (self.Ysd ** 2)
        mvn = MultivariateNormal(mean.squeeze(-1), DiagLinearOperator(var.squeeze(-1)))
        post = GPyTorchPosterior(mvn)
        return posterior_transform(post) if posterior_transform is not None else post


def _fit_ensemble(model, epochs: int = 300, lr: float = 1e-2):
    Xn, Yn = model._Xn, model._Yn
    n = Xn.shape[0]
    for m in model.members:
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        idx = torch.randint(0, n, (n,))  # bootstrap resample
        for _ in range(epochs):
            opt.zero_grad()
            F.mse_loss(m(Xn[idx]), Yn[idx]).backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(False)


def _build_deep_ensemble(input_dim, *, kernel, noise, input_transform,
                         fixed_noise_std, cat_dims=None, dkl_config=None,
                         fidelity_config=None, **kw):
    from scilink.agents.planning_agents.bo_tools import (
        SurrogateSpec, SurrogateCapabilities,
    )
    return SurrogateSpec(
        model_factory=lambda X, y: DeepEnsembleModel(X, y, n_models=5, hidden=64),
        fit_fn=lambda m: _fit_ensemble(m, epochs=300, lr=1e-2),
        capabilities=SurrogateCapabilities(
            supports_fixed_noise=False, supports_warp=False,
            needs_cat_dims=False, supports_thompson=False,
        ),
    )


SURROGATE_SPEC = SurrogateComponent(
    key="deep_ensemble",
    builder=_build_deep_ensemble,
    agents=["bo"],
    description=("Deep-ensemble BNN surrogate (ensemble of MLPs; not a GP). For "
                "larger datasets or non-stationary responses where a GP kernel "
                "struggles and ensemble uncertainty is adequate."),
)
