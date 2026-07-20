"""Skill-contributed optimization engine components (issue #196).

A skill bundle's sibling ``.py`` may declare new engine machinery for the
OptimizationAgent — a custom surrogate or a custom acquisition — and the bundle's
markdown carries the judgment about when to use it. This is the contributor rung
of the extension model:

  - steering skill  : markdown only (any user)        -> reweights the toolkit
  - component skill : markdown + .py helper (in-pkg)  -> ADDS a surrogate/acq
  - core            : bo_tools.py (maintainer)        -> the baseline toolkit

Declarations (module-level attributes in the bundle's .py):
  - ``SURROGATE_SPEC`` / ``SURROGATE_SPECS``     : SurrogateComponent(s)
  - ``ACQUISITION_SPEC`` / ``ACQUISITION_SPECS`` : AcquisitionComponent(s)

Discovery mirrors the TOOL_SPEC registry: per-bundle, IN-PACKAGE ONLY, visible
to the engine only while the skill is active. Uploaded / markdown-only skills
cannot ship code — the deliberate safety boundary against unverifiable injected
posteriors. Structural validity (does it yield a posterior?) can be smoke-tested
while authoring via `smoke_test_surrogate`; calibration is the contributor-
author's responsibility, exactly as for a core surrogate.
"""
import importlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict

_logger = logging.getLogger(__name__)


@dataclass
class SurrogateComponent:
    """A skill-shipped surrogate.

    ``builder`` mirrors the signature of build_surrogate's core branches and
    returns a ``SurrogateSpec``:
        builder(input_dim, *, kernel, noise, input_transform, fixed_noise_std,
                cat_dims=None, dkl_config=None, fidelity_config=None, **kw) -> SurrogateSpec
    Any torch model exposing a BoTorch posterior (mean/variance + rsample)
    composes with the entire acquisition/optimization engine unchanged.
    """
    key: str
    builder: Callable[..., Any]
    description: str = ""
    agents: List[str] = field(default_factory=lambda: ["bo"])


@dataclass
class AcquisitionComponent:
    """A skill-shipped acquisition strategy.

    ``recommend_fn(optimizer, n_candidates, params) -> np.ndarray`` of shape
    (n_candidates, input_dim). It receives the FITTED optimizer, so it can read
    ``optimizer.model`` / ``.bounds`` / ``.X_train`` / ``.y_train`` /
    ``.input_dim`` and build any BoTorch acquisition over the posterior.
    """
    key: str
    recommend_fn: Callable[..., Any]
    description: str = ""
    agents: List[str] = field(default_factory=lambda: ["bo"])


def _collect(mod, single_attr, multi_attr, cls):
    out = []
    one = getattr(mod, single_attr, None)
    if isinstance(one, cls):
        out.append(one)
    for c in (getattr(mod, multi_attr, None) or []):
        if isinstance(c, cls):
            out.append(c)
    return out


def _discover(active_skills, single_attr, multi_attr, cls, agent: str) -> Dict[str, Any]:
    """Components declared by ACTIVE skill bundles, keyed by component key.

    In-package only (walks the package skills dir); gated by bundle membership
    AND the component's ``agents=`` tag. Returns {} when no skill is active.
    """
    if not active_skills:
        return {}
    from scilink.skills.loader import list_all_skills, _SKILLS_DIR

    active = set(active_skills)
    found: Dict[str, Any] = {}
    for domain, names in list_all_skills().items():
        for name in names:
            if name not in active:
                continue
            skill_dir = _SKILLS_DIR / domain / name
            for py in sorted(skill_dir.glob("*.py")):
                if py.stem.startswith("_"):
                    continue
                module_path = f"scilink.skills.{domain}.{name}.{py.stem}"
                try:
                    mod = importlib.import_module(module_path)
                except Exception as exc:
                    _logger.debug("Skipping %s: %s", module_path, exc)
                    continue
                for comp in _collect(mod, single_attr, multi_attr, cls):
                    if agent in comp.agents:
                        found[comp.key] = comp
    return found


def get_surrogate_components(active_skills, agent: str = "bo") -> Dict[str, SurrogateComponent]:
    return _discover(active_skills, "SURROGATE_SPEC", "SURROGATE_SPECS",
                     SurrogateComponent, agent)


def get_acquisition_components(active_skills, agent: str = "bo") -> Dict[str, AcquisitionComponent]:
    return _discover(active_skills, "ACQUISITION_SPEC", "ACQUISITION_SPECS",
                     AcquisitionComponent, agent)


def smoke_test_surrogate(comp: SurrogateComponent, input_dim: int = 3) -> None:
    """Structural check for authoring/tests: the builder yields a model with a
    usable posterior on dummy data. Raises on failure. Not run at runtime — call
    it while authoring a component skill. (Calibration is NOT checked — that is
    the contributor-author's responsibility, as for any core surrogate.)"""
    import numpy as np
    import torch
    spec = comp.builder(input_dim, kernel="matern_2.5", noise="min_noise_low",
                        input_transform="none", fixed_noise_std=None)
    X = torch.tensor(np.random.RandomState(0).uniform(0, 1, (8, input_dim)), dtype=torch.double)
    y = torch.tensor(np.random.RandomState(1).uniform(0, 1, (8, 1)), dtype=torch.double)
    model = spec.model_factory(X, y)
    spec.fit_fn(model)
    post = model.posterior(X[:2])
    # Fully-Bayesian models carry an extra leading MCMC-sample batch dim, so
    # check the q dimension (-2), not dim 0.
    assert post.mean is not None and post.variance is not None, "no usable posterior"
    assert post.mean.shape[-2] == 2, f"unexpected posterior q-dim: {tuple(post.mean.shape)}"
    post.rsample(torch.Size([2]))
