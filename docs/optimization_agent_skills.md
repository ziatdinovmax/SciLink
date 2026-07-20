# Extending the OptimizationAgent with skills

The `OptimizationAgent` runs LLM-driven Bayesian optimization: at each step it
reads the data and history, configures a strategy (surrogate, kernel, noise,
acquisition), fits, recommends the next experiment(s), inspects the diagnostics,
and records the step. This document explains **how to extend it** — both by
steering how it uses what it already has, and by adding genuinely new
capabilities.

## The model: core toolkit + skills

The agent ships with a fixed **baseline toolkit** — everything that can be
composed from BoTorch today:

- surrogates: `single_task`, `mixed` (categoricals), `dkl` (deep-kernel),
  `single_task_multi_fidelity` (multi-fidelity), `saas` (sparse high-D);
- acquisitions: `log_ei`, `ucb`, `thompson`, `max_variance`, `pareto`,
  `weighted`, `mf_kg`;
- plus the kernel / noise / input-transform menus, plateau-escalation and
  budget logic, constrained-batch design, and the visual-inspection checklist.

The agent selects from this toolkit using **data signals** — multi-objective is
chosen when there are multiple targets, `mixed` when categorical inputs are
declared, `saas` when the input dimension is high, multi-fidelity when a
fidelity column is declared, and so on.

**Skills extend the agent in two ways on top of this toolkit:**

| Rung | Author | Bundle ships | Adds |
|---|---|---|---|
| **Steering skill** | anyone | `<name>.md` | judgment that reweights how the agent uses the toolkit |
| **Component skill** | in-package contributor | `<name>.md` **+ `.py` helper** | a new surrogate or acquisition the toolkit doesn't have |

The dividing line for *new capabilities* is simple: **if it can be composed from
BoTorch, it belongs in core; if it is beyond BoTorch (a foreign backend, a
custom torch model, a novel acquisition), it is a component skill.** That is why
multi-fidelity and SAAS are core (native BoTorch) while a deep-ensemble surrogate
or a setpoint-changeover acquisition are component skills.

---

## Steering skills (markdown only)

A steering skill is a single markdown file that injects domain or lab judgment
into the agent's pipeline. It adds *no* machinery — it changes which baseline
options the agent leans toward and how it reads diagnostics.

### Where it lives

```
scilink/skills/optimization/<name>/<name>.md
```

(Skills can also be supplied at runtime via the `skill=` argument / the UI
uploader; uploaded skills are markdown-only.)

### Section vocabulary

The markdown is organized under a fixed set of `## headings`. Three of them are
spliced directly into a pipeline stage:

| Section | Spliced into | Status |
|---|---|---|
| `Surrogate` | the strategy-configuration prompt | injected |
| `Acquisition` | the strategy-configuration prompt | injected |
| `Diagnostics` | the visual-inspection prompt | injected |
| `Overview` / `Setup` / `Interpretation` / `Implementation` | — | recognized by the loader; not yet wired to a stage |

So today a steering skill takes effect through its `Surrogate`, `Acquisition`,
and `Diagnostics` sections. The other headings are part of the vocabulary (the
loader parses them, and they are reserved for future stages) but are not
currently injected — put your active guidance under the three injected sections.
Off-vocabulary headings are preserved under `extras` with a warning.

### Example

```markdown
---
description: Electrochemical CV optimization — high, scan-rate-dependent noise.
---

## Surrogate
This lab's cyclic-voltammetry response has high observation noise (>=8%) that
grows with scan rate. Prefer `min_noise_med`. The potential axis is far smoother
than the scan-rate axis; if LOO residuals stay large on the scan-rate dimension,
warp the inputs.

## Acquisition
Each run is a 6-hour experiment, so treat budget as critical from step 1: favor
`log_ei`, never `max_variance`. Yield and reversibility conflict above 1 V/s —
expect a concave trade-off; do not trust weighted scalarization there.

## Diagnostics
If the Sobol panel shows scan-rate dominating, you are in a kinetics-limited
regime — stop refining potential and explore the scan-rate axis.
```

Activate it with `run_optimization_loop(..., skill="electrochem_cv")`. The agent
reads the injected guidance and selects accordingly — it retains final authority
over the actual choice.

---

## Component skills (markdown + a `.py` helper)

A component skill adds a **new surrogate or acquisition** that the baseline
toolkit does not have. The bundle's `.py` helper declares the machinery; the
bundle's markdown carries the judgment (when to use it, how to read it).

### Where it lives

```
scilink/skills/optimization/<name>/
├── <name>.md        # judgment: when to use it, params, diagnostics
└── <helper>.py      # declares SURROGATE_SPEC and/or ACQUISITION_SPEC
```

Component helpers are discovered **only from skills inside the installed
package** (the same rule as `TOOL_SPEC` tools), and only while the skill is
active. Uploaded / markdown-only skills cannot ship code — this is the safety
boundary: an end user must not inject an unverifiable surrogate posterior that
would silently steer every subsequent experiment.

### Adding a surrogate

Declare a `SurrogateComponent` whose `builder` returns a `SurrogateSpec`. Any
torch model that exposes a BoTorch posterior (`mean` / `variance` / `rsample`)
composes with **every** existing acquisition unchanged.

```python
from scilink.skills._shared._opt_components import SurrogateComponent
from scilink.agents.planning_agents.bo_tools import SurrogateSpec, SurrogateCapabilities

def _build(input_dim, *, kernel, noise, input_transform, fixed_noise_std,
           cat_dims=None, dkl_config=None, fidelity_config=None, **kw):
    def factory(X, y):
        return MyTorchModel(X, y)            # any model with .posterior()
    def fit(model):
        train(model)                          # your fit (Adam, NUTS, ...)
    return SurrogateSpec(model_factory=factory, fit_fn=fit,
                         capabilities=SurrogateCapabilities(
                             supports_fixed_noise=False, supports_warp=False,
                             needs_cat_dims=False, supports_thompson=False))

SURROGATE_SPEC = SurrogateComponent(
    key="my_surrogate", builder=_build, agents=["bo"],
    description="One-line summary the agent sees in the menu.")
```

`builder` has the same signature as `bo_tools.build_surrogate`'s core branches.
The `fit_fn` slot is what makes the seam general — `dkl` uses Adam, `saas` uses
NUTS, an ensemble trains MLPs; any fit works.

The bundle markdown's `## Surrogate` section must describe **when** to select
`"my_surrogate"` (so the agent knows) and the `## Diagnostics` section how to
judge its fit.

### Adding an acquisition

Declare an `AcquisitionComponent` whose `recommend_fn` returns candidate points.
It receives the fitted optimizer, so it can read `optimizer.model`,
`optimizer.bounds`, `optimizer.X_train`, etc., and build any BoTorch acquisition.

```python
from scilink.skills._shared._opt_components import AcquisitionComponent

def _recommend(optimizer, n_candidates, params):
    # params come from the LLM's acquisition_strategy.params
    ...
    return candidates_np   # shape (n_candidates, input_dim)

ACQUISITION_SPEC = AcquisitionComponent(
    key="my_acq", recommend_fn=_recommend, agents=["bo"],
    description="One-line summary, including any params the agent should set.")
```

The bundle markdown's `## Acquisition` section tells the agent when to select
`"my_acq"` and how to set its `params`.

### Validation

- A skill-contributed `key` is **only selectable while its skill is active**
  (config validation gates on it), and is described to the agent through the
  bundle's markdown.
- For a surrogate, smoke-test the structural contract while authoring:

  ```python
  from scilink.skills._shared._opt_components import get_surrogate_components, smoke_test_surrogate
  smoke_test_surrogate(get_surrogate_components(["my_skill"])["my_surrogate"])
  ```

  This builds and fits on dummy data and asserts a usable posterior.
  *Calibration* (whether the posterior is trustworthy) is the author's
  responsibility, exactly as for a core surrogate.

### A note on the helper

The `.py` helper ships **composition and judgment, never the underlying
numerics**. The neural-network library, the GP, the MCMC sampler — those come
from `torch` / `botorch` / `pyro` (dependencies). If a component needs an
optional library that is not installed, importing the helper fails and the
component simply does not register (it degrades gracefully).

---

## Worked examples (in this repo)

| Bundle | Adds | Why a skill (not core) |
|---|---|---|
| `bo_deep_ensemble` | `deep_ensemble` surrogate — an ensemble of bootstrap-trained MLPs exposing a Gaussian posterior | BoTorch ships GPs, not a trained NN ensemble |
| `bo_changeover` | `changeover_ei` acquisition — discounts EI by how far a candidate moves expensive-to-change setpoints | BoTorch's cost-aware utilities model *evaluation* cost, not setpoint changeover between runs |

Read those two bundles as templates.

## When is it core, not a skill?

If the capability is native BoTorch, add it to `bo_tools.py` instead — it is part
of the baseline toolkit, surfaced by a data signal. Multi-fidelity
(`single_task_multi_fidelity` + `mf_kg`, surfaced when a fidelity column is
declared) and SAAS (`saas`, surfaced when the input dimension is high) are the
reference examples. End users cannot inject surrogates regardless; new core
surrogates/acquisitions are contributor changes via the `SurrogateSpec` /
`recommend` seams.

## Quick reference

- **Steer the existing toolkit** → markdown-only skill (any user).
- **Add a torch surrogate / novel acquisition beyond BoTorch** → component skill
  with a `.py` helper (in-package contributor).
- **A native-BoTorch method** → core (`bo_tools.py`), surfaced by a data signal.
- A `.py` helper ships composition + judgment; the numerics are dependencies.
- Component keys are visible to the agent only while the skill is active.
