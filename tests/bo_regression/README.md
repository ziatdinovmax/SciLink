# BO core regression net

A pre-merge gate for the OptimizationAgent foundationalization (issue #196).
It guards the **deterministic core** of `bo_tools.py`
(`get_optimizer → fit → recommend`) against regressions from stage extraction,
the `BOAgent → OptimizationAgent` rename, and Tier-2 surrogate/acquisition
expansion — **without any LLM call** (no API key needed).

## Why an ensemble, not a golden master

The GP/BO core is irreducibly nondeterministic downstream of `fit()`: even
single-threaded, fully seeded, double precision, the marginal-likelihood fit
lands in different hyperparameter optima run-to-run (trajectories drift ~20,
posteriors ~30 — reproduce with `probe` / `probe_fwd`). So no byte-exact
golden is possible.

But the **seed-mean final-best value is stable** (measured floor 0.0 on the
deterministic phase-transition landscapes, ≤0.36% on the noisy scientific
ones), because the optimizer reliably finds the same best value via different
paths. The net therefore gates the *ensemble aggregate*, not the trajectory.

## How it works

`freeze` runs the classical-BO ensemble **twice** — once as the baseline,
once to *measure* each combo's run-to-run noise floor — and sets a per-combo
margin = `max(3×floor, 1%×|mean|, 0.02)`. `check` re-runs the candidate code
on the same seed set and applies a **paired** non-inferiority gate: lower
final-best is better, so a regression makes the ensemble's best value larger
by more than the margin. Self-calibrating margins mean low-noise combos are
sensitive sentinels (tight ~0.02 margins) while noisy combos can't false-alarm.

Coverage: 8 problems × {single_task, dkl} × {log_ei, ucb, thompson,
max_variance} = 64 combos, 10 seeds. The 4 non-smooth phase-transition
landscapes (pitchfork, phase_diagram, critical_cusp, first_order_step) are the
tightest sentinels — that's where kernel-adaptation regressions surface.

## Running

Single-threaded (the multi-threaded BLAS reduction order is one of the noise
sources), in the torch-enabled env:

```bash
cd tests/bo_regression
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    conda run -n scilink python regression_ensemble.py check
```

- `check` — paired non-inferiority gate vs `ensemble_baseline.json` (exit 1 on
  regression). Run this on every foundationalization PR branch.
- `freeze` — regenerate `ensemble_baseline.json` (only when the benchmark set
  or `_benchmarks.py` legitimately changes).
- `selftest_regression` — injects a synthetic regression to prove the gate
  FAILs (sensitivity check).
- `probe` / `probe_fwd` / `noise_floor` — diagnostics documenting why this is
  an ensemble and not a golden master.

`_benchmarks.py` is vendored verbatim from the SciLink benchmarking suite;
do not edit it without re-freezing (the baseline is keyed to those exact
functions and their shared `_noisy_rng`).

## Scope

This is the **classical / deterministic-core** tier. The LLM-driven agent
layer is guarded separately by the agent ensemble against the frozen
benchmark agent runs (API-budgeted, not here).
