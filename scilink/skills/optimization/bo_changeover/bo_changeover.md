---
description: Changeover-aware optimization for labs where moving a setpoint (temperature, solvent, substrate) between experiments is slow/expensive. Ships the `changeover_ei` acquisition.
category: modifier
domain: experimental
---

## Overview

In real wet-lab and process campaigns, the dominant cost between consecutive
experiments is often not the measurement itself but the **setpoint changeover**:
re-equilibrating a furnace or reactor to a new temperature can take hours;
switching solvent, substrate, or catalyst means cleaning and re-priming the rig.
Standard BO ignores this and jumps to the global acquisition optimum regardless
of how far it is from the current setpoint, so a campaign burns most of its wall
time on changeovers. The `changeover_ei` acquisition keeps making progress while
staying close — in the expensive-to-change inputs — to the last experiment.

## Acquisition

Select `acquisition_strategy.type: "changeover_ei"` when the user indicates that
some inputs are slow or costly to change between runs (temperature ramp/soak,
solvent swap, electrode swap, anything with a long re-stabilization). Set its
params from the problem:

- `expensive_dims`: the integer indices (in `input_cols` order) of the
  slow/expensive-to-change inputs. List only those — cheap-to-change inputs
  (e.g. a dosed volume, a software parameter) should NOT be penalized.
- `changeover_weight` (λ ≥ 0): how hard to penalize moves. Start near 1.0. Raise
  it when changeover time dominates the experiment (e.g. a multi-hour thermal
  soak versus a 5-minute measurement); lower it (→ toward standard EI) late in
  the campaign when you must exploit regardless of changeover.

Do not use it when every input is cheap to change, or for a cold start where
broad exploration matters more than changeover economy — plain `log_ei` /
`max_variance` are better there. It is single-objective, and it assumes
continuous inputs: candidates are drawn from a continuous Sobol pool, so it does
not honor categorical dimensions (`cat_dims`).

## Diagnostics

Watch the trajectory of the expensive inputs across steps: with `changeover_ei`
they should move in gradual steps rather than jumping across the range each
iteration, while the best-found value still improves. If the expensive inputs
are jumping wildly, raise `changeover_weight`; if the best-found value has
stalled because the search is trapped near the last setpoint, lower it.
