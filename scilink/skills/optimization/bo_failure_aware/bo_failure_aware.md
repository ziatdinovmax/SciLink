---
description: Failure-aware optimization for campaigns where experiments can fail outright (zero-yield syntheses, failed prints, crashed runs). Ships the `failure_aware_ei` acquisition, which weights expected improvement by a learned probability of feasibility.
category: modifier
domain: experimental
---

## Overview

In many experimental campaigns an experiment can **fail outright** — a print
that doesn't adhere, a synthesis that yields nothing, a run that crashes. Such
failures are usually recorded as a sentinel outcome (exactly zero, or a
repeated identical worst value). A standard GP treats those sentinels as valid
smooth field values: the posterior mean is dragged down across their
neighborhoods, exploration is attracted to the (high-uncertainty) failure
regions, and expected improvement is computed against a best-so-far that the
failure cluster has distorted. The result is budget spent confirming failures
and exploitation aimed at the wrong basins.

`failure_aware_ei` separates the two questions the data is answering: *does
the experiment work here?* (a classifier) and *how good is it when it works?*
(the GP). Acquisition = expected improvement × probability of feasibility.

## Surrogate

When the data shows a failure signature (several exact-duplicate worst-case
outcomes), the landscape is discontinuous by construction. Prefer
`matern_1.5` or `matern_0.5` over smooth kernels, and avoid
`min_noise_high` (it smears the failure boundary into the feasible region).

## Acquisition

Select `acquisition_strategy.type: "failure_aware_ei"` when the observed data
contains ≥3 identical worst-case outcomes (e.g., exact zeros) — the signature
of failed experiments rather than a smooth low region. Params:

- `failure_value` (float, optional): the sentinel outcome marking a failed
  experiment, in the internal maximize orientation. Default: auto-detect —
  the worst observed value, treated as the sentinel when it repeats ≥
  `min_failures` times exactly.
- `min_failures` (int, default 3): minimum number of sentinel repeats before
  feasibility weighting activates; below it the acquisition is plain log-EI.
  RAISE if legitimate (non-failed) measurements can tie exactly at the worst
  value; LOWER to 2 when failures are certain and rare.
- `feasibility_weight` (float ≥ 0, default 1.0): γ in
  log-EI + γ·log P(feasible). RAISE (2–5) when the campaign keeps probing
  failure zones; LOWER toward 0 to recover plain EI when the classifier is
  unreliable (very few observations) or failures are informative near-misses
  rather than hard invalidity.

The improvement baseline (best_f) is computed over **feasible observations
only**, so a failure-heavy dataset does not deflate it. Candidate selection
honors a discrete candidate pool when one is provided; otherwise candidates
are drawn from a continuous Sobol pool (categorical dims are not honored).
Single-objective only.

## Diagnostics

Watch the fraction of new measurements that land on the sentinel value: with
`failure_aware_ei` it should drop well below the failure base rate within a
few steps as the classifier localizes the infeasible region. If good
candidates near the failure boundary are being avoided too aggressively
(best-found stalls while P(feasible) hugs 0 nearby), lower
`feasibility_weight`; if the campaign still wastes measurements on failures,
raise it.
