#!/usr/bin/env python
"""Classical-BO ensemble regression net for the OptimizationAgent
foundationalization (issue #196).

This is NOT a paper benchmark. It exercises the core of ``bo_tools.py``
(``get_optimizer -> fit -> recommend``) with pinned configs across the
surrogate x acquisition registry, with NO LLM in the loop, and gates a
candidate build against a frozen ensemble baseline. Run it before merging any
foundationalization PR (stage extraction, rename, Tier-2 surrogate/acquisition
expansion): existing combos must stay non-inferior to the baseline.

Why an ensemble and not a byte-golden: the GP/BO core is irreducibly
nondeterministic downstream of ``fit()`` -- even single-threaded, fully
seeded, double precision, the marginal-likelihood fit lands in different
hyperparameter optima run-to-run (trajectories drift ~20, posteriors ~30; see
the `probe` / `probe_fwd` diagnostics). But the seed-MEAN final-best value is
stable (measured floor 0.0 on the deterministic landscapes, <=0.36% on the
noisy ones), because the optimizer finds the same best value via different
paths. So we gate the *ensemble aggregate*, not the trajectory.

The net: `freeze` runs the ensemble twice -- once as the baseline, once to
MEASURE each combo's run-to-run noise floor -- and sets a per-combo margin at
max(3x floor, 1% of |mean|, 0.02). `check` re-runs the candidate code on the
same seed set and applies a PAIRED non-inferiority gate (lower final-best is
better, so a regression makes the ensemble's best value larger by more than
the margin). The agent / LLM layer is guarded separately against the frozen
benchmark agent runs (API-budgeted, not here).

Run single-threaded for stability (multi-threaded BLAS reduction order is one
of the noise sources the margins absorb), in the torch-enabled env:

    cd tests/bo_regression
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        conda run -n scilink python regression_ensemble.py <mode>

Modes:
    probe        # diagnostic: trajectory determinism (expect nondeterministic)
    probe_fwd    # diagnostic: forward-pass determinism (expect nondeterministic)
    noise_floor  # diagnostic: across-repeat stability of seed-mean final
    freeze       # write ensemble_baseline.json (self-calibrating margins)
    check        # run candidate, paired non-inferiority gate (exit 1 on regression)
    selftest_regression  # inject a synthetic regression, confirm the gate FAILs
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scilink.agents.planning_agents.bo_tools import get_optimizer
from _benchmarks import (
    branin, ackley_2d, catalytic_yield, catalytic_yield_true,
    alloy_hardness, alloy_hardness_true, generate_initial_data,
    first_order_step_2d, pitchfork_bifurcation_2d,
    phase_diagram_2d, critical_cusp_2d,
)

# Structurally-diverse coverage: 2D smooth, 2D many-minima, 2D heteroscedastic
# noise, 6D-with-irrelevant-dims, plus four non-smooth phase-transition
# landscapes (the structure where kernel-adaptation regressions surface).
PROBLEMS = {
    "branin_2d": dict(func=branin, true=None,
                      bounds=[[-5.0, 10.0], [0.0, 15.0]],
                      cols=["x1", "x2"], n_init=8, n_iters=8),
    "ackley_2d": dict(func=ackley_2d, true=None,
                      bounds=[[-5.0, 5.0], [-5.0, 5.0]],
                      cols=["x1", "x2"], n_init=10, n_iters=8),
    "catalytic_yield": dict(func=catalytic_yield, true=catalytic_yield_true,
                            bounds=[[300.0, 600.0], [1.0, 10.0]],
                            cols=["temperature_K", "pressure_atm"],
                            n_init=8, n_iters=8),
    "alloy_hardness_6d": dict(func=alloy_hardness, true=alloy_hardness_true,
                              bounds=[[0.0, 1.0]] * 6,
                              cols=["Cr", "Ni", "Mo", "Si", "Mn", "C"],
                              n_init=15, n_iters=8),
    "pitchfork_bifurcation": dict(func=pitchfork_bifurcation_2d, true=None,
                                  bounds=[[-2.0, 2.0], [-1.0, 1.0]],
                                  cols=["order_param", "control"],
                                  n_init=8, n_iters=12),
    "phase_diagram": dict(func=phase_diagram_2d, true=None,
                          bounds=[[0.0, 1.0], [300.0, 1500.0]],
                          cols=["composition", "temperature_K"],
                          n_init=10, n_iters=12),
    "critical_cusp": dict(func=critical_cusp_2d, true=None,
                          bounds=[[0.0, 2.0], [0.0, 2.0]],
                          cols=["x1", "x2"], n_init=8, n_iters=12),
    "first_order_step": dict(func=first_order_step_2d, true=None,
                             bounds=[[-3.0, 3.0], [-3.0, 3.0]],
                             cols=["x1", "x2"], n_init=8, n_iters=12),
}

# Continuous-only registry slice (skip 'mixed' -> needs cat_dims).
SURROGATES = ["single_task", "dkl"]
ACQUISITIONS = ["log_ei", "ucb", "thompson", "max_variance"]
SEEDS = [0, 1, 2]   # diagnostics only; the gate uses ENSEMBLE_SEEDS below.


def _pin(seed: int) -> None:
    """Pin every RNG the deterministic path can touch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def run_combo(problem: dict, surrogate: str, strategy: str, seed: int) -> list:
    """One pinned BO trajectory: best-so-far (true value) per iteration."""
    _pin(seed)
    func, true_func = problem["func"], problem["true"]
    bounds, cols = problem["bounds"], problem["cols"]
    eval_func = true_func or func

    df = generate_initial_data(func, bounds, problem["n_init"], cols, seed=seed)
    optimizer = get_optimizer(is_moo=False, device="cpu")
    true_vals = [eval_func(*row) for row in df[cols].values]
    best_history = []

    for _step in range(problem["n_iters"]):
        # Re-pin before each model interaction so the trajectory is a pure
        # function of (code, seed) regardless of how many torch draws upstream
        # steps consumed.
        _pin(seed * 1000 + _step)
        X = df[cols].values.astype(np.float64)
        y_train = -df[["y"]].values.astype(np.float64)  # maximize internally
        model_config = {"surrogate": surrogate, "kernel": "matern_2.5",
                        "noise": "min_noise_low"}
        try:
            optimizer.fit(X, y_train, np.array(bounds, dtype=np.float64),
                          model_config, cols)
            cand = optimizer.recommend(n_candidates=1, strategy=strategy, params={})
            x_new = cand[0]
        except Exception as exc:
            last = best_history[-1] if best_history else min(true_vals)
            best_history += [last] * (problem["n_iters"] - len(best_history))
            best_history.append(f"ERROR:{type(exc).__name__}")
            break

        y_new = func(*x_new)
        true_vals.append(eval_func(*x_new))
        row = {cols[i]: x_new[i] for i in range(len(cols))}
        row["y"] = y_new
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        best_history.append(float(min(true_vals)))

    return best_history


def sweep() -> dict:
    out = {}
    for pname, problem in PROBLEMS.items():
        out[pname] = {}
        for surr in SURROGATES:
            for acq in ACQUISITIONS:
                key = f"{surr}/{acq}"
                out[pname][key] = [run_combo(problem, surr, acq, s) for s in SEEDS]
                print(f"  {pname:18s} {key:24s} done")
    return out


def _max_delta(a: dict, b: dict):
    """Max abs delta across all numeric trajectory points; flags shape/err mismatch."""
    worst = 0.0
    mism = []
    for p in a:
        for k in a[p]:
            for si, (ta, tb) in enumerate(zip(a[p][k], b[p][k])):
                if len(ta) != len(tb):
                    mism.append(f"{p}/{k}/seed{si}: len {len(ta)}!={len(tb)}"); continue
                for va, vb in zip(ta, tb):
                    if isinstance(va, str) or isinstance(vb, str):
                        if va != vb:
                            mism.append(f"{p}/{k}/seed{si}: {va}!={vb}")
                    else:
                        worst = max(worst, abs(va - vb))
    return worst, mism


def fwd_quantities(problem: dict, surrogate: str, seed: int) -> dict:
    """Forward-pass-only signature: fit once on seeded init data, then read
    posterior mean/var AND every acquisition value on a FIXED Sobol design.
    No optimize_acqf / argmax -> excludes the stochastic multistart search,
    isolating exactly the surrogate+acquisition numerics a Tier-2 expansion
    would perturb."""
    from botorch.utils.sampling import draw_sobol_samples
    _pin(seed)
    func, bounds, cols = problem["func"], problem["bounds"], problem["cols"]
    df = generate_initial_data(func, bounds, problem["n_init"], cols, seed=seed)
    X = df[cols].values.astype(np.float64)
    y = -df[["y"]].values.astype(np.float64)
    optimizer = get_optimizer(is_moo=False, device="cpu")
    optimizer.fit(X, y, np.array(bounds, dtype=np.float64),
                  {"surrogate": surrogate, "kernel": "matern_2.5",
                   "noise": "min_noise_low"}, cols)
    # Fixed evaluation design (seeded, independent of fit).
    _pin(10_000 + seed)
    b = torch.tensor(bounds, dtype=torch.double).T
    Xt = draw_sobol_samples(bounds=b, n=32, q=1).squeeze(1)
    post = optimizer.model.posterior(Xt)
    out = {"post_mean": post.mean.detach().cpu().numpy().ravel().tolist(),
           "post_var": post.variance.detach().cpu().numpy().ravel().tolist()}
    Xnp = Xt.detach().cpu().numpy()
    for acq in ["log_ei", "ucb", "max_variance"]:  # thompson has no point-value
        try:
            optimizer.recommend(n_candidates=1, strategy=acq, params={})  # sets acq_func
            out[f"acq_{acq}"] = optimizer.evaluate_acquisition(Xnp).ravel().tolist()
        except Exception as exc:
            out[f"acq_{acq}"] = f"ERROR:{type(exc).__name__}"
    return out


def sweep_fwd() -> dict:
    out = {}
    for pname, problem in PROBLEMS.items():
        out[pname] = {}
        for surr in SURROGATES:
            out[pname][surr] = [fwd_quantities(problem, surr, s) for s in SEEDS]
            print(f"  {pname:18s} {surr:12s} fwd done")
    return out


def _max_delta_fwd(a, b):
    worst, mism = 0.0, []
    perkey = {}  # (surrogate, quantity) -> worst delta
    for p in a:
        for surr in a[p]:
            for si, (da, db) in enumerate(zip(a[p][surr], b[p][surr])):
                for k in da:
                    va, vb = da[k], db[k]
                    if isinstance(va, str) or isinstance(vb, str):
                        if va != vb: mism.append(f"{p}/{surr}/{k}/seed{si}")
                    else:
                        d = max(abs(x - y) for x, y in zip(va, vb))
                        worst = max(worst, d)
                        kk = (surr, k)
                        perkey[kk] = max(perkey.get(kk, 0.0), d)
    print("  per (surrogate, quantity) worst |delta|:")
    for (surr, k), d in sorted(perkey.items()):
        print(f"    {surr:12s} {k:12s} {d:.3e}")
    return worst, mism


def cmd_probe_fwd():
    print("FWD Run 1/2 ..."); a = sweep_fwd()
    print("FWD Run 2/2 ..."); b = sweep_fwd()
    worst, mism = _max_delta_fwd(a, b)
    print("\n=== FORWARD-PASS DETERMINISM PROBE (posterior + acq values, no argmax) ===")
    print(f"max|delta|: {worst:.3e}")
    if mism: print("MISMATCHES:", *mism, sep="\n  ")
    print("VERDICT:", "DETERMINISTIC (forward-pass golden viable)" if worst == 0.0 and not mism
          else f"NONDETERMINISTIC (>= {worst:.1e})")


def cmd_selftest_regression():
    """Sensitivity test: inject a synthetic regression (cripple the acquisition
    optimizer) WITHOUT editing core code, then run the gate. A trustworthy net
    must FAIL here. Monkeypatches optimize_acqf to a near-blind search."""
    import scilink.agents.planning_agents.bo_tools as bt
    _orig = bt.optimize_acqf
    def _crippled(*a, **k):
        k["num_restarts"] = 1
        k["raw_samples"] = 4
        return _orig(*a, **k)
    bt.optimize_acqf = _crippled
    print("Injected synthetic regression: optimize_acqf num_restarts=1, raw_samples=4")
    if not ENSEMBLE_PATH.exists():
        sys.exit("No baseline; run `freeze` first.")
    ref = json.loads(ENSEMBLE_PATH.read_text())
    cur = ensemble_finals(ref["seeds"])
    print("\n=== SENSITIVITY SELF-TEST (expect FAILs on search-sensitive combos) ===")
    print(f"{'combo':40s} {'base':>9s} {'cand':>9s} {'pairD':>9s} {'margin':>8s}  verdict")
    fails = 0
    for k, c in ref["combos"].items():
        base = np.array([v for v in c["per_seed"] if v is not None], dtype=float)
        cand = np.array([v for v in cur.get(k, []) if v is not None], dtype=float)
        if len(cand) != len(base):
            print(f"{k:40s}  SHAPE-MISMATCH"); continue
        paired = float((cand - base).mean()); margin = c["margin"]
        ok = paired <= margin; fails += 0 if ok else 1
        print(f"{k:40s} {base.mean():9.4f} {cand.mean():9.4f} {paired:+9.4f} "
              f"{margin:8.4f}  {'pass' if ok else 'FAIL <<<'}")
    bt.optimize_acqf = _orig
    print(f"\n{fails} combo(s) flagged. Sensitivity: "
          f"{'OK -- gate catches the regression' if fails > 0 else 'WEAK -- gate missed it'}")


def cmd_noise_floor():
    """The crux for a STATISTICAL net: is seed-averaged final regret stable
    run-to-run even though trajectories aren't? Runs the sweep R times and
    reports, per combo, the across-repeat spread of the seed-mean final best
    value vs its magnitude. Small relative spread => a non-inferiority gate
    can detect a real regression above the noise."""
    R = 4
    repeats = []
    for r in range(R):
        print(f"repeat {r+1}/{R} ...")
        rec = {}
        for pname, problem in PROBLEMS.items():
            for surr in ["single_task"]:           # cheapest, most-used path
                for acq in ["log_ei", "ucb"]:
                    finals = []
                    for s in SEEDS:
                        h = run_combo(problem, surr, acq, s)
                        finals.append(h[-1] if isinstance(h[-1], float) else float("nan"))
                    rec[f"{pname}/{surr}/{acq}"] = float(np.nanmean(finals))
        repeats.append(rec)
    print("\n=== STATISTICAL NOISE FLOOR (seed-mean final best, across-repeat) ===")
    print(f"{'combo':40s} {'mean':>12s} {'std':>10s} {'rel%':>8s}")
    for k in repeats[0]:
        vals = np.array([rp[k] for rp in repeats])
        m, sd = vals.mean(), vals.std()
        rel = 100 * sd / (abs(m) + 1e-12)
        print(f"{k:40s} {m:12.4f} {sd:10.4f} {rel:7.2f}%")


def cmd_probe():
    print("Run 1/2 ...");  a = sweep()
    print("Run 2/2 ...");  b = sweep()
    worst, mism = _max_delta(a, b)
    print("\n=== DETERMINISM PROBE (same code, run twice) ===")
    print(f"max|delta| across all trajectories: {worst:.3e}")
    if mism:
        print("MISMATCHES:", *mism, sep="\n  ")
    print("VERDICT:", "DETERMINISTIC (byte golden viable)" if worst == 0.0 and not mism
          else f"NONDETERMINISTIC -> use tolerance gate (>= {worst:.1e})")


# ------------------------------------------------------------------------- #
#  Paired classical-BO ENSEMBLE regression net.
#
#  The GP/BO core is nondeterministic downstream of fit() (see probe/probe_fwd:
#  trajectories drift ~20, posteriors ~30), so no byte-golden exists. But the
#  seed-MEAN final-best value is stable. This net freezes a classical-BO
#  ensemble, MEASURES each combo's noise floor by running twice, and gates a
#  candidate's paired per-seed difference at 3x that floor. Lower final-best =
#  better (minimization), so a regression = the best value getting LARGER
#  beyond the margin.
# ------------------------------------------------------------------------- #

ENSEMBLE_PATH = Path(__file__).parent / "ensemble_baseline.json"
ENSEMBLE_SEEDS = list(range(10))
REL_MARGIN = 0.01   # 1% of |mean| ...
ABS_MARGIN = 0.02   # ... or this absolute, whichever is larger ...
FLOOR_K = 3.0       # ... or 3x the measured run-to-run floor, whichever is largest.


def ensemble_finals(seeds) -> dict:
    """Per-combo list of final-best values across the seed set (classical BO,
    pinned configs, no LLM)."""
    out = {}
    for pname, problem in PROBLEMS.items():
        for surr in SURROGATES:
            for acq in ACQUISITIONS:
                key = f"{pname}/{surr}/{acq}"
                finals = []
                for s in seeds:
                    h = run_combo(problem, surr, acq, s)
                    finals.append(h[-1] if isinstance(h[-1], float) else None)
                out[key] = finals
        print(f"  {pname:18s} ensemble done")
    return out


def cmd_freeze():
    print("Freeze run A/2 (baseline) ..."); A = ensemble_finals(ENSEMBLE_SEEDS)
    print("Freeze run B/2 (floor calibration) ..."); B = ensemble_finals(ENSEMBLE_SEEDS)
    combos = {}
    for k in A:
        a = np.array([v for v in A[k] if v is not None], dtype=float)
        b = np.array([v for v in B[k] if v is not None], dtype=float)
        mean_a = float(a.mean()) if len(a) else float("nan")
        floor = abs(mean_a - float(b.mean())) if len(b) else float("nan")
        margin = max(FLOOR_K * floor, REL_MARGIN * abs(mean_a), ABS_MARGIN)
        combos[k] = {"per_seed": A[k], "seed_mean": mean_a,
                     "measured_floor": floor, "margin": margin}
    out = {"seeds": ENSEMBLE_SEEDS, "n_seeds": len(ENSEMBLE_SEEDS),
           "policy": {"rel": REL_MARGIN, "abs": ABS_MARGIN, "floor_k": FLOOR_K},
           "combos": combos}
    ENSEMBLE_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nFroze ensemble baseline -> {ENSEMBLE_PATH}")
    print(f"{'combo':40s} {'seed_mean':>11s} {'floor':>9s} {'margin':>9s}")
    for k, c in combos.items():
        print(f"{k:40s} {c['seed_mean']:11.4f} {c['measured_floor']:9.4f} {c['margin']:9.4f}")


def cmd_check():
    if not ENSEMBLE_PATH.exists():
        sys.exit("No ensemble baseline; run `freeze` first.")
    ref = json.loads(ENSEMBLE_PATH.read_text())
    seeds = ref["seeds"]
    print(f"Candidate run on the frozen seed set ({len(seeds)} seeds) ...")
    cur = ensemble_finals(seeds)
    print("\n=== PAIRED NON-INFERIORITY CHECK (lower final-best = better) ===")
    print(f"{'combo':40s} {'base':>9s} {'cand':>9s} {'pairD':>9s} {'margin':>8s}  verdict")
    fails = 0
    for k, c in ref["combos"].items():
        base = np.array([v for v in c["per_seed"] if v is not None], dtype=float)
        cand_raw = cur.get(k, [])
        cand = np.array([v for v in cand_raw if v is not None], dtype=float)
        if len(cand) != len(base):
            print(f"{k:40s} {'':9s} {'':9s} {'':9s} {'':8s}  SHAPE-MISMATCH"); fails += 1; continue
        paired = float((cand - base).mean())      # >0 => candidate worse (larger min)
        margin = c["margin"]
        ok = paired <= margin
        fails += 0 if ok else 1
        print(f"{k:40s} {base.mean():9.4f} {cand.mean():9.4f} {paired:+9.4f} "
              f"{margin:8.4f}  {'PASS' if ok else 'FAIL <<<'}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} COMBO(S) REGRESSED'}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"probe": cmd_probe, "probe_fwd": cmd_probe_fwd,
     "noise_floor": cmd_noise_floor, "selftest_regression": cmd_selftest_regression,
     "freeze": cmd_freeze, "check": cmd_check}[cmd]()
