"""Synthetic objective functions for the classical-BO regression net.

Vendored VERBATIM from the SciLink benchmarking suite
(scilink-benchmarking/BOAgent: test_bo_comparison.py + test_phase_transition_bo.py)
so this regression gate is self-contained inside the package and carries no
dependency on the external benchmark repo.

DO NOT edit these bodies or the module-level ``_noisy_rng`` seed: the frozen
``ensemble_baseline.json`` was produced against these exact functions and this
exact RNG. Any change here silently invalidates the baseline (re-freeze if you
must change them). The noisy functions share one module-level RandomState that
advances across calls — which is why a fresh process reproduces the baseline
while an in-process re-run does not.
"""
import numpy as np
import pandas as pd

# Module-level RNG for the noisy functions: each call within a seed's loop
# gets a different noise draw, but the sequence is reproducible from a fresh
# import. (Reset point for fresh-process reproduction of the baseline.)
_noisy_rng = np.random.RandomState(0)


# --------------------------------------------------------------------------- #
#  Scientific robustness landscapes
# --------------------------------------------------------------------------- #

def branin(x1, x2):
    """Branin — 3 global minima ~ 0.398. Domain: x1 in [-5,10], x2 in [0,15]."""
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s


def ackley_2d(x1, x2):
    """Ackley 2D — many local minima, global minimum = 0 at origin.
    Domain: [-5, 5]^2. Tests exploration vs exploitation."""
    a, b, c = 20, 0.2, 2 * np.pi
    s1 = -a * np.exp(-b * np.sqrt(0.5 * (x1**2 + x2**2)))
    s2 = -np.exp(0.5 * (np.cos(c * x1) + np.cos(c * x2)))
    return s1 + s2 + a + np.e


def catalytic_yield(temp, pressure):
    """Simulated catalytic reactor yield with heteroscedastic GC noise.
    Domain: temp in [300, 600] K, pressure in [1, 10] atm. Optimum ~(420, 5.5)."""
    t_norm = (temp - 300) / 300
    yield_temp = 95 * np.exp(-2.0 * (t_norm - 0.4) ** 2) * (1 - 0.5 * t_norm ** 3)
    yield_press = 1 - 0.02 * (pressure - 5.5) ** 2
    yield_clean = yield_temp * yield_press
    noise_std = 1.0 + 8.0 * t_norm ** 2
    return -(yield_clean + _noisy_rng.normal(0, noise_std))


def catalytic_yield_true(temp, pressure):
    """Noiseless version for true regret computation."""
    t_norm = (temp - 300) / 300
    yield_temp = 95 * np.exp(-2.0 * (t_norm - 0.4) ** 2) * (1 - 0.5 * t_norm ** 3)
    yield_press = 1 - 0.02 * (pressure - 5.5) ** 2
    return -(yield_temp * yield_press)


def alloy_hardness(cr, ni, mo, si, mn, c):
    """Simulated steel alloy hardness; only Cr, C, Mo matter (3 of 6 dims).
    Domain: all in [0, 1]. Optimum Cr~0.6, C~0.7, Mo~0.5."""
    hardness = 95 * np.exp(-3.0 * (cr - 0.6)**2
                           - 4.0 * (c - 0.7)**2
                           - 2.5 * (mo - 0.5)**2)
    hardness -= 0.5 * (ni - 0.5)**2
    hardness -= 0.3 * (si - 0.5)**2
    hardness -= 0.2 * (mn - 0.5)**2
    noise = _noisy_rng.normal(0, 1.5)
    return -(hardness + noise)


def alloy_hardness_true(cr, ni, mo, si, mn, c):
    """Noiseless version for true regret."""
    hardness = 95 * np.exp(-3.0 * (cr - 0.6)**2
                           - 4.0 * (c - 0.7)**2
                           - 2.5 * (mo - 0.5)**2)
    hardness -= 0.5 * (ni - 0.5)**2
    hardness -= 0.3 * (si - 0.5)**2
    hardness -= 0.2 * (mn - 0.5)**2
    return -hardness


# --------------------------------------------------------------------------- #
#  Non-smooth phase-transition landscapes (deterministic, no observation noise)
# --------------------------------------------------------------------------- #

def first_order_step_2d(x1, x2):
    """First-order discontinuity: two smooth basins separated by a step at x1=0.
    Domain: [-3,3]^2. Optimum (1.0, -0.5) -> 0.0 (right basin)."""
    left = 0.5 + (x1 + 1.0) ** 2 + (x2 - 0.5) ** 2
    right = (x1 - 1.0) ** 2 + (x2 + 0.5) ** 2
    return left if x1 < 0 else right


def pitchfork_bifurcation_2d(x, control):
    """Landau free energy F = x^4/4 - control*x^2/2; optimum branches at control=0.
    Domain: x in [-2,2], control in [-1,1]. Optimum control=1, x=+/-1 -> -0.25."""
    return 0.25 * x ** 4 - 0.5 * control * x ** 2


def phase_diagram_2d(composition, temperature):
    """Piecewise alpha/beta/liquid phase diagram with a curved liquidus boundary.
    Domain: composition in [0,1], temperature in [300,1500] K. Returns -hardness."""
    liquidus = 800.0 + 500.0 * (1.0 - 4.0 * (composition - 0.5) ** 2)
    if temperature >= liquidus:
        hardness = 10.0 - 0.002 * (temperature - liquidus)
    elif composition < 0.5:
        hardness = 70.0 - 0.05 * (temperature - 400.0) - 80.0 * (composition - 0.2) ** 2
    else:
        under = max(liquidus - temperature, 0.0)
        hardness = 50.0 + 60.0 * (composition - 0.5) - 0.0002 * (under - 300.0) ** 2
    return -hardness


def critical_cusp_2d(x1, x2):
    """Non-smooth critical point: |x1-1|^0.6 cusp at the optimum.
    Domain: [0,2]^2. Optimum (1.0, 1.0) -> 0.0."""
    return np.abs(x1 - 1.0) ** 0.6 + 0.5 * (x2 - 1.0) ** 2


# --------------------------------------------------------------------------- #
#  Seeded initial-data generator
# --------------------------------------------------------------------------- #

def generate_initial_data(func, bounds, n, col_names, seed=42):
    """Random initial data with controlled seed."""
    rng = np.random.RandomState(seed)
    n_dims = len(bounds)
    X = np.zeros((n, n_dims))
    for i, (lo, hi) in enumerate(bounds):
        X[:, i] = rng.uniform(lo, hi, n)
    y = np.array([func(*row) for row in X])
    data = {col_names[i]: X[:, i] for i in range(n_dims)}
    data["y"] = y
    return pd.DataFrame(data)
