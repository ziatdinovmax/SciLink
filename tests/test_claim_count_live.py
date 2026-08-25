"""Live test for the scientific_claims count cap.

Runs CurveFittingAgent twice — once on a single XPS-like spectrum, once on
a 4-spectrum XPS series — and prints ``len(scientific_claims)``. Pre-cap
runs emitted 3-5 claims per analysis; the cap (PR #204) targets 1 for
single, ≤2 for series.

Pass / fail (soft):
  - single: len(scientific_claims) == 1 expected (≤2 allowed)
  - series: len(scientific_claims) <= 2 required

Run:
  UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \\
      python tests/test_claim_count_live.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_single_xps_spectrum(n_points: int = 401, seed: int = 11) -> np.ndarray:
    """Single XPS-like C 1s spectrum (2 Gaussians on flat background)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, n_points)
    y = (
        _gauss(x, 1.0, 284.6, 0.45)
        + _gauss(x, 0.55, 286.5, 0.50)
        + 0.05
        + rng.normal(0, 0.01, size=n_points)
    )
    return np.column_stack([x, y])


def _make_xps_series(n_spectra: int = 4, n_points: int = 401, seed: int = 7):
    """4-spectrum XPS series with per-spectrum shift simulating a dose step."""
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, n_points)
    shifts = np.linspace(-0.05, 0.10, n_spectra)
    stack = np.zeros((n_spectra, 2, n_points), dtype=np.float64)
    for i, shift in enumerate(shifts):
        y = (
            _gauss(x, 1.0, 284.6 + shift, 0.45)
            + _gauss(x, 0.55, 286.5 + shift, 0.50)
            + 0.05
            + rng.normal(0, 0.01, size=n_points)
        )
        stack[i, 0] = x
        stack[i, 1] = y
    return stack, shifts.tolist()


def _make_agent(out_dir: str):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    return CurveFittingAgent(
        api_key=api_key,
        model_name="claude-opus-4-6",
        output_dir=out_dir,
        enable_human_feedback=False,
        use_literature=False,
        run_preprocessing=False,
        r2_threshold=0.95,
        max_verification_iterations=2,
    )


def _count_claims(result: dict) -> tuple[int, list[dict]]:
    """Return (count, claims) — the latter for spot-checking content."""
    claims = result.get("scientific_claims", []) or []
    return len(claims), claims


def run_single() -> int:
    print("\n=== SINGLE SPECTRUM (XPS C 1s, 2-Gaussian) ===")
    spectrum = _make_single_xps_spectrum()
    out_dir = tempfile.mkdtemp(prefix="claim_count_single_")
    print(f"output: {out_dir}")

    agent = _make_agent(out_dir)
    result = agent.analyze(
        spectrum,
        system_info={
            "technique": "XPS",
            "edge": "C 1s",
            "sample": "synthetic two-Gaussian benchmark",
        },
        objective="Fit with two Gaussian peaks on a flat background.",
    )

    n, claims = _count_claims(result)
    print(f"\n[single] len(scientific_claims) = {n}")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c.get('claim','?')[:140]}")

    if n == 1:
        print("[single] PASS (n=1, ideal)")
        return 0
    elif n == 2:
        print("[single] PASS-with-caveat (n=2, allowed only for two genuinely independent findings)")
        return 0
    else:
        print(f"[single] FAIL (n={n}, exceeds cap of 2)")
        return 1


def run_series() -> int:
    print("\n=== SERIES (4 XPS spectra with shifting peaks) ===")
    stack, shifts = _make_xps_series(n_spectra=4)
    out_dir = tempfile.mkdtemp(prefix="claim_count_series_")
    print(f"output: {out_dir}")
    print(f"per-spectrum shifts (eV): {shifts}")

    agent = _make_agent(out_dir)
    result = agent.analyze(
        stack,
        system_info={
            "technique": "XPS",
            "edge": "C 1s",
            "sample": "synthetic two-Gaussian dose series",
        },
        series_metadata={
            "variable": "dose_step",
            "values": list(range(stack.shape[0])),
            "unit": "",
        },
        objective="Fit each spectrum with two Gaussians; report the trend in peak centers across the series.",
    )

    n, claims = _count_claims(result)
    print(f"\n[series] len(scientific_claims) = {n}")
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c.get('claim','?')[:140]}")

    if n <= 2:
        print(f"[series] PASS (n={n}, within cap of 2)")
        return 0
    else:
        print(f"[series] FAIL (n={n}, exceeds cap of 2)")
        return 1


if __name__ == "__main__":
    rc = run_single() + run_series()
    sys.exit(rc)
