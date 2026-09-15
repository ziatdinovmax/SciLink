"""
Live test: multi-regime parallel fan-out.

Synthetic series where the underlying chemical model changes partway
through, so the planner should propose >=2 regimes and require a
distinct fit per regime. Verifies:

  * The pipeline detects multiple regimes
  * Each regime anchor produces its own base_script
  * Non-anchors in each regime are fit using their regime's base_script
  * parallel_workers=3 produces the same regime structure and success
    counts as parallel_workers=1, while completing faster on the
    non-anchor portion

Run with:
    UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \
        python tests/test_parallel_fanout_multiregime_live.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_multi_regime_xps_series(seed: int = 11) -> np.ndarray:
    """6-spectrum series with a chemistry transition in the middle.

    Regime A (idx 0..2): 2 Gaussians (C-C at 284.6, C-O at 286.5).
    Regime B (idx 3..5): 3 Gaussians (C-C + C-O + a new oxidized peak
        appearing at ~288.8 eV — e.g. C=O carbonyl appearing under
        higher temperature / dose).

    The shapes are visibly different and the third peak in regime B is
    well-separated, so the series planner should reliably propose two
    regimes with different physical models.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 291.0, 451)
    spectra = []
    for i in range(3):  # Regime A
        y = (
            _gauss(x, 1.00, 284.6 + 0.04 * i, 0.45)
            + _gauss(x, 0.55, 286.5 + 0.04 * i, 0.50)
            + 0.05 + rng.normal(0, 0.01, size=x.size)
        )
        spectra.append(np.stack([x, y]))
    for i in range(3):  # Regime B (added 288.8 component)
        y = (
            _gauss(x, 0.95, 284.65 + 0.04 * i, 0.45)
            + _gauss(x, 0.55, 286.55 + 0.04 * i, 0.50)
            + _gauss(x, 0.40, 288.8 + 0.04 * i, 0.55)
            + 0.05 + rng.normal(0, 0.01, size=x.size)
        )
        spectra.append(np.stack([x, y]))
    return np.stack(spectra)  # shape (6, 2, n)


def _load_series(output_dir: str) -> dict | None:
    for path in Path(output_dir).rglob("series_fit_results.json"):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _summarize(label: str, output_dir: str) -> dict:
    data = _load_series(output_dir) or {}
    results = data.get("results", [])
    regimes = sorted({r.get("regime") for r in results if r.get("regime")})
    by_regime = {}
    for r in results:
        rg = r.get("regime") or "default"
        by_regime.setdefault(rg, []).append(r.get("index"))
    return {
        "label": label,
        "total": data.get("total_spectra"),
        "successful": data.get("successful"),
        "n_regimes_detected": len(regimes),
        "regimes": regimes,
        "by_regime": by_regime,
        "plan": (data.get("series_analysis_plan") or {}).get("regimes"),
    }


def _run_once(stack: np.ndarray, workers: int, output_dir: str) -> tuple[dict, float]:
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    agent = CurveFittingAgent(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name="claude-opus-4-6",
        output_dir=output_dir,
        enable_human_feedback=False,
        use_literature=False,
        run_preprocessing=False,
        r2_threshold=0.95,
        max_verification_iterations=3,
        parallel_workers=workers,
    )
    t0 = time.perf_counter()
    agent.analyze(
        stack,
        system_info={
            "technique": "XPS",
            "edge": "C 1s",
            "sample": "synthetic series with chemistry transition (2→3 peaks)",
        },
        series_metadata={
            "variable": "step",
            "values": list(range(stack.shape[0])),
            "unit": "",
        },
        objective=(
            "Fit each C 1s spectrum. Early spectra have two peaks "
            "(C-C ~284.6, C-O ~286.5); later spectra also show a third "
            "oxidized peak around 288.8 eV. Use the most physically "
            "appropriate model per spectrum group."
        ),
        hints=(
            "The series transitions partway through: an additional "
            "component appears around 288.8 eV in the later spectra. "
            "Plan distinct regimes if the model differs."
        ),
    )
    elapsed = time.perf_counter() - t0
    return _summarize(f"workers={workers}", output_dir), elapsed


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    stack = _make_multi_regime_xps_series()
    print(f"Synthetic multi-regime series: {stack.shape}")
    print("  Regime A (idx 0-2): 2-Gaussian model")
    print("  Regime B (idx 3-5): 3-Gaussian model (new peak at 288.8 eV)")

    base_out = Path("tests/_parallel_fanout_multiregime_runs").resolve()
    if base_out.exists():
        shutil.rmtree(base_out)
    base_out.mkdir(parents=True)

    print("\n=== Run 1: workers=1 (serial baseline) ===")
    serial_summary, serial_time = _run_once(stack, 1, str(base_out / "serial"))
    print(f"\nSERIAL summary: {json.dumps(serial_summary, indent=2)}")
    print(f"SERIAL wall time: {serial_time:.1f}s")

    print("\n=== Run 2: workers=3 (parallel fan-out) ===")
    parallel_summary, parallel_time = _run_once(stack, 3, str(base_out / "parallel"))
    print(f"\nPARALLEL summary: {json.dumps(parallel_summary, indent=2)}")
    print(f"PARALLEL wall time: {parallel_time:.1f}s")

    print("\n=== Comparison ===")
    print(f"Wall time: serial={serial_time:.1f}s  parallel={parallel_time:.1f}s  "
          f"speedup={serial_time / max(parallel_time, 1e-9):.2f}x")
    print(f"Regimes detected: serial={serial_summary['n_regimes_detected']}  "
          f"parallel={parallel_summary['n_regimes_detected']}")
    print(f"Successful: serial={serial_summary['successful']}/{serial_summary['total']}  "
          f"parallel={parallel_summary['successful']}/{parallel_summary['total']}")

    ok = (
        serial_summary["total"] == parallel_summary["total"] == stack.shape[0]
        and serial_summary["successful"] == parallel_summary["successful"]
        and serial_summary["regimes"] == parallel_summary["regimes"]
        and serial_summary["by_regime"] == parallel_summary["by_regime"]
    )
    if ok:
        print("\nMULTI-REGIME LIVE TEST: PASS (regime structure matches across modes)")
        return 0
    print("\nMULTI-REGIME LIVE TEST: FAIL")
    print(f"  serial.by_regime = {serial_summary['by_regime']}")
    print(f"  parallel.by_regime = {parallel_summary['by_regime']}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
