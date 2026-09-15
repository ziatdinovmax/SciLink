"""
Live test: non-anchor parallel fan-out in UnifiedSeriesProcessingController.

Runs CurveFittingAgent on a synthetic 4-spectrum XPS-like series twice:
1) parallel_workers=1 (serial, baseline behavior)
2) parallel_workers=3 (parallel fan-out for non-anchors)

Compares wall-clock time and verifies result equivalence.

Run with:
    UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \
        python tests/test_parallel_fanout_live.py
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


def _make_synthetic_xps_series(n_spectra: int = 4, n_points: int = 401,
                               seed: int = 7) -> tuple[np.ndarray, list[float]]:
    """Build N XPS-like spectra: 2 Gaussian peaks on a flat background.

    Peak 1 (C-C):   center ≈ 284.6 eV
    Peak 2 (C-O):   center ≈ 286.5 eV
    Small per-spectrum shift simulates a temperature/dose series.
    """
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


def _summarize_result(label: str, result: dict) -> dict:
    """Pull the minimal compare-worthy fields out of a result dict."""
    if not isinstance(result, dict):
        return {"label": label, "status": "non-dict"}
    out = {"label": label, "status": result.get("status")}
    series = (result.get("claims_generation") or {}).get("series_results")
    if series is None:
        # Fall back to disk: series_results.json under output_directory.
        outdir = result.get("output_directory")
        if outdir:
            for candidate in Path(outdir).rglob("series_fit_results.json"):
                try:
                    with open(candidate) as f:
                        series = json.load(f).get("results", [])
                    break
                except Exception:
                    pass
    if series:
        out["spectra"] = [
            {
                "index": r.get("index"),
                "success": r.get("success"),
                "model_type": r.get("model_type"),
                "r_squared": (r.get("fit_quality") or {}).get("r_squared"),
                "n_components": (r.get("fit_quality") or {}).get("n_components")
                or len(r.get("parameters", {}).get("components", []))
                or None,
            }
            for r in series
        ]
        out["successful"] = sum(1 for s in out["spectra"] if s["success"])
        out["count"] = len(out["spectra"])
    return out


def _run_once(stack: np.ndarray, workers: int, output_dir: str) -> tuple[dict, float]:
    """Run CurveFittingAgent on the synthetic stack and return (summary, wall_time)."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    agent = CurveFittingAgent(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name="claude-opus-4-6",
        output_dir=output_dir,
        enable_human_feedback=False,
        use_literature=False,
        run_preprocessing=False,  # skip preprocessing LLM call for a tighter test
        r2_threshold=0.95,
        max_verification_iterations=3,  # cap to keep test runtime bounded
        parallel_workers=workers,
    )
    t0 = time.perf_counter()
    result = agent.analyze(
        stack,
        system_info={
            "technique": "XPS",
            "edge": "C 1s",
            "sample": "synthetic two-Gaussian benchmark",
        },
        series_metadata={
            "variable": "step",
            "values": list(range(stack.shape[0])),
            "unit": "",
        },
        objective="Fit each spectrum with a two-Gaussian model on a flat background.",
    )
    elapsed = time.perf_counter() - t0
    return _summarize_result(f"workers={workers}", result), elapsed


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    stack, shifts = _make_synthetic_xps_series(n_spectra=4)
    print(f"Synthetic series: {stack.shape}, per-spectrum shifts (eV): {shifts}")

    base_out = Path("tests/_parallel_fanout_runs").resolve()
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
    s_ok = serial_summary.get("successful")
    p_ok = parallel_summary.get("successful")
    s_n = serial_summary.get("count")
    p_n = parallel_summary.get("count")
    print(f"Successful: serial={s_ok}/{s_n}  parallel={p_ok}/{p_n}")

    ok = (s_ok == p_ok == s_n == p_n and s_n == stack.shape[0])
    if ok:
        print("\nLIVE TEST: PASS")
        return 0
    print("\nLIVE TEST: FAIL (success counts diverge or are below N)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
