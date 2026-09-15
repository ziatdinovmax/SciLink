"""
Live test that produces a *measurable* wall-clock speedup from the
non-anchor parallel fan-out.

Why the earlier focused test didn't show one:
  - Only 2 non-anchors per regime, each ~1s of subprocess work.
  - Serial: anchor(~50s) + 2×1s = 52s. Parallel saves ~1s. Inside
    LLM-variance noise.

This test:
  - 20 spectra, single regime → 1 anchor + 19 non-anchors.
  - Per-spectrum data is large enough that the subprocess fit is
    real work (not just import overhead).
  - Instruments the controller's logger so we time the *non-anchor
    phase* separately. That's where the change is supposed to help
    — measuring it directly removes anchor-LLM variance from the
    comparison.
  - Compares workers=1 vs workers=6.

Run with:
    UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \
        python tests/test_parallel_speedup_live.py
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("speedup_live")


# ---------------------------------------------------------------------------
# Synthetic data — large enough to make the subprocess fit non-trivial
# ---------------------------------------------------------------------------


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_stack(n_spectra: int = 20, n_points: int = 2000, seed: int = 7) -> np.ndarray:
    """Two-Gaussian XPS-like series. n_points is high so lmfit's fit
    of two Gaussians + constant takes a noticeable fraction of a second
    (the subprocess Python import is ~0.4–0.6s on its own)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, n_points)
    spectra = []
    for i in range(n_spectra):
        shift = 0.04 * (i / max(n_spectra - 1, 1))
        y = (
            _gauss(x, 1.00, 284.6 + shift, 0.45)
            + _gauss(x, 0.55, 286.5 + shift, 0.50)
            + 0.05 + rng.normal(0, 0.005, size=x.size)
        )
        spectra.append(np.stack([x, y]))
    return np.stack(spectra)


# ---------------------------------------------------------------------------
# Phase-timing logging handler
# ---------------------------------------------------------------------------


class PhaseTimer(logging.Handler):
    """Times distinct phases of the series fit by watching for log markers.

    Markers (emitted by UnifiedSeriesProcessingController):
      "Fitting:" of spectrum 0           → start of anchor
      "Base fitting script locked"       → end of anchor / start of non-anchor
      "Fitting complete"                 → end of non-anchor

    In serial mode the non-anchor phase is the interleaved loop work
    that happens after the anchor finishes; same wall-clock observable.
    """

    def __init__(self):
        super().__init__()
        self.t_start = None
        self.t_anchor_done = None
        self.t_end = None

    def emit(self, record):
        msg = record.getMessage()
        now = time.perf_counter()
        if self.t_start is None and "FITTING: SERIES" in msg:
            self.t_start = now
        elif self.t_anchor_done is None and "Base fitting script locked" in msg:
            self.t_anchor_done = now
        elif "Fitting complete" in msg:
            self.t_end = now

    def report(self) -> dict:
        if not (self.t_start and self.t_anchor_done and self.t_end):
            return {"complete": False}
        return {
            "complete": True,
            "anchor_s": round(self.t_anchor_done - self.t_start, 2),
            "non_anchor_s": round(self.t_end - self.t_anchor_done, 2),
            "total_s": round(self.t_end - self.t_start, 2),
        }


# ---------------------------------------------------------------------------
# Controller construction (bypasses agent's downstream pipeline tail)
# ---------------------------------------------------------------------------


def _build_controller(output_dir: Path, parallel_workers: int):
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        UnifiedSeriesProcessingController,
    )
    from scilink.agents.exp_agents.instruct import (
        FITTING_SCRIPT_INSTRUCTIONS,
        FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
        PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
    )
    from scilink.executors import ScriptExecutor
    from scilink.skills._shared.curve_fitting_tools import plot_curve_to_bytes

    def _parse(resp):
        s = (getattr(resp, "text", "") or "").strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else ""
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        try:
            return json.loads(s), None
        except Exception as e:
            return None, str(e)

    output_dir.mkdir(parents=True, exist_ok=True)
    ctrl = UnifiedSeriesProcessingController(
        model=LiteLLMGenerativeModel(model="claude-opus-4-6", api_key=os.environ["ANTHROPIC_API_KEY"]),
        logger=log,
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse,
        executor=ScriptExecutor(timeout=120),
        script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
        correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions=FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
        output_dir=str(output_dir),
        plot_fn=plot_curve_to_bytes,
        r2_threshold=0.95,
        max_verification_iterations=1,
        enable_human_feedback=False,
        conformance_instructions=PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
        parallel_workers=parallel_workers,
    )
    return ctrl, plot_curve_to_bytes


def _two_gaussian_config() -> dict:
    return {
        "analysis_approach": "Fit a sum of two Gaussian peaks on a flat background.",
        "physical_model": (
            "y = A1*exp(-(x-mu1)^2 / (2*sigma1^2)) + "
            "A2*exp(-(x-mu2)^2 / (2*sigma2^2)) + C; "
            "two Gaussian peaks (~284.6 eV and ~286.5 eV) on a flat baseline."
        ),
        "parameters_to_extract": ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2", "C"],
        "fitting_strategy": (
            "Use lmfit GaussianModel x2 + ConstantModel. Initial guesses: "
            "mu1=284.6, mu2=286.5; sigma=0.5; A from data."
        ),
    }


def _build_state(stack: np.ndarray, plot_fn) -> dict:
    return {
        "num_spectra": stack.shape[0],
        "is_single_spectrum": False,
        "spectrum_stack": stack,
        "spectrum_paths": [],
        "system_info": {"technique": "XPS", "edge": "C 1s", "sample": "synthetic"},
        "data_statistics": {
            "n_points": stack.shape[2],
            "x_range": [float(stack[0, 0].min()), float(stack[0, 0].max())],
            "y_range": [float(stack[0, 1].min()), float(stack[0, 1].max())],
            "y_mean": float(stack[0, 1].mean()),
            "y_std": float(stack[0, 1].std()),
            "has_nans": False,
        },
        "original_plot_bytes": plot_fn(stack[0], {"technique": "XPS", "edge": "C 1s"}),
        "locked_fitting_config": _two_gaussian_config(),
        "first_spectrum_preprocessed": False,
        "skill_sections": {},
        "skill_name": "xps",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _run(stack: np.ndarray, workers: int, out_dir: Path) -> dict:
    log.info(f">> workers={workers}")
    ctrl, plot_fn = _build_controller(out_dir, parallel_workers=workers)
    timer = PhaseTimer()
    logging.getLogger().addHandler(timer)
    try:
        t0 = time.perf_counter()
        out = ctrl.execute(_build_state(stack, plot_fn))
        wall = time.perf_counter() - t0
    finally:
        logging.getLogger().removeHandler(timer)
    rs = out["series_results"]
    phases = timer.report()
    success = sum(1 for r in rs if r.get("success"))
    return {
        "workers": workers,
        "n_spectra": stack.shape[0],
        "success": f"{success}/{len(rs)}",
        "wall_total_s": round(wall, 2),
        "phases": phases,
    }


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    N = 20
    WORKERS = 6
    stack = _make_stack(n_spectra=N, n_points=2000)
    log.info(f"Synthetic series: {stack.shape}  (anchor + {N-1} non-anchors)")

    base = Path("tests/_parallel_speedup_runs").resolve()
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    serial = _run(stack, 1, base / "serial")
    print(f"\nSERIAL:   {json.dumps(serial, indent=2)}")

    parallel = _run(stack, WORKERS, base / "parallel")
    print(f"\nPARALLEL: {json.dumps(parallel, indent=2)}")

    # The interesting comparison: non-anchor section only
    if serial["phases"].get("complete") and parallel["phases"].get("complete"):
        s_na = serial["phases"]["non_anchor_s"]
        p_na = parallel["phases"]["non_anchor_s"]
        speedup = s_na / max(p_na, 1e-9)
        print("\n--- WALL-CLOCK COMPARISON ---")
        print(f"Anchor phase (sequential, LLM-bound):")
        print(f"  serial:   {serial['phases']['anchor_s']:>6.2f}s")
        print(f"  parallel: {parallel['phases']['anchor_s']:>6.2f}s")
        print(f"Non-anchor phase ({N-1} spectra, this is what the feature parallelizes):")
        print(f"  serial:   {s_na:>6.2f}s   ({s_na / (N-1):.2f}s per spectrum)")
        print(f"  parallel: {p_na:>6.2f}s   ({p_na / (N-1):.2f}s per spectrum-equiv with {WORKERS} workers)")
        print(f"  speedup:  {speedup:.2f}x   (theoretical max ≈ {min(N-1, WORKERS)}x)")
        print(f"Total wall time:")
        print(f"  serial:   {serial['phases']['total_s']:>6.2f}s")
        print(f"  parallel: {parallel['phases']['total_s']:>6.2f}s")

    ok = (
        serial["success"] == parallel["success"] == f"{N}/{N}"
        and parallel["phases"].get("complete")
        and parallel["phases"]["non_anchor_s"] < 0.7 * serial["phases"]["non_anchor_s"]
    )
    print(f"\nSPEEDUP TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
