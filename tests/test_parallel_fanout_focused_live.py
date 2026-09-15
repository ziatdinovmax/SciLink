"""
Focused live test: directly invokes UnifiedSeriesProcessingController.execute()
with a real Anthropic Claude model, isolating ONLY the code path changed by
the non-anchor parallel fan-out feature.

Bypasses the heavy upstream/downstream pipeline (preprocessing, novelty,
literature, adaptive refit, trend, synthesis, report) so the comparison
reflects the parallel section rather than serial post-processing.

Cases:
  1. Single-regime, 3 spectra: workers=1 vs workers=3 — compares wall-clock
     of the non-anchor section and verifies same success count / order.
  2. Multi-regime, 4 spectra (R0 = idx 0,1 / R1 = idx 2,3): workers=3 —
     verifies each anchor produces its own base_script, each non-anchor
     gets the right one, and results are tagged with the right regime.

Run with:
    UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \
        python tests/test_parallel_fanout_focused_live.py
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
log = logging.getLogger("focused_live")


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_two_peak_stack(n_spectra: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, 301)
    spectra = []
    for i in range(n_spectra):
        shift = 0.05 * i
        y = (
            _gauss(x, 1.0, 284.6 + shift, 0.45)
            + _gauss(x, 0.55, 286.5 + shift, 0.50)
            + 0.05 + rng.normal(0, 0.005, size=x.size)
        )
        spectra.append(np.stack([x, y]))
    return np.stack(spectra)


def _make_three_peak_spectrum(seed: int = 11, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, 301)
    y = (
        _gauss(x, 1.0, 284.6 + shift, 0.45)
        + _gauss(x, 0.55, 286.5 + shift, 0.50)
        + _gauss(x, 0.40, 288.8 + shift, 0.55)
        + 0.05 + rng.normal(0, 0.005, size=x.size)
    )
    return np.stack([x, y])


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

    def _parse_llm_response(resp):
        text = getattr(resp, "text", "") or ""
        # Strip a leading ```json / ``` if present
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else ""
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
        try:
            return json.loads(s), None
        except Exception as e:
            return None, str(e)

    model = LiteLLMGenerativeModel(
        model="claude-opus-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ctrl = UnifiedSeriesProcessingController(
        model=model,
        logger=log,
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_llm_response,
        executor=ScriptExecutor(timeout=120),
        script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
        correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions=FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
        output_dir=str(output_dir),
        plot_fn=plot_curve_to_bytes,
        r2_threshold=0.95,
        max_verification_iterations=1,  # cap LLM iterations per anchor
        enable_human_feedback=False,
        conformance_instructions=PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
        parallel_workers=parallel_workers,
    )
    return ctrl, plot_curve_to_bytes


def _two_gaussian_config() -> dict:
    return {
        "analysis_approach": "Fit a sum of two Gaussian peaks on a flat background.",
        "physical_model": (
            "y = A1 * exp(-(x-mu1)^2 / (2*sigma1^2)) "
            "+ A2 * exp(-(x-mu2)^2 / (2*sigma2^2)) + C, "
            "two Gaussian peaks (C-C ~284.6 eV, C-O ~286.5 eV) on a flat baseline."
        ),
        "parameters_to_extract": ["A1", "mu1", "sigma1", "A2", "mu2", "sigma2", "C"],
        "fitting_strategy": (
            "Use lmfit GaussianModel x2 + ConstantModel. Initial guesses: "
            "mu1=284.6, mu2=286.5; sigma=0.5; A from data near peak; bounds: "
            "sigma in [0.2, 1.0], mu within +/-1 eV of guess."
        ),
    }


def _three_gaussian_config() -> dict:
    return {
        "analysis_approach": "Fit a sum of three Gaussian peaks on a flat background.",
        "physical_model": (
            "y = sum_{i=1..3} Ai * exp(-(x-mui)^2 / (2*sigmai^2)) + C, "
            "three Gaussian peaks (C-C ~284.6, C-O ~286.5, C=O ~288.8 eV) on a "
            "flat baseline."
        ),
        "parameters_to_extract": ["A1","mu1","sigma1","A2","mu2","sigma2","A3","mu3","sigma3","C"],
        "fitting_strategy": (
            "Use lmfit GaussianModel x3 + ConstantModel. Initial guesses: "
            "mu1=284.6, mu2=286.5, mu3=288.8; sigma=0.5; A from data."
        ),
    }


def _build_state(stack: np.ndarray, plot_fn, regimes: list[dict] | None = None,
                 regime_configs: dict[int, dict] | None = None,
                 default_config: dict | None = None) -> dict:
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        UnifiedSeriesProcessingController,
    )
    helper = UnifiedSeriesProcessingController  # for _compute_statistics signature
    state = {
        "num_spectra": stack.shape[0],
        "is_single_spectrum": stack.shape[0] == 1,
        "spectrum_stack": stack,
        "spectrum_paths": [],
        "system_info": {
            "technique": "XPS",
            "edge": "C 1s",
            "sample": "synthetic two-Gaussian benchmark",
        },
        "data_statistics": {
            "n_points": stack.shape[2],
            "x_range": [float(stack[0, 0].min()), float(stack[0, 0].max())],
            "y_range": [float(stack[0, 1].min()), float(stack[0, 1].max())],
            "y_mean": float(stack[0, 1].mean()),
            "y_std": float(stack[0, 1].std()),
            "has_nans": False,
        },
        "original_plot_bytes": plot_fn(stack[0], {"technique": "XPS", "edge": "C 1s"}),
        "locked_fitting_config": default_config or _two_gaussian_config(),
        "first_spectrum_preprocessed": False,
        "skill_sections": {},
        "skill_name": "xps",
    }
    if regimes is not None:
        state["series_analysis_plan"] = {"regimes": regimes}
        state["regime_configs"] = regime_configs or {}
    return state


def run_single_regime(api_dir: Path) -> tuple[dict, dict]:
    log.info("=== Single-regime: 3 spectra, workers=1 vs workers=3 ===")
    stack = _make_two_peak_stack(n_spectra=3)

    serial_dir = api_dir / "single_serial"
    parallel_dir = api_dir / "single_parallel"

    log.info(">> workers=1 (serial)")
    ctrl_s, plot_fn = _build_controller(serial_dir, parallel_workers=1)
    state_s = _build_state(stack, plot_fn)
    t0 = time.perf_counter()
    out_s = ctrl_s.execute(state_s)
    t_serial = time.perf_counter() - t0
    log.info(f"   serial wall time: {t_serial:.1f}s")

    log.info(">> workers=3 (parallel)")
    ctrl_p, _ = _build_controller(parallel_dir, parallel_workers=3)
    state_p = _build_state(stack, plot_fn)
    t0 = time.perf_counter()
    out_p = ctrl_p.execute(state_p)
    t_parallel = time.perf_counter() - t0
    log.info(f"   parallel wall time: {t_parallel:.1f}s")

    def _summary(label, out, elapsed):
        rs = out["series_results"]
        return {
            "label": label,
            "wall_time_s": round(elapsed, 1),
            "count": len(rs),
            "success_count": sum(1 for r in rs if r.get("success")),
            "indices": [r.get("index") for r in rs],
            "r_squared": [(r.get("fit_quality") or {}).get("r_squared") for r in rs],
        }

    return _summary("serial", out_s, t_serial), _summary("parallel", out_p, t_parallel)


def run_multi_regime(api_dir: Path) -> dict:
    log.info("=== Multi-regime: 4 spectra (R0=2-peak, R1=3-peak), workers=3 ===")
    stack = np.stack([
        _make_two_peak_stack(n_spectra=1, seed=7)[0],
        _make_two_peak_stack(n_spectra=1, seed=8)[0],
        _make_three_peak_spectrum(seed=11, shift=0.0),
        _make_three_peak_spectrum(seed=12, shift=0.05),
    ])
    regimes = [
        {"name": "R0_two_peak", "spectrum_indices": [0, 1]},
        {"name": "R1_three_peak", "spectrum_indices": [2, 3]},
    ]
    regime_configs = {
        0: _two_gaussian_config(),
        1: _two_gaussian_config(),
        2: _three_gaussian_config(),
        3: _three_gaussian_config(),
    }

    out_dir = api_dir / "multi_parallel"
    ctrl, plot_fn = _build_controller(out_dir, parallel_workers=3)
    state = _build_state(stack, plot_fn, regimes=regimes,
                         regime_configs=regime_configs,
                         default_config=_two_gaussian_config())
    t0 = time.perf_counter()
    out = ctrl.execute(state)
    elapsed = time.perf_counter() - t0
    log.info(f"   multi-regime parallel wall time: {elapsed:.1f}s")

    rs = out["series_results"]
    summary = {
        "wall_time_s": round(elapsed, 1),
        "count": len(rs),
        "success_count": sum(1 for r in rs if r.get("success")),
        "indices": [r.get("index") for r in rs],
        "regimes": [r.get("regime") for r in rs],
        "scripts_distinct_per_regime": len({
            r.get("script") for r in rs if r.get("regime") == "R0_two_peak"
        }.union({"x"})) >= 1 and len({
            r.get("script") for r in rs if r.get("regime") == "R1_three_peak"
        }.union({"y"})) >= 1,
        "r_squared": [(r.get("fit_quality") or {}).get("r_squared") for r in rs],
    }

    # Verify each regime got its own base script (different anchor script text)
    r0_scripts = {r.get("script") for r in rs if r.get("regime") == "R0_two_peak"}
    r1_scripts = {r.get("script") for r in rs if r.get("regime") == "R1_three_peak"}
    summary["r0_distinct_from_r1"] = bool(r0_scripts and r1_scripts and r0_scripts.isdisjoint(r1_scripts))
    return summary


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    base = Path("tests/_parallel_fanout_focused_runs").resolve()
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    s_serial, s_parallel = run_single_regime(base)
    print("\nSINGLE-REGIME SERIAL  :", json.dumps(s_serial, indent=2))
    print("SINGLE-REGIME PARALLEL:", json.dumps(s_parallel, indent=2))

    multi = run_multi_regime(base)
    print("\nMULTI-REGIME PARALLEL :", json.dumps(multi, indent=2))

    # Pass criteria:
    ok = (
        s_serial["count"] == s_parallel["count"] == 3
        and s_serial["success_count"] == s_parallel["success_count"]
        and s_serial["indices"] == s_parallel["indices"] == [0, 1, 2]
        and multi["count"] == 4
        and multi["indices"] == [0, 1, 2, 3]
        and multi["regimes"] == ["R0_two_peak", "R0_two_peak", "R1_three_peak", "R1_three_peak"]
        and multi["r0_distinct_from_r1"]
    )
    if ok:
        print("\nFOCUSED LIVE TEST: PASS")
        return 0
    print("\nFOCUSED LIVE TEST: FAIL")
    return 2


if __name__ == "__main__":
    sys.exit(main())
