"""
Live test: per-call `parallel_workers` via the orchestrator's run_analysis tool.

Orchestrator is built WITHOUT curve_fit_parallel_workers (session default None
→ controller resolves to 1). The chat prompt asks the LLM to fit the series
using parallel workers; the LLM should pick up the new tool parameter on
run_analysis and pass `parallel_workers=3`.

Verifies the controller's parallel banner reports the per-call value, not
the session default — proving the LLM can override session policy when
appropriate.

Run with:
    UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=sk-ant-... \
        python tests/test_orchestrator_per_call_workers_live.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_series_dir(out_dir: Path, n_spectra: int = 3, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(282.0, 290.0, 401)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_spectra):
        shift = 0.04 * i
        y = (
            _gauss(x, 1.00, 284.6 + shift, 0.45)
            + _gauss(x, 0.55, 286.5 + shift, 0.50)
            + 0.05 + rng.normal(0, 0.005, size=x.size)
        )
        np.savetxt(
            out_dir / f"spec_{i:02d}.csv",
            np.column_stack([x, y]),
            delimiter=",", header="binding_energy,intensity", comments="",
        )
        with open(out_dir / f"spec_{i:02d}.json", "w") as f:
            json.dump({
                "experiment": {"technique": "XPS", "edge": "C 1s"},
                "sample": {"material": "synthetic"},
                "instrument": {"name": "test"},
                "series_index": i,
            }, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "experiment": {"technique": "XPS", "edge": "C 1s"},
            "sample": {"material": "synthetic two-Gaussian benchmark"},
            "instrument": {"name": "test_instrument"},
        }, f)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    base = Path("tests/_per_call_workers_runs").resolve()
    if base.exists():
        shutil.rmtree(base)
    data_dir = base / "data"
    _make_series_dir(data_dir, n_spectra=3)

    log_buf = io.StringIO()
    capture = logging.StreamHandler(log_buf)
    capture.setLevel(logging.INFO)
    capture.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    try:
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent, AnalysisMode,
        )

        # NOTE: curve_fit_parallel_workers is NOT set here — session default
        # is None → controller resolves to serial. The LLM must opt in via
        # the per-call tool param.
        orch = AnalysisOrchestratorAgent(
            base_dir=str(base / "session"),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name="claude-opus-4-6",
            analysis_mode=AnalysisMode.AUTONOMOUS,
        )

        # Explicit instruction: use parallel workers via the run_analysis tool.
        prompt = (
            f"Examine the data directory {data_dir}. "
            f"It contains 3 XPS C 1s spectra (CSV files with "
            f"'binding_energy,intensity' columns) forming a series. "
            f"Load metadata from {data_dir / 'meta.json'}. "
            f"Select the curve-fitting agent (agent_id=0). "
            f"Run the analysis with parallel_workers=3 so the non-anchor "
            f"spectra are fit concurrently. Use a two-Gaussian model on a "
            f"flat background (peaks near 284.6 and 286.5 eV). Keep "
            f"max_verification_iterations to a minimum — this is a smoke test."
        )
        print(f">>> chat prompt:\n{prompt}\n")

        t0 = time.perf_counter()
        reply = orch.chat(prompt)
        elapsed = time.perf_counter() - t0
        print(f"\n<<< chat returned in {elapsed:.1f}s")
        print(f"reply head: {reply[:400]}")
    finally:
        logging.getLogger().removeHandler(capture)

    log_text = log_buf.getvalue()

    # CRITICAL: banner must show 3 workers (per-call override), NOT 1 (session default)
    banner_correct = "Parallel non-anchor fan-out: up to 3 workers" in log_text
    banner_wrong = "Parallel non-anchor fan-out: up to 1 workers" in log_text
    drain = "Parallel non-anchor phase:" in log_text

    print("\n--- LOG SCAN ---")
    print(f"banner 'up to 3 workers' (per-call override): {'FOUND' if banner_correct else 'NOT FOUND'}")
    print(f"banner 'up to 1 workers' (session default leaked through): {'FOUND' if banner_wrong else 'NOT FOUND'}")
    print(f"drain phase log: {'FOUND' if drain else 'NOT FOUND'}")
    print(f"analysis_results count: {len(orch.analysis_results)}")
    if orch.analysis_results:
        last = orch.analysis_results[-1]
        print(f"last result status: {last.get('status')}")

    log_file = base / "session" / "captured.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(log_text)
    print(f"\nfull captured log: {log_file}")

    ok = banner_correct and drain and not banner_wrong
    print(f"\nPER-CALL WORKERS LIVE TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
