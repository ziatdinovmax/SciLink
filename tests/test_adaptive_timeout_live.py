"""
Live test: adaptive timeout escalation in
UnifiedSeriesProcessingController._execute_with_adaptive_timeout.

Constructs the controller directly with a tiny executor timeout
(1 s) and a hand-built state, then runs a non-anchor fit on a
synthetic spectrum. The non-anchor fast path adapts a deliberately
slow base script (`time.sleep(3)` before emitting outputs); first
attempt times out at 1 s, helper escalates to 2 s (still times out),
escalates again to 4 s (succeeds).

Verifies:
  - escalation log lines appear (1s → 2s → 4s)
  - the controller does NOT route through _correct_script (no LLM
    "fix" calls for code that was never broken)
  - the fit ultimately succeeds

Run with:
    UNSAFE_EXECUTION_OK=true python tests/test_adaptive_timeout_live.py
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
from unittest.mock import MagicMock

import numpy as np


def main() -> int:
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    base = Path("tests/_adaptive_timeout_runs").resolve()
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    # Capture log
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    try:
        from scilink.executors import ScriptExecutor
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController,
        )

        # Tiny base timeout — guaranteed to fire on the slow script
        executor = ScriptExecutor(timeout=1)

        ctrl = UnifiedSeriesProcessingController(
            model=MagicMock(),
            logger=logging.getLogger("test_adaptive_timeout"),
            generation_config=None,
            safety_settings=None,
            parse_fn=lambda r: (json.loads(r.text), None),
            executor=executor,
            script_instructions="",
            correction_instructions="",
            quality_instructions="",
            output_dir=str(base),
            plot_fn=MagicMock(return_value=b""),
            r2_threshold=0.9,
            max_verification_iterations=1,
            enable_human_feedback=False,
            conformance_instructions="",
            parallel_workers=1,
        )

        # If _correct_script gets called, that's a bug — a timeout shouldn't
        # route through script-correction. Wrap it to detect any call.
        correction_calls = []
        orig_correct = ctrl._correct_script
        def _intercept(*a, **kw):
            correction_calls.append((a, kw))
            return orig_correct(*a, **kw)
        ctrl._correct_script = _intercept

        # Base script: sleeps 3s, then emits the expected outputs.
        # First call w/ timeout=1 → kills it. Helper retries with 2 (still
        # kills), then with 4 (succeeds).
        base_script_template = """
import time, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
time.sleep(3)
np.load("DATA_PATH_PLACEHOLDER")
fig, ax = plt.subplots()
ax.plot([0,1],[0,1])
fig.savefig("OUTPUT_PREFIX_PLACEHOLDER_fit.png")
print('FIT_RESULTS_JSON:' + json.dumps({
    "model_type": "synthetic",
    "parameters": {},
    "fit_quality": {"r_squared": 0.99}
}))
"""

        # Stage a curve_data file the adapted script will load
        spectrum_idx = 1
        temp_data_path = base / f"temp_spectrum_{spectrum_idx}.npy"
        x = np.linspace(0, 1, 100)
        y = np.exp(-((x - 0.5) ** 2) / 0.02)
        np.save(temp_data_path, np.stack([x, y]))

        # Build a base script string in the shape _adapt_script_for_spectrum
        # expects (it greps for `temp_spectrum_\d+.npy` and `spectrum_\d{4}_fit.png`).
        base_script = base_script_template \
            .replace("DATA_PATH_PLACEHOLDER", str(base / "temp_spectrum_0.npy")) \
            .replace("OUTPUT_PREFIX_PLACEHOLDER", "spectrum_0000")

        state = {
            "num_spectra": 2,
            "is_single_spectrum": False,
            "locked_fitting_config": {
                "analysis_approach": "synthetic",
                "physical_model": "synthetic",
                "parameters_to_extract": [],
                "fitting_strategy": "synthetic",
            },
            "skill_sections": {},
            "skill_name": "synthetic",
        }

        t0 = time.perf_counter()
        result = ctrl._fit_single_spectrum(
            state=state,
            curve_data=np.stack([x, y]),
            data_path=str(temp_data_path),
            spectrum_name=f"spectrum_{spectrum_idx:04d}",
            spectrum_idx=spectrum_idx,
            base_script=base_script,
        )
        elapsed = time.perf_counter() - t0
    finally:
        logging.getLogger().removeHandler(handler)

    log_text = log_buf.getvalue()
    print(f"\nelapsed: {elapsed:.1f}s")
    print(f"result.success: {result.get('success')}")
    print(f"result.error: {(result.get('error') or '')[:80]}")
    print(f"_correct_script invocations: {len(correction_calls)}")

    esc1 = "retrying same script with 2s" in log_text
    esc2 = "retrying same script with 4s" in log_text
    print(f"saw escalation 1 (1→2s): {esc1}")
    print(f"saw escalation 2 (2→4s): {esc2}")

    # Persist log
    log_file = base / "captured.log"
    log_file.write_text(log_text)
    print(f"\nfull log: {log_file}")

    ok = (
        result.get("success") is True
        and esc1 and esc2
        and len(correction_calls) == 0
    )
    print(f"\nADAPTIVE TIMEOUT LIVE TEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
