"""Live validation: full-series change detection reaches the series planner.

16-spectrum synthetic temperature series (300-450 K, 10 K steps). A sharp
composition switch (peak at 900 dies, peak at 1300 grows) is planted at
405 K — between indices 10 and 11, strictly BETWEEN the visual scout
indices {0, 4, 8, 12, 15} (nearest scouts: 380 K and 420 K). The planner
can only place a regime boundary at index 10/11 by reading the computed
change point; the scouts alone bound it no tighter than (8, 12).

Checks:
  - scout log reports change detection on all 16 spectra, change point
    within one series step of 405 K;
  - the series plan has >= 2 regimes;
  - the boundary between regime 1 and regime 2 falls at index 10 or 11.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_scout_reduction_live.py [model]
"""
import io
import logging
import os
import re
import sys
import tempfile

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(11)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _make_stack(temps, t0=405.0, width=1.5):
    x = np.linspace(400, 1800, 700)
    frames = []
    for T in temps:
        w = 1 / (1 + np.exp(-(T - t0) / width))
        y = ((1 - w) * _g(x, 900, 40) + w * _g(x, 1300, 40)
             + RNG.normal(0, 0.004, x.size))
        frames.append(np.column_stack([x, y]))
    return np.stack(frames)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else MODEL

    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    temps = [300.0 + 10.0 * i for i in range(16)]
    stack = _make_stack(temps)
    out = tempfile.mkdtemp(prefix="scout_reduction_live_")
    print(f"output dir: {out}")

    agent = CurveFittingAgent(
        api_key=None, model_name=model_name, output_dir=out,
        enable_human_feedback=False, use_literature=False,
        max_verification_iterations=0,
    )

    capture = io.StringIO()
    handler = logging.StreamHandler(capture)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, capture)
    try:
        result = agent.analyze(
            stack,
            system_info={
                "technique": "in-situ optical spectroscopy during heating",
                "sample": "thin film, temperature ramp",
                "x_axis": "wavenumber (cm-1)",
                "y_axis": "intensity (a.u.)",
            },
            series_metadata={"variable": "temperature", "unit": "K",
                             "values": temps},
            objective=("Fit the spectral peaks across the temperature series "
                       "and track their evolution; report any transition "
                       "temperature you find."),
        )
    finally:
        sys.stdout = old_stdout
        logging.getLogger().removeHandler(handler)
    log = capture.getvalue()

    print("\nchecks:")
    m = re.search(
        r"Change detection \(all 16 spectra\): change point ≈ ([\d.]+)", log)
    check("change detection ran on all 16 spectra", m is not None)
    cp = float(m.group(1)) if m else float("nan")
    print(f"    detected change point: {cp}")
    check("change point within one step of planted 405 K",
          m is not None and abs(cp - 405.0) <= 10.0)

    # The locked-configuration banner is authoritative — the plan can be
    # logged twice (initial + post-validation revision). Fall back to the
    # plan-log / regime-banner formats.
    n_regimes, splits = 0, []
    locked = re.search(r"First-in-regime spectra \(full QC\): \[([^\]]*)\]",
                       log)
    if locked:
        firsts = [int(v) for v in re.findall(r"\d+", locked.group(1))]
        n_regimes, splits = len(firsts), sorted(firsts[1:])
    else:
        groups = (re.findall(r": indices \[([^\]]*)\], model:", log)
                  or re.findall(r"Spectra: indices \[([^\]]*)\]", log))
        regime_indices = [
            sorted(int(v) for v in re.findall(r"\d+", g)) for g in groups
        ]
        n_regimes = len(regime_indices)
        splits = sorted(r[0] for r in regime_indices[1:])
    print(f"    locked regimes: {n_regimes}, split point(s): {splits} "
          "(planted: 11)")
    check("series plan has >= 2 regimes", n_regimes >= 2)
    check("regime boundary at index 10/11 (sub-scout resolution)",
          any(s in (10, 11) for s in splits))

    status = (result or {}).get("status", "no result")
    print(f"    analyze status: {status}")
    check("analyze completed", isinstance(result, dict)
          and status not in ("error", "no result"))

    print("\n" + "=" * 50)
    npass = sum(checks.values())
    print(f"SCOUT REDUCTION LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
