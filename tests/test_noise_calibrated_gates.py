"""Gates calibrated on the reference, in units of its own noise.

Observed on a real STEM-EELS line scan: R² tracked signal strength (the approved
reference sat at 0.91) and the peak-counting fingerprint wobbled on noise, so
three frames in four were flagged and the loop re-anchored twice for nothing.
These tests pin the two calibrations and what the loop does with them.
"""

from pathlib import Path

import numpy as np
import pytest

from scilink.live import MeasurementLoop
from scilink.live.gates import DEFAULT_DRIFT_BAR, calibrate, residual_excess

X = np.linspace(0.0, 10.0, 800)


def peak(center=5.0, height=1.0, width=0.6):
    return height * np.exp(-0.5 * ((X - center) / width) ** 2)


def write_run(d: Path, y, fit):
    (d / "spectrum_0000").mkdir(parents=True, exist_ok=True)
    np.save(d / "spectrum_0000" / "data.npy", np.column_stack([X, y]))
    np.save(d / "spectrum_0000" / "fit.npy", np.asarray(fit))
    return str(d)


def noisy(signal, sigma, seed=0, correlated=False):
    n = np.random.default_rng(seed).normal(0, sigma, X.size)
    if correlated:                                   # a detector point-spread: neighbours share noise
        n = np.convolve(n, [0.25, 0.5, 0.25], mode="same") * 1.6
    return signal + n


class TestResidualExcess:
    def test_about_one_for_a_right_model_whatever_the_signal(self, tmp_path):
        weak = residual_excess(write_run(tmp_path / "w", noisy(peak(height=1), 0.3), peak(height=1)))
        strong = residual_excess(write_run(tmp_path / "s", noisy(peak(height=30), 0.3), peak(height=30)))
        assert weak == pytest.approx(1.0, abs=0.15) and strong == pytest.approx(1.0, abs=0.15)
        r2 = lambda y, f: 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)  # noqa: E731
        assert r2(noisy(peak(height=1), 0.3), peak(height=1)) < 0.6          # ...while R² calls it poor

    def test_jumps_when_the_model_misses_a_feature(self, tmp_path):
        y = noisy(peak() + peak(center=8.0, height=0.6), 0.05)
        assert residual_excess(write_run(tmp_path / "m", y, peak())) > 2.5

    def test_unavailable_is_none_not_an_error(self, tmp_path):
        assert residual_excess(str(tmp_path / "nothing")) is None
        d = write_run(tmp_path / "sub", noisy(peak(), 0.1), peak()[:100])     # fit over a sub-window
        assert residual_excess(d) is None


class TestCalibrate:
    def test_clean_data_keeps_the_constant_bar(self, tmp_path):
        cal = calibrate(write_run(tmp_path / "c", noisy(peak(height=50), 0.05), peak(height=50)))
        assert cal["drift_floor"] == DEFAULT_DRIFT_BAR            # never tightened, not relaxed
        assert cal["residual_excess"] == pytest.approx(1.0, abs=0.15)

    def test_noisy_data_lowers_the_drift_bar_and_is_deterministic(self, tmp_path):
        d = write_run(tmp_path / "n", noisy(peak(height=1), 0.35, correlated=True), peak(height=1))
        cal = calibrate(d)
        assert cal["drift_floor"] < DEFAULT_DRIFT_BAR
        assert cal == calibrate(d)

    def test_no_arrays_no_calibration(self, tmp_path):
        assert calibrate(str(tmp_path)) == {}


# ── in the loop ──────────────────────────────────────────────────

def _loop(tmp_path, ref_y, ref_fit, frames, **kw):
    """A loop armed on a reference run with arrays; each frame's fake agent
    writes that frame's arrays and returns the agent-side verdict given."""
    from tests.test_measurement_loop import make_anchor
    anchor = make_anchor(tmp_path)
    write_run(anchor, ref_y, ref_fit)

    class Agent:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def analyze(self, data, **k):
            y, fit, r2, sim = frames[Path(str(data)).name]
            write_run(Path(self.output_dir), y, fit)
            return {"status": "success", "output_directory": self.output_dir,
                    "fitting_parameters": {"peak_1": {"center": 5.0}},
                    "fit_quality": {"r_squared": r2},
                    "reuse_validity": {"verdict": "good" if r2 >= 0.95 else "poor", "r_squared": r2,
                                       "threshold": 0.95, "fingerprint_similarity": sim,
                                       "drift": "suspected" if sim < 0.92 else "none"},
                    "stage_timings": {"llm_calls": 0}}
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=Agent, **kw)
    loop.setup(anchor=str(anchor))
    return loop


REF_Y, REF_FIT = noisy(peak(height=1), 0.35, correlated=True), peak(height=1)


def test_a_frame_as_good_as_the_reference_is_clean_whatever_its_r_squared(tmp_path):
    loop = _loop(tmp_path, REF_Y, REF_FIT, {
        "weak.csv": (noisy(peak(height=0.8), 0.35, seed=1, correlated=True), peak(height=0.8), 0.88, 0.85)})
    rec = loop.step("weak.csv")
    assert rec["flags"] == [] and rec["gate"]["accepted_in_noise_units"] is True
    assert rec["gate"]["verdict"] == "poor"                   # the agent's own verdict is kept on record
    assert loop.read_log()[0]["gate_calibration"]["drift_floor"] < 0.92


def test_a_missed_feature_is_still_poor_and_a_real_change_still_drifts(tmp_path):
    changed = noisy(peak(height=1) + peak(center=8.0, height=2.0, width=0.3), 0.35, seed=2, correlated=True)
    loop = _loop(tmp_path, REF_Y, REF_FIT, {"changed.csv": (changed, peak(height=1), 0.55, 0.40)},
                 breach_patience=1)
    rec = loop.step("changed.csv")
    assert set(rec["flags"]) == {"gate_poor", "drift_suspected"} and rec["needs_escalation"] is True
    assert rec["gate"]["residual_excess"] > 1.5 * rec["gate"]["reference_excess"]


def test_none_keeps_the_constant_bars(tmp_path):
    loop = _loop(tmp_path, REF_Y, REF_FIT, {
        "weak.csv": (noisy(peak(height=0.8), 0.35, seed=1, correlated=True), peak(height=0.8), 0.88, 0.85)},
        noise_gate_tolerance=None)
    assert set(loop.step("weak.csv")["flags"]) == {"gate_poor", "drift_suspected"}


def test_calibration_survives_a_restart(tmp_path):
    loop = _loop(tmp_path, REF_Y, REF_FIT, {})
    again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=loop._agent_factory)
    assert again._calibration == loop._calibration and again._calibration
