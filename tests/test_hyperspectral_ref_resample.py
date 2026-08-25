"""Offline regression test for auto-resampling a 1D reference operand.

A tabulated reference (cross-section / attenuation / standard) pulled from a
database is sampled on its OWN axis, never the detector's channel grid. The
operand-alignment gate used to drop any 1D auxiliary whose length != channel
count, so such a reference could not be used by the per-pixel codegen unless
the user pre-resampled it. The gate now resamples a 1D reference that carries
its own axis onto the signal axis, yielding an index-aligned per-channel
operand (codegen contract unchanged). These checks are deterministic.

  conda run -n scilink python -m pytest tests/test_hyperspectral_ref_resample.py -q
"""
import numpy as np

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    _resample_ref_to_signal_axis,
)


def _mu_like(n):
    """A step-at-80.8-keV reference (mimics an Au K-edge attenuation table)."""
    e = np.linspace(1.0, 200.0, n)
    v = np.where(e < 80.8, 2.15, 8.34)
    return e, v


def test_resamples_reference_onto_signal_axis():
    ref_axis, ref_vals = _mu_like(1990)            # foreign grid, foreign length
    energy_axis = np.linspace(0.1, 204.8, 2048)    # data signal axis
    out = _resample_ref_to_signal_axis(ref_vals, ref_axis, energy_axis, 2048)
    assert out is not None and out.shape == (2048,)
    # step preserved across the edge
    assert out[np.argmin(np.abs(energy_axis - 79))] < 3.0
    assert out[np.argmin(np.abs(energy_axis - 82))] > 7.0


def test_unsorted_reference_axis_is_handled():
    ref_axis, ref_vals = _mu_like(500)
    perm = np.random.RandomState(0).permutation(ref_axis.size)
    energy_axis = np.linspace(0.1, 204.8, 2048)
    out = _resample_ref_to_signal_axis(ref_vals[perm], ref_axis[perm], energy_axis, 2048)
    assert out is not None
    assert out[np.argmin(np.abs(energy_axis - 82))] > 7.0


def test_guards_return_none():
    energy_axis = np.linspace(0.1, 204.8, 2048)
    ref_axis, ref_vals = _mu_like(1990)
    # no native axis -> cannot resample
    assert _resample_ref_to_signal_axis(ref_vals, None, energy_axis, 2048) is None
    # 2D operand -> not a 1D reference
    assert _resample_ref_to_signal_axis(np.ones((5, 5)), np.arange(5), energy_axis, 2048) is None
    # axis length mismatched to values
    assert _resample_ref_to_signal_axis(ref_vals, ref_axis[:-1], energy_axis, 2048) is None
    # no signal axis available (degenerate)
    assert _resample_ref_to_signal_axis(ref_vals, ref_axis, None, 2048) is None
    # signal axis shape doesn't match channel count
    assert _resample_ref_to_signal_axis(ref_vals, ref_axis, energy_axis, 1024) is None
    # non-overlapping axes (different physical quantity) -> don't fabricate
    assert _resample_ref_to_signal_axis(
        np.ones(100), np.linspace(400, 600, 100), energy_axis, 2048) is None


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
