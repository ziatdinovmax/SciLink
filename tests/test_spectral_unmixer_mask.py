"""Spectral unmixer validity mask + error propagation (issue #381).

The background mask must be scale- and sign-invariant: signed dI/dV-like
cubes (per-pixel integral ~0, magnitudes far below counts scale) previously
dropped ALL pixels and crashed decomposition with "Too few valid pixels (0)",
and the real exception was then masked downstream by the generic
"No iteration results found for synthesis".
"""

import logging

import numpy as np
import pytest

from scilink.skills.hyperspectral.eels.spectral_unmixer import SpectralUnmixer


RNG = np.random.RandomState(0)


def _counts_cube(h=12, w=12, e=32, masked_quarter=True):
    """Non-negative counts-scale cube with an exactly-zero masked region."""
    cube = RNG.rand(h, w, e).astype(float) * 500.0 + 10.0
    if masked_quarter:
        cube[: h // 2, : w // 2, :] = 0.0
    return cube


def _signed_didv_cube(h=12, w=12, e=32, scale=1e-10):
    """Signed, tiny-magnitude cube whose per-pixel integral is ~0 (dI/dV-like)."""
    x = np.linspace(-1, 1, e)
    base = np.sin(np.pi * x)  # antisymmetric → integrates to ~0
    cube = base[None, None, :] * (1 + 0.1 * RNG.rand(h, w, 1))
    cube += 0.01 * RNG.randn(h, w, e)
    return (cube * scale).astype(np.float64)


# ---------------------------------------------------------------------------
# Regression: non-negative counts data (the mask's original modality)
# ---------------------------------------------------------------------------

def test_counts_cube_masked_region_still_dropped():
    cube = _counts_cube()
    h, w, e = cube.shape
    um = SpectralUnmixer(method="nmf", n_components=3)
    components, maps = um.fit(cube)
    assert components.shape == (3, e)
    assert maps.shape == (h, w, 3)
    # the exactly-zero masked quarter must remain excluded → zero abundance
    assert np.allclose(maps[: h // 2, : w // 2, :], 0.0)
    # and the valid region must carry signal
    assert np.abs(maps[h // 2:, :, :]).sum() > 0


def test_counts_mask_matches_old_rule_on_masked_data():
    # On the original modality the new relative-L1 mask must select the same
    # pixels the old absolute-sum rule selected.
    cube = _counts_cube()
    flat = cube.reshape(-1, cube.shape[-1])
    old = flat.sum(axis=1) > 1e-6
    l1 = np.abs(flat).sum(axis=1)
    new = l1 > l1.max() * 1e-9
    assert np.array_equal(old, new)


def test_normalize_unchanged_for_nonnegative_data():
    # |sum| == sum for non-negative spectra, so L1 normalization is identical.
    cube = _counts_cube(masked_quarter=False)
    flat = cube.reshape(-1, cube.shape[-1])
    assert np.allclose(np.abs(flat).sum(axis=1), flat.sum(axis=1))


# ---------------------------------------------------------------------------
# The fix: signed / small-magnitude cubes keep their pixels
# ---------------------------------------------------------------------------

def test_signed_small_magnitude_cube_keeps_all_pixels():
    cube = _signed_didv_cube()
    h, w, e = cube.shape
    um = SpectralUnmixer(method="pca", n_components=4)
    components, maps = um.fit(cube)  # pre-fix: ValueError (0 valid pixels)
    assert components.shape == (4, e)
    assert maps.shape == (h, w, 4)
    assert np.all(np.isfinite(maps))
    # every pixel is real data → no all-zero abundance vector anywhere
    assert (np.abs(maps).sum(axis=-1) > 0).all()


def test_signed_cube_with_normalize_is_finite():
    cube = _signed_didv_cube()
    um = SpectralUnmixer(method="pca", n_components=3, normalize=True)
    components, maps = um.fit(cube)
    assert np.all(np.isfinite(components))
    assert np.all(np.isfinite(maps))


def test_all_zero_cube_raises_cleanly():
    cube = np.zeros((6, 6, 16))
    um = SpectralUnmixer(method="pca", n_components=2)
    with pytest.raises(ValueError, match="Too few valid pixels"):
        um.fit(cube)


def test_scale_invariance():
    # The same cube at counts scale and at 1e-10 scale must select the same
    # pixels and produce the same abundance structure (up to overall scale).
    cube = _counts_cube()
    um1 = SpectralUnmixer(method="pca", n_components=3)
    _, maps1 = um1.fit(cube)
    um2 = SpectralUnmixer(method="pca", n_components=3)
    _, maps2 = um2.fit(cube * 1e-13)
    assert np.allclose(maps1 == 0, maps2 == 0)


# ---------------------------------------------------------------------------
# Error propagation: the original failure must reach the caller
# ---------------------------------------------------------------------------

def test_iteration_error_propagates_to_caller(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        BuildHolisticSynthesisPromptController,
    )

    agent = HyperspectralAnalysisAgent(
        api_key="dummy", base_url="http://localhost:1",
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
    )

    original_error = {"error": "Final spectral unmixing failed",
                      "details": "Too few valid pixels (0) for 6 components."}

    class _FailingController:
        def execute(self, state):
            state["error_dict"] = dict(original_error)
            return state

    agent.iteration_pipeline = [_FailingController()]
    agent.synthesis_pipeline = [
        BuildHolisticSynthesisPromptController(logging.getLogger("t"))
    ]

    cube_path = tmp_path / "cube.npy"
    np.save(cube_path, RNG.rand(4, 4, 8))

    result_json, error_dict = agent._run_analysis_pipeline(
        data_path=str(cube_path), system_info={}, instruction_prompt="x",
    )
    assert result_json is None
    # Pre-fix this came back as the derivative
    # {"error": "No iteration results found for synthesis."}.
    assert error_dict == original_error
