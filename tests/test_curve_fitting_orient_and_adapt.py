"""Tests for the curve-fitting series robustness fixes.

Covers two narrow surfaces both motivated by a real failing run:

  * ``_try_orient_xy`` — column-orientation heuristic for CSV-loaded curves.
    The bug: a TRPL CSV had columns in ``(intensity, time)`` order and the
    generated fit script used positional ``col 0 = x`` and got R² = 0.

  * ``_adapt_script_for_spectrum`` — per-spectrum script adapter. The bug:
    LLM-evolved scripts use ``glob.glob('*.npy')`` as a fallback loader,
    which in parallel fan-out can find a sibling spectrum's temp file.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


# Load the shared curve-fitting helpers without going through the package
# ``__init__`` (which imports torch et al — overkill for a unit test).
_spec = importlib.util.spec_from_file_location(
    "_curve_fitting_tools_under_test",
    "scilink/skills/_shared/curve_fitting_tools.py",
)
_cft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cft)
_try_orient_xy = _cft._try_orient_xy


class TestTryOrientXY:
    def test_swaps_when_only_col1_is_monotonic(self):
        # (intensity, time) — col 0 non-mono, col 1 mono. Should swap.
        intensity = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        time = np.linspace(0.0, 1.0, 5)
        data = np.column_stack([intensity, time])

        out = _try_orient_xy(data)
        assert np.allclose(out[:, 0], time)
        assert np.allclose(out[:, 1], intensity)

    def test_passthrough_when_already_oriented(self):
        # (time, intensity) — col 0 mono, col 1 non-mono. Must not swap.
        time = np.linspace(0.0, 1.0, 5)
        intensity = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        data = np.column_stack([time, intensity])

        out = _try_orient_xy(data)
        assert np.array_equal(out, data)

    def test_passthrough_when_both_monotonic(self):
        # Ambiguous (both mono increasing) — leave as-is.
        data = np.column_stack([np.arange(5.0), np.arange(5.0) * 2])
        out = _try_orient_xy(data)
        assert np.array_equal(out, data)

    def test_passthrough_when_neither_monotonic(self):
        # Ambiguous (neither mono) — leave as-is.
        data = np.column_stack(
            [np.array([1.0, 0.5, 0.3]), np.array([0.2, 0.4, 0.1])]
        )
        out = _try_orient_xy(data)
        assert np.array_equal(out, data)

    def test_recognizes_decreasing_monotonic_as_x(self):
        # Time axes can be reversed (e.g. delay scans). Still an X-column.
        y = np.array([0.1, 0.5, 0.3])
        x = np.array([1.0, 0.5, 0.0])
        data = np.column_stack([y, x])
        out = _try_orient_xy(data)
        assert np.allclose(out[:, 0], x)

    def test_passthrough_for_1d_input(self):
        data = np.arange(10.0)
        assert _try_orient_xy(data) is data

    def test_passthrough_when_data_has_nan(self):
        # NaN/inf disables the heuristic — diff() would be misleading.
        intensity = np.array([0.1, 0.5, 0.3, 0.2, 0.4])
        time = np.linspace(0.0, 1.0, 5)
        data = np.column_stack([intensity, time])
        data[0, 1] = np.nan
        assert _try_orient_xy(data) is data

    def test_passthrough_for_two_row_shape(self):
        # Only (N, 2) handled — (2, N) goes through untouched.
        data = np.array([[0.1, 0.5, 0.3], [1.0, 2.0, 3.0]])
        assert _try_orient_xy(data) is data


class TestAdaptScriptForSpectrum:
    @pytest.fixture
    def adapter(self):
        # Import lazily so importing this test module never triggers the
        # heavy controller import chain unless these tests are selected.
        from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
            UnifiedSeriesProcessingController,
        )
        # _adapt_script_for_spectrum is a pure method — no controller state
        # needed. Bind it via a throwaway instance built from __new__ so we
        # don't run __init__.
        helper = UnifiedSeriesProcessingController.__new__(
            UnifiedSeriesProcessingController
        )
        return helper._adapt_script_for_spectrum

    def test_rewrites_bare_glob_fallback(self, adapter):
        script = (
            "import glob\n"
            "npy_files = glob.glob('*.npy') + glob.glob('**/*.npy', recursive=True)\n"
        )
        out = adapter(script, "/session/results/temp_spectrum_1.npy", "spectrum_0001")
        assert "glob.glob('*.npy')" not in out
        assert "glob.glob('**/*.npy'" not in out
        assert '["/session/results/temp_spectrum_1.npy"]' in out

    def test_leaves_nested_paren_glob_intact(self, adapter):
        # `glob.glob(os.path.join(d, '*.npy'))` would be sliced mid-expression
        # if the regex spanned parens — must be left alone to keep the script
        # syntactically valid.
        script = "files = glob.glob(os.path.join(d, '*.npy'))\n"
        out = adapter(script, "/session/results/temp_spectrum_1.npy", "spectrum_0001")
        assert out == script

    def test_does_not_touch_png_glob(self, adapter):
        script = "images = glob.glob('*.png')\n"
        out = adapter(script, "/session/results/temp_spectrum_1.npy", "spectrum_0001")
        assert out == script

    def test_rewrites_possible_paths_list_literal(self, adapter):
        # The list-literal pattern observed in the real failing run.
        script = (
            "possible_paths = [\n"
            "    '/session/results/temp_spectrum_0.npy',\n"
            "    '/session/results/temp_spectrum_0.npy',\n"
            "    'spectrum_0.npy',\n"
            "]\n"
        )
        out = adapter(script, "/session/results/temp_spectrum_1.npy", "spectrum_0001")
        assert "temp_spectrum_0.npy" not in out
        assert out.count("temp_spectrum_1.npy") == 2

    def test_windows_data_path_does_not_crash_regex(self, adapter):
        # `\U` in a Windows path becomes `re.error: bad escape \U` if the
        # path is interpolated into a replacement string without normalizing
        # backslashes.
        script = "data = np.load('/session/results/temp_spectrum_0.npy')\n"
        out = adapter(script, r"C:\Users\me\session\temp_spectrum_1.npy", "spectrum_0001")
        assert "C:/Users/me/session/temp_spectrum_1.npy" in out
