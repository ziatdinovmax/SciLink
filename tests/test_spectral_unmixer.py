"""Focused tests for hyperspectral background masking and normalization."""

import importlib.util

import numpy as np


# Avoid importing the package root, which eagerly imports optional heavy dependencies.
_spec = importlib.util.spec_from_file_location(
    "_spectral_unmixer_under_test",
    "scilink/skills/hyperspectral/eels/spectral_unmixer.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SpectralUnmixer = _module.SpectralUnmixer


class _RecordingModel:
    def fit_transform(self, values):
        self.values = values.copy()
        self.components_ = np.ones((1, values.shape[1]))
        return np.ones((values.shape[0], 1))


def test_signed_small_magnitude_spectra_are_not_treated_as_background():
    spectrum = np.array([1e-12, -1e-12, 2e-12])
    cube = np.array([[spectrum, np.zeros(3)]])
    model = _RecordingModel()
    unmixer = SpectralUnmixer(method="pca", n_components=1, normalize=True)
    unmixer.model = model

    _, abundance_maps = unmixer.fit(cube)

    assert np.allclose(model.values, [spectrum / np.abs(spectrum).sum()])
    assert abundance_maps[0, 0, 0] == np.abs(spectrum).sum()
    assert abundance_maps[0, 1, 0] == 0
