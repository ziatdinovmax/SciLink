"""Deterministic fixture data for the QC golden harness.

Everything is generated from fixed formulas / a fixed RNG seed so repeated
runs produce byte-identical inputs (and therefore byte-identical prompts and
deterministic fit numerics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_gaussian_spectrum(path: Path) -> Path:
    """Two-column CSV: a clean Gaussian peak on a linear background.

    Noise-free on purpose — the canned fit script recovers it exactly, giving
    stable statistics in prompts and a deterministic R².
    """
    x = np.linspace(0.0, 10.0, 201)
    y = 5.0 * np.exp(-0.5 * ((x - 4.0) / 0.8) ** 2) + 0.3 * x + 1.0
    arr = np.column_stack([x, y])
    header = "energy,intensity"
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.6f")
    return path


def write_blob_image(path: Path) -> Path:
    """Deterministic 2D grayscale .npy: three Gaussian blobs on a gradient.

    Saved as .npy (not PNG) so no image-codec variability enters the goldens.
    """
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    img = 0.05 * (xx / w)  # gentle gradient background
    for (cy, cx, s, a) in [(32, 40, 6.0, 1.0), (80, 70, 9.0, 0.8), (50, 100, 5.0, 0.6)]:
        img += a * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)))
    rng = np.random.default_rng(42)
    img += rng.normal(0.0, 0.01, size=img.shape)  # seeded, reproducible
    np.save(path, img.astype(np.float32))
    return path
