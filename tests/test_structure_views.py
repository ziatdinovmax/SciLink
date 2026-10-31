"""Viewing angles for structure renders (agent views and the file preview)."""

import numpy as np
import pytest

from scilink.utils.structure_views import _get_optimal_rotations


def test_wide_slab_is_viewed_down_its_vacuum():
    """A supercell wider than its cell height is still seen from the top."""
    build = pytest.importorskip("ase.build")
    sheet = build.mx2("MoS2", size=(6, 6, 1), vacuum=7.5)
    # The case a longest-cell-vector rule got wrong: in-plane a > height c.
    assert np.linalg.norm(sheet.cell[0]) > np.linalg.norm(sheet.cell[2])
    assert _get_optimal_rotations(sheet)["surface"] == "0x,0y,0z"
