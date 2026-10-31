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


def test_one_vacuum_reading_for_the_drawn_views_and_the_3d_up():
    """The drawn views' axis and the 3D preview's "up" come from one helper:
    the vacuum axis by empty space, the normal from the in-plane vectors
    (so a sheared c does not tilt it), pointing toward the vacuum."""
    build = pytest.importorskip("ase.build")
    from scilink.utils.structure_views import (
        _is_slab_structure,
        _slab_surface_normal,
        _slab_vacuum_axis,
    )

    slab = build.fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    cell = slab.get_cell().array.copy()
    cell[2] += 0.5 * cell[0]  # shear c by half of a
    slab.set_cell(cell)
    pos = slab.get_positions()
    assert _is_slab_structure(cell, pos) and _slab_vacuum_axis(cell, pos) == 2
    np.testing.assert_allclose(_slab_surface_normal(cell, pos), [0, 0, 1], atol=1e-9)
    assert _get_optimal_rotations(slab)["surface"] == "0x,0y,0z"

    flipped = cell.copy()
    flipped[2] *= -1  # vacuum vector pointing down: the normal follows it
    np.testing.assert_allclose(_slab_surface_normal(flipped, pos), [0, 0, -1], atol=1e-9)

    bulk = build.bulk("Cu", cubic=True)
    assert not _is_slab_structure(bulk.get_cell().array, bulk.get_positions())
