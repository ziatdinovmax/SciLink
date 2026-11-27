"""Structure files keep what refinement needs: the declared space group and the
atomic displacement parameters (COD entries are materialized verbatim; the
engine reads U from the file and keeps the declared symmetry)."""
import numpy as np
import pytest

from scilink.skills.structure_matching.xrd import _gsas_engine as ge

pymatgen = pytest.importorskip("pymatgen")

_HEAD = """data_t
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 221
_symmetry_space_group_name_H-M 'P m -3 m'
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
"""


def _cif(tmp_path, body, name="t.cif"):
    p = tmp_path / name
    p.write_text(_HEAD + body)
    return str(p)


def test_reads_u_iso(tmp_path):
    p = _cif(tmp_path, """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Sr1 Sr2+ 0 0 0 0.0080(3)
Ti1 Ti4+ 0.5 0.5 0.5 0.0050(2)
""")
    u = ge._cif_displacements(p)
    assert u["Sr"] == pytest.approx(0.008) and u["Ti"] == pytest.approx(0.005)


def test_reads_b_iso_and_converts(tmp_path):
    p = _cif(tmp_path, """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_B_iso_or_equiv
Cs1 Cs 0 0 0 0.79
""")
    assert ge._cif_displacements(p)["Cs"] == pytest.approx(0.79 / (8 * np.pi ** 2), rel=1e-6)


def test_anisotropic_only_gives_u_eq(tmp_path):
    p = _cif(tmp_path, """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cs1 Cs 0 0 0
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
Cs1 0.006 0.009 0.012
""")
    assert ge._cif_displacements(p)["Cs"] == pytest.approx(0.009)


def test_zero_placeholders_are_not_values(tmp_path):
    p = _cif(tmp_path, """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Cs1 Cs 0 0 0 0.0
""")
    assert ge._cif_displacements(p) == {}


def test_declared_space_group_is_kept_over_loose_redetection(tmp_path):
    # a tetragonal cell 0.2% off cubic: the 0.1 A re-detection calls it cubic,
    # the file declares P4/mmm (123) and the tight tolerance keeps it
    from pymatgen.core import Lattice, Structure
    s = Structure.from_spacegroup("P4/mmm", Lattice.tetragonal(4.0, 4.008), ["Ba", "Ti"],
                                  [[0, 0, 0], [0.5, 0.5, 0.5]])
    src = tmp_path / "tet.cif"
    from pymatgen.io.cif import CifWriter
    CifWriter(s, symprec=1e-4).write_file(str(src))
    assert ge._declared_space_group(str(src)) == 123
    out = ge._canonicalize_cif(str(src), str(tmp_path), symprec=0.1)
    assert ge._declared_space_group(out) == 123


def test_materialize_copies_the_source_cif_verbatim(tmp_path):
    from pathlib import Path
    from pymatgen.core import Lattice, Structure
    from scilink.skills.structure_matching._backends._base import StructureCandidate
    from scilink.skills.structure_matching.xrd.search_structures import _materialize_cifs
    src = tmp_path / "2300450.cif"
    src.write_text(_HEAD + "# the database's own file\n")
    st = Structure(Lattice.cubic(4.0), ["Cs"], [[0, 0, 0]])
    with_src = StructureCandidate(id="1", source="cod", formula="Cs", space_group="Pm-3m",
                                  metadata={"_structure": st, "_source_cif": str(src)})
    without = StructureCandidate(id="2", source="mp", formula="Cs", space_group="Pm-3m",
                                 metadata={"_structure": st})
    warnings = []
    _materialize_cifs([with_src, without], tmp_path / "out", warnings)
    assert Path(with_src.structure_path).read_text() == src.read_text()
    assert "generated using pymatgen" in Path(without.structure_path).read_text()
    assert not warnings
