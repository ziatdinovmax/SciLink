"""Contradiction framework + LAMMPS selection-realizable realization.

Exercises the engine-neutral dispatch (`check_requirements` over a
`selection_realizable` requirement) and its LAMMPS realization
(`map_selections_to_types` / `split_shared_types`) on a synthetic multi-component
data file where two species share one oxygen atom type — the collision the
species-resolved-RDF case hits when a force field types two chemically distinct
oxygens identically.
"""

import json
import textwrap

import pytest

from scilink.agents.sim_agents.contradictions import (
    Requirement, check_requirements, implemented_kinds,
)
from scilink.skills.molecular_dynamics.lammps import lammps as lt


# Two species (anion, solvent) whose oxygen shares atom type 1; a metal is type 2,
# solvent H is type 3. atom_style full (id mol type q x y z); molecules packed in
# manifest order. Bonds present so the style is detected as full (as real
# OpenFF/Interchange systems always are).
_DATA = textwrap.dedent("""\
    Test system

    8 atoms
    2 bonds

    3 atom types
    1 bond types

    0.0 10.0 xlo xhi
    0.0 10.0 ylo yhi
    0.0 10.0 zlo zhi

    Masses

    1 15.999
    2 65.409
    3 1.008

    Pair Coeffs

    1 0.21 3.04
    2 0.0125 2.169
    3 0.0 1.0

    Bond Coeffs

    1 500.0 1.0

    Atoms

    1 1 2 2.0 1.0 1.0 1.0
    2 2 2 2.0 2.0 2.0 2.0
    3 3 1 -1.0 3.0 3.0 3.0
    4 4 1 -1.0 4.0 4.0 4.0
    5 5 1 -0.8 5.0 5.0 5.0
    6 5 3 0.4 5.5 5.0 5.0
    7 6 1 -0.8 6.0 6.0 6.0
    8 6 3 0.4 6.5 6.0 6.0

    Bonds

    1 1 5 6
    2 1 7 8
    """)

_COMPONENTS = {"components": [
    {"name": "metal", "smiles": "[M]", "count": 2},
    {"name": "anion", "smiles": "[O-]", "count": 2},
    {"name": "solvent", "smiles": "O", "count": 2},
]}

_SELECTIONS = ["metal", "anion:O", "solvent:O"]


@pytest.fixture
def system(tmp_path):
    data = tmp_path / "system.data"
    data.write_text(_DATA)
    comps = tmp_path / "components.json"
    comps.write_text(json.dumps(_COMPONENTS))
    return str(data), str(comps)


def test_selection_realizable_is_registered():
    assert "selection_realizable" in implemented_kinds()


def test_mapping_finds_the_shared_oxygen_type(system):
    data, comps = system
    m = lt.map_selections_to_types(data, comps, _SELECTIONS)
    assert m["metal"] == [2]
    # anion and solvent oxygens both resolve to the single shared O type.
    assert m["anion:O"] == [1]
    assert m["solvent:O"] == [1]


def test_check_detects_the_collision(system):
    data, comps = system
    req = Requirement("solvation RDF", "selection_realizable",
                      params={"selections": _SELECTIONS})
    arts = {"data_file": data, "components_json": comps, "engine_tools": lt}
    cons = check_requirements([req], arts, active_skills=["lammps"])
    assert len(cons) == 1
    c = cons[0]
    assert c.resolvable and c.resolution["tool"] == "split_shared_types"
    assert c.resolution["kwargs"]["collisions"] == [["anion:O", "solvent:O"]]


def test_split_resolves_the_collision(system, tmp_path):
    data, comps = system
    res = lt.split_shared_types(data, comps, [["anion:O", "solvent:O"]])
    # anion keeps type 1, solvent gets a fresh type; physics-identical.
    assert res["type_map"]["anion:O"] == [1]
    new_type = res["type_map"]["solvent:O"][0]
    assert new_type == 4 and res["new_types"][new_type] == 1

    split = tmp_path / "split.data"
    split.write_text(res["data_file_text"])
    info = lt.parse_data_file(str(split))
    assert info["atom_types"] == 4 and info["atom_count"] == 8
    # duplicated mass (same element/mass as the source type)
    assert info["mass_map"][4][0] == info["mass_map"][1][0]

    # exactly the two solvent oxygens moved to the new type
    moved = sum(1 for r in lt._atoms_section_rows(str(split)) if r.split()[2] == "4")
    assert moved == 2

    # re-check: the three selections are now distinct → no contradiction
    m2 = lt.map_selections_to_types(str(split), comps, _SELECTIONS)
    assert m2["anion:O"] == [1] and m2["solvent:O"] == [4]
    req = Requirement("solvation RDF", "selection_realizable",
                      params={"selections": _SELECTIONS})
    cons = check_requirements(
        [req], {"data_file": str(split), "components_json": comps, "engine_tools": lt},
        active_skills=["lammps"])
    assert cons == []


def test_no_false_positive_when_distinct(system):
    # metal vs a single oxygen selection never collides.
    data, comps = system
    req = Requirement("x", "selection_realizable",
                      params={"selections": ["metal", "anion:O"]})
    arts = {"data_file": data, "components_json": comps, "engine_tools": lt}
    assert check_requirements([req], arts, active_skills=["lammps"]) == []
