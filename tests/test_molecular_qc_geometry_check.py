"""Tests for the deterministic NWChem geometry-consistency check (issue #400)."""
import textwrap

import pytest

from scilink.skills.molecular_qc.nwchem.nwchem_geometry import (
    check_geometry_consistency,
)

_CO2_XYZ = textwrap.dedent("""\
    3
    CO2
    O 0.0 0.0 0.0
    C 0.0 0.0 1.16
    O 0.0 0.0 2.32
""")

_DECK_OK = textwrap.dedent("""\
    title "co2"
    geometry units angstrom
      symmetry c1
      O 0.0 0.0 0.0
      C 0.0 0.0 1.16
      O 0.0 0.0 2.32
    end
    task scf energy
""")

# One oxygen dropped from the deck.
_DECK_DROPPED = textwrap.dedent("""\
    geometry units angstrom
      O 0.0 0.0 0.0
      C 0.0 0.0 1.16
    end
""")

# Carbon transcribed as nitrogen.
_DECK_WRONG_EL = textwrap.dedent("""\
    geometry units angstrom
      O 0.0 0.0 0.0
      N 0.0 0.0 1.16
      O 0.0 0.0 2.32
    end
""")

_DECK_NO_GEOM = "task scf energy\n"


@pytest.fixture
def co2_xyz(tmp_path):
    p = tmp_path / "co2.xyz"
    p.write_text(_CO2_XYZ)
    return str(p)


def test_matching_deck_ok(co2_xyz):
    r = check_geometry_consistency(input_files={"m.nw": _DECK_OK},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"
    assert r["n_atoms"] == 3
    assert r["composition"] == {"C": 1, "O": 2}


def test_dropped_atom_mismatch(co2_xyz):
    r = check_geometry_consistency(input_files={"m.nw": _DECK_DROPPED},
                                   structure_file=co2_xyz)
    assert r["status"] == "mismatch"
    assert "atom count 2" in r["reason"]
    assert r["deck_composition"] == {"C": 1, "O": 1}
    assert r["source_composition"] == {"C": 1, "O": 2}


def test_wrong_element_mismatch(co2_xyz):
    r = check_geometry_consistency(input_files={"m.nw": _DECK_WRONG_EL},
                                   structure_file=co2_xyz)
    assert r["status"] == "mismatch"
    # same atom count, but composition differs (N appears, one C missing)
    assert "N:" in r["reason"] and "C:" in r["reason"]


def test_no_geometry_block_skipped(co2_xyz):
    r = check_geometry_consistency(input_files={"m.nw": _DECK_NO_GEOM},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"


def test_unreadable_structure_skipped():
    r = check_geometry_consistency(input_files={"m.nw": _DECK_OK},
                                   structure_file="/no/such/file.xyz")
    assert r["status"] == "skipped"


def test_composition_is_order_independent(co2_xyz):
    reordered = textwrap.dedent("""\
        geometry units angstrom
          C 0.0 0.0 1.16
          O 0.0 0.0 2.32
          O 0.0 0.0 0.0
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": reordered},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"


def test_symmetry_group_skipped(co2_xyz):
    """A non-C1 symmetry group may list only the asymmetric unit -> skip, not fail."""
    deck = textwrap.dedent("""\
        geometry units angstrom
          symmetry d2h
          O 0.0 0.0 0.0
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "symmetry" in r["reason"]


def test_zmatrix_skipped(co2_xyz):
    """A Z-matrix geometry isn't a plain Cartesian list -> skip, not false-fail."""
    deck = textwrap.dedent("""\
        geometry
          zmatrix
            O
            C 1 1.16
            O 1 1.16 2 180.0
          end
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "zmatrix" in r["reason"]


def test_ghost_atoms_excluded(co2_xyz):
    """BSSE ghost (bq) / dummy centers are not physical atoms -> excluded, still ok."""
    deck = textwrap.dedent("""\
        geometry units angstrom
          O  0.0 0.0 0.0
          C  0.0 0.0 1.16
          O  0.0 0.0 2.32
          Bq 5.0 5.0 5.0
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"
    assert r["composition"] == {"C": 1, "O": 2}


def test_lowercase_deck_ok(co2_xyz):
    """NWChem element tags are case-insensitive — a lowercase deck is valid."""
    deck = textwrap.dedent("""\
        geometry units angstrom
          o 0.0 0.0 0.0
          c 0.0 0.0 1.16
          o 0.0 0.0 2.32
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"
    assert r["composition"] == {"C": 1, "O": 2}


def test_mixed_case_element_ok(co2_xyz):
    """`CL`, `cl`, `Cl` all normalize to the same element."""
    deck = textwrap.dedent("""\
        geometry units angstrom
          O 0.0 0.0 0.0
          C 0.0 0.0 1.16
          o 0.0 0.0 2.32
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok" and r["composition"] == {"C": 1, "O": 2}


def test_unrecognized_atom_line_skipped(co2_xyz):
    """A >=4-token line that doesn't parse as a real element -> skip, not fail."""
    deck = textwrap.dedent("""\
        geometry units angstrom
          O  0.0 0.0 0.0
          Zz 0.0 0.0 1.16
          O  0.0 0.0 2.32
        end
    """)
    r = check_geometry_consistency(input_files={"m.nw": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "unrecognized" in r["reason"]


def test_registry_resolves_the_tool():
    from scilink.skills._shared._registry import get_tool_function
    fn = get_tool_function("check_geometry_consistency", active_skills=["nwchem"])
    assert callable(fn)


def test_agent_fails_loud_on_mismatch(co2_xyz, monkeypatch):
    """generate_inputs returns status='error' when the deck drops an atom."""
    from scilink.agents.sim_agents.molecular_qc_agent import MolecularQCAgent

    # base_url set -> no vendor-credential check at construction; no network.
    agent = MolecularQCAgent(api_key="dummy", base_url="http://localhost:0")

    class _Resp:
        text = "ignored"

    monkeypatch.setattr(agent.model, "generate_content",
                        lambda *a, **k: _Resp())
    monkeypatch.setattr(agent, "_parse_response",
                        lambda text: {"input_files": {"co2.nw": _DECK_DROPPED}})

    result = agent.generate_inputs(structure_file=co2_xyz, request="scf energy")
    assert result["status"] == "error"
    assert "inconsistent" in result["message"]
    assert result["geometry_check"]["status"] == "mismatch"
