"""Tests for the deterministic ORCA geometry-consistency check (issue #400).

The ORCA twin of ``test_molecular_qc_geometry_check.py``: it parses ORCA's
``* xyz <charge> <mult> ... *`` coordinate block and compares composition
against the source structure, failing generation loudly on a silent
transcription error.
"""
import textwrap

import pytest

from scilink.skills.molecular_qc.orca.orca_geometry import (
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
    # CO2 single point
    ! B3LYP def2-SVP RIJCOSX def2/J TightSCF
    %pal nprocs 4 end
    * xyz 0 1
      O 0.0 0.0 0.0
      C 0.0 0.0 1.16
      O 0.0 0.0 2.32
    *
""")

# One oxygen dropped from the deck.
_DECK_DROPPED = textwrap.dedent("""\
    ! B3LYP def2-SVP
    * xyz 0 1
      O 0.0 0.0 0.0
      C 0.0 0.0 1.16
    *
""")

# Carbon transcribed as nitrogen.
_DECK_WRONG_EL = textwrap.dedent("""\
    ! B3LYP def2-SVP
    * xyz 0 1
      O 0.0 0.0 0.0
      N 0.0 0.0 1.16
      O 0.0 0.0 2.32
    *
""")

_DECK_NO_GEOM = "! B3LYP def2-SVP\n%pal nprocs 4 end\n"


@pytest.fixture
def co2_xyz(tmp_path):
    p = tmp_path / "co2.xyz"
    p.write_text(_CO2_XYZ)
    return str(p)


def test_matching_deck_ok(co2_xyz):
    r = check_geometry_consistency(input_files={"orca.inp": _DECK_OK},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"
    assert r["n_atoms"] == 3
    assert r["composition"] == {"C": 1, "O": 2}


def test_dropped_atom_mismatch(co2_xyz):
    r = check_geometry_consistency(input_files={"orca.inp": _DECK_DROPPED},
                                   structure_file=co2_xyz)
    assert r["status"] == "mismatch"
    assert "atom count 2" in r["reason"]
    assert r["deck_composition"] == {"C": 1, "O": 1}
    assert r["source_composition"] == {"C": 1, "O": 2}


def test_wrong_element_mismatch(co2_xyz):
    r = check_geometry_consistency(input_files={"orca.inp": _DECK_WRONG_EL},
                                   structure_file=co2_xyz)
    assert r["status"] == "mismatch"
    assert "N:" in r["reason"] and "C:" in r["reason"]


def test_no_coordinate_block_skipped(co2_xyz):
    r = check_geometry_consistency(input_files={"orca.inp": _DECK_NO_GEOM},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"


def test_unreadable_structure_skipped():
    r = check_geometry_consistency(input_files={"orca.inp": _DECK_OK},
                                   structure_file="/no/such/file.xyz")
    assert r["status"] == "skipped"


def test_composition_is_order_independent(co2_xyz):
    reordered = textwrap.dedent("""\
        ! HF def2-SVP
        * xyz 0 1
          C 0.0 0.0 1.16
          O 0.0 0.0 2.32
          O 0.0 0.0 0.0
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": reordered},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"


def test_star_no_space_ok(co2_xyz):
    """`*xyz 0 1` (no space after the star) is valid ORCA syntax."""
    deck = textwrap.dedent("""\
        ! B3LYP def2-SVP
        *xyz 0 1
          O 0.0 0.0 0.0
          C 0.0 0.0 1.16
          O 0.0 0.0 2.32
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok" and r["composition"] == {"C": 1, "O": 2}


def test_xyzfile_skipped(co2_xyz):
    """External-coordinate `* xyzfile ...` can't be counted -> skip, not fail."""
    deck = "! B3LYP def2-SVP Opt\n* xyzfile 0 1 prev.xyz\n"
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "xyzfile" in r["reason"]


def test_internal_coordinates_skipped(co2_xyz):
    """A Z-matrix / internal-coordinate block -> skip, not false-fail."""
    deck = textwrap.dedent("""\
        ! B3LYP def2-SVP Opt
        * int 0 1
          O 0 0 0  0.0 0.0 0.0
          C 1 0 0  1.16 0.0 0.0
          O 1 2 0  1.16 180.0 0.0
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "int" in r["reason"]


def test_ghost_atoms_skip_the_check(co2_xyz):
    """A block with ghost atoms (`O:` trailing colon) is a counterpoise/BSSE
    fragment — skip, don't atom-count against the full source."""
    deck = textwrap.dedent("""\
        ! B3LYP def2-SVP
        * xyz 0 1
          O  0.0 0.0 0.0
          C  0.0 0.0 1.16
          O  0.0 0.0 2.32
          H: 5.0 5.0 5.0
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "ghost" in r["reason"] or "counterpoise" in r["reason"]


def test_dummy_atom_skips_the_check(co2_xyz):
    """ORCA dummy centres (`DA`) also make the atom count a non-comparison."""
    deck = textwrap.dedent("""\
        ! B3LYP def2-SVP
        * xyz 0 1
          O  0.0 0.0 0.0
          C  0.0 0.0 1.16
          O  0.0 0.0 2.32
          DA 5.0 5.0 5.0
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"


def test_counterpoise_fragment_skips_not_mismatch(co2_xyz):
    """The parity-sweep failure: a BSSE fragment deck (one monomer real, the
    other as ghosts) has fewer physical atoms than the full source by design —
    it must SKIP, not report a mismatch and fail generation."""
    # Source is CO2 (C + 2 O). A counterpoise fragment: one O real, C + one O
    # ghost — 1 physical atom vs 3 in the source.
    deck = textwrap.dedent("""\
        ! DLPNO-CCSD(T) def2-TZVP def2-TZVP/C
        * xyz 0 1
          O  0.0 0.0 0.0
          C: 0.0 0.0 1.16
          O: 0.0 0.0 2.32
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"          # NOT "mismatch"


def test_lowercase_deck_ok(co2_xyz):
    """ORCA element tags are case-insensitive — a lowercase deck is valid."""
    deck = textwrap.dedent("""\
        ! b3lyp def2-svp
        * xyz 0 1
          o 0.0 0.0 0.0
          c 0.0 0.0 1.16
          o 0.0 0.0 2.32
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "ok"
    assert r["composition"] == {"C": 1, "O": 2}


def test_unrecognized_atom_line_skipped(co2_xyz):
    """A >=4-token coordinate line that isn't a real element -> skip, not fail."""
    deck = textwrap.dedent("""\
        ! B3LYP def2-SVP
        * xyz 0 1
          O  0.0 0.0 0.0
          Zz 0.0 0.0 1.16
          O  0.0 0.0 2.32
        *
    """)
    r = check_geometry_consistency(input_files={"orca.inp": deck},
                                   structure_file=co2_xyz)
    assert r["status"] == "skipped"
    assert "unrecognized" in r["reason"]


def test_registry_resolves_the_tool():
    from scilink.skills._shared._registry import get_tool_function
    fn = get_tool_function("check_geometry_consistency", active_skills=["orca"])
    assert callable(fn)


def test_orca_is_a_supported_engine():
    from scilink.agents.sim_agents.molecular_qc_agent import MolecularQCAgent
    assert "orca" in MolecularQCAgent.supported_software()


def test_agent_fails_loud_on_mismatch(co2_xyz, monkeypatch):
    """generate_inputs(software='orca') returns status='error' on a dropped atom."""
    from scilink.agents.sim_agents.molecular_qc_agent import MolecularQCAgent

    # base_url set -> no vendor-credential check at construction; no network.
    agent = MolecularQCAgent(api_key="dummy", base_url="http://localhost:0")

    class _Resp:
        text = "ignored"

    monkeypatch.setattr(agent.model, "generate_content",
                        lambda *a, **k: _Resp())
    monkeypatch.setattr(agent, "_parse_response",
                        lambda text: {"input_files": {"orca.inp": _DECK_DROPPED}})

    result = agent.generate_inputs(structure_file=co2_xyz, request="single point",
                                   software="orca")
    assert result["status"] == "error"
    assert "inconsistent" in result["message"]
    assert result["geometry_check"]["status"] == "mismatch"
