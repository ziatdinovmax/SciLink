"""
Elements of LAMMPS data files reach the MLIP path, and atom types are never
guessed.

``ase.io.read`` cannot read ``.data`` files, and without a ``Masses``
section ASE numbers atom types as H, He, ... These tests pin the fixes:
elements are read from LAMMPS data, missing elements make the MLIP path
refuse rather than pick blind, the ASE run script refuses an ambiguous
file, and LAMMPS ``pair_coeff`` follows the data file's type order.
No LLM call or MLIP backend is needed.
"""

import os
import runpy
import tempfile

import pytest
from ase.build import bulk
from ase.io import write

NACL = bulk("NaCl", "rocksalt", a=5.64, cubic=True)


def _data(td, name, **kw):
    path = os.path.join(td, name)
    write(path, NACL, format="lammps-data", **kw)
    return path


def _potential(elements, backend="mace", model_file=""):
    from scilink.agents.sim_agents._potential import (
        ASECalculatorSpec, DeployedPotential)
    return DeployedPotential(
        kind="mlip", backend=backend, model_name="m", model_file=model_file,
        elements=list(elements),
        ase_calculator=ASECalculatorSpec(
            "from ase.calculators.emt import EMT", "EMT()", "X_DEVICE"))


# ── reading elements ─────────────────────────────────────────────────

def test_structure_elements_reads_lammps_data():
    from scilink.skills._shared.structure_io import structure_elements
    with tempfile.TemporaryDirectory() as td:
        atomic = _data(td, "a.data", masses=True, atom_style="atomic")
        full = _data(td, "f.data", masses=True, atom_style="full")
        nomass = _data(td, "n.data", masses=False)
        xyz = os.path.join(td, "s.xyz")
        write(xyz, NACL, format="extxyz")
        assert structure_elements(atomic) == (["Cl", "Na"], 8)
        assert structure_elements(full) == (["Cl", "Na"], 8)
        assert structure_elements(nomass) == ([], 8)      # unknown, not H/He
        assert structure_elements(xyz) == (["Cl", "Na"], 8)
        assert structure_elements(os.path.join(td, "missing.data"))[0] == []


def test_lammps_type_elements_follows_type_numbers():
    from scilink.skills._shared.structure_io import lammps_type_elements
    with tempfile.TemporaryDirectory() as td:
        f = _data(td, "s.data", masses=True, specorder=["Na", "Cl"])
        assert lammps_type_elements(f) == {1: "Na", 2: "Cl"}
        assert lammps_type_elements(_data(td, "n.data", masses=False)) is None


def test_deploy_pretrained_refuses_unknown_elements():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, api_key="sk-not-real",
                          model_name="gpt-4o-mini")
        agent._select_pretrained_model = lambda *a, **k: pytest.fail(
            "must not pick a model without elements")
        nomass = _data(td, "n.data", masses=False)
        with pytest.raises(ValueError, match="Could not determine"):
            agent.deploy_pretrained(system_info={"elements": {}},
                                    research_goal="NaCl MD",
                                    structure_file=nomass)


def test_deploy_pretrained_fills_elements_from_lammps_data():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    seen = {}
    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, api_key="sk-not-real",
                          model_name="gpt-4o-mini")

        def stop(elements, goal, task_name=None, runner=None):
            seen["elements"] = elements
            raise RuntimeError("stop after selection")

        agent._select_pretrained_model = stop
        f = _data(td, "s.data", masses=True)
        with pytest.raises(RuntimeError, match="stop after selection"):
            agent.deploy_pretrained(system_info={"elements": {}},
                                    research_goal="NaCl MD", structure_file=f)
    assert seen["elements"] == ["Cl", "Na"]


# ── the ASE run script ───────────────────────────────────────────────

def _load_in_script(td, structure_name, elements):
    from scilink.agents.sim_agents._ase_runner import generate_ase_script
    # Callers pass the real path; the script itself reads the basename.
    path = generate_ase_script(_potential(elements), td,
                               structure_file=os.path.join(td, structure_name),
                               task="relax", device="cpu")
    cwd = os.getcwd()
    os.chdir(td)
    try:
        ns = runpy.run_path(path, run_name="not_main")
        return ns["_load_structure"]()
    finally:
        os.chdir(cwd)


def test_run_script_refuses_lammps_data_without_masses():
    """Refused when the script is generated, before anything runs."""
    with tempfile.TemporaryDirectory() as td:
        _data(td, "n.data", masses=False)
        with pytest.raises(ValueError, match="no Masses section"):
            _load_in_script(td, "n.data", ["Cl", "Na"])


def test_run_script_reads_full_style_and_checks_elements():
    with tempfile.TemporaryDirectory() as td:
        _data(td, "f.data", masses=True, atom_style="full")
        atoms = _load_in_script(td, "f.data", ["Cl", "Na"])
        assert sorted(set(atoms.get_chemical_symbols())) == ["Cl", "Na"]
        with pytest.raises(SystemExit, match="deployed for"):
            _load_in_script(td, "f.data", ["Cu"])


def test_run_script_single_type_without_masses_is_unambiguous():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cu.data")
        write(path, bulk("Cu", "fcc", a=3.6, cubic=True),
              format="lammps-data", masses=False)
        atoms = _load_in_script(td, "cu.data", ["Cu"])
        assert set(atoms.get_chemical_symbols()) == {"Cu"}


# ── LAMMPS pair_coeff ────────────────────────────────────────────────

def _pair_coeff(td, structure_file, elements):
    from scilink.skills.molecular_dynamics.lammps.lammps import run_with_potential
    model = os.path.join(td, "m.model")
    open(model, "w").write("x")
    run_with_potential(_potential(elements, model_file=model),
                       structure_file=structure_file, working_dir=td,
                       task="md", n_steps=10)
    for name in os.listdir(td):
        p = os.path.join(td, name)
        if os.path.isfile(p) and not name.endswith((".model", ".data")):
            for line in open(p, errors="ignore"):
                if line.startswith("pair_coeff"):
                    return line.split()[4:]
    raise AssertionError("no pair_coeff written")


def test_pair_coeff_follows_data_file_type_order():
    with tempfile.TemporaryDirectory() as td:
        f = _data(td, "s.data", masses=True, specorder=["Na", "Cl"])
        assert _pair_coeff(td, f, ["Cl", "Na"]) == ["Na", "Cl"]


def test_pair_coeff_refuses_ambiguous_or_mismatched_data():
    from scilink.skills.molecular_dynamics.lammps.lammps import _pair_coeff_elements
    with tempfile.TemporaryDirectory() as td:
        nomass = _data(td, "n.data", masses=False)
        with pytest.raises(ValueError, match="element of every atom type is not known"):
            _pair_coeff_elements(nomass, ["Cl", "Na"])
        f = _data(td, "s.data", masses=True)
        with pytest.raises(ValueError, match="deployed for"):
            _pair_coeff_elements(f, ["Cu"])


# ── smaller fixes ────────────────────────────────────────────────────

def test_uma_template_keeps_gpu_for_indexed_cuda():
    from scilink.skills._shared import mlip_tools as mt
    spec = mt.deploy_spec("uma")
    expr = spec["construct_expr"].format(name="uma-s-1p2", task="omat")
    dev = "(" + expr.split("device=(", 1)[1].split(" else 'cpu')", 1)[0] + " else 'cpu')"
    assert eval(dev, {"DEVICE": "cuda:1"}) == "cuda"
    assert eval(dev, {"DEVICE": "cpu"}) == "cpu"


def test_picker_accepts_alias_and_case_variants():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    from scilink.skills._shared import mlip_tools as mt

    class All:
        def get(self, domain, engine):
            return {"available": True}

    for reply, expected in [("orb", "orb-v3-conservative-inf-omat"),
                            ("MACE-OMAT-0", "mace-omat-0")]:
        with tempfile.TemporaryDirectory() as td:
            agent = MLIPAgent(working_dir=td, api_key="sk-not-real",
                              model_name="gpt-4o-mini")
            agent._backends = mt.check_backends(available_software=All())
            agent._generate_json = lambda p, r=reply: {
                "backend": "x", "model_name": r, "justification": "fits"}
            result = agent._select_pretrained_model(["Cu"], "Cu MD")
        assert result["model_name"] == expected
        assert "fallback" not in result["justification"]


# ── review: data files the ASE mass guess got wrong ──────────────────

_HAND_WRITTEN = """{title}
2 atoms
{ntypes} atom types
0 4 xlo xhi
0 4 ylo yhi
0 4 zlo zhi

Masses

{masses}

Atoms # atomic

1 1 0 0 0
2 {t2} 2 2 2
"""


def _hand(td, name, title="LAMMPS data", ntypes=2, t2=2,
          masses="1 22.98977 # Na\n2 35.453 # Cl"):
    path = os.path.join(td, name)
    open(path, "w").write(_HAND_WRITTEN.format(
        title=title, ntypes=ntypes, t2=t2, masses=masses))
    return path


def test_unused_declared_type_is_kept():
    from scilink.skills._shared.structure_io import (
        lammps_type_elements, structure_elements)
    from scilink.skills.molecular_dynamics.lammps.lammps import _pair_coeff_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "u.data", ntypes=3,
                  masses="1 22.98977 # Na\n2 35.453 # Cl\n3 39.0983 # K")
        assert lammps_type_elements(f) == {1: "Na", 2: "Cl", 3: "K"}
        assert structure_elements(f) == (["Cl", "Na"], 2)    # present atoms
        assert _pair_coeff_elements(f, ["Cl", "Na"]) == ["Na", "Cl", "K"]


def test_mass_label_wins_over_nearest_mass():
    """'59.0 # Ni' is Ni; the nearest atomic mass would say Co."""
    from scilink.skills._shared.structure_io import structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "ni.data", ntypes=1, t2=1, masses="1 59.0 # Ni")
        assert structure_elements(f)[0] == ["Ni"]
        atoms = _load_in_script(td, "ni.data", ["Ni"])
        assert set(atoms.get_chemical_symbols()) == {"Ni"}


def test_title_line_starting_with_atoms():
    from scilink.skills._shared.structure_io import lammps_type_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "t.data", title="Atoms of NaCl supercell", ntypes=3,
                  masses="1 22.98977 # Na\n2 35.453 # Cl\n3 39.0983 # K")
        assert lammps_type_elements(f) == {1: "Na", 2: "Cl", 3: "K"}


def test_masses_missing_a_declared_type_is_unknown():
    from scilink.skills._shared.structure_io import structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "m.data", ntypes=3)       # type 3 has no Masses line
        assert structure_elements(f)[0] == []


# ── review round 2: labels, supersets, styles, unknown vs none ──────

def test_full_element_names_and_uppercase_labels():
    """'Carbon' is C (not Ca), 'Nitrogen' N (not Ni), 'NA' at 22.99 is Na."""
    from scilink.skills._shared.structure_io import lammps_type_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "names.data", ntypes=3, t2=2,
                  masses="1 12.011 # Carbon\n2 14.007 # Nitrogen\n3 22.98977 # NA")
        assert lammps_type_elements(f) == {1: "C", 2: "N", 3: "Na"}
        atoms = _load_in_script(td, "names.data", ["C", "N", "Na"])
        assert set(atoms.get_chemical_symbols()) == {"C", "N"}


def test_label_that_names_no_element_of_that_mass_is_unknown():
    """A united-atom 'CH2' at 14.03 must not become N (nearest mass)."""
    from scilink.skills._shared.structure_io import read_structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "ua.data", ntypes=1, t2=1, masses="1 14.027 # CH2")
        elements, _, problem = read_structure_elements(f)
        assert elements == [] and "CH2" in problem


def test_coarse_grained_bead_is_not_an_element():
    from scilink.skills._shared.structure_io import read_structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "x.data", ntypes=1, t2=1, masses="1 1.0 # X")
        elements, _, problem = read_structure_elements(f)
        assert elements == [] and "Masses" in problem


def test_superset_potential_is_accepted():
    """A model fine-tuned on {Cl, Na, O, P} may run a NaCl file."""
    from scilink.skills.molecular_dynamics.lammps.lammps import _pair_coeff_elements
    with tempfile.TemporaryDirectory() as td:
        f = _data(td, "s.data", masses=True, specorder=["Na", "Cl"])
        assert _pair_coeff_elements(f, ["Cl", "Na", "O", "P"]) == ["Na", "Cl"]
        atoms = _load_in_script(td, "s.data", ["Cl", "Na", "O", "P"])
        assert sorted(set(atoms.get_chemical_symbols())) == ["Cl", "Na"]


_CHARGE_NO_HEADER = """LAMMPS data
2 atoms
2 atom types
0 6 xlo xhi
0 6 ylo yhi
0 6 zlo zhi

Masses

1 22.98977 # Na
2 35.453 # Cl

Atoms

1 1 1.0 0 0 0
2 2 -1.0 2.8 0 0
"""


def test_six_column_file_without_style_header_is_read():
    """ASE cannot guess 6 columns; the style parse_data_file infers is used."""
    from scilink.skills._shared.structure_io import (
        inspect_lammps_data, structure_elements)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "q.data")
        open(f, "w").write(_CHARGE_NO_HEADER)
        assert structure_elements(f) == (["Cl", "Na"], 2)
        assert inspect_lammps_data(f)["atom_style"] == "charge"
        atoms = _load_in_script(td, "q.data", ["Cl", "Na"])
        assert sorted(set(atoms.get_chemical_symbols())) == ["Cl", "Na"]


def test_unreadable_atoms_are_unknown_not_none():
    """Masses resolved but Atoms unreadable: pair_coeff from Masses, no false mismatch."""
    from scilink.skills._shared.structure_io import read_structure_elements
    from scilink.skills.molecular_dynamics.lammps.lammps import _pair_coeff_elements
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "bad.data")
        open(f, "w").write(_CHARGE_NO_HEADER.replace(
            "1 1 1.0 0 0 0\n2 2 -1.0 2.8 0 0", "1 1 0 0\n2 2 2.8 0"))
        elements, _, problem = read_structure_elements(f)
        assert elements == [] and "could not be read" in problem
        assert _pair_coeff_elements(f, ["Cl", "Na"]) == ["Na", "Cl"]


def test_pair_coeff_parses_the_data_file_once(monkeypatch):
    import scilink.skills.molecular_dynamics.lammps.lammps as L
    calls = []
    real = L.parse_data_file
    monkeypatch.setattr(L, "parse_data_file",
                        lambda p: calls.append(p) or real(p))
    with tempfile.TemporaryDirectory() as td:
        f = _data(td, "s.data", masses=True)
        L._pair_coeff_elements(f, ["Cl", "Na"])
    assert len(calls) == 1


def test_lowercase_type_names_fall_back_to_the_mass():
    """'opls_140' / 'type1' are type names, not element claims: use the mass."""
    from scilink.skills._shared.structure_io import lammps_type_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "tn.data", ntypes=2,
                  masses="1 1.008 # opls_140\n2 15.999 # type1")
        assert lammps_type_elements(f) == {1: "H", 2: "O"}


def test_capitalised_label_contradicting_mass_is_unknown():
    """'Water' at 15.999 is not W (tungsten); refuse rather than guess."""
    from scilink.skills._shared.structure_io import read_structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "w.data", ntypes=1, t2=1, masses="1 15.999 # Water oxygen")
        elements, _, problem = read_structure_elements(f)
        assert elements == [] and "Water" in problem


@pytest.mark.parametrize("label,mass,expected", [
    # common force-field type names keep their element
    ("OW", 15.9994, "O"), ("HW", 1.008, "H"), ("CT", 12.011, "C"),
    ("HC", 1.008, "H"), ("c3", 12.011, "C"), ("ca", 12.011, "C"),
    ("CT1", 12.011, "C"), ("HGA1", 1.008, "H"), ("Na+", 22.99, "Na"),
    ("Cl-", 35.45, "Cl"), ("Zn2+", 65.38, "Zn"), ("Cl1", 35.45, "Cl"),
    # first-letters readings that the mass now corrects
    ("Carbon", 12.011, "C"), ("Nitrogen", 14.007, "N"), ("NA+", 22.99, "Na"),
    ("CL", 35.45, "Cl"), ("CLA", 35.45, "Cl"), ("CA", 40.078, "Ca"),
    ("SI", 28.086, "Si"), ("FE", 55.845, "Fe"), ("MG", 24.305, "Mg"),
    # established ion type names (CHARMM)
    ("SOD", 22.99, "Na"), ("POT", 39.098, "K"),
    # no reading matches the mass
    ("CH2", 14.027, None), ("H2O", 18.015, None), ("D", 2.014, None),
    ("X", 1.0, None), ("opls_140", 1.008, None), ("1", 22.99, None),
])
def test_element_from_label(label, mass, expected):
    from scilink.skills.molecular_dynamics.lammps.lammps import element_from_label
    assert element_from_label(label, mass) == expected


# ── review round 3 (independent pass) ────────────────────────────────

_WRITE_DATA_STYLE = """LAMMPS data file via write_data

3 atoms
2 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.9994 # O
2 1.008 # H

{coeffs_header}

1 0.102 3.188
2 0 0

Atoms # atomic

1 1 5.0 5.0 5.0
2 2 5.96 5.0 5.0
3 2 4.76 5.92 5.0
"""


@pytest.mark.parametrize("header", ["Pair Coeffs # lj/cut/coul/long",
                                    "PairIJ Coeffs # lj/cut"])
def test_coefficient_section_after_masses_does_not_overwrite_them(header):
    """write_data headers carry a style comment; their rows are not masses."""
    from scilink.skills._shared.structure_io import inspect_lammps_data
    from scilink.skills.molecular_dynamics.lammps.lammps import _pair_coeff_elements
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "w.data")
        open(f, "w").write(_WRITE_DATA_STYLE.format(coeffs_header=header))
        info = inspect_lammps_data(f)
        assert info["type_elements"] == {1: "O", 2: "H"}
        assert info["present"] == ["H", "O"]
        assert _pair_coeff_elements(f, ["H", "O"]) == ["O", "H"]


_SIX_COLUMNS = """LAMMPS data
3 atoms
2 atom types
0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.9994 # O
2 1.008 # H

Atoms

1 1 1 5.0 5.0 5.0
2 1 2 5.96 5.0 5.0
3 1 2 4.76 5.92 5.0
"""


def test_ambiguous_six_column_file_is_refused():
    """id mol type x y z vs id type q x y z: no header, no bonds -> refuse."""
    from scilink.skills._shared.structure_io import read_structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "m.data")
        open(f, "w").write(_SIX_COLUMNS)
        elements, _, problem = read_structure_elements(f)
        assert elements == [] and "atom style" in problem


def test_script_refuses_multi_type_file_staged_after_generation():
    """No type map at generation -> the script must not guess by mass."""
    from scilink.agents.sim_agents._ase_runner import generate_ase_script
    with tempfile.TemporaryDirectory() as td:
        path = generate_ase_script(_potential(["Co", "Ni"]), td,
                                   structure_file=os.path.join(td, "later.data"),
                                   task="relax", device="cpu")
        _hand(td, "later.data", masses="1 59.0 # Ni\n2 58.0 # Co")
        cwd = os.getcwd()
        os.chdir(td)
        try:
            ns = runpy.run_path(path, run_name="not_main")
            with pytest.raises(SystemExit, match="not resolved"):
                ns["_load_structure"]()
        finally:
            os.chdir(cwd)


@pytest.mark.parametrize("masses", [
    "1 15.035 # ch3",        # lowercase united-atom: nearest mass is O
    "1 13.019 # CH",         # united-atom CH: 1 amu from C
    "1 3.024 # HE1",         # repartitioned hydrogen: not helium
    "1 15.035",              # no label, mass far from any element
])
def test_mass_not_close_to_any_element_is_unknown(masses):
    from scilink.skills._shared.structure_io import read_structure_elements
    with tempfile.TemporaryDirectory() as td:
        f = _hand(td, "ua.data", ntypes=1, t2=1, masses=masses)
        assert read_structure_elements(f)[0] == []
