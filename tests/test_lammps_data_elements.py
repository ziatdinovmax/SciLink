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

        def stop(elements, goal, task_name=None):
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
    path = generate_ase_script(_potential(elements), td,
                               structure_file=structure_name, task="relax",
                               device="cpu")
    cwd = os.getcwd()
    os.chdir(td)
    try:
        ns = runpy.run_path(path, run_name="not_main")
        return ns["_load_structure"]()
    finally:
        os.chdir(cwd)


def test_run_script_refuses_lammps_data_without_masses():
    with tempfile.TemporaryDirectory() as td:
        _data(td, "n.data", masses=False)
        with pytest.raises(SystemExit, match="no Masses section"):
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
        with pytest.raises(ValueError, match="no Masses section"):
            _pair_coeff_elements(nomass, ["Cl", "Na"])
        f = _data(td, "s.data", masses=True)
        with pytest.raises(ValueError, match="deployed for"):
            _pair_coeff_elements(f, ["Cu"])


# ── smaller fixes ────────────────────────────────────────────────────

def test_uma_template_keeps_gpu_for_indexed_cuda():
    from scilink.skills._shared import mlip_tools as mt
    spec = mt._skill_meta("uma")["deploy"]
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
