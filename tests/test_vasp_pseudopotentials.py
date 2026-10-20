"""Pseudopotential contract (#786): resolve, VASP hooks, prep script, agent + pipeline wiring.

Uses synthetic POTCAR headers only — no licensed POTCAR content is in the repo.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from scilink.skills._shared.pseudopotentials import element_of, resolve_pseudopotentials
from scilink.skills.periodic_dft.vasp import vasp_pseudopotentials as vp

BASH = shutil.which("bash")


def _fake_potcar(d: Path, variant: str, zval: float, enmax: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "POTCAR").write_text(
        f"  PAW_PBE {variant} 01Jan2000\n {zval:.14f}\n parameters from PSCTR are:\n"
        f"   TITEL  = PAW_PBE {variant} 01Jan2000\n"
        f"   POMASS =   1.000; ZVAL   =    {zval:.3f}    mass and valenz\n"
        f"   ENMAX  =  {enmax:.3f}; ENMIN  =  100.000 eV\n End of Dataset\n"
    )


@pytest.fixture
def library(tmp_path):
    """A library with plain Na, Na_pv, Cl, Ca_sv (no plain Ca), Si, Ge, Ge_d."""
    lib = tmp_path / "lib"
    for v, z, e in [("Na", 1, 102), ("Na_pv", 7, 260), ("Cl", 7, 262), ("Ca_sv", 10, 267),
                    ("Si", 4, 245), ("Ge", 4, 174), ("Ge_d", 14, 310), ("O", 6, 400)]:
        _fake_potcar(lib / v, v, z, e)
    return lib


POSCAR_NACL = """NaCl
1.0
5.6 0 0
0 5.6 0
0 0 5.6
Na Cl
1 1
Direct
0 0 0
0.5 0.5 0.5
"""

POSCAR_REPEATED = POSCAR_NACL.replace("Na Cl\n1 1", "Si Ge Si\n1 1 1").replace(
    "0.5 0.5 0.5\n", "0.5 0.5 0.5\n0.25 0.25 0.25\n")


# ── resolve ──────────────────────────────────────────────────────────

def test_element_of():
    assert element_of("Na_pv") == "Na"
    assert element_of("H.75") == "H"
    assert element_of("Ge_d") == "Ge"
    assert element_of("C") == "C"
    assert element_of("_x") is None


def test_resolve_precedence_user_over_agent_over_default():
    choice, warnings = resolve_pseudopotentials(
        ["Na", "Cl", "Ca"], {"Na": "Na_pv", "Cl": "Cl", "Ca": "Ca_sv"},
        agent_choice={"Na": {"variant": "Na_sv", "reason": "high pressure"}, "Cl": "Cl_h"},
        user_map={"Cl": "Cl"})
    assert choice["Na"] == {"variant": "Na_sv", "source": "agent", "reason": "high pressure"}
    assert choice["Cl"]["source"] == "user" and choice["Cl"]["variant"] == "Cl"
    assert choice["Ca"] == {"variant": "Ca_sv", "source": "default", "reason": None}
    assert warnings == []


def test_resolve_rejects_agent_variant_of_wrong_element_or_missing():
    inv = {"Na": ["Na", "Na_pv"]}
    choice, warnings = resolve_pseudopotentials(
        ["Na"], {"Na": "Na_pv"}, agent_choice={"Na": "Cl_h"}, inventory=inv)
    assert choice["Na"]["variant"] == "Na_pv" and choice["Na"]["source"] == "default"
    assert any("not a variant of Na" in w for w in warnings)
    choice, warnings = resolve_pseudopotentials(
        ["Na"], {"Na": "Na_pv"}, agent_choice={"Na": "Na_sv"}, inventory=inv)
    assert choice["Na"]["variant"] == "Na_pv"
    assert any("not in the library" in w for w in warnings)


def test_resolve_falls_back_when_default_missing():
    inv = {"Cu": ["Cu"], "Ca": ["Ca_pv", "Ca_sv"], "Xx": []}
    choice, warnings = resolve_pseudopotentials(
        ["Cu", "Ca", "Xx"], {"Cu": "Cu_pv", "Ca": "Ca_h", "Xx": "Xx"}, inventory=inv)
    assert choice["Cu"] == {"variant": "Cu", "source": "fallback", "reason": None}
    assert choice["Ca"]["variant"] == "Ca_pv" and choice["Ca"]["source"] == "fallback"
    assert any("no pseudopotential for Xx" in w for w in warnings)
    assert len(warnings) == 3


def test_resolve_without_inventory_trusts_choice():
    choice, warnings = resolve_pseudopotentials(["Ca"], {"Ca": "Ca_sv"}, inventory=None)
    assert choice["Ca"]["variant"] == "Ca_sv" and warnings == []


# ── VASP hooks ───────────────────────────────────────────────────────

def test_default_pseudopotentials_mp_set():
    d = vp.default_pseudopotentials(["Ca", "Na", "Si", "Zz"])
    assert d == {"Ca": "Ca_sv", "Na": "Na_pv", "Si": "Si", "Zz": "Zz"}


def test_list_pseudopotentials_direct_layout(library):
    listing = vp.list_pseudopotentials(["Na", "Ca", "Ge"], library=str(library))
    assert [a["variant"] for a in listing["Na"]] == ["Na", "Na_pv"]
    assert [a["variant"] for a in listing["Ca"]] == ["Ca_sv"]
    na_pv = listing["Na"][1]
    assert na_pv["zval"] == 7 and na_pv["enmax"] == pytest.approx(260)
    assert na_pv["titel"].startswith("PAW_PBE Na_pv")


def test_list_pseudopotentials_functional_subdir_layout(tmp_path):
    _fake_potcar(tmp_path / "potpaw_PBE" / "O", "O", 6, 400)
    _fake_potcar(tmp_path / "potpaw" / "O", "O", 6, 350)       # LDA set
    listing = vp.list_pseudopotentials(["O"], library=str(tmp_path))
    assert [a["enmax"] for a in listing["O"]] == [pytest.approx(400)]
    listing = vp.list_pseudopotentials(["O"], library=str(tmp_path), gga="LDA")
    assert [a["enmax"] for a in listing["O"]] == [pytest.approx(350)]


def test_list_pseudopotentials_none_without_library(monkeypatch, tmp_path):
    monkeypatch.delenv("VASP_PP_PATH", raising=False)
    assert vp.list_pseudopotentials(["Na"]) is None
    assert vp.list_pseudopotentials(["Na"], library=str(tmp_path / "nope")) is None


def test_poscar_species():
    assert vp.poscar_species(POSCAR_NACL) == ["Na", "Cl"]
    assert vp.poscar_species(POSCAR_REPEATED) == ["Si", "Ge", "Si"]
    assert vp.poscar_species("too\nshort") == []


def test_render_spec_one_line_per_species():
    files = vp.render_pseudopotential_prep(
        {"Si": {"variant": "Si"}, "Ge": {"variant": "Ge_d"}}, ["Si", "Ge", "Si"])
    assert files[vp.SPEC_FILE] == "Si\nGe_d\n"
    assert vp.PREP_SCRIPT in files and "VASP_PP_PATH" in files[vp.PREP_SCRIPT]


def test_check_encut_floor():
    assert vp.check_encut_floor("ENCUT = 300\n", {"Na": 260.0}) is not None   # 300 < 338
    assert vp.check_encut_floor("ENCUT = 520\n", {"Na": 260.0}) is None
    assert vp.check_encut_floor("ISMEAR = 0\n", {"Na": 260.0}) is None       # no ENCUT
    assert vp.check_encut_floor("ENCUT = 300\n", {"Na": None}) is None       # ENMAX unknown


# ── prep script, executed ────────────────────────────────────────────

def _run_prep(run_dir: Path, files: dict, env_lib):
    for name, text in files.items():
        (run_dir / name).write_text(text)
    env = {"PATH": "/usr/bin:/bin"}
    if env_lib is not None:
        env["VASP_PP_PATH"] = str(env_lib)
    return subprocess.run([BASH, vp.PREP_SCRIPT], cwd=run_dir, env=env,
                          capture_output=True, text=True)


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_prep_script_assembles_in_poscar_order_with_repeats(tmp_path, library):
    files = {"POSCAR": POSCAR_REPEATED, "INCAR": "GGA = PE\n",
             **vp.render_pseudopotential_prep({"Si": "Si", "Ge": "Ge_d"}, ["Si", "Ge", "Si"])}
    r = _run_prep(tmp_path, files, library)
    assert r.returncode == 0, r.stderr
    titels = [ln.split("=", 1)[1].strip() for ln in (tmp_path / "POTCAR").read_text().splitlines()
              if "TITEL" in ln]
    assert titels == ["PAW_PBE Si 01Jan2000", "PAW_PBE Ge_d 01Jan2000", "PAW_PBE Si 01Jan2000"]
    used = vp.describe_pseudopotentials(str(tmp_path))
    assert used["Ge"]["variant"] == "Ge_d" and used["Ge"]["enmax"] == pytest.approx(310)
    assert used["Si"]["titel"] == "PAW_PBE Si 01Jan2000"
    assert len(used["Si"]["sha256"]) == 64


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_prep_script_functional_subdir(tmp_path):
    lib = tmp_path / "lib"
    _fake_potcar(lib / "potpaw_PBE" / "Na_pv", "Na_pv", 7, 260)
    _fake_potcar(lib / "potpaw_PBE" / "Cl", "Cl", 7, 262)
    run = tmp_path / "run"; run.mkdir()
    files = {"POSCAR": POSCAR_NACL, "INCAR": "GGA = PE\n",
             **vp.render_pseudopotential_prep({"Na": "Na_pv", "Cl": "Cl"}, ["Na", "Cl"])}
    r = _run_prep(run, files, lib)
    assert r.returncode == 0, r.stderr
    assert (run / "POTCAR").read_text().count("TITEL") == 2


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_prep_script_missing_variant_is_a_clear_pre_run_error(tmp_path, library):
    """The case that motivated #786: no plain Ca in the library."""
    poscar = POSCAR_NACL.replace("Na Cl", "Ca O")
    files = {"POSCAR": poscar, "INCAR": "GGA = PE\n",
             **vp.render_pseudopotential_prep({"Ca": "Ca", "O": "O"}, ["Ca", "O"])}
    r = _run_prep(tmp_path, files, library)
    assert r.returncode == vp.MISSING_EXIT_CODE
    assert "MISSING PSEUDOPOTENTIAL Ca:Ca" in r.stderr
    assert not (tmp_path / "POTCAR").exists()


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_prep_script_requires_library_env(tmp_path):
    files = {"POSCAR": POSCAR_NACL, "INCAR": "",
             **vp.render_pseudopotential_prep({"Na": "Na", "Cl": "Cl"}, ["Na", "Cl"])}
    r = _run_prep(tmp_path, files, None)
    assert r.returncode == vp.MISSING_EXIT_CODE and "VASP_PP_PATH is not set" in r.stderr


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_prep_script_honours_an_edited_spec(tmp_path, library):
    """A critic fix that edits POTCAR.spec takes effect on the next run."""
    files = {"POSCAR": POSCAR_NACL, "INCAR": "",
             **vp.render_pseudopotential_prep({"Na": "Na", "Cl": "Cl"}, ["Na", "Cl"])}
    files[vp.SPEC_FILE] = "Na_pv\nCl\n"
    r = _run_prep(tmp_path, files, library)
    assert r.returncode == 0, r.stderr
    assert vp.describe_pseudopotentials(str(tmp_path))["Na"]["variant"] == "Na_pv"


# ── agent wiring ─────────────────────────────────────────────────────

def _agent():
    from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent
    a = PeriodicDFTAgent.__new__(PeriodicDFTAgent)
    import logging
    a.logger = logging.getLogger("test")
    return a


def test_agent_injects_spec_prep_and_records_sources(library):
    a = _agent()
    ctx = {"elements": ["Na", "Cl"], "defaults": {"Na": "Na_pv", "Cl": "Cl"},
           "listing": vp.list_pseudopotentials(["Na", "Cl"], library=str(library))}
    result = {"input_files": {"INCAR": "GGA = PE\nENCUT = 300\n", "POSCAR": POSCAR_NACL,
                              "POTCAR": "llm-authored"}}
    a._inject_pseudopotentials(result, ctx, "vasp",
                               agent_choice={"Cl": {"variant": "Cl", "reason": "standard"}})
    files = result["input_files"]
    assert files[vp.SPEC_FILE] == "Na_pv\nCl\n"
    assert vp.PREP_SCRIPT in files and "POTCAR" not in files
    rec = result["pseudopotentials"]
    assert rec["choice"]["Na"]["source"] == "default"
    assert rec["choice"]["Cl"]["source"] == "agent"
    assert rec["enmax"] == {"Na": pytest.approx(260), "Cl": pytest.approx(262)}
    assert any("ENCUT" in w for w in rec["warnings"])          # 300 < 1.3 * 262
    assert result["pre_run_command"] == "bash prepare_pseudopotentials.sh"


def test_agent_prompt_block_lists_defaults_and_inventory(library):
    ctx = {"elements": ["Na"], "defaults": {"Na": "Na_pv"},
           "listing": vp.list_pseudopotentials(["Na"], library=str(library))}
    block = _agent()._pseudo_prompt_block(ctx)
    assert "recommended `Na_pv`" in block and "`Na_pv` (ZVAL 7, ENMAX 260 eV)" in block
    no_lib = _agent()._pseudo_prompt_block({**ctx, "listing": None})
    assert "not readable here" in no_lib
    assert _agent()._pseudo_prompt_block(None) == ""


def test_context_is_none_for_engine_without_contract(tmp_path):
    from ase.build import bulk
    from ase.io import write
    p = tmp_path / "s.extxyz"; write(p, bulk("Si"))
    a = _agent()
    assert a._pseudopotential_context(str(p), {"skill_sections": {"meta": {}}}, "qe") is None
    ctx = a._pseudopotential_context(
        str(p), {"skill_sections": {"meta": {"pseudopotentials": {"library_env": "VASP_PP_PATH"}}}},
        "vasp")
    assert ctx["elements"] == ["Si"] and ctx["defaults"] == {"Si": "Si"}


# ── pipeline wiring ──────────────────────────────────────────────────

def test_with_pre_run_prefixes_once():
    from scilink.agents.sim_agents.simulation_pipeline import _with_pre_run
    g = {"pre_run_command": "bash prepare_pseudopotentials.sh"}
    once = _with_pre_run("mpirun vasp_std", g)
    assert once == "bash prepare_pseudopotentials.sh && mpirun vasp_std"
    assert _with_pre_run(once, g) == once
    assert _with_pre_run("mpirun vasp_std", {}) == "mpirun vasp_std"
    assert _with_pre_run(None, g) is None


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_local_executor_runs_prep_then_engine(tmp_path, library, monkeypatch):
    """End to end without VASP: the executor materializes the inputs, the prep
    step assembles POTCAR, the 'engine' sees it, and provenance is recorded."""
    from scilink.agents.sim_agents.refinement import LocalExecutor
    from scilink.agents.sim_agents.simulation_pipeline import (
        _record_pseudopotential_provenance, _with_pre_run)
    monkeypatch.setenv("VASP_PP_PATH", str(library))
    a = _agent()
    ctx = {"elements": ["Na", "Cl"], "defaults": {"Na": "Na_pv", "Cl": "Cl"},
           "listing": vp.list_pseudopotentials(["Na", "Cl"])}
    gen = {"input_files": {"INCAR": "GGA = PE\n", "POSCAR": POSCAR_NACL}}
    a._inject_pseudopotentials(gen, ctx, "vasp")
    cmd = _with_pre_run("grep -c TITEL POTCAR", gen)
    out = LocalExecutor(timeout=60).run(gen["input_files"], cmd, str(tmp_path))
    assert (tmp_path / LocalExecutor.STDOUT_FILE).read_text().strip() == "2", out
    result = {}
    _record_pseudopotential_provenance(result, gen, str(tmp_path), "vasp")
    assert result["pseudopotentials"]["used"]["Na"]["variant"] == "Na_pv"
    assert result["pseudopotentials"]["choice"]["Cl"]["source"] == "default"


# ── UI submit script ─────────────────────────────────────────────────

def test_ui_submit_script_uses_prep_script_when_spec_present():
    from scilink.ui.components.vasp_workflow import _build_slurm_script
    kw = dict(job_name="j", account="", partition="", time_limit="1:00:00", nodes=1,
              ntasks_per_node=4, extra_sbatch="", work_dir="/w", module_commands="",
              pseudo_dir="/lib/pbe", elements=["Na", "Cl"], vasp_command="mpirun vasp_std")
    legacy = _build_slurm_script(**kw)
    assert 'cat "$PSEUDO_DIR/$e/POTCAR"' in legacy and "prepare_pseudopotentials" not in legacy
    prep = _build_slurm_script(**kw, use_prep_script=True)
    assert "export VASP_PP_PATH='/lib/pbe'" in prep
    assert prep.index("bash prepare_pseudopotentials.sh") < prep.index("mpirun vasp_std")
    assert "PSEUDO_DIR" not in prep
