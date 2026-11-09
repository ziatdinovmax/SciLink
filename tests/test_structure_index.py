"""The shared structure index (#804) and plan mode's index_structures tool.

Offline: hand-written CIFs (mixed and split sites, symmetry to expand, a
large P1 cell, broken files), POSCAR and xyz files, and a planning
orchestrator built with a dummy key. No model calls.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

ase = pytest.importorskip("ase")
pd = pytest.importorskip("pandas")

from scilink.utils import structure_index as si  # noqa: E402


def _cif(name, cell, sites, *, formula=None, sg_number=None, hm=None, ops=None, extra=""):
    """A CIF: ``sites`` = (label, type, x, y, z, occupancy)."""
    a, b, c, al, be, ga = cell
    out = [f"data_{name}"]
    if formula:
        out.append(f"_chemical_formula_sum '{formula}'")
    out += [f"_cell_length_a {a}", f"_cell_length_b {b}", f"_cell_length_c {c}",
            f"_cell_angle_alpha {al}", f"_cell_angle_beta {be}", f"_cell_angle_gamma {ga}"]
    if sg_number:
        out.append(f"_space_group_IT_number {sg_number}")
    if hm:
        out.append(f"_symmetry_space_group_name_H-M '{hm}'")
    if extra:
        out.append(extra)
    if ops:
        out += ["loop_", "_symmetry_equiv_pos_as_xyz"] + [f"'{o}'" for o in ops]
    out += ["loop_", "_atom_site_label", "_atom_site_type_symbol", "_atom_site_fract_x",
            "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    out += [" ".join(str(v) for v in s) for s in sites]
    return "\n".join(out) + "\n"


CUBIC = lambda a: (a, a, a, 90, 90, 90)  # noqa: E731

NACL = _cif("nacl", CUBIC(5.64), [("Na1", "Na+", 0, 0, 0, 1), ("Cl1", "Cl-", 0.5, 0.5, 0.5, 1)],
            formula="Cl Na", sg_number=225, hm="F m -3 m",
            extra="_chemical_name_mineral Halite\n_cod_database_code 1000041")
MIXED = _cif("mixed", CUBIC(4.2), [("Fe1", "Fe2+", 0, 0, 0, 0.5), ("Ni1", "Ni2+", 0, 0, 0, 0.5),
                                   ("O1", "O2-", 0.5, 0.5, 0.5, 1)],
             formula="Fe0.5 Ni0.5 O", sg_number=225, hm="F m -3 m")
OVERLAP = _cif("overlap", CUBIC(5.0), [("Na1", "Na", 0, 0, 0, 1), ("Cl1", "Cl", 0.04, 0, 0, 1),
                                       ("Cl2", "Cl", 0.5, 0.5, 0.5, 1)], ops=["x, y, z"])
SPLIT = _cif("split", CUBIC(5.0), [("O1a", "O", 0, 0, 0, 0.5), ("O1b", "O", 0.06, 0, 0, 0.5),
                                   ("Ti1", "Ti", 0.5, 0.5, 0.5, 1)], ops=["x, y, z"])
NO_H = _cif("noh", CUBIC(10.0), [("C1", "C", 0.1, 0.1, 0.1, 1), ("C2", "C", 0.25, 0.1, 0.1, 1),
                                 ("O1", "O", 0.4, 0.1, 0.1, 1)],
            formula="C2 H6 O", ops=["x, y, z"])
TIO2 = _cif("rutile", (4.594, 4.594, 2.959, 90, 90, 90),
            [("Ti1", "Ti4+", 0, 0, 0, 1), ("O1", "O2-", 0.3048, 0.3048, 0, 1)],
            sg_number=136, hm="P 42/m n m")


def _write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(text)
    return p


def _row(path, **kw):
    return si.parse_structure_file(path, path.parent, compute_symmetry=kw.get("cs", False))


# ── one file ─────────────────────────────────────────────────────────────

def test_symmetry_is_expanded_and_tags_are_read(tmp_path):
    r = _row(_write(tmp_path, "nacl.cif", NACL))
    assert r["parse_error"] is None
    assert r["n_atoms"] == 8.0 and r["n_sites"] == 2
    assert r["formula"] == "Cl Na" and r["formula_source"] == "declared"
    assert r["reduced_formula"] == "NaCl" and r["elements"] == "Cl-Na" and r["metals"] == "Na"
    assert r["spacegroup_number"] == 225 and r["crystal_system"] == "cubic"
    assert r["mineral_name"] == "Halite" and r["database_id"] == "COD:1000041"
    assert r["density"] == pytest.approx(2.165, abs=0.01)
    assert r["min_distance"] == pytest.approx(2.82, abs=0.01)
    assert r["sites_match_formula"] is True and not r["partial_occupancy"]


def test_a_mixed_site_keeps_every_species(tmp_path):
    # ase.io.read keeps one species per mixed site; the index must not
    r = _row(_write(tmp_path, "mixed.cif", MIXED))
    assert r["elements"] == "Fe-Ni-O"
    assert r["reduced_formula"] == "FeNiO2" and r["n_atoms"] == 8.0
    assert r["partial_occupancy"] and r["sites_match_formula"] is True
    # Fe and Ni share a site: alternatives, not overlapping atoms
    assert not r["overlapping_atoms"] and r["min_distance"] == pytest.approx(2.1, abs=0.01)


def test_overlapping_atoms_are_flagged_with_the_distance(tmp_path):
    r = _row(_write(tmp_path, "overlap.cif", OVERLAP))
    assert r["overlapping_atoms"] and r["min_distance"] == pytest.approx(0.2, abs=1e-3)
    assert r["formula_source"] == "counted" and r["formula"] == "NaCl2"


def test_a_split_site_is_partial_occupancy_not_overlap(tmp_path):
    r = _row(_write(tmp_path, "split.cif", SPLIT))
    assert r["partial_occupancy"] and not r["overlapping_atoms"]


def test_missing_hydrogens_are_flagged(tmp_path):
    r = _row(_write(tmp_path, "noh.cif", NO_H))
    assert r["carbon_without_hydrogen"] and r["sites_match_formula"] is False
    assert r["formula"] == "C2 H6 O"           # the declared formula is kept as written
    assert r["elements"] == "C-O"              # what the structure actually holds


def test_p1_cells_skip_symmetry_expansion(tmp_path, monkeypatch):
    from ase.build import bulk
    import ase.io
    from ase.io.cif import CIFBlock
    big = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((8, 8, 8))     # 2048 atoms
    ase.io.write(tmp_path / "big.cif", big, format="cif")

    def no(*a, **k):
        raise AssertionError("P1 must not go through get_atoms / get_spacegroup")
    monkeypatch.setattr(CIFBlock, "get_atoms", no)
    monkeypatch.setattr(CIFBlock, "get_spacegroup", no)
    t = time.monotonic()
    r = _row(tmp_path / "big.cif")
    assert time.monotonic() - t < 5
    assert r["parse_error"] is None and r["n_atoms"] == 2048.0
    assert r["density"] == pytest.approx(8.97, abs=0.02)


def test_unreadable_files_get_a_row_and_are_listed(tmp_path):
    _write(tmp_path, "nacl.cif", NACL)
    _write(tmp_path, "broken.cif", "data_x\n_cell_length_a 4\nthis is not a cif\n")
    _write(tmp_path, "tags_only.cif", "data_y\n_chemical_formula_sum 'Cu'\n")
    res = si.build_index(tmp_path, compute_symmetry=False)
    assert len(res.df) == 3                                   # nothing dropped
    assert [Path(p).name for p, _ in res.failures] == ["broken.cif"]
    tags = res.df[res.df["path"].str.endswith("tags_only.cif")].iloc[0]
    assert tags["formula"] == "Cu" and tags["parse_error"].startswith("sites:")
    assert "no fractional atom sites" in tags["parse_error"]
    summary = si.summarize(res)
    assert summary["n_failed"] == 1 and summary["quality"]["sites_not_read"] == 1
    json.dumps(summary)                                       # tool-result ready


def test_other_formats(tmp_path):
    from ase.build import bulk, molecule
    import ase.io
    ase.io.write(tmp_path / "POSCAR_si", bulk("Si", "diamond", a=5.43), format="vasp")
    ase.io.write(tmp_path / "water.xyz", molecule("H2O"), format="extxyz")
    rows = {Path(r["path"]).name: r for r in
            si.build_index(tmp_path, compute_symmetry=False).df.to_dict("records")}
    assert rows["POSCAR_si"]["file_format"] == "vasp"
    assert rows["POSCAR_si"]["density"] == pytest.approx(2.33, abs=0.01)
    assert rows["water.xyz"]["formula"] == "H2O" and pd.isna(rows["water.xyz"]["density"])


def test_formulas_read_the_usual_way():
    assert si._formula(si._reduce({"O": 4, "Ti": 2})) == "TiO2"
    assert si._formula(si._reduce({"Cl": 4, "Na": 4})) == "NaCl"
    assert si._formula({"C": 13, "H": 18, "O": 2}) == "C13H18O2"
    assert si._formula({"Li": 1, "Fe": 1, "P": 1, "O": 4}) == "LiFePO4"


def test_computed_space_group_and_mismatch(tmp_path):
    pymatgen = pytest.importorskip("pymatgen.core")
    s = pymatgen.Structure.from_spacegroup("Fd-3m", pymatgen.Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    (tmp_path / "si.cif").write_text(s.to(fmt="cif"))       # pymatgen writes P1
    r = si.parse_structure_file(tmp_path / "si.cif", tmp_path, compute_symmetry=True)
    assert r["spacegroup_number"] == 1 and r["computed_spacegroup_number"] == 227
    assert r["spacegroup_mismatch"] is True


# ── the index ────────────────────────────────────────────────────────────

def _tree(root: Path):
    _write(root, "nacl.cif", NACL)
    _write(root / "oxides", "rutile.cif", TIO2)
    _write(root / "oxides" / "doped", "mixed.cif", MIXED)
    _write(root / ".hidden", "skip.cif", NACL)


def test_subfolders_are_indexed_and_hidden_ones_skipped(tmp_path):
    _tree(tmp_path)
    df = si.build_index(tmp_path, compute_symmetry=False).df
    assert sorted(df["subfolder"]) == ["", "oxides", "oxides/doped"]
    top = si.build_index(tmp_path, recursive=False, compute_symmetry=False).df
    assert list(top["subfolder"]) == [""]


def test_incremental_rebuild(tmp_path):
    _tree(tmp_path / "s")
    out = tmp_path / "index.csv"
    first = si.build_index(tmp_path / "s", out, compute_symmetry=False)
    assert first.n_parsed == 3 and out.is_file()
    again = si.build_index(tmp_path / "s", out, compute_symmetry=False)
    assert again.n_parsed == 0 and again.n_reused == 3
    time.sleep(0.02)
    _write(tmp_path / "s", "nacl.cif", MIXED)                  # changed
    (tmp_path / "s" / "oxides" / "rutile.cif").unlink()          # removed
    _write(tmp_path / "s", "new.cif", SPLIT)                     # added
    third = si.build_index(tmp_path / "s", out, compute_symmetry=False)
    assert third.n_parsed == 2 and third.n_reused == 1 and len(third.df) == 3
    row = third.df[third.df["path"].str.endswith("/nacl.cif")].iloc[0]
    assert row["elements"] == "Fe-Ni-O"
    assert si.build_index(tmp_path / "s", out, refresh=True, compute_symmetry=False).n_parsed == 3


def test_an_index_from_an_older_version_is_rebuilt(tmp_path, monkeypatch):
    _tree(tmp_path / "s")
    out = tmp_path / "index.csv"
    si.build_index(tmp_path / "s", out, compute_symmetry=False)
    monkeypatch.setattr(si, "INDEX_VERSION", si.INDEX_VERSION + 1)
    assert si.build_index(tmp_path / "s", out, compute_symmetry=False).n_parsed == 3


def test_workers_give_the_same_rows(tmp_path, monkeypatch):
    for i in range(6):
        _write(tmp_path / "s", f"nacl_{i}.cif", NACL)
    serial = si.build_index(tmp_path / "s", compute_symmetry=False, workers=1).df
    monkeypatch.setattr(si, "PARALLEL_MIN_FILES", 2)
    parallel = si.build_index(tmp_path / "s", compute_symmetry=False, workers=2).df
    cols = ["path", "formula", "n_atoms", "density", "min_distance"]
    pd.testing.assert_frame_equal(serial[cols], parallel[cols])


# ── plan mode ────────────────────────────────────────────────────────────

@pytest.fixture
def planner(tmp_path, monkeypatch):
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
    orch = PlanningOrchestratorAgent(api_key="sk-dummy", data_dir=str(tmp_path / "data"),
                                     base_dir=str(tmp_path / "pl"))
    return orch


def _collection(root: Path, n=12):
    for i in range(n):
        _write(root / f"batch_{i % 3}", f"s{i:02d}.cif", NACL if i % 2 else TIO2)


def test_index_structures_writes_a_queryable_csv(planner, tmp_path):
    _collection(tmp_path / "cifs")
    r = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "cifs")))
    assert r["status"] == "success", r
    index = Path(r["index_file"])
    assert index.parent == Path(planner.base_dir) / "knowledge" and index.suffix == ".csv"
    assert r["n_rows"] == 12 and r["n_failed"] == 0
    df = pd.read_csv(index)
    assert set(df["reduced_formula"]) == {"NaCl", "TiO2"}
    assert "query_knowledge_data" in r["next_steps"] and str(index) in r["next_steps"]
    again = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "cifs")))
    assert again["n_parsed"] == 0 and again["n_reused"] == 12


def test_index_structures_finds_a_folder_by_name_in_the_knowledge_folder(planner):
    _collection(Path(planner.base_dir) / "knowledge" / "my_cifs", n=3)
    r = json.loads(planner.tools.execute_tool("index_structures", directory="my_cifs"))
    assert r["status"] == "success" and r["n_rows"] == 3


def test_index_structures_says_when_there_is_nothing_to_index(planner, tmp_path):
    (tmp_path / "empty").mkdir()
    r = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "empty")))
    assert r["status"] == "error" and "No structure files" in r["message"]


def test_a_structure_folder_query_is_sent_to_the_index_first(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval",
                        lambda *a, **k: pytest.fail("no consent asked: no code would run"))
    _collection(tmp_path / "cifs")          # in subfolders only: the top level holds no files
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="cubic structures", file_name=str(tmp_path / "cifs")))
    assert r["status"] == "index_structures_first"
    assert "index_structures(directory=" in r["message"] and r["structure_files"] == "12"
    idx = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "cifs")))
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="cubic structures", file_name=str(tmp_path / "cifs")))
    assert r["status"] == "index_structures_first" and idx["index_file"] in r["message"]


def test_screen_database_on_a_structure_folder_points_to_the_index(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda *a, **k: True)
    _collection(tmp_path / "cifs")
    r = json.loads(planner.tools.execute_tool(
        "screen_database", query="densest", database_path=str(tmp_path / "cifs")))
    assert r["status"] == "index_structures_first" and r["tool"] == "screen_database"


def test_a_folder_of_other_records_is_not_redirected(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda *a, **k: False)
    folder = tmp_path / "records"
    folder.mkdir()
    for i in range(12):
        (folder / f"r{i}.json").write_text(json.dumps({"id": i}))
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="count", file_name=str(folder)))
    assert r["status"] == "error" and "approval declined" in r["message"]   # the usual path


# ── review fixes ─────────────────────────────────────────────────────────

def _xyz_ops(number, setting=1):
    """A group's operations as CIF 'x,y,z' strings."""
    from fractions import Fraction
    from ase.spacegroup import Spacegroup
    out = []
    for rot, trans in Spacegroup(number, setting).get_symop():
        terms = []
        for r, t in zip(rot, trans):
            s = "".join(f"{'+' if c > 0 else '-'}{v}" for c, v in zip(r, "xyz") if c)
            if abs(t) > 1e-9:
                s += f"+{Fraction(t).limit_denominator(12)}"
            terms.append(s.lstrip("+"))
        out.append(",".join(terms))
    return out


CORUNDUM_CELL = (4.7589, 4.7589, 12.991, 90, 90, 120)
CORUNDUM_SITES = [("Al1", "Al3+", 0, 0, 0.35216, 1), ("O1", "O2-", 0.30624, 0, 0.25, 1)]


def test_listed_operations_need_no_group_name(tmp_path):
    # ops only, no IT number or H-M symbol: ASE's lookup raised and every site was lost
    r = _row(_write(tmp_path, "ops.cif", _cif("ops", CORUNDUM_CELL, CORUNDUM_SITES,
                                              ops=_xyz_ops(167))))
    assert r["parse_error"] is None and r["n_atoms"] == 30.0
    assert r["reduced_formula"] == "Al2O3" and r["density"] == pytest.approx(3.99, abs=0.02)
    # ops with a full H-M symbol ASE does not recognise
    r = _row(_write(tmp_path, "full_hm.cif", _cif("hm", CORUNDUM_CELL, CORUNDUM_SITES,
                                                  ops=_xyz_ops(167), hm="R -3 2/c")))
    assert r["parse_error"] is None and r["n_atoms"] == 30.0


def test_a_rhombohedral_group_takes_its_axes_from_the_cell(tmp_path):
    # no ops, "trigonal" tag, hexagonal cell: ASE alone used rhombohedral axes (24 atoms)
    r = _row(_write(tmp_path, "cor.cif", _cif(
        "cor", CORUNDUM_CELL, CORUNDUM_SITES, formula="Al2 O3", sg_number=167, hm="R -3 c :H",
        extra="_space_group_crystal_system trigonal")))
    assert r["n_atoms"] == 30.0 and r["sites_match_formula"] is True


@pytest.mark.parametrize("label,element", [
    ("Wat", "O"), ("Ow1", "O"), ("ow1", "O"), ("Hw1", "H"), ("CL1", "Cl"), ("SM1", "Sm"),
    ("Cs1", "Cs"), ("NB1", "Nb"), ("O2-", "O"), ("Fe3+", "Fe"), ("D1", "H"), ("N1A", "N"),
])
def test_labels_are_read_as_pymatgen_reads_them(label, element):
    assert si._element(None, label) == element


def test_the_xrd_view_filters_on_pymatgens_elements(tmp_path):
    pytest.importorskip("pymatgen.core")
    from scilink.skills.structure_matching._backends import _cif_index
    # label-only sites (no _atom_site_type_symbol): "Wat" is water oxygen, not W
    _write(tmp_path, "wat.cif", "\n".join([
        "data_w", "_cell_length_a 6.0", "_cell_length_b 6.0", "_cell_length_c 6.0",
        "_cell_angle_alpha 90", "_cell_angle_beta 90", "_cell_angle_gamma 90",
        "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'",
        "loop_", "_atom_site_label", "_atom_site_fract_x", "_atom_site_fract_y",
        "_atom_site_fract_z", "Na1 0 0 0", "Wat 0.5 0.5 0.5", ""]))
    df = _cif_index._xrd_view(si.build_index(tmp_path, compute_symmetry=True).df)
    assert df.iloc[0]["computed_elements"] == "Na-O"           # pymatgen's reading
    assert df.iloc[0]["elements"] == "Na-O"                    # what the pre-filter uses


def test_an_unchanged_index_is_not_rewritten(tmp_path):
    _tree(tmp_path / "s")
    out = tmp_path / "index.parquet"
    pytest.importorskip("pyarrow")
    si.build_index(tmp_path / "s", out, compute_symmetry=False)
    before = out.stat().st_mtime_ns
    time.sleep(0.02)
    si.build_index(tmp_path / "s", out, compute_symmetry=False)
    assert out.stat().st_mtime_ns == before
    assert not list(tmp_path.glob("*.tmp"))


def test_the_summary_is_strict_json(tmp_path):
    _write(tmp_path, "hm_only.cif", _cif("h", CUBIC(4.0), [("Cu1", "Cu", 0, 0, 0, 1)],
                                        hm="F m -3 m"))
    json.dumps(si.summarize(si.build_index(tmp_path, compute_symmetry=False)), allow_nan=False)


def test_a_folder_with_other_records_too_is_not_redirected(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda *a, **k: False)
    _collection(tmp_path / "mixed", n=12)
    for i in range(10):
        (tmp_path / "mixed" / f"run_{i}.csv").write_text("x,y\n1,2\n")
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="count", file_name=str(tmp_path / "mixed")))
    assert r["status"] == "error" and "approval declined" in r["message"]


def test_screen_database_redirects_before_asking_consent(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval",
                        lambda *a, **k: pytest.fail("no consent asked: no code would run"))
    _collection(tmp_path / "cifs")
    r = json.loads(planner.tools.execute_tool(
        "screen_database", query="densest", database_path=str(tmp_path / "cifs")))
    assert r["status"] == "index_structures_first"


def test_scanned_cifs_are_not_embedded_but_a_named_one_is(tmp_path, capsys):
    from scilink.parsers.ingestor import ingest_files
    for i in range(30):                        # one CIF per folder: no folder reaches 10
        _write(tmp_path / "tree" / f"m{i:02d}", "s.cif", NACL)
    assert ingest_files([str(tmp_path / "tree")], is_code_mode=False) == []
    assert "30 structure files (CIF/POSCAR/xyz) not embedded" in capsys.readouterr().out
    one = _write(tmp_path, "one.cif", NACL)
    chunks = ingest_files([str(one)], is_code_mode=False)
    assert chunks and "_chemical_formula_sum" in chunks[0]["text"]


# ── cases from real CIF collections ──────────────────────────────────────

def test_one_operation_written_as_a_plain_tag(tmp_path):
    text = NACL.replace("_space_group_IT_number 225\n", "").replace(
        "_symmetry_space_group_name_H-M 'F m -3 m'\n", "_symmetry_equiv_pos_as_xyz x,y,z\n")
    r = _row(_write(tmp_path, "one_op.cif", text))
    assert r["parse_error"] is None and r["n_atoms"] == 2.0          # P1: the two sites as given


def test_a_mixture_type_symbol_keeps_both_species(tmp_path):
    # Pauling File style: one site, type symbol '0.648(4)Mn + 0.352(4)Fe'
    r = _row(_write(tmp_path, "mix.cif", _cif(
        "mix", CUBIC(3.0), [("T", "'0.648(4)Mn + 0.352(4)Fe'", 0, 0, 0, 1)], ops=["x, y, z"])))
    assert r["elements"] == "Fe-Mn" and r["n_atoms"] == pytest.approx(1.0)
    assert r["partial_occupancy"] and not r["overlapping_atoms"]


def test_icsd_dots_are_zero_coordinates(tmp_path):
    r = _row(_write(tmp_path, "icsd.cif", _cif(
        "icsd", CUBIC(4.19), [("Mn1", "Mn2+", ".", ".", ".", 1), ("K1", "K1+", .5, .5, .5, 1),
                              ("F1", "F1-", .5, ".", ".", 1)], sg_number=221, hm="P m -3 m")))
    assert r["parse_error"] is None and r["n_atoms"] == 5.0 and r["reduced_formula"] == "KMnF3"


def test_a_full_monoclinic_symbol_falls_back_to_its_short_form(tmp_path):
    # no operations listed, only 'P 1 21/c 1' (ASE lists 'P 21/c')
    r = _row(_write(tmp_path, "mono.cif", _cif(
        "mono", (5.0, 6.0, 7.0, 90, 98.3, 90), [("O1", "O", 0.1, 0.2, 0.3, 1)], hm="P 1 21/c 1")))
    assert r["parse_error"] is None and r["n_atoms"] == 4.0


def test_a_cif_ase_cannot_read_is_read_by_pymatgen(tmp_path):
    pytest.importorskip("pymatgen.core")
    # the alpha value on the line after its tag: ASE leaves the cell unread
    text = NACL.replace("_cell_angle_alpha 90", "_cell_angle_alpha\n90").replace(
        "_space_group_IT_number 225\n", "")
    r = _row(_write(tmp_path, "split_value.cif", text))
    if r["parse_error"] and r["parse_error"].startswith("sites read by pymatgen"):
        assert r["n_atoms"] == 8.0 and r["elements"] == "Cl-Na"
    else:                                      # ASE read it after all: fine too
        assert r["parse_error"] is None and r["n_atoms"] == 8.0


# ── second review ────────────────────────────────────────────────────────

def test_fraction_coordinates_and_non_numbers():
    assert si._num("1/3") == pytest.approx(1 / 3) and si._num("2/3") == pytest.approx(2 / 3)
    assert si._num("0.35216(1)") == pytest.approx(0.35216) and si._num(".5") == 0.5
    assert si._num("1.2abc") is None and si._num("?") is None


def test_hcp_with_fraction_coordinates(tmp_path):
    r = _row(_write(tmp_path, "zn.cif", _cif(
        "zn", (2.66, 2.66, 4.95, 90, 90, 120), [("Zn1", "Zn", "1/3", "2/3", "1/4", 1)],
        sg_number=194, hm="P 63/m m c")))
    assert r["parse_error"] is None and r["n_atoms"] == 2.0
    assert r["min_distance"] == pytest.approx(2.66, abs=0.05)


def test_a_rhombohedral_symbol_alone_on_rhombohedral_axes(tmp_path):
    # Bi: R-3m on rhombohedral axes (a=b=c, alpha=57.2), symbol only: 2 atoms, not 3x
    r = _row(_write(tmp_path, "bi.cif", _cif(
        "bi", (4.746, 4.746, 4.746, 57.23, 57.23, 57.23), [("Bi1", "Bi", 0.234, 0.234, 0.234, 1)],
        hm="R -3 m")))
    assert r["n_atoms"] == 2.0 and not r["overlapping_atoms"]


def test_origin_choice_two(tmp_path):
    r = _row(_write(tmp_path, "si2.cif", _cif(
        "si2", CUBIC(5.431), [("Si1", "Si", 0.125, 0.125, 0.125, 1)],
        sg_number=227, hm="F d -3 m :2")))
    assert r["n_atoms"] == 8.0 and r["density"] == pytest.approx(2.33, abs=0.01)


def test_a_leading_digit_is_not_a_mixture_and_rounded_mixtures_are_one_site(tmp_path):
    assert si._species("2H", "O1") == [("O", 1.0)]
    r = _row(_write(tmp_path, "mix2.cif", _cif(
        "mix2", CUBIC(3.0), [("T", "'0.65Mn + 0.36Fe'", 0, 0, 0, 1)], ops=["x, y, z"])))
    assert not r["overlapping_atoms"] and r["n_atoms"] == pytest.approx(1.0)


def test_companion_files_do_not_stop_the_redirect(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda *a, **k: False)
    _collection(tmp_path / "ccdc", n=30)
    for i in range(30):                         # .hkl / .fcf beside each CIF
        (tmp_path / "ccdc" / f"batch_{i % 3}" / f"s{i:02d}.hkl").write_text("1 0 0 10.0\n")
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="count", file_name=str(tmp_path / "ccdc")))
    assert r["status"] == "index_structures_first"


# ── PR review (#822) ─────────────────────────────────────────────────────

def test_a_folder_of_calculation_runs_is_not_redirected(planner, tmp_path, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval", lambda *a, **k: False)
    from ase.build import bulk
    import ase.io
    for i in range(20):                       # VASP runs: POSCAR + CONTCAR + OUTCAR each
        run = tmp_path / "runs" / f"run_{i:02d}"
        run.mkdir(parents=True)
        for name in ("POSCAR", "CONTCAR"):
            ase.io.write(run / name, bulk("Si", "diamond", a=5.43), format="vasp")
        (run / "OUTCAR").write_text("free  energy   TOTEN  =  -10.8 eV\n")
    r = json.loads(planner.tools.execute_tool(
        "query_knowledge_data", query="lowest energy", file_name=str(tmp_path / "runs")))
    assert r["status"] == "error" and "approval declined" in r["message"]   # the usual path


def test_a_top_level_index_never_replaces_the_full_one(planner, tmp_path):
    _collection(tmp_path / "cifs")
    _write(tmp_path / "cifs", "top.cif", NACL)
    full = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "cifs")))
    top = json.loads(planner.tools.execute_tool("index_structures", directory=str(tmp_path / "cifs"),
                                                recursive=False))
    assert full["index_file"] != top["index_file"] and top["n_rows"] == 1
    assert len(pd.read_csv(full["index_file"])) == 13


def test_screen_database_by_folder_name_redirects_before_consent(planner, monkeypatch):
    import scilink.executors as ex
    monkeypatch.setattr(ex, "require_sandbox_approval",
                        lambda *a, **k: pytest.fail("no consent asked: no code would run"))
    for i in range(12):                       # a flat folder: discovery lists it by name
        _write(Path(planner.base_dir) / "knowledge" / "cod_subset", f"s{i:02d}.cif", NACL)
    r = json.loads(planner.tools.execute_tool(
        "screen_database", query="densest", database_path="cod_subset"))
    assert r["status"] == "index_structures_first"


def test_an_impossible_space_group_number_keeps_the_row(tmp_path):
    r = _row(_write(tmp_path, "bad_sg.cif", NACL.replace(
        "_space_group_IT_number 225", "_space_group_IT_number 999")))
    assert r["crystal_system"] is None and r["formula"] == "Cl Na" and r["a"] == 5.64


def test_text_columns_survive_the_csv_round_trip(tmp_path):
    _tree(tmp_path / "s")
    _write(tmp_path / "s", "123.cif", NACL.replace("data_nacl", "data_123"))
    out = tmp_path / "index.csv"
    si.build_index(tmp_path / "s", out, compute_symmetry=False)
    _write(tmp_path / "s", "new.cif", SPLIT)
    df = si.build_index(tmp_path / "s", out, compute_symmetry=False).df    # mixes reused + new
    assert (df[df["path"].str.endswith(("/nacl.cif", "/new.cif", "/123.cif"))]["subfolder"] == "").all()
    assert df[df["path"].str.endswith("/123.cif")].iloc[0]["data_block"] == "123"


def test_symmetry_is_skipped_for_huge_cells_unless_asked(tmp_path, monkeypatch):
    pytest.importorskip("pymatgen.core")
    from ase.build import bulk
    import ase.io
    ase.io.write(tmp_path / "small.cif", bulk("Cu", "fcc", a=3.61, cubic=True), format="cif")
    ase.io.write(tmp_path / "big.cif", bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2)),
                 format="cif")
    monkeypatch.setattr(si, "SYMMETRY_MAX_ATOMS", 10)
    auto = {Path(r["path"]).name: r for r in si.build_index(tmp_path).df.to_dict("records")}
    assert auto["small.cif"]["computed_spacegroup_number"] == 225
    assert pd.isna(auto["big.cif"]["computed_spacegroup_number"])
    asked = si.parse_structure_file(tmp_path / "big.cif", tmp_path, compute_symmetry=True)
    assert asked["computed_spacegroup_number"] == 225


def test_the_xrd_view_leaves_out_files_nobody_can_read(tmp_path):
    from scilink.skills.structure_matching._backends import _cif_index
    _write(tmp_path, "nacl.cif", NACL)
    _write(tmp_path, "broken.cif", "data_x\nthis is not a cif\n")
    df = _cif_index._xrd_view(si.build_index(tmp_path, compute_symmetry=True).df)
    assert [Path(p).name for p in df["path"]] == ["nacl.cif"]
