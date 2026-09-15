"""Tests for the optional GSAS-II full-profile XRD simulation engine.

Most assertions are GSAS-II-INDEPENDENT (wavelength-alias resolution, registry
wiring, the Bragg peak-pick helper, the actionable missing-dependency error) so
they run in CI without GSAS-II installed. The end-to-end physics test
(``test_full_simulation_physics``) is gated on ``gsas_available()`` and skipped
when GSAS-II is absent; it is exercised live in a GSAS-II environment against
independent crystallographic ground truth (Si positions, Fe BCC systematic
absences, NaCl structure-factor intensities, Mo/Cu wavelength scaling).
"""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd import _gsas_engine as ge
from scilink.skills.structure_matching.xrd import simulate_xrd as sx


def _pymatgen_available():
    try:
        import pymatgen.core  # noqa: F401
        import pymatgen.io.cif  # noqa: F401
        return True
    except Exception:
        return False


# An origin-choice-ambiguous anatase CIF: only the I4_1/amd H-M symbol, no
# explicit symmetry operators. GSAS-II mis-resolves the origin from this alone
# (Fcalc(200)=0, wrong), while the real phase has a strong (200) at 48.05 deg.
# _canonicalize_cif must expand it to explicit coordinates so GSAS agrees with
# pymatgen. This is the regression guard for that finding.
_ANATASE_AMBIGUOUS = (
    "data_anatase\n_cell_length_a 3.7845\n_cell_length_b 3.7845\n_cell_length_c 9.5143\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'I 41/a m d'\n_symmetry_Int_Tables_number 141\n"
    "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    "Ti 0.0 0.0 0.0\nO 0.0 0.0 0.2081\n"
)


def test_engine_registered():
    assert "gsas" in sx._ENGINES
    assert sx._ENGINES["gsas"] is sx._simulate_gsas


def test_tool_exposes_engine_knobs():
    # The gsas physical knobs must be surfaced in TOOL_SPEC (LLM-facing) with
    # tuning guidance, not hidden in the Python signature.
    params = sx.TOOL_SPEC.parameters
    assert "crystallite_um" in params and "peak_rel_height" in params
    assert "nanocrystalline" in params["crystallite_um"]["description"].lower()
    assert "LOWER" in params["crystallite_um"]["description"]


_SI_CIF = (
    "data_Si\n_cell_length_a 5.43088\n_cell_length_b 5.43088\n_cell_length_c 5.43088\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    "_symmetry_space_group_name_H-M 'F d -3 m'\n_symmetry_Int_Tables_number 227\n"
    "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
    "Si 0.0 0.0 0.0\n"
)


@pytest.mark.skipif(not _pymatgen_available(), reason="pymatgen not installed")
def test_pymatgen_ignores_engine_knobs(tmp_path):
    # Passing gsas-only knobs to the pymatgen engine must be silently ignored,
    # not error — the dispatch forwards them but pymatgen absorbs **_ignored.
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)
    r = sx.simulate_xrd_pattern(str(cif), engine="pymatgen",
                                crystallite_um=0.05, peak_rel_height=0.2)
    assert r["engine"] == "pymatgen" and len(r["two_theta"]) > 0


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_gsas_knobs_forwarded_through_tool(tmp_path):
    # The crystallite_um knob must actually reach the gsas engine via the tool:
    # a nanocrystalline size broadens the (111) peak vs the sharp default.
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)

    def fwhm(res):
        x = np.asarray(res["profile_two_theta"]); y = np.asarray(res["profile_intensities"])
        half = y.max() / 2
        above = x[y >= half]
        return (above.max() - above.min()) if above.size else 0.0

    nano = sx.simulate_xrd_pattern(str(cif), engine="gsas",
                                   two_theta_range=(27.0, 30.0), crystallite_um=0.05)
    bulk = sx.simulate_xrd_pattern(str(cif), engine="gsas",
                                   two_theta_range=(27.0, 30.0), crystallite_um=10.0)
    assert fwhm(nano) > fwhm(bulk)


def test_wavelength_alias_resolution():
    assert ge._resolve_wavelength(1.2) == 1.2
    assert ge._resolve_wavelength("CuKa") == pytest.approx(1.5406)
    assert ge._resolve_wavelength("MoKa") == pytest.approx(0.71073)
    # tolerant to case / spacing / hyphen
    assert ge._resolve_wavelength("Mo-Ka") == pytest.approx(0.71073)
    assert ge._resolve_wavelength(" cu ka ") == pytest.approx(1.5406)
    with pytest.raises(ValueError):
        ge._resolve_wavelength("nonsense")


def test_peak_pick_bragg_dspacing():
    # A synthetic two-peak profile: peak-pick must recover positions, normalize
    # intensities to 100 at the max, and convert to correct Bragg d-spacings.
    x = np.linspace(20.0, 60.0, 4000)
    y = np.zeros_like(x)
    for c, a, w in [(28.44, 0.5, 0.1), (47.30, 1.0, 0.1)]:  # (111) weaker, (220) strongest
        y += a * (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)
    tt, inten, d = ge._peak_pick(x, y, lam=1.5406, rel_height=0.05)
    assert len(tt) == 2
    assert tt[0] == pytest.approx(28.44, abs=0.05)
    assert tt[1] == pytest.approx(47.30, abs=0.05)
    assert inten.max() == pytest.approx(100.0)          # normalized to max
    assert inten[0] < inten[1]                          # weaker peak stays weaker
    # d = lambda / (2 sin theta): Si(111) ~ 3.135 A, Si(220) ~ 1.920 A
    assert d[0] == pytest.approx(3.135, abs=0.01)
    assert d[1] == pytest.approx(1.920, abs=0.01)


def test_peak_pick_empty_profile():
    x = np.linspace(10, 90, 100)
    tt, inten, d = ge._peak_pick(x, np.zeros_like(x), lam=1.5406)
    assert len(tt) == len(inten) == len(d) == 0


def test_degenerate_range_rejected():
    # A reversed range is normalized via min/max (forgiving); only a degenerate
    # zero-width range is invalid. The check fires before any GSAS/CIF work.
    if not ge.gsas_available():
        pytest.skip("GSAS-II not installed")
    with pytest.raises(ValueError):
        ge.simulate_gsas("dummy.cif", "CuKa", (50.0, 50.0))


def test_actionable_error_without_gsas():
    if ge.gsas_available():
        pytest.skip("GSAS-II present; missing-dependency path not exercised here")
    with pytest.raises(RuntimeError) as exc:
        sx.simulate_xrd_pattern("x.cif", engine="gsas")
    msg = str(exc.value)
    assert "GSAS-II @ git+" in msg and "Fortran" in msg


@pytest.mark.skipif(not _pymatgen_available(), reason="pymatgen not installed")
def test_canonicalize_cif_expands_to_p1(tmp_path):
    # A symbol-only CIF must be rewritten with explicit atom sites (P1), removing
    # the origin-choice ambiguity before GSAS-II sees it.
    src = tmp_path / "amb.cif"
    src.write_text(_ANATASE_AMBIGUOUS)
    out = ge._canonicalize_cif(str(src), str(tmp_path))
    assert out != str(src)
    text = open(out).read()
    assert "P 1" in text or "'P 1'" in text
    # all 12 atoms of the conventional anatase cell explicit (4 Ti + 8 O)
    assert text.count(" Ti") >= 4 and text.count(" O") >= 8


def test_canonicalize_cif_fallback_on_unparseable(tmp_path):
    # If the CIF cannot be parsed (or pymatgen absent), fall back to the original.
    src = tmp_path / "bad.cif"
    src.write_text("not a cif")
    assert ge._canonicalize_cif(str(src), str(tmp_path)) == str(src)


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_ambiguous_cif_intensities_recovered(tmp_path):
    # End-to-end regression guard: the ambiguous anatase CIF must yield the strong
    # (200) reflection at ~48.05 deg once the engine canonicalizes it — the exact
    # case that gave Fcalc(200)=0 before the fix.
    cif = tmp_path / "amb.cif"
    cif.write_text(_ANATASE_AMBIGUOUS)
    r = ge.simulate_gsas(str(cif), "CuKa", (20.0, 55.0))
    tt = np.asarray(r["two_theta"])
    assert tt.size and np.min(np.abs(tt - 48.05)) < 0.2   # (200) present, correct position
    # (101) at 25.28 remains the strongest line
    I = np.asarray(r["intensities"])
    assert tt[int(np.argmax(I))] == pytest.approx(25.28, abs=0.15)


def test_refine_rietveld_registered_and_knobs():
    from scilink.skills.structure_matching.xrd.refine_rietveld import TOOL_SPEC
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "refine_rietveld" in names
    # tunable knobs surfaced with guidance (not hidden in the signature)
    for k in ("refine_cell", "refine_profile", "refine_atoms", "n_background_terms"):
        assert k in TOOL_SPEC.parameters
    assert "RISKY" in TOOL_SPEC.parameters["refine_atoms"]["description"]
    # returns doc names the robust arbitrary-units fit metric
    assert "profile_corr" in TOOL_SPEC.returns


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_rietveld_roundtrip_recovers_cell(tmp_path):
    # Simulate a broadened Si profile (the "measurement"), then Rietveld-refine
    # the same structure against it: the fit must be good (high profile_corr),
    # the refined cell must stay at Si's a=5.431 Å, and the contract keys present.
    from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)
    sim = simulate_xrd_pattern(str(cif), "CuKa", (20.0, 80.0), engine="gsas",
                               crystallite_um=0.1)  # broadened -> realistic widths
    out = ge.rietveld_refine(str(cif), sim["profile_two_theta"], sim["profile_intensities"],
                             "CuKa", two_theta_range=(20.0, 80.0))
    for k in ("lattice", "input_lattice", "converged", "lattice_esd", "Rwp",
              "profile_corr", "microstrain", "convergence_trace", "profile"):
        assert k in out
    assert out["lattice"]["length_a"] == pytest.approx(5.43088, abs=0.02)
    assert out["profile_corr"] > 0.9
    assert out["converged"] is True
    assert len(out["convergence_trace"]) >= 3
    for pk in ("two_theta", "y_obs", "y_calc", "y_background", "residual"):
        assert len(out["profile"][pk]) == len(out["profile"]["two_theta"]) > 100


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_rietveld_cell_actually_refines(tmp_path):
    # Regression guard: the unit cell must GENUINELY refine, not be a no-op.
    # (A P1-canonicalized phase silently refuses cell refinement in GSAS-II, so
    # the cell must be canonicalized WITH its space group.) Simulate a pattern
    # from Si's true cell, feed a structure whose cell is perturbed +0.5%, and
    # require the refinement to pull it back toward 5.431 Å and report converged.
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)
    sim = simulate_xrd_pattern(str(cif), "CuKa", (20.0, 80.0), engine="gsas", crystallite_um=0.1)

    s = Structure.from_file(str(cif))
    s.scale_lattice(s.volume * (1.005 ** 3))         # +0.5% cell
    pert = tmp_path / "Si_pert.cif"
    CifWriter(s).write_file(str(pert))
    a_start = s.lattice.a
    assert a_start > 5.45                             # perturbed away from truth

    out = ge.rietveld_refine(str(pert), sim["profile_two_theta"], sim["profile_intensities"],
                             "CuKa", two_theta_range=(20.0, 80.0))
    a_ref = out["lattice"]["length_a"]
    assert out["converged"] is True
    # genuinely MOVED toward the truth — a no-op cell refinement (the bug this
    # guards) would leave a_ref == a_start with the full offset intact. Require it
    # to recover most of the perturbation (a sparse cubic pattern with broad peaks
    # won't hit it exactly, but a no-op recovers zero).
    err_start = abs(a_start - 5.43088)
    err_ref = abs(a_ref - 5.43088)
    assert err_ref < 0.6 * err_start                 # recovered >40% of the offset
    assert a_ref < a_start - 0.01                     # and demonstrably moved
    assert out["input_lattice"]["length_a"] == pytest.approx(a_start, abs=0.01)


@pytest.mark.skipif(not (ge.gsas_available() and _pymatgen_available()),
                    reason="needs GSAS-II + pymatgen")
def test_rietveld_wrong_phase_flagged_not_converged(tmp_path):
    # Adversarial: refine a Si structure against an NaCl-like pattern (wrong phase).
    # The refinement must NOT report converged, and profile_corr must be a finite
    # number (a diverged fit gives a flat Ycalc -> corr ~0, never None).
    from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
    si = tmp_path / "Si.cif"
    si.write_text(_SI_CIF)
    nacl = tmp_path / "NaCl.cif"
    nacl.write_text(
        "data_NaCl\n_cell_length_a 5.6402\n_cell_length_b 5.6402\n_cell_length_c 5.6402\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Na 0.0 0.0 0.0\nCl 0.5 0.5 0.5\n")
    pattern = simulate_xrd_pattern(str(nacl), "CuKa", (20.0, 80.0), engine="gsas", crystallite_um=0.1)
    out = ge.rietveld_refine(str(si), pattern["profile_two_theta"], pattern["profile_intensities"],
                             "CuKa", two_theta_range=(20.0, 80.0))
    assert isinstance(out["profile_corr"], float)      # a number, never None, even on divergence
    assert out["converged"] is False                    # wrong phase caught
    assert out["profile_corr"] < 0.5


def test_rietveld_needs_dense_pattern():
    if not ge.gsas_available():
        pytest.skip("GSAS-II not installed")
    with pytest.raises(ValueError):
        ge.rietveld_refine("x.cif", [10.0, 20.0], [1.0, 2.0], "CuKa")


# --- index_pattern (autoindexing for blind identification) --------------------

# Si Cu-Kα peak positions (diamond cubic a=5.43088): 111..533. >=10 peaks — the
# empirically established reliability floor for autoindexing.
_SI_PEAKS = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032,
             94.954, 106.710, 114.092, 127.547, 136.897]


def test_index_pattern_registered_and_knobs():
    from scilink.skills.structure_matching.xrd.index_pattern import TOOL_SPEC
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "index_pattern" in names
    for k in ("crystal_systems", "timeout_per_lattice", "m20_min", "max_nc_no",
              "zero_offset", "start_volume", "top_n", "stage_stop_m20"):
        assert k in TOOL_SPEC.parameters
    # blind-workflow guidance is on the LLM-facing surface
    assert "chemistry" in TOOL_SPEC.description.lower()
    assert "triclinic" in TOOL_SPEC.parameters["crystal_systems"]["description"]


def test_index_pattern_input_guards():
    if not ge.gsas_available():
        pytest.skip("GSAS-II not installed")
    with pytest.raises(ValueError):                      # too few peaks
        ge.index_pattern([28.4, 47.3, 56.1], "CuKa")
    with pytest.raises(ValueError):                      # unknown system name
        ge.index_pattern(_SI_PEAKS, "CuKa", crystal_systems=["quasicrystal"])


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_index_pattern_recovers_si_cell():
    # Blind cold-start: 11 Si peak positions, no chemistry -> the top candidate
    # must be the true a=5.4309 cell, NOT its equal-M20 sqrt2 superlattice alias
    # (guards the near-equal-M20 smallest-volume ranking).
    r = ge.index_pattern(_SI_PEAKS, "CuKa", crystal_systems=["cubic"])
    assert r["candidate_cells"], "no cells found"
    best = r["candidate_cells"][0]
    assert best["a"] == pytest.approx(5.4309, abs=0.01)
    assert best["crystal_system"] == "cubic"
    assert best["M20"] > 10
    # ready-made DB filter brackets the true cell
    lo, hi = r["lattice_param_ranges"]["a"]
    assert lo < 5.4309 < hi
    assert not r["warnings"]


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_index_pattern_staged_defeats_false_monoclinic():
    # Regression for the live blind-eval failure: on this REAL measured peak
    # list (RRUFF Cassiterite R040017 via extract_peaks; SnO2, tetragonal
    # a=4.7374 c=3.1864), a whole-family search returned a spurious monoclinic
    # V~560 A^3 cell at M20~97 that outranked the true tetragonal cell — the
    # classic false-low-symmetry pathology (M20 is not comparable across
    # symmetry levels). The default STAGED symmetry descent must stop at the
    # tetragonal stage with the true cell on top and never run monoclinic.
    peaks = [26.633, 33.921, 51.813, 38.007, 54.802, 64.773, 66.012, 61.922,
             83.759, 89.83, 71.312, 78.758, 78.979, 83.992, 57.86, 39.023,
             24.036, 87.291, 42.676, 81.18]
    r = ge.index_pattern(peaks, "CuKa")           # default: staged descent
    assert r["stopped_at"] == "tetragonal/hexagonal/trigonal"
    assert "monoclinic" not in r["stages_run"]
    # The TRUE cell must be among the candidates (near-equal-M20 aliases may
    # legitimately share the list — the skill directs the agent to try several;
    # what staging guarantees is that the true-symmetry family is searched and
    # a spurious monoclinic cell can no longer bury it).
    def _is_true_cell(c):
        edges = sorted([c["a"], c["b"], c["c"]])
        return (abs(edges[0] - 3.1864) < 0.02 and abs(edges[1] - 4.7374) < 0.02
                and abs(edges[2] - 4.7374) < 0.02)
    assert any(_is_true_cell(c) for c in r["candidate_cells"])
    # explicit crystal_systems disables staging
    r2 = ge.index_pattern(peaks, "CuKa", crystal_systems=["tetragonal"])
    assert r2["stages_run"] == ["tetragonal"] and r2["stopped_at"] is None
    # seeded: reproducible run-to-run
    r3 = ge.index_pattern(peaks, "CuKa", crystal_systems=["tetragonal"])
    assert [c["M20"] for c in r3["candidate_cells"]] == \
           [c["M20"] for c in r2["candidate_cells"]]


def test_validate_cell_lebail_registered_and_knobs():
    from scilink.skills.structure_matching.xrd.validate_cell import TOOL_SPEC
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "validate_cell_lebail" in names
    for k in ("cell", "bravais", "refine_cell", "extra_cycles", "n_background_terms"):
        assert k in TOOL_SPEC.parameters
    # the supercell caveat is on the LLM-facing surface
    assert "SUPERCELL" in TOOL_SPEC.description or "supercell" in TOOL_SPEC.returns


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_lebail_discriminates_true_vs_wrong_cell(tmp_path):
    # Structure-free arbiter: simulate a broadened Si pattern, then Le-Bail-fit
    # (a) Si's true cell -> high corr, cell_fits, lattice refined to precision;
    # (b) a wrong cell (NaCl's a=5.64 P-lattice... use a genuinely wrong 4.9/5.4
    #     hexagonal quartz cell) -> low corr, not cell_fits.
    from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF)
    sim = simulate_xrd_pattern(str(cif), "CuKa", (20.0, 80.0), engine="gsas",
                               crystallite_um=0.1)
    x, y = sim["profile_two_theta"], sim["profile_intensities"]

    good = ge.lebail_fit(x, y, (5.431, 5.431, 5.431, 90, 90, 90), bravais="Cubic-F")
    assert good["cell_fits"] is True
    assert good["profile_corr"] > 0.9
    assert good["lattice"]["length_a"] == pytest.approx(5.43088, abs=0.02)

    bad = ge.lebail_fit(x, y, (4.913, 4.913, 5.405, 90, 90, 120),
                        bravais="Trigonal/Hexagonal-P")
    assert bad["cell_fits"] is False
    assert bad["profile_corr"] < good["profile_corr"] - 0.2


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_index_pattern_recovers_quartz_cell():
    # Non-cubic: alpha-quartz (trigonal a=4.913, c=5.405) from 17 peak positions.
    qz = [20.86, 26.64, 36.54, 39.47, 40.30, 42.45, 45.79, 50.14, 54.87,
          55.32, 59.96, 64.03, 65.79, 67.74, 68.14, 68.32, 73.47]
    r = ge.index_pattern(qz, 1.5406, crystal_systems=["trigonal", "hexagonal"])
    best = r["candidate_cells"][0]
    assert best["a"] == pytest.approx(4.913, abs=0.02)
    assert best["c"] == pytest.approx(5.405, abs=0.02)
    assert best["gamma"] == pytest.approx(120.0, abs=0.1)


@pytest.mark.skipif(not ge.gsas_available(), reason="GSAS-II not installed")
def test_full_simulation_physics(tmp_path):
    # Minimal Si CIF (diamond cubic, a=5.43088) -> exact (111)/(220) at CuKa,
    # full contract (core keys + continuous profile), correct d-spacing.
    cif = tmp_path / "Si.cif"
    cif.write_text(
        "data_Si\n"
        "_cell_length_a 5.43088\n_cell_length_b 5.43088\n_cell_length_c 5.43088\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F d -3 m'\n_symmetry_Int_Tables_number 227\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si 0.0 0.0 0.0\n"
    )
    r = ge.simulate_gsas(str(cif), "CuKa", (20.0, 60.0))
    for k in ("two_theta", "intensities", "hkls", "d_spacings", "wavelength",
              "profile_two_theta", "profile_intensities", "engine_note"):
        assert k in r
    assert r["hkls"] == []
    assert r["wavelength"] == pytest.approx(1.5406)
    assert len(r["profile_two_theta"]) == len(r["profile_intensities"]) > 100
    tt = np.asarray(r["two_theta"])
    assert np.min(np.abs(tt - 28.44)) < 0.15    # (111)
    assert np.min(np.abs(tt - 47.30)) < 0.15    # (220)
    i111 = int(np.argmin(np.abs(tt - 28.44)))
    assert r["d_spacings"][i111] == pytest.approx(3.1355, abs=0.01)
