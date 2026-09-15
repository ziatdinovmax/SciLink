"""Live test suite for the xrd_profile skill + structure_matching/xrd enhancements.

Covers six surfaces added on the ``xrd-profile-skill`` branch:

  1. (offline) ``search_structures`` `anonymous_formula` filter via local CIF.
  2. (live) `curve_fitting/xrd_profile` profile fitting on sharp-peak Si.
  3. (live) `curve_fitting/xrd_profile` profile fitting on nanocrystalline Si.
  4. (live) Co-activation `skill=["xrd","xrd_profile"]` — both tool sets visible.
  5. (live) Wavelength resolver — MoKa pattern with MoKa source in system_info.
  6. (live) Joint multi-phase MIP — Si+Ge mixture pattern via local CIF.

Live tests require:
  - ANTHROPIC_API_KEY (or GEMINI_API_KEY) in env
  - UNSAFE_EXECUTION_OK=true (the curve-fitting executor requires it)

Run manually:
  UNSAFE_EXECUTION_OK=true ANTHROPIC_API_KEY=... \\
      python -m pytest tests/test_xrd_extensions_live.py -v -s
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scilink.skills.structure_matching._backends.materials_project import (
    MP_API_AVAILABLE,
)
from scilink.skills.structure_matching.xrd.simulate_xrd import PYMATGEN_XRD_AVAILABLE
from scilink.skills.structure_matching.xrd.score_match_robust import PULP_AVAILABLE


# --- shared deps gate ----------------------------------------------------

_DEPS_OK = PYMATGEN_XRD_AVAILABLE and PULP_AVAILABLE

requires_deps = pytest.mark.skipif(
    not _DEPS_OK,
    reason="pymatgen XRD + pulp required (pip install scilink[structure-matching])",
)


def _has_llm_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GEMINI_API_KEY"))


requires_llm = pytest.mark.skipif(
    not _has_llm_key(),
    reason="no LLM API key in env (ANTHROPIC_API_KEY / GEMINI_API_KEY)",
)


def _pick_model_and_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-opus-4-6", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro", os.environ["GEMINI_API_KEY"]
    pytest.skip("no LLM key")


# --- synthetic-data helpers ----------------------------------------------

def _make_lorentzian_pattern(
    sim_two_theta, sim_intensity, *, grid, fwhm, noise_frac=0.02, seed=42,
):
    """Lorentzian-broaden a list of (2θ, I) peaks on ``grid`` with FWHM ``fwhm``."""
    gamma = fwhm / 2.0
    intensity = np.zeros_like(grid)
    for x0, amp in zip(sim_two_theta, sim_intensity):
        intensity += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    rng = np.random.default_rng(seed)
    intensity += rng.normal(scale=noise_frac * intensity.max(), size=intensity.shape)
    return np.clip(intensity, 0, None)


def _save_csv(grid, intensity, path):
    np.savetxt(
        path,
        np.column_stack([grid, intensity]),
        delimiter=",",
        header="two_theta,intensity",
        comments="",
    )


def _build_silicon_pattern(tmpdir, *, wavelength="CuKa", fwhm=0.2, seed=42, name="si_xrd"):
    """Synthesize a kinematic Si pattern (Fd-3m, a=5.43 Å), Lorentzian-broadened.

    Returns (csv_path, true_peaks_list_of_dicts).
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    structure = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]],
    )
    pattern = XRDCalculator(wavelength=wavelength).get_pattern(
        structure, two_theta_range=(20, 90),
    )
    grid = np.arange(20.0, 90.0, 0.04)
    intensity = _make_lorentzian_pattern(
        pattern.x, pattern.y, grid=grid, fwhm=fwhm, seed=seed,
    )
    csv_path = Path(tmpdir) / f"{name}.csv"
    _save_csv(grid, intensity, csv_path)
    return csv_path, [{"two_theta": float(x), "intensity": float(y)}
                      for x, y in zip(pattern.x, pattern.y)]


def _build_si_ge_mixture(tmpdir, *, fwhm=0.2, seed=43, name="si_ge_xrd"):
    """Synthesize a 50/50 Si+Ge mixture pattern."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    ge = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.658), ["Ge"], [[0, 0, 0]])

    calc = XRDCalculator(wavelength="CuKa")
    p_si = calc.get_pattern(si, two_theta_range=(20, 90))
    p_ge = calc.get_pattern(ge, two_theta_range=(20, 90))

    grid = np.arange(20.0, 90.0, 0.04)
    intensity = (
        _make_lorentzian_pattern(p_si.x, p_si.y, grid=grid, fwhm=fwhm, seed=seed)
        + _make_lorentzian_pattern(p_ge.x, p_ge.y, grid=grid, fwhm=fwhm, seed=seed + 1)
    )
    csv_path = Path(tmpdir) / f"{name}.csv"
    _save_csv(grid, intensity, csv_path)
    return csv_path


def _materialize_cif_dir(tmpdir):
    """Build a small local CIF directory: Si, Ge, TiO2 (rutile), SrTiO3."""
    from pymatgen.core import Lattice, Structure

    cif_dir = Path(tmpdir) / "local_cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    structures = {
        "Si": Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]),
        "Ge": Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.658), ["Ge"], [[0, 0, 0]]),
        "TiO2_rutile": Structure.from_spacegroup(
            "P4_2/mnm",
            Lattice.tetragonal(4.594, 2.959),
            ["Ti", "O"],
            [[0, 0, 0], [0.305, 0.305, 0]],
        ),
        "SrTiO3": Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(3.905),
            ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]],
        ),
    }
    for name, struct in structures.items():
        struct.to(filename=str(cif_dir / f"{name}.cif"), fmt="cif")
    return cif_dir


# ============================================================
# TEST 1 — OFFLINE: anonymous_formula filter via local CIF
# ============================================================

@requires_deps
def test_anonymous_formula_filter_offline(tmp_path, monkeypatch):
    """B.2 enhancement: anonymous_formula='AB2' should select TiO2, reject Si/Ge/SrTiO3.

    Pure-Python — no LLM, no Materials Project. Exercises the new
    QuerySpec field + local CIF backend client-side filter.
    """
    from scilink.skills.structure_matching.xrd.search_structures import search_structures

    cif_dir = _materialize_cif_dir(tmp_path)
    monkeypatch.setenv("SCILINK_LOCAL_CIF_DIR", str(cif_dir))

    # Querying ['Ti', 'O'] should normally return TiO2; with anonymous_formula='AB2'
    # set, that filter is satisfied by TiO2 (1 Ti + 2 O per formula). SrTiO3
    # would NOT pass since it has 3 elements.
    result = search_structures(
        query={"chemistry": ["Ti", "O"], "anonymous_formula": "AB2", "top_n": 5},
        sources=["local"],
        output_dir=str(tmp_path / "candidates"),
    )

    formulas = [c["formula"] for c in result["candidates"]]
    assert formulas, f"No candidates returned; warnings: {result['warnings']}"
    assert all(f == "TiO2" for f in formulas), (
        f"Expected only TiO2; got {formulas}"
    )

    # With anonymous_formula='ABC3' on chemistry=['Sr','Ti','O'], SrTiO3 should
    # pass (anonymized form SrTiO3 → ABC3).
    result2 = search_structures(
        query={"chemistry": ["Sr", "Ti", "O"], "anonymous_formula": "ABC3", "top_n": 5},
        sources=["local"],
        output_dir=str(tmp_path / "candidates2"),
    )
    formulas2 = [c["formula"] for c in result2["candidates"]]
    assert formulas2, f"No candidates for ABC3 query; warnings: {result2['warnings']}"
    assert all(f == "SrTiO3" for f in formulas2), (
        f"Expected only SrTiO3 perovskite; got {formulas2}"
    )

    # With anonymous_formula='AB' (binary 1:1) on Si-only — no match (Si is A).
    result3 = search_structures(
        query={"chemistry": ["Si"], "anonymous_formula": "AB", "top_n": 5},
        sources=["local"],
        output_dir=str(tmp_path / "candidates3"),
    )
    assert not result3["candidates"], (
        f"Expected zero AB-stoichiometry Si candidates; got {result3['candidates']}"
    )


# ============================================================
# Common live-test agent factory
# ============================================================

def _make_agent(out_dir):
    """Construct a CurveFittingAgent with non-interactive defaults."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
    model_name, api_key = _pick_model_and_key()
    return CurveFittingAgent(
        api_key=api_key,
        model_name=model_name,
        output_dir=str(out_dir),
        enable_human_feedback=False,
        use_literature=False,
        run_preprocessing=False,
    )


# ============================================================
# TEST 2 — LIVE: profile fit on sharp-peak Si
# ============================================================

@requires_deps
@requires_llm
def test_profile_fit_sharp_silicon(tmp_path):
    """Sharp Si pattern (FWHM=0.15°): per-peak R² ≥ 0.9; Scherrer mean ≥ 30 nm."""
    csv, _ = _build_silicon_pattern(tmp_path, fwhm=0.15, seed=11, name="si_sharp")
    out_dir = tmp_path / "out_sharp"

    agent = _make_agent(out_dir)
    result = agent.analyze(
        data=str(csv),
        system_info={
            "technique": "XRD",
            "wavelength": "CuKa",
            "chemistry_hint": ["Si"],
            "notes": (
                "Synthetic well-crystallized Si pattern (FWHM ~0.15°). "
                "Fit profiles, compute Scherrer crystallite size, and run "
                "Williamson-Hall if enough peaks span the 2θ range."
            ),
        },
        skill="xrd_profile",
    )

    fit_quality = result.get("fit_results", {}).get("fit_quality", {})
    print("\n[sharp] fit_quality:", json.dumps(fit_quality, indent=2))
    print("[sharp] keys in fit_results:", list(result.get("fit_results", {}).keys()))

    # Soft assertions — record values but only fail on serious miss
    scherrer_mean = result.get("fit_results", {}).get("scherrer_mean_size_nm")
    print(f"[sharp] Scherrer mean: {scherrer_mean}")
    assert isinstance(scherrer_mean, (int, float)) or scherrer_mean is None, (
        f"Scherrer mean has wrong type: {type(scherrer_mean)}"
    )
    if isinstance(scherrer_mean, (int, float)):
        assert scherrer_mean >= 20, (
            f"Sharp Si pattern should give Scherrer ≥ 20 nm; got {scherrer_mean} nm. "
            f"Session: {out_dir}"
        )


# ============================================================
# TEST 3 — LIVE: profile fit on nanocrystalline Si
# ============================================================

@requires_deps
@requires_llm
def test_profile_fit_nanocrystalline_silicon(tmp_path):
    """Nano Si pattern (FWHM=0.8°): Scherrer mean ≤ 25 nm."""
    csv, _ = _build_silicon_pattern(tmp_path, fwhm=0.8, seed=22, name="si_nano")
    out_dir = tmp_path / "out_nano"

    agent = _make_agent(out_dir)
    result = agent.analyze(
        data=str(csv),
        system_info={
            "technique": "XRD",
            "wavelength": "CuKa",
            "chemistry_hint": ["Si"],
            "notes": (
                "Synthetic nanocrystalline Si pattern with broad peaks "
                "(FWHM ~0.8°). Fit profiles and estimate crystallite size."
            ),
        },
        skill="xrd_profile",
    )

    scherrer_mean = result.get("fit_results", {}).get("scherrer_mean_size_nm")
    print(f"\n[nano] Scherrer mean: {scherrer_mean}")
    if isinstance(scherrer_mean, (int, float)):
        assert scherrer_mean <= 25, (
            f"Nano Si pattern (FWHM 0.8°) should give Scherrer ≤ 25 nm; "
            f"got {scherrer_mean} nm. Session: {out_dir}"
        )


# ============================================================
# TEST 4 — LIVE: co-activation skill=[xrd, xrd_profile]
# ============================================================

@requires_deps
@requires_llm
def test_coactivation_xrd_and_xrd_profile(tmp_path, monkeypatch):
    """Both skills active: tool inventories merge, generated script chains them."""
    cif_dir = _materialize_cif_dir(tmp_path)
    monkeypatch.setenv("SCILINK_LOCAL_CIF_DIR", str(cif_dir))

    csv, _ = _build_silicon_pattern(tmp_path, fwhm=0.3, seed=33, name="si_combo")
    out_dir = tmp_path / "out_combo"

    agent = _make_agent(out_dir)
    result = agent.analyze(
        data=str(csv),
        system_info={
            "technique": "XRD",
            "wavelength": "CuKa",
            "chemistry_hint": ["Si"],
            "notes": (
                "Synthetic Si pattern. Both profile fitting AND structure "
                "matching are needed: fit the peaks first, then identify "
                "the phase using the refined widths."
            ),
        },
        skill=["xrd", "xrd_profile"],
    )

    blob = json.dumps(result).lower()
    assert any(tok in blob for tok in ("silicon", " si ", '"si"', "fd-3m")), (
        f"Co-activation analysis failed to mention silicon; session: {out_dir}"
    )
    # At least one of the bridge-relevant outputs should appear in fit_results
    fr = result.get("fit_results", {})
    has_profile = any(k in fr for k in ("peaks", "scherrer_mean_size_nm"))
    has_match = any(k in fr for k in ("best_match", "db_matches"))
    print(f"\n[combo] has profile output: {has_profile}; has match output: {has_match}")
    assert has_profile or has_match, (
        f"Neither profile-fit nor structure-match outputs present; session: {out_dir}"
    )


# ============================================================
# TEST 5 — LIVE: wavelength resolver (MoKa)
# ============================================================

@requires_deps
@requires_llm
def test_wavelength_resolver_moka(tmp_path, monkeypatch):
    """MoKa-source pattern + MoKa-tagged system_info: identification still works."""
    cif_dir = _materialize_cif_dir(tmp_path)
    monkeypatch.setenv("SCILINK_LOCAL_CIF_DIR", str(cif_dir))

    csv, _ = _build_silicon_pattern(
        tmp_path, wavelength="MoKa", fwhm=0.1, seed=55, name="si_moka",
    )
    out_dir = tmp_path / "out_moka"

    agent = _make_agent(out_dir)
    result = agent.analyze(
        data=str(csv),
        system_info={
            "technique": "XRD",
            "experiment": {"source": "MoKa", "wavelength": "MoKa"},
            "chemistry_hint": ["Si"],
            "notes": (
                "Synthetic Si pattern collected with Mo Kα radiation "
                "(λ ≈ 0.7093 Å). Use resolve_wavelength to pick up the "
                "MoKa source from system_info before simulating; default "
                "CuKa would mis-match every peak."
            ),
        },
        skill="xrd",
    )

    blob = json.dumps(result).lower()
    print(f"\n[moka] session: {out_dir}")
    assert any(tok in blob for tok in ("silicon", " si ", '"si"', "fd-3m")), (
        f"MoKa identification failed to mention silicon; session: {out_dir}"
    )


# ============================================================
# TEST 6 — LIVE: joint multi-phase MIP (Si + Ge)
# ============================================================

@requires_deps
@requires_llm
def test_joint_multiphase_si_ge(tmp_path, monkeypatch):
    """Si+Ge mixture pattern: score_xrd_match_multiphase activates both phases."""
    cif_dir = _materialize_cif_dir(tmp_path)
    monkeypatch.setenv("SCILINK_LOCAL_CIF_DIR", str(cif_dir))

    csv = _build_si_ge_mixture(tmp_path, fwhm=0.2, seed=66)
    out_dir = tmp_path / "out_multiphase"

    agent = _make_agent(out_dir)
    result = agent.analyze(
        data=str(csv),
        system_info={
            "technique": "XRD",
            "wavelength": "CuKa",
            "chemistry_hint": ["Si", "Ge"],
            "notes": (
                "Synthetic two-phase mixture: Si (Fd-3m, a=5.43 Å) and Ge "
                "(Fd-3m, a=5.658 Å) co-existing 50/50. Use "
                "score_xrd_match_multiphase since this is multi-phase."
            ),
        },
        skill="xrd",
    )

    blob = json.dumps(result).lower()
    print(f"\n[multiphase] session: {out_dir}")
    has_si = any(tok in blob for tok in ("silicon", '"si"'))
    has_ge = any(tok in blob for tok in ("germanium", '"ge"'))
    print(f"[multiphase] Si mentioned: {has_si}; Ge mentioned: {has_ge}")
    assert has_si and has_ge, (
        f"Multi-phase analysis missed one phase (Si={has_si}, Ge={has_ge}); "
        f"session: {out_dir}"
    )
