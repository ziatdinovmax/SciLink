"""Offline tests for the structure_matching/xrd tools.

Tools are pure Python and fully unit-testable without an LLM or network.
MP-related paths are mocked; local-only paths use a tmp CIF directory.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching._backends import StructureCandidate
from scilink.skills.structure_matching.xrd.search_structures import (
    TOOL_SPEC,
    _candidate_to_dict,
    _dedupe,
    search_structures,
)
from scilink.skills.structure_matching.xrd.simulate_xrd import (
    PYMATGEN_XRD_AVAILABLE,
    TOOL_SPEC as SIM_TOOL_SPEC,
    simulate_xrd_pattern,
)

_skip_no_xrd = pytest.mark.skipif(
    not PYMATGEN_XRD_AVAILABLE,
    reason="pymatgen XRD analysis module not installed; pip install scilink[structure-matching]",
)
from scilink.skills.structure_matching.xrd.score_match_fast import (
    TOOL_SPEC as SCORE_FAST_TOOL_SPEC,
    score_xrd_match_fast,
)


# --- TOOL_SPEC shape ----------------------------------------------------------

def test_tool_spec_renders_prompt_block():
    block = TOOL_SPEC.to_prompt()
    assert "search_structures" in block
    assert "query" in block
    assert "Returns" in block


# --- Fixture helpers ----------------------------------------------------------

def _silicon() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])


def _diamond() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(3.57), ["C"], [[0, 0, 0]])


def _write_cif(path: Path, structure: Structure) -> None:
    path.write_text(structure.to(fmt="cif"))


def _capture_stdout(func, *args, **kwargs):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = saved
    return result, buf.getvalue()


# --- search_structures end-to-end (local only) --------------------------------

def test_search_local_only_returns_matching_candidates(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "c.cif", _diamond())

    out_dir = tmp_path / "candidates"

    monkey_env = {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}
    with patch.dict("os.environ", monkey_env):
        result = search_structures(
            query={"chemistry": ["Si"]},
            sources=["local"],
            output_dir=str(out_dir),
        )

    assert result["sources_queried"] == ["local"]
    assert len(result["candidates"]) == 1
    cand = result["candidates"][0]
    assert cand["formula"] == "Si"
    assert cand["source"] == "local"
    assert Path(cand["structure_path"]).is_file()


def test_search_cell_only_blind_query(tmp_path):
    # Blind identification: NO chemistry — the lattice filter (from
    # index_pattern's recovered cell) is the search key. Si (a=5.43) must be
    # found and diamond (a=3.57) excluded by the cell window alone.
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "c.cif", _diamond())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result = search_structures(
            query={"lattice_param_ranges": {"a": (5.32, 5.54),
                                            "volume": (150.6, 169.9)}},
            sources=["local"],
            output_dir=str(tmp_path / "candidates"),
        )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["formula"] == "Si"


def test_search_cell_only_volume_only(tmp_path):
    # The permutation-invariant volume window alone must also work (the
    # recommended non-cubic form, where a database setting's axes may be a
    # permutation of the indexed cell's).
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "c.cif", _diamond())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result = search_structures(
            query={"lattice_param_ranges": {"volume": (40.0, 50.0)}},  # diamond ~45.5
            sources=["local"],
            output_dir=str(tmp_path / "candidates"),
        )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["formula"] == "C"


def test_search_requires_chemistry_or_cell():
    with pytest.raises(ValueError):
        search_structures(query={}, sources=["local"], output_dir="/tmp/x")
    with pytest.raises(ValueError):
        search_structures(query={"top_n": 5}, sources=["local"], output_dir="/tmp/x")


def test_queryspec_cell_only_validation():
    from scilink.skills.structure_matching._backends._base import QuerySpec
    QuerySpec(chemistry=[], lattice_param_ranges={"a": (5.3, 5.5)})  # cell-only OK
    with pytest.raises(ValueError):
        QuerySpec(chemistry=[])                                      # neither: invalid


def test_chemistry_search_honors_lattice_filter(tmp_path):
    # A chemistry query WITH a lattice filter must apply the filter: Si passes
    # its own cell window and fails a wrong one.
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        hit = search_structures(
            query={"chemistry": ["Si"], "lattice_param_ranges": {"a": (5.3, 5.5)}},
            sources=["local"], output_dir=str(tmp_path / "c1"))
        miss = search_structures(
            query={"chemistry": ["Si"], "lattice_param_ranges": {"a": (3.0, 3.2)}},
            sources=["local"], output_dir=str(tmp_path / "c2"))
    assert len(hit["candidates"]) == 1
    assert len(miss["candidates"]) == 0


def test_cod_query_volume_center():
    # Cell-only ranking helper: explicit volume window wins; else the a/b/c
    # midpoint product; None when edges are incomplete.
    from scilink.skills.structure_matching._backends.cod import _query_volume_center
    assert _query_volume_center({"volume": (150.0, 170.0)}) == pytest.approx(160.0)
    assert _query_volume_center(
        {"a": (5.3, 5.5), "b": (5.3, 5.5), "c": (5.3, 5.5)}
    ) == pytest.approx(5.4 ** 3)
    assert _query_volume_center({"a": (5.3, 5.5)}) is None


def test_search_warns_when_no_backends_available(tmp_path):
    with patch.dict("os.environ", {}, clear=False):
        # Unset env vars that might enable a backend
        with patch.dict("os.environ", {
            "MP_API_KEY": "",
            "MATERIALS_PROJECT_API_KEY": "",
            "SCILINK_LOCAL_CIF_DIR": "",
        }):
            result = search_structures(
                query={"chemistry": ["Si"]},
                sources=["local"],
                output_dir=str(tmp_path / "out"),
            )
    assert result["candidates"] == []
    assert any("not available" in w for w in result["warnings"])


def test_search_unknown_source_is_warned(tmp_path):
    result = search_structures(
        query={"chemistry": ["Si"]},
        sources=["not_a_real_backend"],
        output_dir=str(tmp_path / "out"),
    )
    assert any("Unknown source" in w for w in result["warnings"])


def test_search_query_requires_chemistry(tmp_path):
    with pytest.raises(ValueError, match="chemistry"):
        search_structures(
            query={"top_n": 5},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )


def test_search_multi_chemistry_hypothesis(tmp_path):
    """list[list[str]] dispatches one query per sublist and merges."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "diamond.cif", _diamond())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result = search_structures(
            query={"chemistry": [["Si"], ["C"]], "top_n": 3},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )

    formulas = {c["formula"] for c in result["candidates"]}
    assert "Si" in formulas
    assert "C" in formulas
    assert len(result["candidates"]) == 2  # one per chemistry


def test_search_single_chemistry_unchanged(tmp_path):
    """list[str] (legacy form) still works identically."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "diamond.cif", _diamond())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result = search_structures(
            query={"chemistry": ["Si"]},  # single hypothesis
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )
    formulas = {c["formula"] for c in result["candidates"]}
    assert formulas == {"Si"}


def test_search_multi_chemistry_validates_shape(tmp_path):
    with pytest.raises(ValueError, match="list\\[str\\]"):
        search_structures(
            query={"chemistry": [1, 2, 3]},  # ints, not strings
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )
    with pytest.raises(ValueError, match="list\\[str\\]"):
        search_structures(
            query={"chemistry": [["Si"], "C"]},  # mixed shapes
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )
    with pytest.raises(ValueError, match="list\\[str\\]"):
        search_structures(
            query={"chemistry": [[]]},  # empty sublist
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )


def test_search_multi_chemistry_warns_per_backend_failure(tmp_path):
    """If a backend raises on chemistry=['Si'] but succeeds on ['C'], we get
    one warning referencing 'Si' and candidates from 'C'."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "diamond.cif", _diamond())

    from scilink.skills.structure_matching._backends import (
        LocalCIFBackend, QuerySpec, StructureCandidate,
    )

    original_query = LocalCIFBackend.query

    def selective_raise(self, spec: QuerySpec):
        if spec.chemistry == ["Si"]:
            raise RuntimeError("simulated MP timeout for Si")
        return original_query(self, spec)

    with patch.object(LocalCIFBackend, "query", selective_raise), \
         patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result = search_structures(
            query={"chemistry": [["Si"], ["C"]]},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )

    formulas = {c["formula"] for c in result["candidates"]}
    assert "C" in formulas
    assert any("chemistry=['Si']" in w for w in result["warnings"])


def test_search_emits_db_matches_json(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result, captured = _capture_stdout(
            search_structures,
            query={"chemistry": ["Si"]},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )

    assert "DB_MATCHES_JSON:" in captured
    json_line = [
        line[len("DB_MATCHES_JSON:"):].strip()
        for line in captured.splitlines()
        if line.startswith("DB_MATCHES_JSON:")
    ][0]
    parsed = json.loads(json_line)
    assert parsed["candidates"] == result["candidates"]


# --- Dedup --------------------------------------------------------------------

def test_dedup_prefers_mp_over_local():
    cands = [
        StructureCandidate(id="local_x", source="local", formula="Si", space_group="Fd-3m", rank_score=0.5),
        StructureCandidate(id="mp-149", source="mp", formula="Si", space_group="Fd-3m", rank_score=1.0),
    ]
    result = _dedupe(cands)
    assert len(result) == 1
    assert result[0].source == "mp"


def test_dedup_keeps_distinct_space_groups():
    cands = [
        StructureCandidate(id="mp-1", source="mp", formula="C", space_group="Fd-3m"),
        StructureCandidate(id="mp-2", source="mp", formula="C", space_group="P6_3/mmc"),
    ]
    assert len(_dedupe(cands)) == 2


def test_dedup_keeps_distinct_formulas():
    cands = [
        StructureCandidate(id="mp-1", source="mp", formula="TiO2"),
        StructureCandidate(id="mp-2", source="mp", formula="Ti2O3"),
    ]
    assert len(_dedupe(cands)) == 2


# --- Candidate serialization --------------------------------------------------

def test_candidate_to_dict_strips_private_metadata():
    cand = StructureCandidate(
        id="mp-1", source="mp", formula="Si",
        metadata={"energy_above_hull": 0.0, "_structure": "pretend-pymatgen-obj"},
    )
    d = _candidate_to_dict(cand)
    assert "_structure" not in d["metadata"]
    assert d["metadata"]["energy_above_hull"] == 0.0


# --- MP + local combined (mocked MP) ------------------------------------------

def _mp_record(material_id, formula, sg_symbol, sg_number, e_hull, structure):
    return SimpleNamespace(
        material_id=material_id,
        formula_pretty=formula,
        symmetry=SimpleNamespace(symbol=sg_symbol, number=sg_number),
        energy_above_hull=e_hull,
        structure=structure,
    )


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_search_dedupes_mp_and_local_same_phase(mock_mprester, tmp_path):
    # Local CIF for Si
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si_local.cif", _silicon())

    # Mock MP returning same Si (Fd-3m)
    si_struct = _silicon()
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = [
        _mp_record("mp-149", "Si", "Fd-3m", 227, 0.0, si_struct),
    ]

    with patch.dict("os.environ", {
        "MP_API_KEY": "fake",
        "SCILINK_LOCAL_CIF_DIR": str(cif_dir),
    }):
        result = search_structures(
            query={"chemistry": ["Si"]},
            sources=["mp", "local"],
            output_dir=str(tmp_path / "out"),
        )

    assert result["sources_queried"] == ["mp", "local"]
    assert len(result["candidates"]) == 1
    # Dedup keeps MP (preferred source)
    assert result["candidates"][0]["source"] == "mp"
    assert result["candidates"][0]["id"] == "mp-149"


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_search_truncates_to_top_n(mock_mprester, tmp_path):
    records = [
        _mp_record(f"mp-{i}", "TiO2", "Pbca", 61, 0.01 * i, _silicon())  # synthetic, rank by e_hull
        for i in range(20)
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    # Dedup will collapse same-formula+sg entries, but each "mp-i" has the same
    # formula+sg so dedup collapses to 1. Use distinct sgs to test top_n.
    records = [
        _mp_record(f"mp-{i}", "TiO2", f"sg-{i}", 100 + i, 0.01 * i, _silicon())
        for i in range(20)
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
        result = search_structures(
            query={"chemistry": ["Ti", "O"], "top_n": 5},
            sources=["mp"],
            output_dir=str(tmp_path / "out"),
        )

    assert len(result["candidates"]) == 5
    # Should be the ones with lowest e_hull (best rank_score)
    assert result["candidates"][0]["id"] == "mp-0"
    assert result["candidates"][-1]["id"] == "mp-4"


# --- simulate_xrd_pattern -----------------------------------------------------

def test_simulate_tool_spec_renders():
    block = SIM_TOOL_SPEC.to_prompt()
    assert "simulate_xrd_pattern" in block
    assert "structure_path" in block


@_skip_no_xrd
def test_simulate_returns_silicon_peaks(tmp_path):
    cif = tmp_path / "si.cif"
    _write_cif(cif, _silicon())

    out = simulate_xrd_pattern(str(cif), wavelength="CuKa", two_theta_range=(20, 80))

    # Silicon (Fd-3m, a=5.43 Å) with CuKa has the (111) peak near 28.4°.
    assert len(out["two_theta"]) > 0
    assert len(out["two_theta"]) == len(out["intensities"]) == len(out["hkls"]) == len(out["d_spacings"])
    assert min(out["two_theta"]) >= 20
    assert max(out["two_theta"]) <= 80
    assert any(abs(x - 28.4) < 0.5 for x in out["two_theta"])
    assert all(20 <= x <= 80 for x in out["two_theta"])
    assert max(out["intensities"]) == pytest.approx(100.0, abs=0.5)
    # hkls should be 3-int lists
    assert all(len(h) == 3 and all(isinstance(v, int) for v in h) for h in out["hkls"])


@_skip_no_xrd
def test_simulate_two_theta_range_clips(tmp_path):
    cif = tmp_path / "si.cif"
    _write_cif(cif, _silicon())

    out = simulate_xrd_pattern(str(cif), two_theta_range=(40, 60))
    assert all(40 <= x <= 60 for x in out["two_theta"])


@_skip_no_xrd
def test_simulate_wavelength_shifts_peaks(tmp_path):
    cif = tmp_path / "si.cif"
    _write_cif(cif, _silicon())

    cu = simulate_xrd_pattern(str(cif), wavelength="CuKa")
    mo = simulate_xrd_pattern(str(cif), wavelength="MoKa")
    # MoKa (~0.71 Å) vs CuKa (~1.54 Å): shorter wavelength → smaller 2θ for same plane.
    assert min(mo["two_theta"]) < min(cu["two_theta"])


# --- score_xrd_match_fast (cross-correlation) ---------------------------------

def test_score_fast_tool_spec_renders():
    block = SCORE_FAST_TOOL_SPEC.to_prompt()
    assert "score_xrd_match_fast" in block
    assert "cross-correlation" in block.lower() or "shift" in block.lower()


def _synthetic_pattern(peak_positions, peak_intensities, grid=None, fwhm=0.15, noise=0.0):
    """Build a synthetic experimental pattern by broadening peaks with Lorentzians."""
    if grid is None:
        grid = np.arange(10.0, 90.0, 0.05)
    gamma = fwhm / 2.0
    y = np.zeros_like(grid)
    for x0, amp in zip(peak_positions, peak_intensities):
        y += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    if noise:
        rng = np.random.default_rng(0)
        y = y + rng.normal(scale=noise * max(y), size=y.shape)
    return grid.tolist(), y.tolist()


def test_score_fast_perfect_match_yields_accept():
    peaks_x = [28.4, 47.3, 56.1]
    peaks_y = [100.0, 60.0, 30.0]
    grid, exp = _synthetic_pattern(peaks_x, peaks_y)

    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=peaks_x, sim_intensity=peaks_y,
    )

    assert out["verdict"] == "accept"
    assert out["correlation"] > 0.95
    # No shift/scale needed when sim == exp
    assert abs(out["fitted_shift"]) < 0.1
    assert abs(out["fitted_scale"] - 1.0) < 0.005


def test_score_fast_recovers_known_shift():
    """If exp is offset by Δ from sim, fast tier must recover Δ from the lag."""
    peaks_x = np.array([28.4, 47.3, 56.1])
    peaks_y = [100.0, 60.0, 30.0]
    shift = 0.15  # degrees — modest zero-shift typical of real lab data
    grid, exp = _synthetic_pattern(peaks_x + shift, peaks_y)

    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=peaks_x.tolist(), sim_intensity=peaks_y,
    )
    assert out["verdict"] == "accept"
    assert abs(out["fitted_shift"] - shift) < 0.05
    assert out["correlation"] > 0.9


def test_score_fast_recovers_known_scale():
    """Mild lattice-parameter offset shifts all peaks proportionally — scale grid catches it."""
    base_peaks = np.array([28.4, 47.3, 56.1, 69.1])
    peaks_y = [100.0, 60.0, 30.0, 25.0]
    true_scale = 1.005
    grid, exp = _synthetic_pattern((base_peaks * true_scale).tolist(), peaks_y)

    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=base_peaks.tolist(), sim_intensity=peaks_y,
        scale_search=(0.99, 1.01, 0.001),
    )
    assert out["verdict"] == "accept"
    assert abs(out["fitted_scale"] - true_scale) < 0.003


def test_score_fast_total_mismatch_yields_reject():
    grid, exp = _synthetic_pattern([28.4, 47.3, 56.1], [100, 60, 30])
    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=[70.0, 80.0], sim_intensity=[100.0, 50.0],
    )
    assert out["verdict"] == "reject"
    assert out["correlation"] < 0.6


def test_score_fast_empty_simulation_returns_reject():
    grid, exp = _synthetic_pattern([28.4, 47.3], [100, 60])
    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=[], sim_intensity=[],
    )
    assert out["verdict"] == "reject"
    assert out["correlation"] == 0.0


def test_score_fast_disables_scale_search():
    """Passing scale_search=None forces scale=1.0; fitted_scale must be exactly 1."""
    peaks_x = np.array([28.4, 47.3])
    peaks_y = [100.0, 60.0]
    grid, exp = _synthetic_pattern((peaks_x * 1.005).tolist(), peaks_y)

    out = score_xrd_match_fast(
        exp_two_theta=grid, exp_intensity=exp,
        sim_two_theta=peaks_x.tolist(), sim_intensity=peaks_y,
        scale_search=None,
    )
    assert out["fitted_scale"] == 1.0


def test_score_fast_validates_array_shapes():
    with pytest.raises(ValueError, match="same length"):
        score_xrd_match_fast(
            exp_two_theta=[10.0, 20.0, 30.0] * 10,
            exp_intensity=[1.0, 2.0],
            sim_two_theta=[15.0],
            sim_intensity=[1.0],
        )


def test_score_fast_validates_fwhm():
    grid = np.linspace(10, 50, 100).tolist()
    intensity = (np.zeros(100) + 1).tolist()
    with pytest.raises(ValueError, match="fwhm"):
        score_xrd_match_fast(
            exp_two_theta=grid, exp_intensity=intensity,
            sim_two_theta=[15.0], sim_intensity=[1.0],
            fwhm=0,
        )


def test_score_fast_unknown_background_raises():
    grid = np.linspace(10, 50, 100).tolist()
    intensity = (np.zeros(100) + 1).tolist()
    with pytest.raises(ValueError, match="background"):
        score_xrd_match_fast(
            exp_two_theta=grid, exp_intensity=intensity,
            sim_two_theta=[15.0], sim_intensity=[1.0],
            background="weird",
        )
