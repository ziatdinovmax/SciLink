"""Axis-range metadata synonyms fold deterministically into canonical form.

Externally-authored hyperspectral metadata (instrument sidecars) often
carries the signal-axis range as ``spectral_axis: {min, max, units}`` (or an
``energy_range`` spelled with min/max) instead of the canonical
``energy_range: {start, end, units}``. Such dicts used to die at the agent's
fail-fast axis precondition ("Axis precondition failed") when passed straight
to ``analyze(system_info=...)``, because the Tier-1 normalizer (a) had no
nested-range aliases and (b) was never invoked on that path.

These assertions are deterministic (no LLM):
- the fold is strictly failure->success: canonical metadata passes through
  byte-identical (the no-op proof), and canonical always wins when both
  shapes are present;
- the observed ``spectral_axis`` sidecar shape resolves end to end
  (``create_axis`` builds the axis after normalization);
- ``HyperspectralAnalysisAgent._handle_system_info`` applies the fold, so
  every metadata consumer on that agent sees the canonical shape.

  conda run -n scilink python -m pytest tests/test_metadata_axis_synonyms.py -q
"""
import copy
import logging

import numpy as np

from scilink.agents.exp_agents.metadata_converter import (
    check_schema_conformance,
    normalize_metadata_dict,
)
from scilink.skills.hyperspectral.eels.eels import create_axis


def _sidecar(**overrides):
    """The literal externally-authored sidecar shape that failed live."""
    meta = {
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "Raman microscopy",
                       "details": "532.0 nm laser excitation"},
        "sample": {"material": "carbon nanowalls on diamond"},
        "spatial_info": {"field_of_view_x": 6.05, "field_of_view_y": 6.9,
                         "units": "mm"},
        "spectral_axis": {"axis_name": "Raman shift", "units": "cm^-1",
                          "min": 175.02, "max": 3412.72},
    }
    meta.update(overrides)
    return meta


# --- the no-op proof: canonical metadata is untouched -----------------------

def test_canonical_metadata_passes_through_identical():
    meta = {
        "experiment_type": "Spectroscopy",
        "experiment": {"technique": "STEM-EELS spectrum image"},
        "sample": {"material": "TiOx"},
        "energy_range": {"start": 450.0, "end": 550.0, "units": "eV"},
    }
    snapshot = copy.deepcopy(meta)
    normalized, was_modified = normalize_metadata_dict(meta)
    assert was_modified is False
    assert normalized is meta          # fast-path returns the same object
    assert meta == snapshot            # and it was not mutated


def test_canonical_wins_when_both_shapes_present():
    meta = _sidecar(energy_range={"start": 100.0, "end": 3500.0,
                                  "units": "cm^-1"})
    normalized, _ = normalize_metadata_dict(meta)
    # spectral_axis must NOT override an already-resolvable energy_range.
    assert normalized["energy_range"]["start"] == 100.0
    assert normalized["energy_range"]["end"] == 3500.0


# --- the folds --------------------------------------------------------------

def test_spectral_axis_min_max_folds_to_energy_range():
    normalized, was_modified = normalize_metadata_dict(_sidecar())
    assert was_modified is True
    er = normalized["energy_range"]
    assert er["start"] == 175.02 and er["end"] == 3412.72
    assert er["units"] == "cm^-1"
    # The source block is preserved (unrecognized keys are kept).
    assert normalized["spectral_axis"]["axis_name"] == "Raman shift"


def test_energy_range_min_max_spelling_folds():
    meta = _sidecar()
    meta.pop("spectral_axis")
    # Prose must describe spectral imaging for conformance to route the dict
    # into the normalizer at all (scope of the fold: hyperspectral metadata).
    meta["experiment"] = {"technique": "Raman hyperspectral mapping"}
    meta["energy_range"] = {"min": 175.02, "max": 3412.72, "units": "cm^-1"}
    normalized, was_modified = normalize_metadata_dict(meta)
    assert was_modified is True
    assert normalized["energy_range"]["start"] == 175.02
    assert normalized["energy_range"]["end"] == 3412.72


def test_half_specified_range_does_not_half_fold():
    meta = _sidecar()
    meta["spectral_axis"] = {"units": "cm^-1", "min": 175.02}  # no max
    normalized, _ = normalize_metadata_dict(meta)
    er = normalized.get("energy_range") or {}
    assert er.get("start") is None and er.get("end") is None


# --- conformance routing ----------------------------------------------------

def test_spectral_axis_block_marks_dict_spectral():
    # Technique prose ("Raman microscopy") misses the spectral-imaging word
    # list, but a declared spectral_axis IS a spectral-axis declaration: the
    # dict must be flagged non-conformant so the normalizer runs (no
    # fast-path around the fold), and be conformant once folded.
    meta = _sidecar()
    conformant, issues = check_schema_conformance(meta)
    assert conformant is False
    assert any("energy axis" in i for i in issues)
    normalized, _ = normalize_metadata_dict(meta)
    assert check_schema_conformance(normalized)[0] is True


# --- end to end: the axis actually builds -----------------------------------

def test_create_axis_succeeds_after_normalization():
    normalized, _ = normalize_metadata_dict(_sidecar())
    axis, xlabel, has_info = create_axis(617, normalized, axis_index=2)
    assert has_info is True
    assert axis.shape == (617,)
    assert np.isclose(axis[0], 175.02) and np.isclose(axis[-1], 3412.72)


# --- the agent seam ---------------------------------------------------------

def test_hyperspectral_handle_system_info_applies_fold():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    # Bare instance without __init__ (no credentials/LLM client needed):
    # _handle_system_info touches only self.logger and super().
    agent = HyperspectralAnalysisAgent.__new__(HyperspectralAnalysisAgent)
    agent.logger = logging.getLogger("test_axis_synonyms")
    handle = agent._handle_system_info
    si = handle(_sidecar())
    assert si["energy_range"]["start"] == 175.02
    assert si["energy_range"]["end"] == 3412.72
    # Canonical input still passes through unchanged.
    canonical = {"experiment_type": "Spectroscopy",
                 "experiment": {"technique": "EELS spectrum image"},
                 "sample": {"material": "TiOx"},
                 "energy_range": {"start": 1.0, "end": 2.0, "units": "eV"}}
    assert handle(copy.deepcopy(canonical)) == canonical


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
