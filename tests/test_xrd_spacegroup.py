"""Offline tests for systematic-absence analysis (determine_space_group).

Ground truth is textbook crystallography: Si (Fd-3m) peaks must be consistent
with F-centering and violate I/R; real measured cassiterite (P4_2/mnm) peaks
must violate every centering (forcing P) and rank the true group first, while
quarantining the pattern's genuine impurity line as unassigned."""

from __future__ import annotations

import pytest

from scilink.skills.structure_matching.xrd.determine_space_group import (
    determine_space_group, TOOL_SPEC)

SI_PEAKS = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032, 94.954]
SI_CELL = [5.43088, 5.43088, 5.43088, 90, 90, 90]
# real measured cassiterite positions (RRUFF R040017 via extract_peaks);
# 24.036 is a genuine non-SnO2 line in that pattern
SN_PEAKS = [24.036, 26.633, 33.921, 38.007, 39.023, 42.676, 51.813, 54.802,
            57.86, 61.922, 64.773, 66.012, 71.312, 78.758, 83.759]
SN_CELL = {"a": 4.7374, "b": 4.7374, "c": 3.1864,
           "alpha": 90, "beta": 90, "gamma": 90}


def test_registered_and_knobs():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "determine_space_group" in names
    for k in ("tol_deg", "min_evidence", "crystal_system"):
        assert k in TOOL_SPEC.parameters
    assert "not exhaustive" in TOOL_SPEC.returns.lower() or \
           "NOT exhaustive" in TOOL_SPEC.returns


def test_si_fcc_extinctions():
    r = determine_space_group(SI_PEAKS, SI_CELL, "cubic")
    assert r["centering"]["F"]["verdict"] == "consistent"
    assert r["centering"]["I"]["verdict"] == "violated"
    sgs = [c["space_group"] for c in r["consistent_space_groups"]]
    assert "Fd-3m" in sgs                      # the truth survives
    # F-groups precede P-groups (frequency order preserved within evidence rank)
    assert sgs.index("Fm-3m") < sgs.index("Pm-3m")


def test_cassiterite_primitive_and_true_group_first():
    r = determine_space_group(SN_PEAKS, SN_CELL, "tetragonal")
    # every centering violated -> primitive lattice forced
    assert all(v["verdict"] == "violated" for v in r["centering"].values())
    sgs = [c["space_group"] for c in r["consistent_space_groups"]]
    assert sgs and sgs[0] == "P4_2/mnm"        # the true group ranks first
    # the pattern's genuine impurity line is quarantined, not force-assigned
    assert 24.036 in r["unassigned_peaks"]
    assert r["warnings"]


def test_guards():
    with pytest.raises(ValueError):
        determine_space_group(SI_PEAKS, SI_CELL, "quasiperiodic")
    with pytest.raises(ValueError):
        determine_space_group([28.4, 47.3], SI_CELL, "cubic")   # too few peaks
    # trigonal/hexagonal aliasing accepted
    r = determine_space_group(SI_PEAKS, SI_CELL, "hexagonal")
    assert "consistent_space_groups" in r