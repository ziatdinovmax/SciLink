"""Unit tests for the engine-neutral physical-sanity (health) comparator.

Pure: no engine, no I/O, no LLM. Verifies :func:`evaluate_health` and the
frontmatter coercion in :func:`parse_health_specs`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.health import (  # noqa: E402
    HealthBand,
    evaluate_health,
    observable_names,
    parse_health_specs,
)

# A loose density band like lammps.md declares.
DENSITY_SPEC = [{"observable": "density", "min": 0.02, "max": 30.0}]


def test_within_band_is_healthy():
    assert evaluate_health({"density": 0.83}, DENSITY_SPEC) == []


def test_below_min_is_a_violation():
    # The box-explosion case: density collapsed to a near-vacuum.
    violations = evaluate_health({"density": 0.003}, DENSITY_SPEC)
    assert len(violations) == 1
    assert violations[0].observable == "density"
    assert "min" in violations[0].reason


def test_above_max_is_a_violation():
    violations = evaluate_health({"density": 500.0}, DENSITY_SPEC)
    assert len(violations) == 1
    assert "max" in violations[0].reason


def test_none_value_is_skipped_not_failed():
    # An unreadable observable can't be judged — the gate skips it, it is not
    # treated as a violation (no fabricated failures).
    assert evaluate_health({"density": None}, DENSITY_SPEC) == []


def test_missing_observable_is_skipped():
    assert evaluate_health({}, DENSITY_SPEC) == []


def test_non_finite_is_always_a_violation():
    for bad in (float("nan"), float("inf"), float("-inf")):
        violations = evaluate_health({"density": bad}, DENSITY_SPEC)
        assert len(violations) == 1
        assert "non-finite" in violations[0].reason


def test_empty_specs_gate_disabled():
    assert evaluate_health({"density": 0.003}, []) == []


def test_one_sided_band():
    spec = [{"observable": "density", "min": 0.02}]  # no max
    assert evaluate_health({"density": 1e6}, spec) == []
    assert len(evaluate_health({"density": 0.0}, spec)) == 1


def test_multiple_observables():
    spec = [
        {"observable": "density", "min": 0.02, "max": 30.0},
        {"observable": "temperature", "min": 1.0, "max": 1e5},
    ]
    violations = evaluate_health({"density": 0.001, "temperature": 300.0}, spec)
    assert [v.observable for v in violations] == ["density"]


# ── frontmatter coercion ────────────────────────────────────────────────────

def test_parse_accepts_min_max_aliases():
    bands = parse_health_specs([{"observable": "d", "minimum": 1, "maximum": 2}])
    assert bands == [HealthBand("d", 1.0, 2.0)]


def test_parse_skips_malformed():
    specs = [
        {"observable": "density", "min": 0.02, "max": 30.0},  # ok
        {"min": 1, "max": 2},                                  # no observable
        {"observable": "nobounds"},                            # no bounds
        "not-a-dict",                                          # wrong type
    ]
    bands = parse_health_specs(specs)
    assert [b.observable for b in bands] == ["density"]


def test_parse_non_list_is_empty():
    assert parse_health_specs(None) == []
    assert parse_health_specs({"observable": "x"}) == []


def test_observable_names_dedup_preserves_order():
    specs = [
        {"observable": "density", "min": 0},
        {"observable": "temperature", "max": 1e5},
        {"observable": "density", "max": 30},
    ]
    assert observable_names(specs) == ["density", "temperature"]
