"""Pure unit tests for the system-aware health-band machinery (issue #783).

No LLM, no API key. Two pieces are exercised:

* the CLAMP (``health_bounds``) — the model supplies only the expected CENTER,
  CODE owns the K-slack WIDTH, and an explicit band is only ever WIDENED. This is
  the accuracy-creep guard: the model is structurally incapable of tightening the
  gate into an accuracy check.
* the MERGE (``refinement._merge_health_bounds``) — a per-observable override of
  the frontmatter fallback by the derived bands.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.health import evaluate_health  # noqa: E402
from scilink.agents.sim_agents.health_bounds import (  # noqa: E402
    _SLACK_K,
    clamp_center_to_band,
)
from scilink.agents.sim_agents.refinement import _merge_health_bounds  # noqa: E402


# ── the clamp: center -> K-slack band ────────────────────────────────────────

def test_center_becomes_K_slack_band():
    assert _SLACK_K == 5.0
    spec = clamp_center_to_band("density", 1.0)
    assert spec["observable"] == "density"
    assert spec["min"] == pytest.approx(0.2)
    assert spec["max"] == pytest.approx(5.0)


def test_center_band_is_exact_ratio_even_for_tiny_values():
    # Regression: the center path must not route through sqrt(e*e), which
    # underflows to 0.0 for e <~ 1e-154 and collapses the band to [0.0, e].
    e = 1e-200
    spec = clamp_center_to_band("x", e)
    assert spec is not None
    assert spec["min"] > 0.0
    assert spec["min"] == pytest.approx(e / 5.0)
    assert spec["max"] == pytest.approx(e * 5.0)


def test_slack_factor_sets_band_width():
    # The caller-level health_slack_factor owns the WIDTH; the model only ever
    # supplies the center. K=10 -> [e/10, e*10].
    spec = clamp_center_to_band("density", 1.0, slack_factor=10.0)
    assert spec["min"] == pytest.approx(0.1)
    assert spec["max"] == pytest.approx(10.0)


def test_slack_factor_must_exceed_one():
    # A degenerate K <= 1 (or non-finite) would not widen the center into a
    # band — drop it so the frontmatter fallback stands.
    assert clamp_center_to_band("density", 1.0, slack_factor=1.0) is None
    assert clamp_center_to_band("density", 1.0, slack_factor=0.0) is None
    assert clamp_center_to_band("density", 1.0, slack_factor=float("inf")) is None


def test_nonpositive_or_nonfinite_center_is_dropped():
    assert clamp_center_to_band("density", 0) is None
    assert clamp_center_to_band("density", -1.0) is None
    assert clamp_center_to_band("density", float("nan")) is None
    assert clamp_center_to_band("density", float("inf")) is None
    assert clamp_center_to_band("density", "not a number") is None
    assert clamp_center_to_band("density", None) is None
    assert clamp_center_to_band("", 1.0) is None


# ── the merge: derived over frontmatter ──────────────────────────────────────

FM = [{"observable": "density", "min": 0.02, "max": 30.0}]


def test_merge_derived_overrides_frontmatter_same_observable():
    derived = [{"observable": "density", "min": 0.2, "max": 5.0}]
    merged = _merge_health_bounds(FM, derived)
    by_obs = {d["observable"]: d for d in merged}
    assert by_obs["density"]["min"] == 0.2 and by_obs["density"]["max"] == 5.0
    assert len(merged) == 1


def test_merge_keeps_frontmatter_observable_with_no_derived():
    fm = [{"observable": "density", "min": 0.02, "max": 30.0},
          {"observable": "pressure", "min": -5000, "max": 5000}]
    derived = [{"observable": "density", "min": 0.2, "max": 5.0}]
    merged = _merge_health_bounds(fm, derived)
    by_obs = {d["observable"]: d for d in merged}
    assert by_obs["pressure"]["min"] == -5000     # untouched frontmatter kept
    assert by_obs["density"]["min"] == 0.2


def test_merge_adds_derived_only_observable():
    derived = [{"observable": "lattice_constant", "min": 1.0, "max": 25.0}]
    merged = _merge_health_bounds(FM, derived)
    by_obs = {d["observable"]: d for d in merged}
    assert "density" in by_obs and "lattice_constant" in by_obs
    assert by_obs["lattice_constant"]["max"] == 25.0


def test_merge_tags_provenance_source():
    # The merge is the one place that knows which list is frontmatter and which
    # is derived — it tags each surviving spec so a later violation can say
    # whether the BAND or the RUN is suspect.
    fm = [{"observable": "density", "min": 0.02, "max": 30.0},
          {"observable": "pressure", "min": -5000, "max": 5000}]
    derived = [{"observable": "density", "min": 0.2, "max": 5.0}]
    merged = _merge_health_bounds(fm, derived)
    by_obs = {d["observable"]: d for d in merged}
    assert by_obs["density"]["source"] == "derived"      # overridden by derived
    assert by_obs["pressure"]["source"] == "frontmatter"  # untouched fallback


def test_merge_none_is_identity():
    assert _merge_health_bounds(FM, None) is FM
    assert _merge_health_bounds(FM, []) is FM


def test_malformed_derived_dropped_not_raised():
    derived = ["notadict", {"no": "observable"}, {"observable": ""},
               {"observable": "density", "min": 0.2, "max": 5.0}]
    merged = _merge_health_bounds(FM, derived)
    by_obs = {d["observable"]: d for d in merged if isinstance(d, dict)}
    assert by_obs["density"]["min"] == 0.2
    # only the one well-formed derived spec overrode; nothing malformed landed
    assert set(by_obs) == {"density"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
