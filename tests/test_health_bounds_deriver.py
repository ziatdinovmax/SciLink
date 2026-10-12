"""Deriver tests for ``HealthBoundsDeriver`` (issue #783).

The one LLM hop (``_generate_json``) is monkeypatched; no API key, no network.
Covers that the deriver turns the model's expected CENTER values into K-slack
bands via the clamp, fails open to ``[]`` on any error, and reuses the
``validation`` skill section (the same one ``ReferencePropertyCritic`` reads).
"""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents import refinement  # noqa: E402
from scilink.agents.sim_agents.health_bounds import HealthBoundsDeriver  # noqa: E402

DENSITY_FM = [{"observable": "density", "min": 0.02, "max": 30.0,
               "units": "g/cm^3"}]


def _deriver():
    return HealthBoundsDeriver(api_key="test-key")


def _with_frontmatter(monkeypatch, specs):
    """Wire a frontmatter health block + a silent skill-section load.

    ``_frontmatter_bands`` reads through ``refinement._health_specs``; the skill
    section load is stubbed so the test needs no real skill bundle.
    """
    monkeypatch.setattr(refinement, "_health_specs",
                        lambda skill, domain: specs)
    d = _deriver()
    d._load_skill_section = lambda skill, domain: ""
    return d


def test_parse_applies_slack_clamp():
    d = _deriver()
    # Model returns CENTERS only; code must widen each into min=e/5, max=e*5,
    # and drop the non-positive one.
    d._generate_json = lambda p: {"bounds": [
        {"observable": "density", "expected": 1.0},
        {"observable": "lattice_constant", "expected": 4.0},
        {"observable": "bad", "expected": 0.0},   # dropped: non-positive
    ]}
    specs = d.derive("equilibrate water", domain="molecular_dynamics")
    by_obs = {s["observable"]: s for s in specs}
    assert set(by_obs) == {"density", "lattice_constant"}
    assert by_obs["density"]["min"] == pytest.approx(0.2)
    assert by_obs["density"]["max"] == pytest.approx(5.0)
    assert by_obs["lattice_constant"]["min"] == pytest.approx(0.8)
    assert by_obs["lattice_constant"]["max"] == pytest.approx(20.0)


def test_derive_failure_returns_empty():
    d = _deriver()

    def boom(prompt):
        raise RuntimeError("llm down")

    d._generate_json = boom
    assert d.derive("goal", domain="molecular_dynamics") == []


def test_uses_validation_section():
    assert HealthBoundsDeriver.SKILL_SECTION == "validation"


def test_requires_domain_when_skill_given():
    d = _deriver()
    d._generate_json = lambda p: {"bounds": []}
    with pytest.raises(ValueError):
        d.derive("goal", skill="lammps")


def test_derived_specs_tagged_source_derived():
    d = _deriver()
    d._generate_json = lambda p: {"bounds": [
        {"observable": "density", "expected": 1.0}]}
    specs = d.derive("equilibrate water", domain="molecular_dynamics")
    assert specs[0]["source"] == "derived"


# ── units-slip guard against the frontmatter fallback (issue #783) ───────────

def test_center_outside_frontmatter_band_is_dropped(monkeypatch):
    # A density center of 997 (kg/m^3 slip) is outside the frontmatter g/cm^3
    # band 0.02–30, so the derived band is DROPPED — the frontmatter fallback
    # stands rather than a 199–4985 band that trips every healthy run.
    d = _with_frontmatter(monkeypatch, DENSITY_FM)
    d._generate_json = lambda p: {"bounds": [
        {"observable": "density", "expected": 997.0}]}
    specs = d.derive("equilibrate water", skill="lammps",
                     domain="molecular_dynamics")
    assert specs == []


def test_center_inside_frontmatter_band_is_kept(monkeypatch):
    d = _with_frontmatter(monkeypatch, DENSITY_FM)
    d._generate_json = lambda p: {"bounds": [
        {"observable": "density", "expected": 1.0}]}
    specs = d.derive("equilibrate water", skill="lammps",
                     domain="molecular_dynamics")
    by_obs = {s["observable"]: s for s in specs}
    assert by_obs["density"]["min"] == pytest.approx(0.2)
    assert by_obs["density"]["max"] == pytest.approx(5.0)


def test_observable_without_frontmatter_band_is_kept(monkeypatch):
    # The model grounds an observable the frontmatter does not declare: there is
    # nothing to sanity-check the center against, so the derived band is kept.
    d = _with_frontmatter(monkeypatch, DENSITY_FM)
    d._generate_json = lambda p: {"bounds": [
        {"observable": "lattice_constant", "expected": 4.0}]}
    specs = d.derive("relax a crystal", skill="lammps",
                     domain="molecular_dynamics")
    by_obs = {s["observable"]: s for s in specs}
    assert "lattice_constant" in by_obs
    assert by_obs["lattice_constant"]["max"] == pytest.approx(20.0)


def test_prompt_includes_units_and_deck(monkeypatch):
    d = _with_frontmatter(monkeypatch, DENSITY_FM)
    captured = {}

    def _capture(prompt):
        captured["p"] = prompt
        return {"bounds": []}

    d._generate_json = _capture
    d.derive("equilibrate water", skill="lammps",
             domain="molecular_dynamics",
             input_files={"in.lmp": "velocity all create 300 42"})
    assert "g/cm^3" in captured["p"]          # frontmatter units surfaced
    assert "in.lmp" in captured["p"]          # the generated deck is shown
    assert "velocity all create 300 42" in captured["p"]


def test_slack_factor_threads_to_band_width(monkeypatch):
    d = _with_frontmatter(monkeypatch, DENSITY_FM)
    d._generate_json = lambda p: {"bounds": [
        {"observable": "density", "expected": 1.0}]}
    specs = d.derive("equilibrate water", skill="lammps",
                     domain="molecular_dynamics", slack_factor=3.0)
    by_obs = {s["observable"]: s for s in specs}
    assert by_obs["density"]["min"] == pytest.approx(1.0 / 3.0)
    assert by_obs["density"]["max"] == pytest.approx(3.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
