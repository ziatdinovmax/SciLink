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

from scilink.agents.sim_agents.health_bounds import HealthBoundsDeriver  # noqa: E402


def _deriver():
    return HealthBoundsDeriver(api_key="test-key")


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
