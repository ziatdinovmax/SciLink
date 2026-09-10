"""AvailableSoftware.auto() self-heals a stale cache.

A cached YAML was previously trusted forever, so a package installed AFTER
the cache was built stayed invisible until a manual refresh — the live
fairchem-core / UMA bug: `pip install fairchem-core` succeeded, `import
fairchem.core` worked, yet the agent kept reporting UMA unavailable because
the cache (built pre-install) said `available: false`.

auto() now re-probes the not-yet-available entries on load and persists any
that flipped, while leaving available / user_confirmed entries untouched.
"""

import textwrap
from pathlib import Path

import pytest

from scilink.utils import available_software as asw
from scilink.utils.available_software import AvailableSoftware


@pytest.fixture
def yaml_path(tmp_path):
    return tmp_path / "available_software.yaml"


def _write(path: Path, body: str):
    path.write_text(textwrap.dedent(body).lstrip())


def test_auto_reprobes_and_heals_a_stale_unavailable_entry(yaml_path, monkeypatch):
    # Cache built before the package was installed.
    _write(yaml_path, """
        machine_learning_potentials:
          uma:
            available: false
            reason: 'no match: python modules [fairchem.core]'
    """)

    # Simulate the package now being importable: the frontmatter probe reports
    # available. Patch the probe so the test is independent of what's actually
    # installed in the test environment.
    def fake_probe(domain, engine):
        if (domain, engine) == ("machine_learning_potentials", "uma"):
            return {"available": True, "python_module": "fairchem.core",
                    "source": "importlib.util.find_spec"}
        return {"available": False, "source": "test-stub"}

    monkeypatch.setattr(asw, "_probe_from_frontmatter", fake_probe)

    cfg = AvailableSoftware.auto(yaml_path)
    assert cfg.has("machine_learning_potentials", "uma")           # healed in memory
    # ...and persisted, so the next process sees it without re-probing
    assert AvailableSoftware.load(yaml_path).has(
        "machine_learning_potentials", "uma")


def test_auto_does_not_reprobe_available_or_user_confirmed(yaml_path, monkeypatch):
    # chgnet already available; deepmd manually forced true by the user.
    _write(yaml_path, """
        machine_learning_potentials:
          chgnet:
            available: true
            python_module: chgnet
          deepmd:
            available: true
            user_confirmed: true
    """)

    calls = []

    def spy_probe(domain, engine):
        calls.append((domain, engine))
        # If it WERE re-probed, it'd now report unavailable — proving the
        # trusted entries would be clobbered if we didn't skip them.
        return {"available": False, "source": "test-stub"}

    monkeypatch.setattr(asw, "_probe_from_frontmatter", spy_probe)

    cfg = AvailableSoftware.auto(yaml_path)
    # neither trusted entry was re-probed ...
    assert ("machine_learning_potentials", "chgnet") not in calls
    assert ("machine_learning_potentials", "deepmd") not in calls
    # ... and both remain available
    assert cfg.has("machine_learning_potentials", "chgnet")
    assert cfg.has("machine_learning_potentials", "deepmd")


def test_reprobe_leaves_still_missing_entries_false(yaml_path, monkeypatch):
    _write(yaml_path, """
        machine_learning_potentials:
          orb:
            available: false
    """)

    # Everything still missing → probe keeps reporting unavailable.
    monkeypatch.setattr(asw, "_probe_from_frontmatter",
                        lambda d, e: {"available": False, "source": "test-stub"})

    cfg = AvailableSoftware.auto(yaml_path)
    assert not cfg.has("machine_learning_potentials", "orb")
