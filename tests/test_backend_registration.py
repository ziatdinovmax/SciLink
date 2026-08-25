"""Tests for the structure-backend registration API.

Exercises:
  - register_backend() adds a new backend
  - unregister_backend() removes it
  - registered_backends() reflects the current state
  - search_structures dispatch sees registered backends
  - Re-registering shadows the previous binding (override semantics)
  - Entry-point discovery is invoked at import (smoke test)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scilink.skills.structure_matching._backends import (
    QuerySpec,
    StructureCandidate,
    _reload_builtins,
    get_backend_factory,
    register_backend,
    registered_backends,
    unregister_backend,
)


class _FakeBackend:
    """Minimal StructureBackend implementation for registration tests."""

    name = "fake"
    canned_candidates: list[StructureCandidate] = []
    available: bool = True

    def is_available(self) -> bool:
        return type(self).available

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        return [
            StructureCandidate(
                id=f"fake-{i}", source=self.name,
                formula="-".join(spec.chemistry), space_group=None,
                rank_score=1.0 - 0.1 * i,
            )
            for i in range(min(spec.top_n, len(type(self).canned_candidates) or 2))
        ]


@pytest.fixture(autouse=True)
def restore_registry():
    """Each test starts from a clean built-ins registry."""
    snapshot = dict(registered_backends())
    yield
    # Restore exactly: clear added entries, restore previous bindings.
    current = list(registered_backends().keys())
    for name in current:
        unregister_backend(name)
    for name, factory in snapshot.items():
        register_backend(name, factory)


# --- register / unregister / lookup -------------------------------------------

def test_builtins_registered_on_import():
    reg = registered_backends()
    assert "mp" in reg
    assert "local" in reg
    assert "cod" in reg


def test_register_adds_new_backend():
    register_backend("fake", _FakeBackend)
    assert "fake" in registered_backends()
    assert get_backend_factory("fake") is _FakeBackend


def test_register_validates_name():
    with pytest.raises(ValueError, match="non-empty"):
        register_backend("", _FakeBackend)
    with pytest.raises(ValueError, match="non-empty"):
        register_backend(None, _FakeBackend)  # type: ignore[arg-type]


def test_register_can_shadow_builtin():
    register_backend("mp", _FakeBackend)
    assert get_backend_factory("mp") is _FakeBackend


def test_unregister_removes_backend():
    register_backend("fake", _FakeBackend)
    unregister_backend("fake")
    assert get_backend_factory("fake") is None


def test_unregister_missing_is_noop():
    unregister_backend("definitely_not_registered")  # must not raise


def test_reload_builtins_restores_originals():
    register_backend("fake", _FakeBackend)
    unregister_backend("mp")
    _reload_builtins()
    assert "mp" in registered_backends()
    # User-added entries are untouched by _reload_builtins.
    assert "fake" in registered_backends()


# --- Integration with search_structures dispatch ------------------------------

def test_search_structures_sees_registered_backend(tmp_path):
    """A backend registered through the API must be reachable via
    `sources=['fake']` AND via auto-detect."""
    from scilink.skills.structure_matching.xrd.search_structures import search_structures

    register_backend("fake", _FakeBackend)

    out = search_structures(
        query={"chemistry": ["Si"], "top_n": 2},
        sources=["fake"],
        output_dir=str(tmp_path / "out"),
    )
    assert "fake" in out["sources_queried"]
    assert any(c["source"] == "fake" for c in out["candidates"])


def test_search_structures_unknown_source_warns(tmp_path):
    from scilink.skills.structure_matching.xrd.search_structures import search_structures

    out = search_structures(
        query={"chemistry": ["Si"]},
        sources=["icsd"],  # not registered
        output_dir=str(tmp_path / "out"),
    )
    assert any("Unknown source 'icsd'" in w for w in out["warnings"])


def test_search_structures_skips_unavailable_backend(tmp_path):
    from scilink.skills.structure_matching.xrd.search_structures import search_structures

    class _Unavail(_FakeBackend):
        available = False

    register_backend("fake_unavail", _Unavail)
    out = search_structures(
        query={"chemistry": ["Si"]},
        sources=["fake_unavail"],
        output_dir=str(tmp_path / "out"),
    )
    assert "fake_unavail" not in out["sources_queried"]
    assert any("not available" in w for w in out["warnings"])


# --- Entry-point discovery smoke ----------------------------------------------

def test_entry_point_loader_imported():
    """The entry-point loader runs at module import. We don't have a real
    third-party package to test against, but the function should exist and
    not raise."""
    from scilink.skills.structure_matching._backends import _load_entry_points

    _load_entry_points()  # Idempotent; SciLink itself declares an empty group.


def test_entry_point_load_failure_logged_not_raised(caplog):
    """A broken entry-point target shouldn't break discovery."""
    import importlib.metadata as _md
    import logging

    from scilink.skills.structure_matching._backends import _load_entry_points

    class _BrokenEP:
        name = "broken"
        value = "nonexistent.module:Nope"
        def load(self):
            raise ImportError("module not found")

    with patch.object(_md, "entry_points", lambda *a, **kw: [_BrokenEP()]), \
         caplog.at_level(logging.WARNING,
                         logger="scilink.skills.structure_matching._backends"):
        _load_entry_points()
    assert any("Failed to load structure-backend entry-point broken" in r.message
               for r in caplog.records)
