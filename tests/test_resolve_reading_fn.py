"""Tests for the skill loader's resolve_reading_fn helper.

A skill declares its live-mode reading function in frontmatter::

    live_reading:
      enabled: true
      reading_fn: my_pkg.live_reading:my_tick

resolve_reading_fn(meta) imports the dotted path and returns the callable.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from scilink.skills.loader import resolve_reading_fn


# ---------------------------------------------------------------------------
# Inert / disabled / missing — should return None
# ---------------------------------------------------------------------------


def test_none_meta_returns_none():
    assert resolve_reading_fn(None) is None


def test_empty_meta_returns_none():
    assert resolve_reading_fn({}) is None


def test_missing_live_tick_block_returns_none():
    assert resolve_reading_fn({"description": "..."}) is None


def test_disabled_explicitly_returns_none():
    meta = {"live_reading": {"enabled": False, "reading_fn": "some.where:func"}}
    assert resolve_reading_fn(meta) is None


def test_live_tick_block_without_tick_fn_returns_none():
    """A `live_reading:` block missing the reading_fn key is treated as not-yet-
    configured rather than an error."""
    meta = {"live_reading": {"enabled": True}}
    assert resolve_reading_fn(meta) is None


def test_non_dict_live_tick_returns_none():
    """Defensive: live_reading: 'yes' (string instead of mapping) doesn't crash."""
    meta = {"live_reading": "yes"}
    assert resolve_reading_fn(meta) is None


# ---------------------------------------------------------------------------
# Malformed dotted paths
# ---------------------------------------------------------------------------


def test_dotted_path_without_colon_raises_value_error():
    meta = {"live_reading": {"enabled": True, "reading_fn": "missing.colon"}}
    with pytest.raises(ValueError, match="module.path:function_name"):
        resolve_reading_fn(meta)


def test_dotted_path_empty_module_raises():
    meta = {"live_reading": {"enabled": True, "reading_fn": ":just_attr"}}
    with pytest.raises(ValueError, match="module.path:function_name"):
        resolve_reading_fn(meta)


def test_dotted_path_empty_attr_raises():
    meta = {"live_reading": {"enabled": True, "reading_fn": "module.only:"}}
    with pytest.raises(ValueError, match="module.path:function_name"):
        resolve_reading_fn(meta)


# ---------------------------------------------------------------------------
# Import failures
# ---------------------------------------------------------------------------


def test_missing_module_raises_import_error():
    meta = {
        "live_reading": {
            "enabled": True,
            "reading_fn": "nonexistent_module_xyz_qqq:some_func",
        }
    }
    with pytest.raises(ImportError, match="nonexistent_module_xyz_qqq"):
        resolve_reading_fn(meta)


def test_missing_attribute_raises_attribute_error(monkeypatch):
    """Inject a fake module that exists but lacks the attribute."""
    fake = types.ModuleType("fake_live_tick_holder")
    fake.real_func = lambda *a, **kw: None  # exists, but we'll request a different name
    monkeypatch.setitem(sys.modules, "fake_live_tick_holder", fake)
    meta = {
        "live_reading": {
            "enabled": True,
            "reading_fn": "fake_live_tick_holder:absent_func",
        }
    }
    with pytest.raises(AttributeError, match="absent_func"):
        resolve_reading_fn(meta)


def test_non_callable_attribute_raises_type_error(monkeypatch):
    fake = types.ModuleType("fake_non_callable_holder")
    fake.its_a_string = "not callable"  # exists but not callable
    monkeypatch.setitem(sys.modules, "fake_non_callable_holder", fake)
    meta = {
        "live_reading": {
            "enabled": True,
            "reading_fn": "fake_non_callable_holder:its_a_string",
        }
    }
    with pytest.raises(TypeError, match="not callable"):
        resolve_reading_fn(meta)


# ---------------------------------------------------------------------------
# Happy path — valid module + callable
# ---------------------------------------------------------------------------


def test_valid_dotted_path_returns_callable(monkeypatch):
    fake = types.ModuleType("fake_valid_tick_holder")
    sentinel = lambda *a, **kw: "ran-reading"  # noqa: E731
    fake.my_tick = sentinel
    monkeypatch.setitem(sys.modules, "fake_valid_tick_holder", fake)
    meta = {
        "live_reading": {
            "enabled": True,
            "reading_fn": "fake_valid_tick_holder:my_tick",
        }
    }
    resolved = resolve_reading_fn(meta)
    assert resolved is sentinel
    assert resolved() == "ran-reading"


def test_enabled_defaults_to_true(monkeypatch):
    """A reading_fn without an explicit enabled key is treated as enabled."""
    fake = types.ModuleType("fake_default_enabled_holder")
    fake.my_tick = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "fake_default_enabled_holder", fake)
    meta = {
        "live_reading": {
            # no `enabled:` field
            "reading_fn": "fake_default_enabled_holder:my_tick",
        }
    }
    assert resolve_reading_fn(meta) is fake.my_tick


def test_dotted_path_with_submodule(monkeypatch):
    """Handles dotted module paths like pkg.sub.module:func."""
    parent = types.ModuleType("fake_parent_module")
    child = types.ModuleType("fake_parent_module.child")
    child.reading = lambda *a, **kw: "child-reading"
    parent.child = child
    monkeypatch.setitem(sys.modules, "fake_parent_module", parent)
    monkeypatch.setitem(sys.modules, "fake_parent_module.child", child)
    meta = {
        "live_reading": {
            "enabled": True,
            "reading_fn": "fake_parent_module.child:reading",
        }
    }
    assert resolve_reading_fn(meta) is child.reading
