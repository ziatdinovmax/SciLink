"""Tests for the skill-owned trigger taxonomy loader helper.

After the refactor, deterministic triggers live in skill frontmatter
under ``live_reading.triggers:``. The framework provides the protocol
and reusable primitives; skills decide which to use (and can add
modality-specific ones in sibling .py files).
"""
from __future__ import annotations

import sys
import types

import pytest

from scilink.skills.loader import resolve_triggers_from_skill


# --- Inert returns ------------------------------------------------------------

def test_none_meta_returns_empty():
    assert resolve_triggers_from_skill(None) == []


def test_empty_meta_returns_empty():
    assert resolve_triggers_from_skill({}) == []


def test_missing_live_reading_block_returns_empty():
    assert resolve_triggers_from_skill({"description": "..."}) == []


def test_non_dict_live_reading_returns_empty():
    """Defensive: live_reading: 'yes' (string) shouldn't crash."""
    assert resolve_triggers_from_skill({"live_reading": "yes"}) == []


def test_missing_triggers_key_returns_empty():
    assert resolve_triggers_from_skill({"live_reading": {"enabled": True}}) == []


def test_empty_triggers_list_returns_empty():
    meta = {"live_reading": {"triggers": []}}
    assert resolve_triggers_from_skill(meta) == []


# --- Validation errors --------------------------------------------------------

def test_non_list_triggers_raises_type_error():
    with pytest.raises(TypeError, match="must be a list"):
        resolve_triggers_from_skill({"live_reading": {"triggers": "not a list"}})


def test_non_dict_entry_raises():
    meta = {"live_reading": {"triggers": ["just a string"]}}
    with pytest.raises(TypeError, match="must be a mapping"):
        resolve_triggers_from_skill(meta)


def test_missing_type_raises():
    meta = {"live_reading": {"triggers": [{"config": {}}]}}
    with pytest.raises(ValueError, match="'type' missing"):
        resolve_triggers_from_skill(meta)


def test_malformed_type_no_colon_raises():
    meta = {"live_reading": {"triggers": [{"type": "no.colon.in.this.path"}]}}
    with pytest.raises(ValueError, match="module.path:ClassName"):
        resolve_triggers_from_skill(meta)


def test_missing_module_raises():
    meta = {
        "live_reading": {"triggers": [{"type": "nonexistent_module_xyz:Foo"}]}
    }
    with pytest.raises(ImportError, match="nonexistent_module_xyz"):
        resolve_triggers_from_skill(meta)


def test_missing_attribute_raises(monkeypatch):
    fake = types.ModuleType("fake_trigger_mod_attr_test")
    fake.RealClass = type("RealClass", (), {})
    monkeypatch.setitem(sys.modules, "fake_trigger_mod_attr_test", fake)
    meta = {
        "live_reading": {
            "triggers": [{"type": "fake_trigger_mod_attr_test:AbsentClass"}]
        }
    }
    with pytest.raises(AttributeError, match="AbsentClass"):
        resolve_triggers_from_skill(meta)


def test_non_mapping_config_raises(monkeypatch):
    fake = types.ModuleType("fake_trigger_mod_cfg_test")
    fake.MyTrigger = type("MyTrigger", (), {})
    monkeypatch.setitem(sys.modules, "fake_trigger_mod_cfg_test", fake)
    meta = {
        "live_reading": {"triggers": [{
            "type": "fake_trigger_mod_cfg_test:MyTrigger",
            "config": "not a mapping",
        }]}
    }
    with pytest.raises(TypeError, match="'config' must be a mapping"):
        resolve_triggers_from_skill(meta)


def test_bad_kwargs_raises_clear_error(monkeypatch):
    """Skill passes a kwarg the trigger doesn't accept → clear error."""
    fake = types.ModuleType("fake_trigger_mod_bad_kw")

    class StrictTrigger:
        def __init__(self): pass

    fake.StrictTrigger = StrictTrigger
    monkeypatch.setitem(sys.modules, "fake_trigger_mod_bad_kw", fake)
    meta = {
        "live_reading": {"triggers": [{
            "type": "fake_trigger_mod_bad_kw:StrictTrigger",
            "config": {"unexpected_kw": 42},
        }]}
    }
    with pytest.raises(TypeError, match="failed to instantiate"):
        resolve_triggers_from_skill(meta)


# --- Happy paths --------------------------------------------------------------

def test_resolves_built_in_verdict_change_trigger():
    meta = {
        "live_reading": {
            "triggers": [
                {"type": "scilink.agents.exp_agents.live_triggers:VerdictChangeTrigger"},
            ]
        }
    }
    out = resolve_triggers_from_skill(meta)
    assert len(out) == 1
    assert type(out[0]).__name__ == "VerdictChangeTrigger"


def test_resolves_with_config_kwargs():
    meta = {
        "live_reading": {
            "triggers": [
                {
                    "type": "scilink.agents.exp_agents.live_triggers:ConfidenceReversalTrigger",
                    "config": {"window": 7, "min_reversal": 0.1},
                },
            ]
        }
    }
    out = resolve_triggers_from_skill(meta)
    assert len(out) == 1
    t = out[0]
    assert t.window == 7
    assert t.min_reversal == 0.1


def test_resolves_multiple_triggers_in_order():
    meta = {
        "live_reading": {
            "triggers": [
                {"type": "scilink.agents.exp_agents.live_triggers:VerdictChangeTrigger"},
                {"type": "scilink.agents.exp_agents.live_triggers:NewFeatureTrigger",
                 "config": {"lookback": 3}},
                {"type": "scilink.agents.exp_agents.live_triggers:ConfidenceReversalTrigger"},
            ]
        }
    }
    out = resolve_triggers_from_skill(meta)
    names = [type(t).__name__ for t in out]
    assert names == [
        "VerdictChangeTrigger", "NewFeatureTrigger", "ConfidenceReversalTrigger",
    ]
    assert out[1].lookback == 3


def test_resolves_custom_skill_specific_trigger(monkeypatch):
    """Skills can define their own triggers in Python and reference them."""
    fake = types.ModuleType("fake_skill_custom_triggers")

    class LatticeDriftTrigger:
        def __init__(self, drift_threshold_pct=1.0):
            self.drift_threshold_pct = drift_threshold_pct
            self.name = "lattice_drift"

        def evaluate(self, history): return None
        def reset(self): pass

    fake.LatticeDriftTrigger = LatticeDriftTrigger
    monkeypatch.setitem(sys.modules, "fake_skill_custom_triggers", fake)

    meta = {
        "live_reading": {
            "triggers": [
                {"type": "fake_skill_custom_triggers:LatticeDriftTrigger",
                 "config": {"drift_threshold_pct": 1.5}},
            ]
        }
    }
    out = resolve_triggers_from_skill(meta)
    assert len(out) == 1
    assert isinstance(out[0], LatticeDriftTrigger)
    assert out[0].drift_threshold_pct == 1.5


def test_diagnostics_skill_triggers_resolve():
    """Smoke: the shipped diagnostics skill's frontmatter declares
    triggers that should resolve cleanly."""
    from scilink.skills.loader import load_skill

    skill = load_skill("live_passthrough", domain="diagnostics")
    resolved = resolve_triggers_from_skill(skill.get("meta"))
    assert len(resolved) >= 3  # at least VerdictChange, NewFeature, ConfidenceReversal
    names = {type(t).__name__ for t in resolved}
    assert "VerdictChangeTrigger" in names
    assert "NewFeatureTrigger" in names
    assert "ConfidenceReversalTrigger" in names
