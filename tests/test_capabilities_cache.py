"""The meta's capability-inventory disk cache: a hit must skip the live
build (which constructs all three children), a changed extension set must
miss, and cache failures must fall back to the live path."""

import json

from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _bare_meta(tmp_path, extensions):
    meta = object.__new__(MetaOrchestratorAgent)
    meta._shared_extensions = extensions
    meta.logger = _Logger()
    meta._CAPABILITIES_CACHE = tmp_path / "capabilities_cache.json"
    return meta


def test_cache_hit_skips_live_build(tmp_path):
    meta = _bare_meta(tmp_path, [])
    calls = {"n": 0}

    def live():
        calls["n"] += 1
        return "**SPECIALIST CAPABILITIES** block"

    meta._build_capabilities_block_live = live

    assert meta._build_capabilities_block() == "**SPECIALIST CAPABILITIES** block"
    assert calls["n"] == 1
    # Second call (fresh session, same build+extensions): served from disk.
    meta2 = _bare_meta(tmp_path, [])
    meta2._build_capabilities_block_live = live
    assert meta2._build_capabilities_block() == "**SPECIALIST CAPABILITIES** block"
    assert calls["n"] == 1
    assert meta._CAPABILITIES_CACHE.exists()


def test_extension_change_misses_cache(tmp_path):
    meta = _bare_meta(tmp_path, [])
    meta._build_capabilities_block_live = lambda: "plain"
    meta._build_capabilities_block()

    with_skill = _bare_meta(
        tmp_path, [{"kind": "skill", "skill_path": "/x/custom.md"}])
    with_skill._build_capabilities_block_live = lambda: "with custom skill"
    assert with_skill._build_capabilities_block() == "with custom skill"
    # Both variants now cached under distinct keys.
    cached = json.loads(meta._CAPABILITIES_CACHE.read_text())
    assert sorted(cached.values()) == ["plain", "with custom skill"]


def test_corrupt_cache_falls_back_to_live(tmp_path):
    meta = _bare_meta(tmp_path, [])
    meta._CAPABILITIES_CACHE.write_text("{not json")
    meta._build_capabilities_block_live = lambda: "rebuilt"
    assert meta._build_capabilities_block() == "rebuilt"
    # ...and the rewrite leaves a valid cache behind.
    assert json.loads(meta._CAPABILITIES_CACHE.read_text())


def test_cache_bounded(tmp_path):
    meta = _bare_meta(tmp_path, [])
    meta._build_capabilities_block_live = lambda: "block"
    # Pre-fill beyond the cap; a new write must trim to the newest entries.
    meta._CAPABILITIES_CACHE.write_text(json.dumps(
        {f"key{i}": "old" for i in range(12)}))
    meta._build_capabilities_block()
    cached = json.loads(meta._CAPABILITIES_CACHE.read_text())
    assert len(cached) <= MetaOrchestratorAgent._CAPABILITIES_CACHE_MAX
    assert "block" in cached.values()
