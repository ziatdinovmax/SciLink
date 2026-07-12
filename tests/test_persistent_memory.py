

# ──────────────────────────────────────────────────────────────
# Demote — the reverse of promote (PR #348 follow-up)
# ──────────────────────────────────────────────────────────────

class TestDemote:
    def _write_bundle(self, root, domain, name, provisional):
        d = root / domain / name
        d.mkdir(parents=True)
        fm = "---\nprovisional: true\n---\n" if provisional else ""
        (d / f"{name}.md").write_text(fm + "## overview\nbody text stays\n")

    def test_promote_demote_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        from scilink.skills.loader import graduated_skills_dir

        self._write_bundle(graduated_skills_dir(), "curve_fitting", "sk", True)
        _memory.promote_memory("curve_fitting", "sk")
        assert all(not r["provisional"] for r in _memory.list_memory())

        out = _memory.demote_memory("curve_fitting", "sk")
        assert out["provisional"] is True
        rows = _memory.list_memory()
        assert len(rows) == 1 and rows[0]["provisional"] is True
        # Section bodies untouched by the frontmatter rewrite.
        assert "body text stays" in _memory.show_memory("curve_fitting", "sk")
        # Idempotent + re-promotable.
        _memory.demote_memory("curve_fitting", "sk")
        _memory.promote_memory("curve_fitting", "sk")
        assert all(not r["provisional"] for r in _memory.list_memory())

    def test_demote_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            _memory.demote_memory("curve_fitting", "nope")


# ──────────────────────────────────────────────────────────────
# Fork built-in skills (copy-on-write) + additivity guard
# ──────────────────────────────────────────────────────────────

class TestForkBuiltin:
    def test_fork_shadows_and_upgradable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        # Shadowing requires the persistent store to be ACTIVE — with memory
        # off the graduated root is not on the skill search path at all.
        monkeypatch.setenv("SCILINK_MEMORY", "1")
        from scilink.skills import loader
        from scilink.skills._shared import _memory

        builtin_text = (loader._SKILLS_DIR / "curve_fitting" / "raman"
                        / "raman.md").read_text()
        out = _memory.fork_builtin("curve_fitting", "raman")
        assert out["status"] == "success"
        assert out["has_sibling_tools"] is False
        # Byte-identical copy -> diff-builtin starts clean.
        d = _memory.diff_builtin("curve_fitting", "raman")
        assert d["identical"] is True
        # The fork appears in the persistent store (upgrade target)…
        assert any(r["name"] == "raman" for r in _memory.list_memory())
        # …and SHADOWS the built-in: edit the fork, loader serves the edit.
        fork_md = tmp_path / "graduated_skills" / "curve_fitting" / "raman" / "raman.md"
        fork_md.write_text(builtin_text + "\nFORK_SENTINEL_LINE\n")
        loaded = loader.load_skill("raman", domain="curve_fitting")
        assert "FORK_SENTINEL_LINE" in str(loaded)
        d = _memory.diff_builtin("curve_fitting", "raman")
        assert d["identical"] is False and "FORK_SENTINEL_LINE" in d["diff"]

    def test_double_fork_refused_and_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        assert _memory.fork_builtin("curve_fitting", "raman")["status"] == "success"
        again = _memory.fork_builtin("curve_fitting", "raman")
        assert again["status"] == "error" and "already forked" in again["message"]
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            _memory.fork_builtin("curve_fitting", "no_such_skill")
        with _pytest.raises(FileNotFoundError):
            _memory.diff_builtin("curve_fitting", "xps")  # not forked

    def test_sibling_tools_flagged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
        from scilink.skills._shared import _memory
        out = _memory.fork_builtin("curve_fitting", "xrd_profile")
        assert out["status"] == "success"
        assert out["has_sibling_tools"] is True
        assert "fit_pattern.py" in out["sibling_tools"]


class TestAdditivityGuard:
    def test_regression_warnings(self):
        from scilink.skills._shared._staging import _regression_warnings
        existing = ("## overview\nlong text " + "word " * 100
                    + "\n## planning\nrules\n## validation\nchecks\n")
        clean = existing + "\nnew lesson appended\n"
        assert _regression_warnings(existing, clean) == []
        dropped = "## overview\n" + "word " * 100
        warns = _regression_warnings(existing, dropped)
        assert any("planning" in w for w in warns)
        assert any("validation" in w for w in warns)
        shrunk = "## overview\nshort\n## planning\nx\n## validation\ny\n"
        assert any("shorter" in w for w in _regression_warnings(existing, shrunk))

    def test_preserve_structure_round_trip(self):
        from scilink.skills._shared._staging import _preserve_structure
        existing = (
            "---\ndescription: old desc\n"
            "technique: [Raman, micro-Raman]\n---\n"
            "# Raman Spectroscopy\n\n"
            "## Overview\nbody A\n\n## Planning\nbody B\n")
        proposed = (  # what the JSON round-trip typically emits
            "---\ndescription: refreshed desc\n---\n"
            "## overview\nbody A plus new lesson\n\n"
            "## analysis\n\n"           # empty artifact section
            "## planning\nbody B\n")
        out = _preserve_structure(existing, proposed)
        assert "technique:" in out and "micro-Raman" in out   # routing meta kept
        assert "refreshed desc" in out                        # LLM description honored
        assert "# Raman Spectroscopy" in out                  # title restored
        assert "## Overview" in out and "## overview" not in out  # casing restored
        assert "## analysis" not in out                       # empty artifact dropped
        assert "new lesson" in out                            # LLM merge kept
        from scilink.skills._shared._staging import _regression_warnings
        assert _regression_warnings(existing, out) == []

    def test_preserve_structure_reinstates_dropped_section(self):
        from scilink.skills._shared._staging import (
            _preserve_structure, _regression_warnings)
        existing = ("---\ndescription: d\n---\n"
                    "## Overview\nA\n\n## Implementation\nCRITICAL RECIPE\n\n"
                    "## Validation\nchecks\n")
        proposed = ("---\ndescription: d\n---\n"
                    "## overview\nA plus lesson\n\n## validation\nchecks\n")
        out = _preserve_structure(existing, proposed)
        assert "## Implementation" in out and "CRITICAL RECIPE" in out
        # reinstated content sits after the surviving sections
        assert out.index("CRITICAL RECIPE") > out.index("A plus lesson")
        assert _regression_warnings(existing, out) == []
