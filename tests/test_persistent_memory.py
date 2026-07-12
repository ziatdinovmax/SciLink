

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
