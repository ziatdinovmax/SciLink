"""Offline tests for the named knowledge-base store (scilink.knowledge.kb_store).

No embedding calls: index building is monkeypatched; import/copy/manifest/
resolution mechanics are exercised against fake persisted-KB files.
"""

import json
from pathlib import Path

import pytest

from scilink.knowledge import kb_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Isolated store under a temp SCILINK_HOME."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    return kb_store.kb_store_dir()


def _fake_legacy_kb(dir_path: Path, sources=None):
    """Write the minimal persisted-KB file set import_kb expects."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "default_kb_docs.faiss").write_bytes(b"FAKEINDEX")
    (dir_path / "default_kb_docs.json").write_text("[]")
    (dir_path / "default_kb_docs.sources.json").write_text(
        json.dumps(sources if sources is not None
                   else [{"path": "/x/papers", "files": ["a.pdf", "b.md"]}])
    )
    return dir_path


def _fake_build(target, embedding_model, api_key, base_url, ocr_model,
                record_as=None):
    """Stand-in for _build_index_into: writes dummy index files."""
    prefix = target / "default_kb_docs"
    prefix.with_suffix(".faiss").write_bytes(b"BUILT")
    prefix.with_suffix(".json").write_text("[]")
    prefix.with_suffix(".sources.json").write_text("[]")
    return 7, 7


# ── names and resolution ────────────────────────────────────────────────

class TestNamesAndResolution:
    @pytest.mark.parametrize("bad", ["", "a/b", "-lead", ".hidden", "x" * 65])
    def test_invalid_names_rejected(self, bad):
        with pytest.raises(ValueError):
            kb_store._validate_name(bad)

    def test_valid_names_accepted(self):
        for ok in ("produced-water", "kb_1", "A.b-c_d", "x"):
            assert kb_store._validate_name(ok) == ok

    def test_existing_dir_beats_store_name(self, store, tmp_path):
        # A store KB and a cwd dir with the same name: the dir wins.
        _fake_legacy_kb(store / "clash")
        local = tmp_path / "clash"
        local.mkdir()
        resolved, manifest = kb_store.resolve_knowledge_source(str(local))
        assert resolved == local.resolve()
        assert manifest is None

    def test_bare_name_resolves_to_store(self, store):
        kb_dir = _fake_legacy_kb(store / "mykb")
        (kb_dir / kb_store.MANIFEST_NAME).write_text(
            json.dumps({"name": "mykb", "embedding_model": "m"})
        )
        resolved, manifest = kb_store.resolve_knowledge_source("mykb")
        assert resolved == kb_dir.resolve()
        assert manifest["embedding_model"] == "m"

    def test_strict_raises_with_available_names(self, store):
        _fake_legacy_kb(store / "onlykb")
        with pytest.raises(FileNotFoundError) as ei:
            kb_store.resolve_knowledge_source("no-such")
        assert "onlykb" in str(ei.value)

    def test_lenient_returns_plain_path(self, store, tmp_path):
        target = tmp_path / "not" / "yet"
        resolved, manifest = kb_store.resolve_knowledge_source(
            str(target), strict=False
        )
        assert resolved == target and manifest is None


# ── compatibility warnings ──────────────────────────────────────────────

class TestCompatWarning:
    def test_matching_model_is_silent(self):
        assert kb_store.embedding_compat_warning(
            {"name": "k", "embedding_model": "gemini-embedding-001"},
            "gemini-embedding-001",
        ) is None

    def test_mismatch_warns_with_rebuild_hint(self):
        msg = kb_store.embedding_compat_warning(
            {"name": "k", "embedding_model": "text-embedding-3-small"},
            "gemini-embedding-001",
        )
        assert "text-embedding-3-small" in msg and "rebuild" in msg

    def test_unknown_model_warns(self):
        msg = kb_store.embedding_compat_warning(
            {"name": "k", "embedding_model": "unknown"}, "any"
        )
        assert "does not record" in msg

    def test_no_manifest_is_silent(self):
        assert kb_store.embedding_compat_warning(None, "any") is None


# ── import / list / delete ──────────────────────────────────────────────

class TestImportListDelete:
    def test_import_copies_index_and_writes_manifest(self, store, tmp_path):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        manifest = kb_store.import_kb("adopted", str(legacy),
                                      embedding_model="gemini-embedding-001",
                                      description="legacy KB")
        kb_dir = store / "adopted"
        assert (kb_dir / "default_kb_docs.faiss").read_bytes() == b"FAKEINDEX"
        assert manifest["embedding_model"] == "gemini-embedding-001"
        assert manifest["sources"] == ["a.pdf", "b.md"]
        assert "imported" in manifest["origin"]

    def test_import_rejects_non_kb_dir(self, store, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            kb_store.import_kb("x", str(tmp_path / "empty"))

    def test_import_no_clobber_without_overwrite(self, store, tmp_path):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        kb_store.import_kb("dup", str(legacy))
        with pytest.raises(FileExistsError):
            kb_store.import_kb("dup", str(legacy))
        kb_store.import_kb("dup", str(legacy), overwrite=True)  # explicit OK

    def test_list_and_delete(self, store, tmp_path):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        kb_store.import_kb("one", str(legacy))
        kb_store.import_kb("two", str(legacy))
        names = [m["name"] for m in kb_store.list_kbs()]
        assert names == ["one", "two"]
        kb_store.delete_kb("one")
        assert [m["name"] for m in kb_store.list_kbs()] == ["two"]
        with pytest.raises(FileNotFoundError):
            kb_store.delete_kb("one")

    def test_staging_dirs_hidden_from_list(self, store, tmp_path):
        (store / ".staging_x").mkdir(parents=True)
        assert kb_store.list_kbs() == []


# ── create / rebuild (build monkeypatched) ──────────────────────────────

class TestCreateRebuild:
    def test_create_stages_sources_and_manifest(self, store, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(kb_store, "_build_index_into", _fake_build)
        doc = tmp_path / "paper.md"
        doc.write_text("# science")
        manifest = kb_store.create_kb("built", [str(doc)],
                                      embedding_model="gemini-embedding-001",
                                      description="test kb")
        kb_dir = store / "built"
        assert (kb_dir / "sources" / "paper.md").read_text() == "# science"
        assert manifest["origin"] == "built"
        assert manifest["n_vectors"] == 7
        assert manifest["sources"] == ["paper.md"]
        # The dir is directly usable as a knowledge_dir (default_kb layout)
        assert (kb_dir / "default_kb_docs.faiss").exists()

    def test_create_failure_leaves_nothing(self, store, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("embedding provider down")
        monkeypatch.setattr(kb_store, "_build_index_into", boom)
        doc = tmp_path / "paper.md"
        doc.write_text("x")
        with pytest.raises(RuntimeError):
            kb_store.create_kb("broken", [str(doc)])
        assert not (store / "broken").exists()
        assert not list(store.glob(".staging_*"))

    def test_create_missing_source_errors(self, store, tmp_path):
        with pytest.raises(FileNotFoundError):
            kb_store.create_kb("x", [str(tmp_path / "nope.pdf")])

    def test_rebuild_requires_sources(self, store, tmp_path, monkeypatch):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        kb_store.import_kb("imported", str(legacy))
        with pytest.raises(ValueError, match="not.*rebuildable|no stored"):
            kb_store.rebuild_kb("imported", embedding_model="m2")

    def test_rebuild_updates_embedding_model(self, store, tmp_path,
                                             monkeypatch):
        monkeypatch.setattr(kb_store, "_build_index_into", _fake_build)
        doc = tmp_path / "p.md"
        doc.write_text("x")
        kb_store.create_kb("rb", [str(doc)], embedding_model="old-model")
        manifest = kb_store.rebuild_kb("rb", embedding_model="new-model")
        assert manifest["embedding_model"] == "new-model"
        assert (store / "rb" / "sources" / "p.md").exists()  # sources kept


# ── incremental add ─────────────────────────────────────────────────────

def _fake_append(target, new_doc_paths, embedding_model, api_key, base_url,
                 ocr_model, record_as):
    """Stand-in for _append_index_into: touches the index, returns counts."""
    (target / "default_kb_docs.faiss").write_bytes(b"GROWN")
    return len(new_doc_paths), 2 * len(new_doc_paths)


class TestAddToKb:
    def _built(self, store, tmp_path, monkeypatch, name="grow"):
        monkeypatch.setattr(kb_store, "_build_index_into", _fake_build)
        doc = tmp_path / "base.md"
        doc.write_text("base")
        kb_store.create_kb(name, [str(doc)])
        return kb_store.kb_path(name)

    def test_add_appends_and_updates_manifest(self, store, tmp_path,
                                              monkeypatch):
        self._built(store, tmp_path, monkeypatch)
        monkeypatch.setattr(kb_store, "_append_index_into", _fake_append)
        new = tmp_path / "extra.md"
        new.write_text("more science")
        manifest = kb_store.add_to_kb("grow", [str(new)])
        assert manifest["n_vectors"] == 7 + 2          # base 7 + fake 2
        assert manifest["n_chunks"] == 7 + 1
        assert "extra.md" in manifest["sources"]
        assert (kb_store.kb_path("grow") / "sources" / "extra.md").exists()

    def test_add_refuses_duplicate_basenames(self, store, tmp_path,
                                             monkeypatch):
        self._built(store, tmp_path, monkeypatch)
        dup = tmp_path / "elsewhere" / "base.md"
        dup.parent.mkdir()
        dup.write_text("changed")
        with pytest.raises(FileExistsError, match="base.md"):
            kb_store.add_to_kb("grow", [str(dup)])

    def test_add_refuses_imported_kb(self, store, tmp_path):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        kb_store.import_kb("frozen", str(legacy),
                           embedding_model="gemini-embedding-001")
        new = tmp_path / "n.md"
        new.write_text("x")
        with pytest.raises(ValueError, match="imported"):
            kb_store.add_to_kb("frozen", [str(new)])

    def test_add_refuses_unknown_embedding_model(self, store, tmp_path,
                                                 monkeypatch):
        kb_dir = self._built(store, tmp_path, monkeypatch)
        mf = json.loads((kb_dir / kb_store.MANIFEST_NAME).read_text())
        mf["embedding_model"] = "unknown"
        (kb_dir / kb_store.MANIFEST_NAME).write_text(json.dumps(mf))
        new = tmp_path / "n.md"
        new.write_text("x")
        with pytest.raises(ValueError, match="embedding model"):
            kb_store.add_to_kb("grow", [str(new)])

    def test_add_failure_is_atomic(self, store, tmp_path, monkeypatch):
        kb_dir = self._built(store, tmp_path, monkeypatch)
        before = (kb_dir / "default_kb_docs.faiss").read_bytes()

        def boom(*a, **k):
            raise RuntimeError("provider down")
        monkeypatch.setattr(kb_store, "_append_index_into", boom)
        new = tmp_path / "n.md"
        new.write_text("x")
        with pytest.raises(RuntimeError):
            kb_store.add_to_kb("grow", [str(new)])
        assert (kb_dir / "default_kb_docs.faiss").read_bytes() == before
        assert not (kb_dir / "sources" / "n.md").exists()
        assert not list(store.glob(".staging_*"))


# ── orchestrator knowledge-path fallback vs prebuilt store KBs ──────────

class TestKnowledgePathFallback:
    """A store KB handed to the orchestrator must never be re-ingested from
    its root (that would re-embed the documents and swallow the index files
    themselves); plain dirs keep legacy incremental-ingest behavior."""

    @staticmethod
    def _tools_with(knowledge_dir):
        from types import SimpleNamespace
        from scilink.agents.planning_agents.orchestrator_tools import (
            OrchestratorTools,
        )
        tools = OrchestratorTools.__new__(OrchestratorTools)
        tools.orch = SimpleNamespace(knowledge_dir=knowledge_dir)
        return tools

    def test_store_kb_falls_back_to_sources_dir(self, store, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr(kb_store, "_build_index_into", _fake_build)
        doc = tmp_path / "p.md"
        doc.write_text("x")
        kb_store.create_kb("fb", [str(doc)])
        kb_dir = kb_store.kb_path("fb")
        resolved = self._tools_with(kb_dir)._resolve_knowledge_paths(None)
        assert resolved == [str(kb_dir / "sources")]

    def test_imported_kb_falls_back_to_nothing(self, store, tmp_path):
        legacy = _fake_legacy_kb(tmp_path / "kb_storage")
        kb_store.import_kb("idx-only", str(legacy))
        kb_dir = kb_store.kb_path("idx-only")
        assert self._tools_with(kb_dir)._resolve_knowledge_paths(None) is None

    def test_plain_dir_keeps_legacy_behavior(self, tmp_path):
        plain = tmp_path / "papers"
        plain.mkdir()
        resolved = self._tools_with(plain)._resolve_knowledge_paths(None)
        assert resolved == [str(plain)]

    def test_explicit_paths_still_win(self, tmp_path):
        resolved = self._tools_with(None)._resolve_knowledge_paths("a, b")
        assert resolved == ["a", "b"]

    def test_created_kb_records_final_sources_path(self, store, tmp_path,
                                                   monkeypatch):
        # _fake_build bypasses recording; use the real recording logic with a
        # stubbed KnowledgeBase to check the path written into sources.json.
        captured = {}

        class FakeKB:
            def __init__(self, **kw):
                self.sources = []
                class _Idx:  # noqa: N801
                    ntotal = 3
                self.index = _Idx()
            def save(self, index_path, chunks_path, sources_path=None, **kw):
                Path(index_path).write_bytes(b"X")
                Path(chunks_path).write_text("[]")
                Path(sources_path).write_text(json.dumps(self.sources))
                captured["sources"] = self.sources
            def build(self, chunks):
                pass

        import scilink.knowledge.knowledge_base as kbmod
        monkeypatch.setattr(kbmod, "KnowledgeBase", FakeKB)
        import scilink.parsers as parsers
        monkeypatch.setattr(parsers, "ingest_files",
                            lambda *a, **k: [{"text": "t", "metadata": {}}])

        doc = tmp_path / "d.md"
        doc.write_text("x")
        kb_store.create_kb("recorded", [str(doc)])
        rec = captured["sources"][0]
        final_sources = str(kb_store.kb_path("recorded") / "sources")
        assert rec["path"] == final_sources, rec
        assert rec["files"] == ["d.md"]


class TestCrashSafeSwap:
    """The publish step must never leave the KB path empty.

    Review (PR #394): rmtree(final) then staging.rename(final) has a window
    where the KB exists at NEITHER path. That is not merely
    recoverable-in-principle — `kb list` hid the stranded `.staging_<name>`,
    nothing promoted it, and the natural "recreate it" response ran
    `kb create`, which clears stale staging first and destroys the last copy.
    """

    def test_live_kb_survives_a_crash_during_publish(self, store, monkeypatch):
        kb = kb_store.kb_path("mykb")
        kb.mkdir(parents=True)
        (kb / "default_kb_docs.faiss").write_bytes(b"ORIGINAL")
        (kb / "manifest.json").write_text(json.dumps({"name": "mykb"}))
        staging = kb_store.kb_store_dir() / ".staging_mykb"
        staging.mkdir()
        (staging / "default_kb_docs.faiss").write_bytes(b"REBUILT")

        real_rename = Path.rename
        failed = []

        def die_on_publish(self, target):
            # fail ONLY the staging -> final move, once; the restore that
            # follows must be allowed to run, as it would after a real crash
            if Path(self).name.startswith(".staging_") and not failed:
                failed.append(1)
                raise RuntimeError("power cut mid-publish")
            return real_rename(self, target)

        monkeypatch.setattr(Path, "rename", die_on_publish)
        with pytest.raises(RuntimeError):
            kb_store._swap_into_place(staging, kb)

        # the ORIGINAL KB is still there, intact and usable
        assert kb.is_dir()
        assert (kb / "default_kb_docs.faiss").read_bytes() == b"ORIGINAL"
        assert kb_store.read_manifest(kb)["name"] == "mykb"

    def test_successful_swap_publishes_and_cleans_up(self, store):
        kb = kb_store.kb_path("mykb")
        kb.mkdir(parents=True)
        (kb / "default_kb_docs.faiss").write_bytes(b"ORIGINAL")
        staging = kb_store.kb_store_dir() / ".staging_mykb"
        staging.mkdir()
        (staging / "default_kb_docs.faiss").write_bytes(b"REBUILT")

        kb_store._swap_into_place(staging, kb)
        assert (kb / "default_kb_docs.faiss").read_bytes() == b"REBUILT"
        assert not staging.exists()
        assert not kb.with_name(kb.name + ".bak").exists()   # no litter

    def test_swap_works_when_there_is_no_previous_kb(self, store):
        kb = kb_store.kb_path("fresh")
        staging = kb_store.kb_store_dir() / ".staging_fresh"
        staging.mkdir(parents=True)
        (staging / "default_kb_docs.faiss").write_bytes(b"NEW")
        kb_store._swap_into_place(staging, kb)
        assert (kb / "default_kb_docs.faiss").read_bytes() == b"NEW"

    def test_a_stale_backup_does_not_block_a_later_swap(self, store):
        kb = kb_store.kb_path("mykb")
        kb.mkdir(parents=True)
        (kb / "x").write_text("live")
        stale = kb.with_name(kb.name + ".bak")
        stale.mkdir()
        (stale / "junk").write_text("from a previous failure")
        staging = kb_store.kb_store_dir() / ".staging_mykb"
        staging.mkdir()
        (staging / "x").write_text("new")

        kb_store._swap_into_place(staging, kb)
        assert (kb / "x").read_text() == "new"
        assert not stale.exists()

    def test_interrupted_build_is_reported_not_hidden(self, store, caplog):
        (kb_store.kb_store_dir() / ".staging_halfbuilt").mkdir(parents=True)
        good = kb_store.kb_path("good")
        good.mkdir(parents=True)
        (good / "manifest.json").write_text(json.dumps({"name": "good"}))

        with caplog.at_level("WARNING"):
            listed = kb_store.list_kbs()
        assert [k["name"] for k in listed] == ["good"]   # not a usable KB
        assert "Leftover build directory" in caplog.text
        assert ".staging_halfbuilt" in caplog.text
