"""Concurrent writers to the shared stores do not destroy each other's work.

Knowledge bases: every build of a KB staged into one fixed `.staging_<name>`
and began by deleting it, so two builds wiped each other's staging; two
concurrent `add_to_kb` calls each copied the live KB, added their document and
swapped in, and the first addition was lost. The sessions index was a
read-modify-write through one fixed temp name under a lock that only covered
threads, so two SciLink processes registering at once lost records.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scilink.knowledge import kb_store

REPO = Path(__file__).resolve().parents[1]


def _fake_build(target, embedding_model, api_key, base_url, ocr_model, record_as=None):
    prefix = target / "default_kb_docs"
    prefix.with_suffix(".faiss").write_bytes(b"BUILT")
    prefix.with_suffix(".json").write_text("[]")
    prefix.with_suffix(".sources.json").write_text("[]")
    return 7, 7


def _slow_append(target, new_doc_paths, embedding_model, api_key, base_url, ocr_model, record_as):
    time.sleep(0.3)                     # embedding takes a while
    (target / "default_kb_docs.faiss").write_bytes(b"GROWN")
    return len(new_doc_paths), 2 * len(new_doc_paths)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(kb_store, "_build_index_into", _fake_build)
    return kb_store.kb_store_dir()


def _doc(tmp_path, name, text="science"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_concurrent_additions_both_land(store, tmp_path, monkeypatch):
    kb_store.create_kb("grow", [_doc(tmp_path, "base.md")])
    monkeypatch.setattr(kb_store, "_append_index_into", _slow_append)
    errors = []

    def add(name):
        try:
            kb_store.add_to_kb("grow", [_doc(tmp_path, name)])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(f"doc{i}.md",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    manifest = kb_store.read_manifest(kb_store.kb_path("grow"))
    assert {"doc0.md", "doc1.md", "doc2.md", "base.md"} <= set(manifest["sources"])
    assert manifest["n_chunks"] == 7 + 3
    assert [p.name for p in store.iterdir() if p.name.startswith(".staging_")] == []


def test_two_creates_of_one_name_do_not_wreck_each_other(store, tmp_path, monkeypatch):
    def slow_build(*a, **k):
        time.sleep(0.3)
        return _fake_build(*a, **k)

    monkeypatch.setattr(kb_store, "_build_index_into", slow_build)
    outcomes = []

    def create(i):
        try:
            kb_store.create_kb("twin", [_doc(tmp_path, f"src{i}.md")])
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")
        except Exception as exc:  # noqa: BLE001
            outcomes.append(repr(exc))

    threads = [threading.Thread(target=create, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["created", "exists"]
    manifest = kb_store.read_manifest(kb_store.kb_path("twin"))
    assert len(manifest["sources"]) == 1           # one build, whole


def test_concurrent_additions_from_separate_processes_both_land(store, tmp_path):
    kb_store.create_kb("shared", [_doc(tmp_path, "base.md")])
    script = textwrap.dedent("""
        import sys, time
        from scilink.knowledge import kb_store
        def slow_append(target, new_doc_paths, *a, **k):
            time.sleep(0.5)
            (target / "default_kb_docs.faiss").write_bytes(b"GROWN")
            return len(new_doc_paths), len(new_doc_paths)
        kb_store._append_index_into = slow_append
        kb_store.add_to_kb("shared", [sys.argv[1]])
    """)
    env = dict(os.environ, PYTHONPATH=str(REPO), SCILINK_HOME=os.environ["SCILINK_HOME"])
    procs = [subprocess.Popen([sys.executable, "-c", script, _doc(tmp_path, f"p{i}.md")],
                              env=env, stderr=subprocess.PIPE) for i in range(3)]
    errs = [p.communicate(timeout=120)[1].decode()[-400:] for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0], errs
    sources = set(kb_store.read_manifest(kb_store.kb_path("shared"))["sources"])
    assert {"p0.md", "p1.md", "p2.md", "base.md"} <= sources


def test_delete_waits_for_a_running_build(store, tmp_path, monkeypatch):
    kb_store.create_kb("busy", [_doc(tmp_path, "base.md")])
    monkeypatch.setattr(kb_store, "_append_index_into", _slow_append)
    t = threading.Thread(target=kb_store.add_to_kb, args=("busy", [_doc(tmp_path, "more.md")]))
    t.start()
    time.sleep(0.05)
    kb_store.delete_kb("busy")          # waits for the addition, then deletes
    t.join()
    assert not kb_store.kb_path("busy").exists()
    assert [p.name for p in store.iterdir() if p.is_dir()] == []


def test_the_sessions_index_keeps_every_record_under_concurrent_processes(tmp_path):
    home = tmp_path / "home"
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        from scilink import sessions as S
        base = Path(sys.argv[1])
        for j in range(8):
            d = base / f"meta_session_2026092{sys.argv[2]}_1000{j:02d}"
            d.mkdir(parents=True)
            S.register_session(d, "meta")
    """)
    env = dict(os.environ, PYTHONPATH=str(REPO), SCILINK_HOME=str(home))
    procs = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path / f"proj{i}"), str(i)],
                              env=env, stderr=subprocess.PIPE) for i in range(5)]
    errs = [p.communicate(timeout=120)[1].decode()[-400:] for p in procs]
    assert [p.returncode for p in procs] == [0] * 5, errs
    lines = [json.loads(l) for l in (home / "sessions.jsonl").read_text().splitlines() if l.strip()]
    assert len({r["path"] for r in lines}) == 40
    assert [p.name for p in home.iterdir() if p.name.endswith(".tmp")] == []


def test_a_home_that_cannot_hold_the_lock_still_registers_nothing_breaks(tmp_path, monkeypatch):
    """The index is a convenience: a read-only home must not break a session."""
    from scilink import sessions as S
    home = tmp_path / "ro_home"
    home.mkdir()
    monkeypatch.setenv("SCILINK_HOME", str(home))
    d = tmp_path / "meta_session_20260928_120000"
    d.mkdir()
    home.chmod(0o555)
    try:
        S.register_session(d, "meta")            # no exception
        assert S.list_sessions("meta", root=tmp_path) is not None
    finally:
        home.chmod(0o755)


# ── review follow-ups: readers, backups, live builds ──────────────────────

def _generation_append(target, new_doc_paths, *a, **k):
    """Stand-in for _append_index_into that writes the index and the chunks
    as one generation, tagged so a reader can tell generations apart."""
    import uuid
    tag = uuid.uuid4().hex
    (target / "default_kb_docs.faiss").write_bytes(tag.encode() * 4000)
    (target / "default_kb_docs.json").write_text(json.dumps([tag] * 4000))
    return len(new_doc_paths), len(new_doc_paths)


def test_a_reader_never_copies_a_mixed_generation(store, tmp_path, monkeypatch):
    kb_store.create_kb("busy", [_doc(tmp_path, "base.md")])
    _generation_append(kb_store.kb_path("busy"), [])        # tag the first generation
    monkeypatch.setattr(kb_store, "_append_index_into", _generation_append)
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            kb_store.add_to_kb("busy", [_doc(tmp_path, f"n{i}.md")])
            i += 1

    w = threading.Thread(target=writer)
    w.start()
    mixed, copies = 0, 0
    try:
        deadline = time.time() + 6
        while time.time() < deadline:
            dest = tmp_path / "caches" / f"c{copies}"
            kb_store.snapshot_kb(kb_store.kb_path("busy"), dest)
            idx = (dest / "default_kb_docs.faiss").read_bytes()[:32].decode()
            chunks = json.loads((dest / "default_kb_docs.json").read_text())[0]
            mixed += idx != chunks
            copies += 1
    finally:
        stop.set()
        w.join()
    assert copies > 20
    assert mixed == 0


def test_a_backup_stranded_by_a_crash_is_restored(store, tmp_path, monkeypatch):
    kb_store.create_kb("vault", [_doc(tmp_path, "base.md")])
    final = kb_store.kb_path("vault")
    final.rename(final.with_name(".bak_vault"))             # crashed between the two renames
    monkeypatch.setattr(kb_store, "_append_index_into", lambda t, n, *a, **k: (1, 1))
    kb_store.add_to_kb("vault", [_doc(tmp_path, "more.md")])
    assert {"base.md", "more.md"} <= set(kb_store.read_manifest(final)["sources"])
    assert not final.with_name(".bak_vault").exists()


def test_publishing_a_kb_does_not_delete_one_named_like_its_backup(store, tmp_path, monkeypatch):
    kb_store.create_kb("foo.bak", [_doc(tmp_path, "keep.md")])
    kb_store.create_kb("foo", [_doc(tmp_path, "base.md")])
    monkeypatch.setattr(kb_store, "_append_index_into", lambda t, n, *a, **k: (1, 1))
    kb_store.add_to_kb("foo", [_doc(tmp_path, "more.md")])
    assert kb_store.read_manifest(kb_store.kb_path("foo.bak"))["sources"] == ["keep.md"]
    assert [m["name"] for m in kb_store.list_kbs()] == ["foo", "foo.bak"]


def test_list_kbs_tells_a_live_build_from_a_dead_one(store, tmp_path, caplog):
    import logging as _logging
    from scilink.utils.file_lock import path_lock
    (store / ".staging_live").mkdir(parents=True)
    (store / ".staging_dead").mkdir(parents=True)
    with caplog.at_level(_logging.INFO, logger="scilink.knowledge.kb_store"):
        with path_lock(kb_store._lock_target("live")):
            kb_store.list_kbs()
    warnings = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any(".staging_dead" in w for w in warnings)
    assert not any(".staging_live" in w for w in warnings)
    assert any("being built by another process" in r.message for r in caplog.records)


def test_an_index_that_disagrees_with_its_chunks_is_not_used_for_dense_retrieval(tmp_path):
    import faiss
    import numpy as np
    from scilink.knowledge.knowledge_base import KnowledgeBase
    prefix = tmp_path / "default_kb_docs"
    idx = faiss.IndexFlatL2(4)
    idx.add(np.random.rand(2, 4).astype("float32"))
    faiss.write_index(idx, str(prefix.with_suffix(".faiss")))
    prefix.with_suffix(".json").write_text(json.dumps(
        [{"text": f"chunk {i}", "metadata": {}} for i in range(3)]))
    prefix.with_suffix(".sources.json").write_text("[]")
    kb = KnowledgeBase(embedding_model=None)
    assert kb.load(str(prefix.with_suffix(".faiss")), str(prefix.with_suffix(".json")),
                   sources_path=str(prefix.with_suffix(".sources.json")))
    assert "2 vectors for 3 chunks" in (kb._dense_disabled_reason or "")


# ── re-review follow-ups ──────────────────────────────────────────────────

def test_delete_also_removes_a_stranded_backup(store, tmp_path):
    kb_store.create_kb("gone", [_doc(tmp_path, "a.md")])
    final = kb_store.kb_path("gone")
    backup = final.with_name(".bak_gone")
    backup.mkdir()
    (backup / "manifest.json").write_text(json.dumps({"name": "gone", "sources": ["old.md"]}))
    kb_store.delete_kb("gone")
    assert not backup.exists()
    kb_store.create_kb("gone", [_doc(tmp_path, "fresh.md")])      # not resurrected, no FileExistsError
    assert kb_store.read_manifest(final)["sources"] == ["fresh.md"]


def test_snapshot_of_a_missing_kb_fails_fast_and_says_so(store, tmp_path):
    t0 = time.time()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        kb_store.snapshot_kb(store / "nope", tmp_path / "cache")
    empty = store / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no index files"):
        kb_store.snapshot_kb(empty, tmp_path / "cache")
    assert time.time() - t0 < 0.5


def test_snapshot_drops_files_the_source_lacks(store, tmp_path):
    kb_store.create_kb("b", [_doc(tmp_path, "b.md")])
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "default_kb_code.faiss").write_bytes(b"from KB a")          # an earlier attach
    kb_store.snapshot_kb(kb_store.kb_path("b"), cache)
    assert not (cache / "default_kb_code.faiss").exists()
    assert (cache / "default_kb_docs.faiss").exists()


def test_a_failed_attach_leaves_the_meta_and_its_planner_on_the_old_kb(store, tmp_path):
    from types import SimpleNamespace
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    broken = store / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(json.dumps({"name": "broken", "embedding_model": "m"}))
    old = tmp_path / "old_kb"
    rebinds = []
    child = SimpleNamespace(base_dir=tmp_path / "planning", knowledge_dir=old, _kb_store_manifest=None,
                            planner=SimpleNamespace(rebind_kb=rebinds.append))
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m.knowledge_dir, m._children, m.embedding_model = old, {"planning": child}, "m"
    m._shared_kb_candidate = None
    with pytest.raises(FileNotFoundError):
        m.attach_knowledge_dir("broken")
    assert m.knowledge_dir == old and child.knowledge_dir == old and rebinds == []
