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
