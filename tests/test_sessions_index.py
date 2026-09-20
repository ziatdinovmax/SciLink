"""The central session index: register, list across folders, resolve an
id from anywhere, prune gone directories."""

import json
from pathlib import Path

from scilink import sessions as S


def _mk(d: Path, checkpoint=True):
    d.mkdir(parents=True)
    if checkpoint:
        (d / "checkpoint.json").write_text(json.dumps({"analysis_results": []}))
    return d


def test_register_list_and_resolve_across_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    a = _mk(tmp_path / "projA" / "meta_session_20260920_100000")
    b = _mk(tmp_path / "projB" / "meta_session_20260920_110000")
    S.register_session(a, "meta", launcher_cwd=a.parent)
    S.register_session(b, "meta", launcher_cwd=b.parent)
    monkeypatch.chdir(tmp_path / "projC")  if (tmp_path / "projC").mkdir() is None else None
    listed = S.list_sessions("meta")
    assert [e["id"] for e in listed] == [b.name, a.name]           # newest first
    assert listed[0]["folder"] == str(b.parent)
    assert S.resolve_session(a.name, "meta") == a                  # by id, from another folder
    assert S.resolve_session(str(b), "meta") == b                  # by path
    assert S.resolve_session("nope", "meta") is None
    assert S.list_sessions("plan") == []                           # mode-filtered


def test_index_prunes_gone_directories_and_keeps_local_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    gone = _mk(tmp_path / "gone" / "meta_session_20260920_090000")
    S.register_session(gone, "meta")
    import shutil; shutil.rmtree(gone)
    local = _mk(tmp_path / "here" / "meta_session_20260920_120000")   # never registered
    listed = S.list_sessions("meta", root=tmp_path / "here")
    assert [e["id"] for e in listed] == [local.name]
    assert str(gone) not in S.index_path().read_text()


def test_touch_updates_name_and_time(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    d = _mk(tmp_path / "meta_session_20260920_130000")
    rec = S.register_session(d, "meta")
    (d / "session_meta.json").write_text(json.dumps({"name": "Grains", "named_by": "user"}))
    S.touch_session(d)
    e = S.list_sessions("meta")[0]
    assert e["label"].startswith("Grains") and e["updated"] >= rec["updated"]
