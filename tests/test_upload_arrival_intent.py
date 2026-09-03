"""Offline tests for #114: sequential uploads are no longer silently merged
into a series — the UI records arrival grouping and the dispatch prompt
hands the series-vs-independent decision to the orchestrator.

  python -m pytest tests/test_upload_arrival_intent.py -v
"""
import types
from pathlib import Path

from scilink.ui.components.chat_uploads import build_analyze_prompt
from scilink.ui.components import sidebar as sb
from scilink.ui.components.sidebar import split_upload_arrivals


def _f(name):
    return types.SimpleNamespace(name=name,
                                 getvalue=lambda: f"data:{name}".encode())


# ------------------------------------------------- split_upload_arrivals

def test_first_batch_defines_membership():
    state = {}
    first, late = split_upload_arrivals([_f("a.txt"), _f("b.txt")], "d",
                                        state)
    assert [x.name for x in first] == ["a.txt", "b.txt"] and late == []
    # Same widget value on a later rerun: still all first, stable.
    first, late = split_upload_arrivals([_f("a.txt"), _f("b.txt")], "d",
                                        state)
    assert [x.name for x in first] == ["a.txt", "b.txt"] and late == []


def test_late_addition_is_partitioned_out():
    state = {}
    split_upload_arrivals([_f("a.txt")], "d", state)
    first, late = split_upload_arrivals([_f("a.txt"), _f("b.txt")], "d",
                                        state)
    assert [x.name for x in first] == ["a.txt"]
    assert [x.name for x in late] == ["b.txt"]
    # Stable across further reruns and further additions.
    first, late = split_upload_arrivals(
        [_f("a.txt"), _f("b.txt"), _f("c.txt")], "d", state)
    assert [x.name for x in late] == ["b.txt", "c.txt"]


def test_clearing_the_uploader_resets():
    state = {}
    split_upload_arrivals([_f("a.txt")], "d", state)
    split_upload_arrivals([], "d", state)
    first, late = split_upload_arrivals([_f("b.txt"), _f("c.txt")], "d",
                                        state)
    assert [x.name for x in first] == ["b.txt", "c.txt"] and late == []


# ------------------------------------------------- build_analyze_prompt

def test_single_file_prompt_unchanged():
    assert build_analyze_prompt("/u/x.txt") == (
        "I uploaded a data file at `/u/x.txt`. Please examine it.")
    assert build_analyze_prompt("/u/x.txt", meta_path="/u/m.json") == (
        "I uploaded a data file at `/u/x.txt` and a metadata file at "
        "`/u/m.json`. Please examine the data and load the metadata.")


def test_batch_prompt_asks_for_series_confirmation():
    p = build_analyze_prompt("/u/series", batch=True)
    assert "uploaded together as one batch" in p
    assert "confirm they actually form one series" in p
    # Composes with the metadata variants too.
    assert "confirm they actually form one series" in build_analyze_prompt(
        "/u/series", meta_path="/u/m.json", batch=True)
    assert "confirm they actually form one series" in build_analyze_prompt(
        "/u/series", has_sidecars=True, batch=True)


def test_late_paths_prompt_lists_arrival_groups():
    p = build_analyze_prompt("/u/b.txt", first_path="/u/a.txt",
                             late_paths=["/u/b.txt"])
    assert "- uploaded first: `/u/a.txt`" in p
    assert "- added later: `/u/b.txt`" in p
    assert "likely independent datasets" in p
    assert "ask me if it is genuinely ambiguous" in p
    p = build_analyze_prompt("/u/b.txt", meta_path="/u/m.json",
                             first_path="/u/a.txt", late_paths=["/u/b.txt"])
    assert "judge which data it describes" in p


# --------------------------------------- file layout through save helpers

class _State(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, k, v):
        self[k] = v


def _fake_st(tmp_path):
    return types.SimpleNamespace(
        session_state=_State(session_dir=str(tmp_path),
                             _processed_uploads=set()),
        sidebar=types.SimpleNamespace(success=lambda *a, **k: None),
    )


def test_sequential_uploads_not_merged_into_series(tmp_path, monkeypatch):
    """The #114 repro, against the real save helpers: file A analyzed, file
    B added later — B must land flat in uploads/, and no series/ appears."""
    fake = _fake_st(tmp_path)
    monkeypatch.setattr(sb, "st", fake)
    state = {}

    first, late = split_upload_arrivals([_f("A.txt")], "d", state)
    sb.save_upload(first[0], "data", auto_dispatch=False)

    first, late = split_upload_arrivals([_f("A.txt"), _f("B.txt")], "d",
                                        state)
    for f in late:
        sb.save_upload(f, "data", auto_dispatch=False)

    uploads = tmp_path / "uploads"
    assert (uploads / "A.txt").exists() and (uploads / "B.txt").exists()
    assert not (uploads / "series").exists()


def test_true_batch_still_becomes_series(tmp_path, monkeypatch):
    fake = _fake_st(tmp_path)
    monkeypatch.setattr(sb, "st", fake)
    state = {}
    first, late = split_upload_arrivals([_f("A.txt"), _f("B.txt")], "d",
                                        state)
    assert not late
    out = sb.save_upload_batch(first, "data", auto_dispatch=False)
    series = tmp_path / "uploads" / "series"
    assert Path(out) == series
    assert (series / "A.txt").exists() and (series / "B.txt").exists()
