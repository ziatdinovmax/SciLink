"""Direct tests for the shared surgical-edit core (utils/file_edit).

The orchestrator-facing behavior is covered by test_edit_file_tool.py;
these pin the utility's own contract, including the parameters future
consumers depend on — notably the suffix policy that must admit
extensionless VASP inputs (POSCAR / INCAR) when the simulation
orchestrator adopts the tool.
"""

from pathlib import Path

import pytest

from scilink.utils.file_edit import (
    DEFAULT_EDITABLE_SUFFIXES, apply_surgical_edit, refresh_pdf_twin)


def edit(path, old, new, *, root, **kw):
    kw.setdefault("backup_dir", Path(root) / "backups")
    return apply_surgical_edit(path, old, new, root=root, **kw)


def test_success_shape_and_backup(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("alpha beta\n")
    out = edit(f, "beta", "gamma", root=tmp_path)
    assert out["status"] == "success" and out["replacements"] == 1
    assert f.read_text() == "alpha gamma\n"
    bak = Path(out["previous_version"])
    assert bak.parent == tmp_path / "backups"
    assert bak.read_text() == "alpha beta\n"


def test_guard_order_and_messages(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("x y x\n")
    outside = tmp_path.parent / "outside.md"
    assert "session directory" in edit(
        outside, "a", "b", root=tmp_path)["message"]
    assert "No such file" in edit(
        tmp_path / "nope.md", "a", "b", root=tmp_path)["message"]
    assert "non-empty" in edit(f, "", "b", root=tmp_path)["message"]
    assert "identical" in edit(f, "x", "x", root=tmp_path)["message"]
    assert "2 places" in edit(f, "x", "z", root=tmp_path)["message"]
    out = edit(f, "x", "z", root=tmp_path, replace_all=True)
    assert out["status"] == "success" and out["replacements"] == 2


def test_cap_uses_the_callers_routing_hint(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("short\n")
    out = edit(f, "short", "y" * 2001, root=tmp_path,
               too_large_message="USE-MODE-SPECIFIC-TOOL")
    assert out["status"] == "error"
    assert out["message"] == "USE-MODE-SPECIFIC-TOOL"   # verbatim, no prefix
    assert f.read_text() == "short\n"


def test_suffix_policy_is_configurable_for_extensionless_vasp(tmp_path):
    incar = tmp_path / "INCAR"
    incar.write_text("ENCUT = 400\n")
    out = edit(incar, "400", "520", root=tmp_path)
    assert out["status"] == "error"          # default policy refuses

    out = edit(incar, "400", "520", root=tmp_path,
               allowed_suffixes=DEFAULT_EDITABLE_SUFFIXES | {""})
    assert out["status"] == "success"
    assert incar.read_text() == "ENCUT = 520\n"


def test_chain_origin_backup_only(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("v1\n")
    o1 = edit(f, "v1", "v2", root=tmp_path)
    o2 = edit(f, "v2", "v3", root=tmp_path)
    baks = sorted(p.name for p in (tmp_path / "backups").iterdir())
    assert baks == ["doc.before_edit.md"]
    assert (tmp_path / "backups" / "doc.before_edit.md").read_text() == "v1\n"
    assert o1["backup_created"] is True and o2["backup_created"] is False


def test_snippet_batch_is_sequential_and_atomic(tmp_path):
    from scilink.utils.file_edit import apply_snippet_edits

    # sequential: edit 2 matches text PRODUCED by edit 1
    out = apply_snippet_edits("a b c", [
        {"old_text": "a b", "new_text": "X Y"},
        {"old_text": "Y c", "new_text": "Y Z"},
    ])
    assert out["status"] == "success"
    assert out["text"] == "X Y Z" and out["n_edits"] == 2

    # atomic: failure names the edit, nothing "applied"
    out = apply_snippet_edits("a b c", [
        {"old_text": "a", "new_text": "A"},
        {"old_text": "missing", "new_text": "x"},
    ])
    assert out["status"] == "error" and out["failed_edit"] == 2
    assert "IN ORDER" in out["message"]

    # per-edit guards carry the edit label
    out = apply_snippet_edits("a a", [{"old_text": "a", "new_text": "b"}])
    assert out["status"] == "error" and "2 places" in out["message"]
    out = apply_snippet_edits("a", [{"old_text": "a", "new_text": "b" * 2001}])
    assert out["status"] == "error" and out["failed_edit"] == 1
    assert apply_snippet_edits("a", [])["status"] == "error"

    # replace_all inside a batch
    out = apply_snippet_edits("x x x", [
        {"old_text": "x", "new_text": "y", "replace_all": True}])
    assert out["status"] == "success" and out["replacements"] == 3


def test_rename_moves_bytes_and_pdf_twin(tmp_path):
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf
    from scilink.utils.file_edit import rename_or_copy_file

    src = tmp_path / "technical_document.md"
    src.write_text("# Companion\n\nbody\n")
    markdown_to_pdf(src)
    dest = tmp_path / "spm_companion.md"

    out = rename_or_copy_file(src, dest, root=tmp_path)
    assert out["status"] == "success" and out["pdf_twin_followed"] is True
    assert not src.exists() and not src.with_suffix(".pdf").exists()
    assert dest.read_text() == "# Companion\n\nbody\n"
    assert dest.with_suffix(".pdf").exists()


def test_copy_keeps_the_original(tmp_path):
    from scilink.utils.file_edit import rename_or_copy_file
    src = tmp_path / "a.md"
    src.write_text("x\n")
    out = rename_or_copy_file(src, tmp_path / "b.md", root=tmp_path,
                              copy=True)
    assert out["status"] == "success" and out["copied"] is True
    assert src.exists() and (tmp_path / "b.md").read_text() == "x\n"


def test_rename_guards(tmp_path):
    from scilink.utils.file_edit import rename_or_copy_file
    src = tmp_path / "a.md"
    src.write_text("x\n")
    taken = tmp_path / "b.md"
    taken.write_text("occupied\n")

    assert "already exists" in rename_or_copy_file(
        src, taken, root=tmp_path)["message"]
    assert taken.read_text() == "occupied\n"          # never clobbered
    assert "session directory" in rename_or_copy_file(
        src, tmp_path.parent / "esc.md", root=tmp_path)["message"]
    assert "No such file" in rename_or_copy_file(
        tmp_path / "nope.md", tmp_path / "c.md", root=tmp_path)["message"]
    assert "identical" in rename_or_copy_file(
        src, src, root=tmp_path)["message"]
    assert src.read_text() == "x\n"                   # untouched throughout


def test_refresh_pdf_twin_states(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf

    doc = tmp_path / "r.md"
    doc.write_text("# R\n\nbody\n")
    assert refresh_pdf_twin(doc) == (False, None)          # no twin
    markdown_to_pdf(doc)
    assert refresh_pdf_twin(doc) == (True, None)           # refreshed

    import scilink.utils.md_to_pdf as mod
    monkeypatch.setattr(mod, "markdown_to_pdf",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    ok, err = refresh_pdf_twin(doc)
    assert ok is False and "boom" in err                   # failure reported
