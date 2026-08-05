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
    assert out == {"status": "error", "message": "USE-MODE-SPECIFIC-TOOL"}
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


def test_backup_counter_never_clobbers(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("v1\n")
    edit(f, "v1", "v2", root=tmp_path)
    edit(f, "v2", "v3", root=tmp_path)
    baks = sorted(p.name for p in (tmp_path / "backups").iterdir())
    assert baks == ["doc.before_edit.md", "doc.before_edit2.md"]


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
