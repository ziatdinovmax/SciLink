"""rename_file: renames stay in place, copies land in the current folder.

Under the meta every delegation writes into its own directory, so a
consolidating delegation that wants to embed a figure an earlier one
produced had no way to bring it alongside its document: rename_file forced
the destination into the SOURCE's folder, so the call collapsed to a
self-rename and came back only with "path and new path are identical".
Live, the agent burned two turns inferring the constraint before falling
back to a relative path.

copy=true now lands the file in the CURRENT output directory — never a
path the agent picks, so a copy still cannot write into a sibling
delegation. Renames are untouched: identity, not location.
"""

import json
from pathlib import Path

import pytest

from tests.test_edit_file_tool import make_tools


def _figure(tmp_path, folder="delegations/01_design", name="campaign_workflow.png"):
    p = tmp_path / folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"figure-bytes" * 64)
    return p


# ------------------------------------------------- copy into current folder


def test_copy_brings_a_sibling_delegations_figure_here(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)

    out = json.loads(cap["rename_file"](str(fig), "campaign_workflow.png",
                                        copy=True))

    assert out["status"] == "success", out
    landed = Path(out["path"])
    assert landed == deleg / "campaign_workflow.png", "copy did not land here"
    assert landed.read_bytes() == fig.read_bytes(), "copy is not byte-exact"
    assert fig.exists(), "copy destroyed the original"


def test_copy_lets_a_local_embed_resolve(tmp_path):
    """The actual point: after the copy, a bare filename reference in the
    consolidating document resolves in its own folder."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)
    cap["rename_file"](str(fig), "campaign_workflow.png", copy=True)

    report = deleg / "report.md"
    report.write_text("# Report\n\n![Workflow](campaign_workflow.png)\n")
    ref = report.parent / "campaign_workflow.png"
    assert ref.is_file(), "embedded reference does not resolve beside the doc"


def test_copy_cannot_write_into_a_sibling_delegation(tmp_path):
    """Destination is this delegation's folder or nowhere — directory
    components in new_name must not steer it elsewhere."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)
    victim = tmp_path / "delegations" / "02_other"
    victim.mkdir(parents=True)

    out = json.loads(cap["rename_file"](
        str(fig), "../02_other/stolen.png", copy=True))

    assert out["status"] == "success", out
    assert Path(out["path"]) == deleg / "stolen.png"
    assert not (victim / "stolen.png").exists(), "copy escaped into a sibling"


def test_copy_refuses_to_clobber(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)
    (deleg / "campaign_workflow.png").write_bytes(b"mine, different")

    out = json.loads(cap["rename_file"](str(fig), "campaign_workflow.png",
                                        copy=True))

    assert out["status"] == "error"
    assert (deleg / "campaign_workflow.png").read_bytes() == b"mine, different"


def test_copy_of_a_file_already_here_under_the_same_name_is_refused(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path, "delegations/09_consolidate")

    out = json.loads(cap["rename_file"](str(fig), "campaign_workflow.png",
                                        copy=True))

    assert out["status"] == "error"
    assert "already this file's name" in out["message"]
    assert fig.exists()


def test_copy_of_a_local_file_under_a_new_name_still_duplicates_here(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path, "delegations/09_consolidate")

    out = json.loads(cap["rename_file"](str(fig), "figure_2.png", copy=True))

    assert out["status"] == "success"
    assert Path(out["path"]) == deleg / "figure_2.png"
    assert fig.exists()


# ------------------------------------------------------ renames unchanged


def test_rename_of_a_foreign_file_still_stays_in_its_own_folder(tmp_path):
    """The pre-existing guarantee: a rename changes identity, not
    location, even when the file lives in another delegation."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)

    out = json.loads(cap["rename_file"](str(fig), "renamed.png"))

    assert out["status"] == "success"
    assert Path(out["path"]) == fig.parent / "renamed.png"
    assert not (deleg / "renamed.png").exists(), "rename leaked into this folder"
    assert not fig.exists()


def test_rename_in_the_current_folder_is_unchanged(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    doc = deleg / "draft.md"
    doc.write_text("# Draft\n\nbody\n")

    out = json.loads(cap["rename_file"](str(doc), "final.md"))

    assert out["status"] == "success"
    assert Path(out["path"]) == deleg / "final.md"
    assert (deleg / "final.md").read_text() == "# Draft\n\nbody\n"
    assert not doc.exists()


# ----------------------------------------------- the error is a signpost


def test_self_rename_of_a_foreign_file_names_the_fix(tmp_path):
    """The live dead-end: the agent wanted the figure HERE, asked for a
    rename, and learned only that two paths matched."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    fig = _figure(tmp_path)
    before = fig.read_bytes()

    out = json.loads(cap["rename_file"](str(fig), "campaign_workflow.png"))

    assert out["status"] == "error"
    assert "copy=true" in out["message"], "error does not name the fix"
    assert "relative path" in out["message"], "error omits the other option"
    assert fig.read_bytes() == before, "failed call touched the file"
    assert not (deleg / "campaign_workflow.png").exists()


def test_self_rename_in_the_current_folder_says_nothing_to_do(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    doc = deleg / "report.md"
    doc.write_text("body\n")

    out = json.loads(cap["rename_file"](str(doc), "report.md"))

    assert out["status"] == "error"
    assert "already this file's name" in out["message"]
    assert "copy=true" not in out["message"], "copy is not the fix here"
    assert doc.read_text() == "body\n"


# ------------------------------------------------------------ PDF twins


def test_pdf_twin_follows_a_cross_folder_copy(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = tmp_path / "delegations" / "01_design" / "memo.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("# Memo\n\nbody\n")
    src.with_suffix(".pdf").write_bytes(b"%PDF-1.4 twin")

    out = json.loads(cap["rename_file"](str(src), "memo.md", copy=True))

    assert out["status"] == "success"
    assert out["pdf_twin_followed"] is True
    assert (deleg / "memo.pdf").read_bytes() == b"%PDF-1.4 twin"
    assert src.with_suffix(".pdf").exists(), "copy moved the twin"


# ------------------------------------------------------------ edge cases


def test_missing_source_still_errors(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    out = json.loads(cap["rename_file"](str(tmp_path / "nope.png"), "x.png",
                                        copy=True))
    assert out["status"] == "error"
    assert "No such file" in out["message"]


def test_source_outside_the_session_is_refused(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    outside = tmp_path.parent / "outside_asset.png"
    outside.write_bytes(b"not ours")
    try:
        out = json.loads(cap["rename_file"](str(outside), "asset.png",
                                            copy=True))
        assert out["status"] == "error"
        assert "session directory" in out["message"]
        assert not (deleg / "asset.png").exists()
    finally:
        outside.unlink()


def test_relative_source_resolves_against_the_current_folder(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    (deleg / "local.png").write_bytes(b"local")

    out = json.loads(cap["rename_file"]("local.png", "copy.png", copy=True))

    assert out["status"] == "success"
    assert Path(out["path"]) == deleg / "copy.png"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
