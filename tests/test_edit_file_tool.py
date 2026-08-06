"""edit_file — surgical in-place document edits + PDF-twin refresh.

The planning agent's only routes to a changed document were save_file
(whole-file rewrite passed as one tool argument) and
write_technical_document(revise_path=...) (LLM re-emission of the whole
document). A one-line image-reference swap cost a full re-authoring, and
the white paper's exported PDF twin silently kept the old content — live,
a diagram swap updated white_paper.md while white_paper.pdf still carried
the overfit image, and the specialist had no tool that could fix it
without violating a content freeze.

edit_file is exact-snippet replacement (mechanical edits only, capped
small so it cannot become a piecewise document rewriter), and both it and
the revise_path branch deterministically re-export a .pdf sibling.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools


def make_tools(tmp_path, deleg_name="delegations/07_render_and_embed"):
    deleg = tmp_path / deleg_name
    deleg.mkdir(parents=True, exist_ok=True)
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        planner=SimpleNamespace(state={}, model=None, kb_docs=None,
                                generation_config=None,
                                _build_skill_context=lambda s: None),
        base_dir=tmp_path,
        _active_output_subdir=deleg,
        lit_agent=None,
    )
    cap = {}
    tools._register_tool = (
        lambda func, name, description, parameters, required=None:
        cap.update({name: func}))
    ot.OrchestratorTools._register_all_tools(tools)
    return tools, cap, deleg


# ------------------------------------------------------------ edit_file


def test_single_replacement_updates_file_and_keeps_backup(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    doc = tmp_path / "delegations" / "01_brainstorm" / "companion.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# Companion\n\n![Workflow](old_diagram.png)\n\nProse.\n")

    out = json.loads(cap["edit_file"](
        str(doc), "![Workflow](old_diagram.png)",
        "![Workflow](campaign_workflow.png)"))

    assert out["status"] == "success"
    assert out["replacements"] == 1
    assert out["pdf_refreshed"] is False          # no twin beside it
    text = doc.read_text()
    assert "campaign_workflow.png" in text and "old_diagram.png" not in text
    assert "Prose." in text                       # rest untouched
    # the delegation MAKING the edit keeps the replaced version
    bak = Path(out["previous_version"])
    assert bak.parent == deleg
    assert "old_diagram.png" in bak.read_text()


def test_relative_path_resolves_into_the_delegation_dir(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    (deleg / "notes.md").write_text("alpha beta\n")
    out = json.loads(cap["edit_file"]("notes.md", "beta", "gamma"))
    assert out["status"] == "success"
    assert (deleg / "notes.md").read_text() == "alpha gamma\n"


def test_missing_snippet_is_an_error_and_file_is_untouched(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("original\n")
    out = json.loads(cap["edit_file"](str(doc), "not present", "x"))
    assert out["status"] == "error"
    assert "not found" in out["message"]
    assert doc.read_text() == "original\n"
    assert not list(deleg.glob("*.before_edit*"))  # no backup for a no-op


def test_ambiguous_snippet_requires_replace_all(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("fig.png here, fig.png there\n")

    out = json.loads(cap["edit_file"](str(doc), "fig.png", "new.png"))
    assert out["status"] == "error"
    assert "2 places" in out["message"]
    assert doc.read_text() == "fig.png here, fig.png there\n"

    out = json.loads(cap["edit_file"](str(doc), "fig.png", "new.png",
                                      replace_all=True))
    assert out["status"] == "success" and out["replacements"] == 2
    assert doc.read_text() == "new.png here, new.png there\n"


def test_large_edits_are_routed_to_the_revision_tool(tmp_path):
    """The size cap is the guard that keeps edit_file from becoming a
    piecewise document rewriter dodging write_technical_document's
    whole-document length guard."""
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("short\n")
    out = json.loads(cap["edit_file"](str(doc), "short", "x" * 2001))
    assert out["status"] == "error"
    assert "write_technical_document" in out["message"]
    assert doc.read_text() == "short\n"


def test_sandbox_refuses_paths_outside_the_session(tmp_path):
    tools, cap, _ = make_tools(tmp_path / "session")
    outside = tmp_path / "elsewhere.md"
    outside.write_text("secret\n")
    out = json.loads(cap["edit_file"](str(outside), "secret", "x"))
    assert out["status"] == "error"
    assert "session directory" in out["message"]
    assert outside.read_text() == "secret\n"


def test_binary_formats_are_refused(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    png = deleg / "diagram.png"
    png.write_bytes(b"\x89PNG fake")
    out = json.loads(cap["edit_file"](str(png), "PNG", "JPG"))
    assert out["status"] == "error"
    assert ".png" in out["message"] or "editable" in out["message"]


def test_degenerate_inputs_are_refused(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("text\n")
    assert json.loads(cap["edit_file"](str(doc), "", "x"))["status"] == "error"
    assert json.loads(
        cap["edit_file"](str(doc), "text", "text"))["status"] == "error"
    assert json.loads(
        cap["edit_file"](str(deleg / "nope.md"), "a", "b"))["status"] == "error"


def test_edit_chain_keeps_only_the_chain_origin_backup(tmp_path):
    """One backup per file per delegation: the state before the FIRST
    edit. Live, an 18-edit chain filed 18 near-identical backups, each
    recorded and re-embedded by the UI."""
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("v1\n")
    out1 = json.loads(cap["edit_file"](str(doc), "v1", "v2"))
    out2 = json.loads(cap["edit_file"](str(doc), "v2", "v3"))
    baks = sorted(p.name for p in deleg.glob("doc.before_edit*"))
    assert baks == ["doc.before_edit.md"]
    assert (deleg / "doc.before_edit.md").read_text() == "v1\n"
    assert out1["backup_created"] is True
    assert out2["backup_created"] is False
    assert out2["previous_version"] == out1["previous_version"]


def test_batched_edits_apply_atomically_in_one_call(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "doc.md"
    doc.write_text("alpha one. beta two. gamma three.\n")

    out = json.loads(cap["edit_file"](str(doc), edits=[
        {"old_text": "alpha one", "new_text": "alpha 1"},
        {"old_text": "beta two", "new_text": "beta 2"},
        {"old_text": "gamma three", "new_text": "gamma 3"},
    ]))
    assert out["status"] == "success" and out["n_edits"] == 3
    assert doc.read_text() == "alpha 1. beta 2. gamma 3.\n"
    assert len(list(deleg.glob("doc.before_edit*"))) == 1

    # a failing edit in the middle aborts the WHOLE batch
    before = doc.read_text()
    out = json.loads(cap["edit_file"](str(doc), edits=[
        {"old_text": "alpha 1", "new_text": "alpha I"},
        {"old_text": "NOT PRESENT", "new_text": "x"},
    ]))
    assert out["status"] == "error" and out["failed_edit"] == 2
    assert doc.read_text() == before

    # neither form provided -> clear error
    out = json.loads(cap["edit_file"](str(doc)))
    assert out["status"] == "error" and "edits list" in out["message"]


# --------------------------------------------------------- rename_file


def test_rename_is_byte_exact_and_stays_in_place(tmp_path):
    """The live failure this replaces: no rename tool, so the agent
    reconstructed a 30 KB document via save_file + append_file chunks,
    dropping content and nesting it in a phantom directory."""
    tools, cap, deleg = make_tools(tmp_path)
    d08 = tmp_path / "delegations" / "08_brainstorm"
    d08.mkdir(parents=True)
    doc = d08 / "technical_document.md"
    doc.write_text("# Companion\n\nfull body, every byte kept\n")

    out = json.loads(cap["rename_file"](
        str(doc), "spm_companion.md", deliverable=True, title="Companion"))
    assert out["status"] == "success"
    dest = Path(out["path"])
    assert dest == d08 / "spm_companion.md"       # same folder, new identity
    assert dest.read_text() == "# Companion\n\nfull body, every byte kept\n"
    assert not doc.exists()

    # directories in new_name are stripped, not nested (the phantom-folder
    # failure came from a path passed as a name)
    src2 = deleg / "notes.md"
    src2.write_text("n\n")
    out2 = json.loads(cap["rename_file"](str(src2), "sub/dir/final.md"))
    assert Path(out2["path"]) == deleg / "final.md"


def test_rename_refuses_to_clobber(tmp_path):
    tools, cap, deleg = make_tools(tmp_path)
    (deleg / "a.md").write_text("a\n")
    (deleg / "b.md").write_text("b\n")
    out = json.loads(cap["rename_file"](str(deleg / "a.md"), "b.md"))
    assert out["status"] == "error"
    assert (deleg / "b.md").read_text() == "b\n"


# ----------------------------------------------------- PDF twin refresh


def _pdf_text(pdf_path):
    fitz = pytest.importorskip("fitz")
    with fitz.open(pdf_path) as doc:
        return "".join(page.get_text() for page in doc)


def test_edit_refreshes_a_stale_pdf_twin(tmp_path):
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf

    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "white_paper.md"
    doc.write_text("# Paper\n\nThe OLDTOKEN result.\n")
    markdown_to_pdf(doc)                          # the forwarded export
    assert "OLDTOKEN" in _pdf_text(doc.with_suffix(".pdf"))

    out = json.loads(cap["edit_file"](str(doc), "OLDTOKEN", "NEWTOKEN"))
    assert out["status"] == "success" and out["pdf_refreshed"] is True
    pdf = _pdf_text(doc.with_suffix(".pdf"))
    assert "NEWTOKEN" in pdf and "OLDTOKEN" not in pdf


def test_pdf_failure_never_loses_the_edit(tmp_path, monkeypatch):
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf

    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "paper.md"
    doc.write_text("# P\n\nOLD.\n")
    markdown_to_pdf(doc)

    import scilink.utils.md_to_pdf as mod

    def boom(*a, **k):
        raise RuntimeError("no renderer")

    monkeypatch.setattr(mod, "markdown_to_pdf", boom)
    out = json.loads(cap["edit_file"](str(doc), "OLD.", "NEW."))
    assert out["status"] == "success" and out["pdf_refreshed"] is False
    assert "NEW." in doc.read_text()


def test_revision_branch_refreshes_the_pdf_twin(tmp_path, monkeypatch):
    """write_technical_document(revise_path=...) rewrote the .md and left
    the exported PDF serving pre-revision content (live)."""
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf

    tools, cap, deleg = make_tools(tmp_path)
    doc = tmp_path / "delegations" / "04_revise" / "white_paper.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("# White Paper\n\nBody with OLDIMAGE reference and "
                   "enough prose to clear the revision length guard.\n")
    markdown_to_pdf(doc)
    assert "OLDIMAGE" in _pdf_text(doc.with_suffix(".pdf"))

    revised = [{"heading": "Body",
                "body": ("Body with NEWIMAGE reference and enough prose "
                         "to clear the revision length guard.")}]
    monkeypatch.setattr(ot, "author_technical_document",
                        lambda **kw: {"sections": revised})
    out = json.loads(cap["write_technical_document"](
        request="swap the image reference only",
        revise_path=str(doc), use_literature=False))

    assert out["status"] == "success"
    assert out["revised_in_place"] is True
    assert out["pdf_refreshed"] is True
    assert "NEWIMAGE" in doc.read_text()
    pdf = _pdf_text(doc.with_suffix(".pdf"))
    assert "NEWIMAGE" in pdf and "OLDIMAGE" not in pdf


def test_save_file_overwrite_refreshes_the_pdf_twin(tmp_path):
    """The invariant holds on EVERY markdown write path: live, the agent
    rewrote a document via save_file (not the revision tool) and the
    exported PDF kept serving the old content."""
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf

    tools, cap, deleg = make_tools(tmp_path)
    doc = deleg / "report.md"
    doc.write_text("# R\n\nOLDBODY.\n")
    markdown_to_pdf(doc)

    out = json.loads(cap["save_file"]("report.md", "# R\n\nNEWBODY.\n"))
    assert out["status"] == "success" and out["pdf_refreshed"] is True
    assert "NEWBODY" in _pdf_text(doc.with_suffix(".pdf"))

    out2 = json.loads(cap["save_file"]("fresh_notes.md", "no twin\n"))
    assert out2["status"] == "success" and out2["pdf_refreshed"] is False


def test_fresh_document_reports_no_pdf_refresh(tmp_path, monkeypatch):
    tools, cap, deleg = make_tools(tmp_path)
    monkeypatch.setattr(ot, "author_technical_document",
                        lambda **kw: {"sections": [
                            {"heading": "H", "body": "C"}]})
    monkeypatch.setattr(OrchestratorTools, "_maybe_embed_workflow_diagram",
                        lambda self, text, out_dir, stem=None: text)
    out = json.loads(cap["write_technical_document"](
        request="write a memo", filename="memo.md", use_literature=False))
    assert out["status"] == "success"
    assert out["pdf_refreshed"] is False
