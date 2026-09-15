"""Markdown -> PDF export.

The white paper is authored as markdown and forwarded as a PDF, so what
matters is that the CONTENT survives the conversion — a PDF that renders but
drops the tables is worse than no PDF.
"""

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("markdown_it")

from scilink.utils.md_to_pdf import (            # noqa: E402
    PdfConversionError, markdown_text_to_pdf, markdown_to_pdf)

DOC = """# Campaign Title

## Executive Summary

Prose with **bold**, *italic* and `inline_code`.

| Phase | Duration | Cost |
|---|---|---|
| Screening | 3 mo | $120k |
| Validation | 6 mo | $340k |

1. First step
2. Second step
   - nested detail

> A block quote.

```python
def objective(x):
    return -(x - 2) ** 2
```

A [link](https://example.org) at the end.
"""


def _text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(p.get_text() for p in doc)


def test_every_kind_of_content_survives(tmp_path):
    md = tmp_path / "white_paper.md"
    md.write_text(DOC)
    out = markdown_to_pdf(md)

    body = _text(out)
    for expected in ("Campaign Title", "Executive Summary", "bold", "italic",
                     "inline_code", "Screening", "$340k", "First step",
                     "nested detail", "block quote", "objective", "link"):
        assert expected in body, f"{expected!r} was dropped"


def test_markdown_syntax_is_rendered_not_printed(tmp_path):
    md = tmp_path / "p.md"
    md.write_text(DOC)
    body = _text(markdown_to_pdf(md))

    assert "# Campaign" not in body, "heading marker printed literally"
    assert "**bold**" not in body
    assert "|---|" not in body, "table divider printed literally"


def test_output_defaults_beside_the_source(tmp_path):
    md = tmp_path / "plan.md"
    md.write_text("# Hi\n")
    assert markdown_to_pdf(md) == tmp_path / "plan.pdf"


def test_explicit_destination_and_nested_dirs(tmp_path):
    md = tmp_path / "plan.md"
    md.write_text("# Hi\n")
    dest = tmp_path / "sub" / "dir" / "out.pdf"
    assert markdown_to_pdf(md, dest) == dest and dest.exists()


def test_long_documents_paginate(tmp_path):
    md = tmp_path / "long.md"
    md.write_text("# T\n\n" + ("Paragraph of body text. " * 40 + "\n\n") * 40)
    assert fitz.open(markdown_to_pdf(md)).page_count > 1


def test_text_is_extractable_not_an_image(tmp_path):
    """A rasterised PDF would defeat search, copy-paste and accessibility."""
    md = tmp_path / "p.md"
    md.write_text(DOC)
    doc = fitz.open(markdown_to_pdf(md))
    assert len(doc[0].get_text().strip()) > 200
    assert doc[0].get_images() == []


def test_links_survive_as_real_links(tmp_path):
    md = tmp_path / "p.md"
    md.write_text("See [the docs](https://example.org/guide).\n")
    doc = fitz.open(markdown_to_pdf(md))
    assert any("example.org" in (l.get("uri") or "")
               for l in doc[0].get_links())


def test_empty_and_whitespace_documents_do_not_raise(tmp_path):
    for body in ("", "\n\n   \n"):
        out = markdown_text_to_pdf(body, tmp_path / "e.pdf")
        assert out.exists() and out.stat().st_size > 0


def test_in_memory_variant_matches(tmp_path):
    md = tmp_path / "p.md"
    md.write_text(DOC)
    a = _text(markdown_to_pdf(md, tmp_path / "a.pdf"))
    b = _text(markdown_text_to_pdf(DOC, tmp_path / "b.pdf"))
    assert a == b


def test_html_in_the_markdown_cannot_break_the_title(tmp_path):
    out = markdown_text_to_pdf("# Hi\n", tmp_path / "t.pdf",
                               title='</title><script>x</script>')
    assert out.exists()
    assert "script" not in _text(out)


def test_a_broken_converter_reports_rather_than_writing_nothing(
        tmp_path, monkeypatch):
    import scilink.utils.md_to_pdf as m
    monkeypatch.setattr(m, "_render_html",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError()))
    with pytest.raises(PdfConversionError, match="markdown-it-py"):
        markdown_text_to_pdf("# Hi\n", tmp_path / "x.pdf")


def test_the_white_paper_gets_a_pdf_twin(tmp_path):
    """The white paper is the document that gets forwarded, so it ships as
    both. The markdown stays the single starred deliverable."""
    from types import SimpleNamespace
    from scilink.agents.planning_agents.orchestrator_tools import (
        OrchestratorTools)
    from scilink.agents.planning_agents.user_interface import load_deliverables

    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        base_dir=tmp_path, _active_output_subdir=None,
        planner=SimpleNamespace(
            state={"current_plan": {"campaign_id": 1,
                                    "literature_search": "refs"}},
            generate_white_paper=lambda audience_context=None: DOC))

    out = tools._write_white_paper()

    assert out.endswith("white_paper.md")
    pdf = tmp_path / "white_paper.pdf"
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"
    assert "Screening" in _text(pdf)

    entries = {e["title"]: e for e in load_deliverables(tmp_path)}
    assert entries["White paper"]["deliverable"] is True
    assert entries["White paper PDF"]["deliverable"] is False


def test_a_pdf_failure_never_costs_the_white_paper(tmp_path, monkeypatch,
                                                   capsys):
    from types import SimpleNamespace
    import scilink.utils.md_to_pdf as m
    from scilink.agents.planning_agents.orchestrator_tools import (
        OrchestratorTools)

    monkeypatch.setattr(m, "markdown_to_pdf", lambda *a, **k: 1 / 0)
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        base_dir=tmp_path, _active_output_subdir=None,
        planner=SimpleNamespace(
            state={"current_plan": {"campaign_id": 1,
                                    "literature_search": "refs"}},
            generate_white_paper=lambda audience_context=None: DOC))

    out = tools._write_white_paper()
    assert Path(out).read_text() == DOC
    assert "PDF version unavailable" in capsys.readouterr().out
