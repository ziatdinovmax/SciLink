"""Markdown -> PDF, using dependencies SciLink already ships.

A white paper or campaign plan is written as markdown, but the thing a
scientist forwards to a sponsor is a PDF. Both halves of the conversion are
already installed: `markdown-it-py` parses the markdown, and PyMuPDF's
`Story` is a real HTML/CSS layout engine with automatic pagination.

`Story` supports a subset of HTML/CSS — block layout, tables, fonts, colours,
links. That covers documents; it is not a browser, so it is deliberately NOT
used for the richer `plan.html` reports.
"""

from __future__ import annotations

from pathlib import Path

# Letter page with 0.75" margins. Story paginates within whatever rect it is
# given; these are the page and the text frame inside it.
_PAGE = "letter"
_MARGIN = 54          # points (0.75")

# A runaway document must not spin forever if Story ever stops making
# progress. Far above any real plan: a 105 KB white paper is ~24 pages.
_MAX_PAGES = 500

# Print styling, not screen styling: dark text on white, with the app's
# purple for headings so the PDF is recognisably SciLink's.
DEFAULT_CSS = """
body { font-family: sans-serif; font-size: 10pt; color: #1a1a1a; }
h1 { font-size: 19pt; color: #1a1a2e; margin-bottom: 2pt; }
h2 { font-size: 14pt; color: #6200EE; margin-top: 14pt; }
h3 { font-size: 11.5pt; color: #333333; margin-top: 10pt; }
h4, h5, h6 { font-size: 10.5pt; color: #333333; }
p { line-height: 1.35; }
a { color: #0645AD; }
code { font-family: monospace; font-size: 9pt; background-color: #F2F2F2; }
pre { font-family: monospace; font-size: 8.5pt; background-color: #F7F7F7;
      color: #1a1a1a; }
blockquote { color: #444444; margin-left: 18pt; }
table { width: 100%; }
th { background-color: #EDEDED; text-align: left; padding: 4px;
     font-size: 9.5pt; }
td { padding: 4px; border-bottom: 1px solid #DDDDDD; font-size: 9.5pt; }
"""


class PdfConversionError(RuntimeError):
    """Raised when the conversion cannot run or produced nothing."""


def _render_html(text: str, title: str = "") -> str:
    from markdown_it import MarkdownIt

    md = (MarkdownIt("commonmark")
          .enable("table")
          .enable("strikethrough"))
    import html as _html
    return (f"<html><head><title>{_html.escape(title)}</title></head>"
            f"<body>{md.render(text)}</body></html>")


def markdown_to_pdf(md_path, pdf_path=None, title: str | None = None,
                    css: str | None = None) -> Path:
    """Convert a markdown FILE to PDF and return the PDF's path.

    `pdf_path` defaults to the markdown file with a .pdf suffix. `title`
    defaults to the file stem and only sets document metadata — it is not
    printed, so a document whose first line is already its title does not
    get it twice.
    """
    md_path = Path(md_path)
    out = Path(pdf_path) if pdf_path else md_path.with_suffix(".pdf")
    text = md_path.read_text(errors="replace")
    # base_dir lets relative image references (e.g. an embedded workflow
    # diagram PNG saved beside the markdown) resolve into the PDF.
    return _write_pdf(text, out, title or md_path.stem, css,
                      base_dir=md_path.parent)


def markdown_text_to_pdf(text: str, pdf_path, title: str = "",
                         css: str | None = None) -> Path:
    """Same conversion, for markdown held in memory."""
    return _write_pdf(text, Path(pdf_path), title, css)


# A figure at least this wide relative to its height cannot be read in a
# portrait text column, so it earns a landscape page of its own.
_WIDE_FIGURE_RATIO = 2.2


def _pull_wide_figures(text: str, base_dir):
    """Strip wide markdown images out of ``text``; return the remaining
    text and ``[(path, caption)]`` for placement on their own pages."""
    import re
    figures = []
    if not base_dir:
        return text, figures

    def take(m):
        alt, src = m.group(1), m.group(2).strip()
        p = Path(src)
        if not p.is_absolute():
            p = Path(base_dir) / src
        try:
            from PIL import Image
            with Image.open(p) as im:
                if im.height and im.width / im.height >= _WIDE_FIGURE_RATIO:
                    figures.append((p, alt))
                    return ""      # placed later, on its own page
        except Exception:  # noqa: BLE001 - not an image we can measure
            pass
        return m.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", take, text), figures


def _append_figure_pages(doc, figures, fitz) -> None:
    """One landscape page per wide figure, image fitted with a caption."""
    for path, caption in figures:
        page = doc.new_page(width=792, height=612)     # letter, landscape
        frame = fitz.Rect(_MARGIN, _MARGIN,
                          792 - _MARGIN, 612 - _MARGIN - 24)
        try:
            page.insert_image(frame, filename=str(path), keep_proportion=True)
        except Exception:  # noqa: BLE001 - a missing figure is not fatal
            continue
        if caption:
            page.insert_textbox(
                fitz.Rect(_MARGIN, 612 - _MARGIN - 20,
                          792 - _MARGIN, 612 - _MARGIN),
                caption, fontsize=9, fontname="helv", align=1)


def _write_pdf(text: str, out: Path, title: str, css: str | None,
               base_dir=None) -> Path:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:      # pragma: no cover - declared dependency
        raise PdfConversionError(
            "PDF export needs PyMuPDF: pip install pymupdf") from exc
    # A wide schematic squeezed into the text column is an inch tall and
    # unreadable. Pull those figures out of the flow and give each its own
    # landscape page after the text, where the full page width is theirs.
    text, wide_figures = _pull_wide_figures(text, base_dir)
    try:
        html = _render_html(text, title)
    except ImportError as exc:
        raise PdfConversionError(
            "PDF export needs markdown-it-py: pip install markdown-it-py"
        ) from exc

    import io

    archive = fitz.Archive(str(base_dir)) if base_dir else None
    story = fitz.Story(html=html, user_css=css or DEFAULT_CSS,
                       archive=archive)
    try:
        # Gives headings ids, so an in-document [jump](#section) resolves.
        story.add_header_ids()
    except Exception:  # noqa: BLE001 - anchors are a bonus, not the document
        pass

    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    page = fitz.paper_rect(_PAGE)
    frame = page + (_MARGIN, _MARGIN, -_MARGIN, -_MARGIN)

    # Link rects are collected DURING layout — a link is only a position once
    # the text has been placed, and a wrapped one occupies several rects.
    positions: list = []
    more, pages = True, 0
    while more and pages < _MAX_PAGES:
        device = writer.begin_page(page)
        more, _ = story.place(frame)
        # A lambda, not positions.append: PyMuPDF introspects __code__, which
        # a builtin method does not have.
        story.element_positions(lambda p: positions.append(p),
                                {"page_num": pages})
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()

    try:
        doc = fitz.Story.add_pdf_links(buf, positions)
    except Exception:  # noqa: BLE001
        # add_pdf_links refuses e.g. an anchor with no target. Clickable links
        # are worth having, but never at the cost of the document itself.
        doc = fitz.open("pdf", buf.getvalue())
    try:
        _append_figure_pages(doc, wide_figures, fitz)
    except Exception:  # noqa: BLE001 - never lose the document over a figure
        pass
    doc.save(str(out))

    if not out.exists() or out.stat().st_size == 0:
        raise PdfConversionError(f"PDF conversion produced nothing: {out}")
    return out
