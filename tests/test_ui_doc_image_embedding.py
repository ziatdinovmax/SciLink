"""A document's relative figure refs must survive every display surface.

Generated white papers reference figures saved beside them
(``![...](campaign_workflow.png)``). Streamlit's markdown renderer treats
image targets as web URLs, so the chat embed and the file-preview pane
dropped such figures, and the file viewer's PDF button regenerated the PDF
from in-memory text in a temp dir where the figure could not resolve.
"""

import base64
from pathlib import Path

import pytest

from scilink.ui.md_images import inline_local_images, pdf_twin

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg==")  # 1x1 px


@pytest.fixture
def doc_dir(tmp_path):
    (tmp_path / "campaign_workflow.png").write_bytes(PNG)
    return tmp_path


def test_relative_ref_is_inlined(doc_dir):
    out = inline_local_images("![d](campaign_workflow.png)", doc_dir)
    assert out.startswith("![d](data:image/png;base64,")
    payload = out.split("base64,")[1].rstrip(")")
    assert base64.b64decode(payload) == PNG


def test_absolute_ref_is_inlined(doc_dir):
    p = doc_dir / "campaign_workflow.png"
    out = inline_local_images(f"![d]({p})", doc_dir)
    assert "data:image/png;base64," in out


def test_titled_ref_keeps_its_title(doc_dir):
    out = inline_local_images('![d](campaign_workflow.png "The workflow")',
                              doc_dir)
    assert out.endswith('"The workflow")')
    assert "data:image/png;base64," in out


def test_remote_and_data_refs_pass_through(doc_dir):
    for target in ("https://x.org/a.png", "http://x.org/a.png",
                   "data:image/png;base64,AAAA"):
        text = f"![d]({target})"
        assert inline_local_images(text, doc_dir) == text


def test_missing_file_and_unknown_suffix_pass_through(doc_dir):
    for text in ("![d](nope.png)", "![d](campaign_workflow.txt)"):
        assert inline_local_images(text, doc_dir) == text


def test_oversize_figure_stays_a_link(doc_dir):
    assert inline_local_images("![d](campaign_workflow.png)", doc_dir,
                               max_bytes=10) == "![d](campaign_workflow.png)"


def test_surrounding_prose_untouched(doc_dir):
    text = "before\n\n![d](campaign_workflow.png)\n\nafter"
    out = inline_local_images(text, doc_dir)
    assert out.startswith("before\n\n") and out.endswith("\n\nafter")


def test_pdf_twin_prefers_fresh_sibling(tmp_path):
    md = tmp_path / "white_paper.md"
    md.write_text("x")
    twin = tmp_path / "white_paper.pdf"
    twin.write_bytes(b"%PDF")
    assert pdf_twin(md) == twin


def test_pdf_twin_rejects_stale_sibling(tmp_path):
    import os
    twin = tmp_path / "white_paper.pdf"
    twin.write_bytes(b"%PDF")
    os.utime(twin, (1, 1))
    md = tmp_path / "white_paper.md"
    md.write_text("revised later")
    assert pdf_twin(md) is None


def test_pdf_twin_none_when_missing(tmp_path):
    md = tmp_path / "white_paper.md"
    md.write_text("x")
    assert pdf_twin(md) is None


# ── the display surfaces actually use the helper ─────────────────────

def test_chat_embed_inlines_images():
    src = Path("scilink/ui/app.py").read_text()
    i = src.find("md_reports")
    assert "inline_local_images" in src[i:i + 1200]


def test_file_preview_inlines_images_and_prefers_twin():
    src = Path("scilink/ui/components/file_viewer.py").read_text()
    assert "inline_local_images" in src
    assert "pdf_twin" in src
    # the in-memory text converter (no base_dir) must be gone
    assert "markdown_text_to_pdf" not in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
