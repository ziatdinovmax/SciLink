"""File Explorer markdown preview.

Reports and white papers were rendered as raw source in the preview pane —
literal `#` headings and pipe-soup tables. These pin the rendered/source
behaviour without a browser, by standing a recording double in for streamlit.
"""

import contextlib
import re

import pytest

from scilink.ui.components import file_viewer


class _Recorder:
    """Minimal streamlit stand-in: records calls, returns the toggle value."""

    def __init__(self, view="Rendered"):
        self._view = view
        self.calls = []

    def __getattr__(self, name):
        def _call(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "segmented_control":
                return self._view
            if name == "container":
                return contextlib.nullcontext()
            if name == "columns":
                n = a[0] if a else 1
                n = n if isinstance(n, int) else len(n)
                return [contextlib.nullcontext() for _ in range(n)]
            return None
        return _call

    def labels(self, kind):
        return [a[0] for name, a, _ in self.calls if name == kind]

    def kinds(self):
        return [c[0] for c in self.calls]

    def doc_markdown(self):
        """Markdown calls that render DOCUMENT content, not injected CSS."""
        return [a[0] for name, a, kw in self.calls
                if name == "markdown" and not kw.get("unsafe_allow_html")]

    def css(self):
        return "\n".join(a[0] for name, a, kw in self.calls
                         if name == "markdown" and kw.get("unsafe_allow_html"))

    def arg(self, kind):
        for name, a, _ in self.calls:
            if name == kind:
                return a[0]
        raise AssertionError(f"{kind} was never called")


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(file_viewer, "st", r)
    return r


def _md(tmp_path, body, name="report.md"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_markdown_is_rendered_not_dumped_as_source(rec, tmp_path):
    body = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    file_viewer.render_file_preview(_md(tmp_path, body))

    assert rec.doc_markdown() == [body], "rendered view must use st.markdown"
    assert "code" not in rec.kinds(), "must not fall through to the code dump"


def test_headings_keep_their_true_levels(rec, tmp_path):
    """Unlike the chat bubble, this pane is the document view — no demotion."""
    file_viewer.render_file_preview(_md(tmp_path, "# H1\n## H2\n"))
    assert rec.doc_markdown()[0].startswith("# H1")


def test_source_toggle_shows_raw_markdown(monkeypatch, tmp_path):
    r = _Recorder(view="Source")
    monkeypatch.setattr(file_viewer, "st", r)
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))

    assert r.arg("code") == "# Title\n"
    assert r.doc_markdown() == []
    lang = [kw for name, _, kw in r.calls if name == "code"][0]["language"]
    assert lang == "markdown"


def test_deselecting_the_toggle_still_renders(monkeypatch, tmp_path):
    """A segmented control can be clicked back to None; that is not a request
    for raw source."""
    r = _Recorder(view=None)
    monkeypatch.setattr(file_viewer, "st", r)
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    assert r.doc_markdown() and "code" not in r.kinds()


def test_huge_file_is_clipped_and_says_so(rec, tmp_path):
    body = "x" * (file_viewer._MD_MAX_CHARS + 5_000)
    file_viewer.render_file_preview(_md(tmp_path, body))

    assert len(rec.doc_markdown()[0]) == file_viewer._MD_MAX_CHARS
    assert "caption" in rec.kinds()


def test_clipping_never_leaves_a_fence_open(rec, tmp_path):
    """An unterminated ``` would render the whole tail as one code block."""
    body = "```python\n" + "y = 1\n" * file_viewer._MD_MAX_CHARS
    file_viewer.render_file_preview(_md(tmp_path, body))

    assert rec.doc_markdown()[0].count("```") % 2 == 0


def test_download_button_survives(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    assert "download_button" in rec.kinds()


def test_markdown_offers_a_pdf_alongside_the_source(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n\nBody.\n"))

    assert rec.labels("download_button") == ["Download", "Download PDF"]
    pdf = [kw for name, a, kw in rec.calls
           if name == "download_button" and a[0] == "Download PDF"][0]
    assert pdf["file_name"] == "report.pdf"
    assert pdf["mime"] == "application/pdf"
    assert pdf["data"][:5] == b"%PDF-", "must hand over a real PDF"


def test_non_markdown_gets_no_pdf_button(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "plain\n", name="n.txt"))
    assert rec.labels("download_button") == ["Download"]


def test_a_failing_conversion_never_breaks_the_preview(monkeypatch, tmp_path):
    """The markdown download and the rendered document must still be there."""
    import scilink.utils.md_to_pdf as m
    monkeypatch.setattr(m, "markdown_text_to_pdf", lambda *a, **k: 1 / 0)
    r = _Recorder()
    monkeypatch.setattr(file_viewer, "st", r)
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))

    assert r.labels("download_button") == ["Download"]
    assert r.doc_markdown(), "the document itself must still render"
    assert any("PDF export unavailable" in str(a[0])
               for name, a, _ in r.calls if name == "caption")


def test_other_text_files_still_use_the_code_view(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "plain\n", name="notes.txt"))
    assert "code" in rec.kinds() and rec.doc_markdown() == []


def test_toggle_restyle_cannot_leak_to_other_widgets(rec, tmp_path):
    """The contrast fix must repaint THIS toggle and nothing else: every
    selector stays behind the keyed container's class."""
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    css = rec.css()
    assert css, "the selected segment needs its contrast fix"

    body = css[css.index("<style>") + 7: css.index("</style>")]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)   # prose has commas too
    selectors = [s.strip() for chunk in body.split("}") if "{" in chunk
                 for s in chunk.split("{")[0].split(",") if s.strip()]
    assert selectors
    scope = f".st-key-{file_viewer._MD_TOGGLE_KEY}"
    assert all(s.startswith(scope) for s in selectors), selectors


def test_the_scope_class_matches_the_container_key(rec, tmp_path):
    """A renamed key with un-renamed CSS silently loses the styling."""
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    keys = [kw.get("key") for name, _, kw in rec.calls if name == "container"]
    assert file_viewer._MD_TOGGLE_KEY in keys
    assert re.search(rf"\.st-key-{file_viewer._MD_TOGGLE_KEY}\b", rec.css())
