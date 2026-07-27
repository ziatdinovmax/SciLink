"""File Explorer markdown preview.

Reports and white papers were rendered as raw source in the preview pane —
literal `#` headings and pipe-soup tables. These pin the rendered/source
behaviour without a browser, by standing a recording double in for streamlit.
"""

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
            return None
        return _call

    def kinds(self):
        return [c[0] for c in self.calls]

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

    assert "markdown" in rec.kinds(), "rendered view must use st.markdown"
    assert "code" not in rec.kinds(), "must not fall through to the code dump"
    assert rec.arg("markdown") == body


def test_headings_keep_their_true_levels(rec, tmp_path):
    """Unlike the chat bubble, this pane is the document view — no demotion."""
    file_viewer.render_file_preview(_md(tmp_path, "# H1\n## H2\n"))
    assert rec.arg("markdown").startswith("# H1")


def test_source_toggle_shows_raw_markdown(monkeypatch, tmp_path):
    r = _Recorder(view="Source")
    monkeypatch.setattr(file_viewer, "st", r)
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))

    assert r.arg("code") == "# Title\n"
    assert "markdown" not in r.kinds()
    lang = [kw for name, _, kw in r.calls if name == "code"][0]["language"]
    assert lang == "markdown"


def test_deselecting_the_toggle_still_renders(monkeypatch, tmp_path):
    """A segmented control can be clicked back to None; that is not a request
    for raw source."""
    r = _Recorder(view=None)
    monkeypatch.setattr(file_viewer, "st", r)
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    assert "markdown" in r.kinds() and "code" not in r.kinds()


def test_huge_file_is_clipped_and_says_so(rec, tmp_path):
    body = "x" * (file_viewer._MD_MAX_CHARS + 5_000)
    file_viewer.render_file_preview(_md(tmp_path, body))

    assert len(rec.arg("markdown")) == file_viewer._MD_MAX_CHARS
    assert "caption" in rec.kinds()


def test_clipping_never_leaves_a_fence_open(rec, tmp_path):
    """An unterminated ``` would render the whole tail as one code block."""
    body = "```python\n" + "y = 1\n" * file_viewer._MD_MAX_CHARS
    file_viewer.render_file_preview(_md(tmp_path, body))

    assert rec.arg("markdown").count("```") % 2 == 0


def test_download_button_survives(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "# Title\n"))
    assert "download_button" in rec.kinds()


def test_other_text_files_still_use_the_code_view(rec, tmp_path):
    file_viewer.render_file_preview(_md(tmp_path, "plain\n", name="notes.txt"))
    assert "code" in rec.kinds() and "markdown" not in rec.kinds()
