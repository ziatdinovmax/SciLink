"""#587 — a metadata dict that carries title/xlabel/ylabel keys with None
values renders with the defaults instead of crashing on None + str."""
import numpy as np
import pytest

from scilink.utils.curve_preview import render_curve_overlay, render_curve_single

CURVES = [{"label": f"s{i}", "curve_data": np.column_stack([np.linspace(0, 1, 20), np.sin(np.linspace(0, 1, 20) + i)])}
          for i in range(3)]
NONE_META = {"title": None, "xlabel": None, "ylabel": None}


@pytest.fixture(autouse=True)
def _agg():
    import matplotlib
    matplotlib.use("Agg", force=True)


def _texts(monkeypatch):
    seen = {}
    import matplotlib.axes
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", lambda self, t, *a, **k: seen.__setitem__("title", t))
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xlabel", lambda self, t, *a, **k: seen.__setitem__("xlabel", t))
    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", lambda self, t, *a, **k: seen.__setitem__("ylabel", t))
    return seen


def test_overlay_with_present_but_none_metadata_uses_the_defaults(monkeypatch):
    seen = _texts(monkeypatch)
    png = render_curve_overlay(CURVES, NONE_META)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen == {"title": "Data — Scout Overlay", "xlabel": "X", "ylabel": "Y"}


def test_overlay_keeps_populated_metadata(monkeypatch):
    seen = _texts(monkeypatch)
    render_curve_overlay(CURVES, {"title": "XPS C 1s", "xlabel": "BE (eV)", "ylabel": "counts"})
    assert seen == {"title": "XPS C 1s — Scout Overlay", "xlabel": "BE (eV)", "ylabel": "counts"}


def test_single_curve_with_none_metadata(monkeypatch):
    seen = _texts(monkeypatch)
    png = render_curve_single(CURVES[0]["curve_data"], NONE_META)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen == {"title": "Data", "xlabel": "X", "ylabel": "Y"}
