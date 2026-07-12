"""Headless render tests for the UI memory panel (skills.py).

Guards the panel's performance contract: Streamlit renders popover/expander
content eagerly on every rerun, so with many memory records the panel used to
re-read every file and re-highlight every script per chat interaction (the
observed slowdown). Heavy content must load only behind the explicit
'Load full content' tick, and long lists must paginate.

Uses streamlit.testing.v1.AppTest (no browser); skipped when streamlit is
not installed.
"""

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PANEL_SCRIPT = """
from scilink.ui.components.skills import _render_memory_section
_render_memory_section()
"""


@pytest.fixture()
def heavy_store(tmp_path, monkeypatch):
    """A store with enough records to expose eager-render regressions."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _script_bank as sb, _staging

    big_script = "\n".join(f"# line {i}" for i in range(200))
    for i in range(40):
        sb.add_record("curve_fitting", {
            "working_script": big_script + f"\n# uniq {i}",
            "data_fingerprint": {"kind": "curve", "n_points": 1000 + i},
            "measurement_context": {"technique": "Raman"},
            "technique_signals": {"model_type": f"model variant {i}"},
            "outcome": {"metric": {"name": "r_squared", "value": 0.99}},
            "provenance": {"session": f"run{i}"}})
    for i in range(30):
        _staging.stage_solution("curve_fitting", f"tech_{i % 5}", {
            "provenance": "t2_solution", "model": f"m{i}",
            "working_script": big_script, "session": f"run{i}",
            "r_squared": 0.99})
    return tmp_path


def test_panel_renders_lazily(heavy_store):
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    # NO script is syntax-highlighted until a viewer is explicitly loaded.
    assert len(at.code) == 0
    # Lazy viewers exist for the paged rows.
    assert len(at.checkbox) > 0


def test_lazy_tick_loads_one_record(heavy_store):
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    at.checkbox[0].check().run()
    assert not at.exception, at.exception
    assert len(at.code) == 1  # exactly the ticked record's script


def test_long_lists_paginate(heavy_store):
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    # 40 bank records in one domain -> a 'Show all' button caps the page.
    labels = " ".join(b.label for b in at.button)
    assert "Show all 40" in labels
