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


def test_panel_survives_hostile_and_degenerate_records(tmp_path, monkeypatch):
    """Markdown/HTML in labels, missing tiers, plain-float metrics, and a
    corrupt JSON file alongside must not break the panel."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _script_bank as sb

    sb.add_record("curve_fitting", {
        "working_script": "# s", "data_fingerprint": {"kind": "curve"},
        "measurement_context": {},
        "technique_signals": {"model_type": "**bold** <script>x</script> `tick`"},
        "outcome": {"metric": 0.97}, "provenance": {"session": "s"}})
    sb.add_record("curve_fitting", {
        "working_script": "# s2", "data_fingerprint": None,
        "measurement_context": None, "technique_signals": None,
        "outcome": None, "provenance": {"session": "s"}})
    (sb._domain_dir("curve_fitting") / "corrupt.json").write_text("{broken")

    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_promote_button_flow_and_dangling_reenable(tmp_path, monkeypatch):
    """Clicking Promote stages the record and disables the button; once the
    staged copy is consumed/pruned, the button re-enables."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _script_bank as sb, _staging

    rid = sb.add_record("curve_fitting", {
        "working_script": "# promoteme", "data_fingerprint": {"kind": "curve"},
        "measurement_context": {}, "technique_signals": {"model_type": "voigt sum"},
        "outcome": {"metric": {"name": "r_squared", "value": 0.99}},
        "provenance": {"session": "s"}})["id"]
    for s in ("s2", "s3"):
        sb.record_success("curve_fitting", rid, session=s)
    key = f"bankprom::curve_fitting/{rid}"

    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    btn = next(b for b in at.button if b.key == key)
    assert not btn.disabled
    btn.click().run()
    assert not at.exception, at.exception
    staged = _staging.list_staged("curve_fitting")
    assert len(staged) == 1 and staged[0]["bank_id"] == rid

    at.run()  # fresh render: button now disabled
    assert next(b for b in at.button if b.key == key).disabled

    _staging.remove_staged("curve_fitting", [staged[0]["id"]])
    at.run()  # staged copy gone: promotable again
    assert not next(b for b in at.button if b.key == key).disabled


def test_long_lists_paginate(heavy_store):
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    # 40 bank records in one domain -> a 'Show all' button caps the page.
    labels = " ".join(b.label for b in at.button)
    assert "Show all 40" in labels


def test_demote_and_destage_buttons(tmp_path, monkeypatch):
    """A promoted skill shows a working Demote button; a staged record's
    viewer offers de-stage (delete without distilling)."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _memory, _staging
    from scilink.skills.loader import graduated_skills_dir

    d = graduated_skills_dir() / "curve_fitting" / "myskill"
    d.mkdir(parents=True)
    (d / "myskill.md").write_text("## overview\npromoted skill body\n")
    sid = _staging.stage_solution("curve_fitting", "voigt", {
        "provenance": "t2_solution", "model": "m", "working_script": "# s",
        "session": "s1", "r_squared": 0.99})

    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    demote = next(b for b in at.button if b.key == "demote::curve_fitting/myskill")
    demote.click().run()
    assert not at.exception, at.exception
    assert [r["provisional"] for r in _memory.list_memory()] == [True]
    # After demotion the row is provisional: Promote replaces Demote.
    at.run()
    assert any(b.key == "promote::curve_fitting/myskill" for b in at.button)

    destage = next(b for b in at.button
                   if b.key == f"destage::curve_fitting/{sid}")
    destage.click().run()
    assert not at.exception, at.exception
    assert _staging.list_staged("curve_fitting") == []


def test_variant_group_suggestion_and_group_promote(tmp_path, monkeypatch):
    """The bank panel suggests same-system variant groups and promotes them
    under one shared technique label; the suggestion clears afterwards."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _script_bank as sb, _staging

    fp = {"kind": "curve", "n_points": 500, "x_range": [0.0, 100.0],
          "peaks": {"count": 2, "top": [
              {"position": 30.0, "fwhm": 5.0, "prominence": 1.0},
              {"position": 70.0, "fwhm": 5.0, "prominence": 0.5}]}}
    ids = [sb.add_record("curve_fitting", {
        "working_script": f"# v{i}", "data_fingerprint": fp,
        "measurement_context": {"technique": "Raman"},
        "technique_signals": {"model_type": "two voigt carbon"},
        "outcome": {"metric": {"name": "r_squared", "value": 0.99 - i * 0.01}},
        "provenance": {"session": f"s{i}"}})["id"] for i in range(2)]

    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    gkey = "::".join(["curve_fitting"] + sorted(ids, key=lambda x: x))
    btns = [b for b in at.button if b.key and b.key.startswith("bankgroup::")]
    assert btns, "group-promote suggestion button missing"
    btns[0].click().run()
    assert not at.exception, at.exception
    staged = _staging.group_by_technique("curve_fitting")
    assert len(staged) == 1 and len(next(iter(staged.values()))) == 2

    at.run()  # suggestion gone once all members are promoted
    assert not [b for b in at.button if b.key and b.key.startswith("bankgroup::")]


def _seed_full_pipeline(tmp_path, monkeypatch):
    """A store exercising all three knowledge streams + the linkage."""
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    from scilink.skills._shared import _script_bank as sb, _staging
    from scilink.skills.loader import graduated_skills_dir

    rid = sb.add_record("curve_fitting", {
        "working_script": "# hot win", "data_fingerprint": {"kind": "curve"},
        "measurement_context": {"technique": "Raman"},
        "technique_signals": {"model_type": "two voigt"},
        "outcome": {"metric": {"name": "r_squared", "value": 0.99}},
        "provenance": {"session": "s1"}})["id"]
    for s in ("s2", "s3"):
        sb.record_success("curve_fitting", rid, session=s)
    out = sb.promote_to_staging("curve_fitting", rid, technique="raman_dg",
                                provenance="t2_hot_win",
                                extra={"deviation_from_plan": "switched model"})
    _staging.stage_solution("curve_fitting", "raman_dg", {
        "provenance": "error_fix", "model": "m",
        "error_lessons": [{"error": "SNIP no converge", "fix": "fixed iter"}],
        "session": "s1"})
    _staging.stage_solution("curve_fitting", "raman_dg", {
        "provenance": "user_correction", "model": "m",
        "user_feedback": "always report baseline fraction", "session": "s1"})
    d = graduated_skills_dir() / "curve_fitting" / "myskill"
    d.mkdir(parents=True)
    (d / "myskill.md").write_text(
        "---\nprovisional: true\n---\n## overview\nskill body\n")
    return rid, out["staged_id"]


def test_panel_pipeline_order_and_summary(tmp_path, monkeypatch):
    _seed_full_pipeline(tmp_path, monkeypatch)
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    texts = [m.value for m in at.markdown]
    joined = "\n".join(texts)
    # summary strip with counts (plain labels + bold counts — the stage
    # numbers live on the section headers only, so the strip doesn't read
    # like arithmetic: "bank: 2 → 2 · Review…")
    assert "Script bank **1**" in joined and "★1 proven" in joined
    assert "Review inbox **3**" in joined and "ready to distill" in joined
    assert "Skills **1**" in joined
    # pipeline order top-to-bottom
    i_bank = next(i for i, t in enumerate(texts) if t.startswith("**1 · Script bank** —"))
    i_inbox = next(i for i, t in enumerate(texts) if t.startswith("**2 · Review inbox**"))
    i_skills = next(i for i, t in enumerate(texts) if t.startswith("**3 · Skills**"))
    assert i_bank < i_inbox < i_skills


def test_panel_badges_verbs_and_crosslinks(tmp_path, monkeypatch):
    rid, sid = _seed_full_pipeline(tmp_path, monkeypatch)
    at = AppTest.from_string(PANEL_SCRIPT, default_timeout=60)
    at.run()
    captions = "\n".join(c.value for c in at.caption)
    # three knowledge-type badges in the inbox
    assert "📜" in captions and "🐛" in captions and "💬" in captions
    # cross-links both ways
    assert f"from bank `{rid}`" in captions and "succeeded in 3 sessions" in captions
    assert f"in review inbox (`{sid}`)" in captions
    # one verb per pipeline stage
    labels = [b.label for b in at.button]
    assert any("Nominate for review" in l for l in labels)
    assert any("Approve for routing" in l for l in labels)
    assert any("Discard" in l for l in labels)
    assert not any(l.strip() == "Promote" for l in labels)  # overload gone
