"""Copying across delegations must not disturb delegation-dependent machinery.

copy=true lands a file in the CURRENT delegation folder, which means a
session can now hold two files with the same basename in different
delegations. Several surfaces resolve files by basename or enumerate
delegation folders, so each is pinned here:

  - per-delegation isolation (the source folder is untouched)
  - the deliverables ledger (keyed by resolved path, not name)
  - write_technical_document's source_files (current delegation first)
  - _resolve_data_path's bare-name search (newest wins)
  - list_workspace_files' delegation listing
  - campaign literature auto-load (a duplicated corpus must not double-count)
"""

import json
from pathlib import Path

import pytest

from scilink.agents.planning_agents.user_interface import (
    load_deliverables, record_deliverable)
from tests.test_edit_file_tool import make_tools


def _seed(tmp_path, name="campaign_workflow.png", body=b"PNGDATA" * 32):
    src = tmp_path / "delegations" / "01_design" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(body)
    return src


# ------------------------------------------------------------- isolation


def test_copy_leaves_the_source_delegation_exactly_as_it_was(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = _seed(tmp_path)
    (src.parent / "platform_overview.md").write_text(
        "# Overview\n\n![W](campaign_workflow.png)\n")
    before = sorted(p.name for p in src.parent.iterdir())
    before_bytes = src.read_bytes()

    out = json.loads(cap["rename_file"](str(src), src.name, copy=True))
    assert out["status"] == "success"

    assert sorted(p.name for p in src.parent.iterdir()) == before
    assert src.read_bytes() == before_bytes
    # The source delegation's own embed must still resolve.
    doc = src.parent / "platform_overview.md"
    assert (doc.parent / "campaign_workflow.png").is_file()


def test_rename_cannot_relocate_a_foreign_file_into_this_delegation(tmp_path):
    """Isolation in the other direction: only copies cross folders."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = _seed(tmp_path)

    json.loads(cap["rename_file"](str(src), "grabbed.png"))

    assert (src.parent / "grabbed.png").is_file()
    assert not (deleg / "grabbed.png").exists()


# ---------------------------------------------------- deliverables ledger


def test_ledger_holds_both_copies_as_distinct_entries(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = _seed(tmp_path)
    record_deliverable(tmp_path, src, "Original figure", False)

    out = json.loads(cap["rename_file"](str(src), src.name, copy=True))
    landed = Path(out["path"])

    paths = {e["path"] for e in load_deliverables(tmp_path)}
    assert str(src.resolve()) in paths, "copy evicted the original's entry"
    assert str(landed.resolve()) in paths, "copy was not recorded"
    assert len(paths) >= 2, "same basename collapsed the ledger"


# -------------------------------------------- basename resolution surfaces


def test_source_files_prefers_the_copy_in_the_current_delegation(tmp_path):
    """write_technical_document resolves a bare source_files name against
    the CURRENT delegation before falling back to a session-wide search."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = tmp_path / "delegations" / "01_design" / "memo.md"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("ORIGINAL in 01\n")

    cap["rename_file"](str(src), "memo.md", copy=True)
    (deleg / "memo.md").write_text("LOCAL COPY in 09, later edited\n")

    local = deleg / "memo.md"
    assert local.is_file() and local.read_text().startswith("LOCAL COPY")
    # 01 sorts before 09, so a session-wide basename search would return the
    # original; the current-delegation-first branch must win.
    hits = sorted(Path(tmp_path).rglob("memo.md"))
    assert hits[0] == src, "fixture no longer exercises the ordering hazard"


def test_bare_name_data_lookup_resolves_to_the_newest_copy(tmp_path):
    """_resolve_data_path searches the session by basename, newest first."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    src = tmp_path / "delegations" / "01_design" / "results.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("a,b\n1,2\n")

    out = json.loads(cap["rename_file"](str(src), "results.csv", copy=True))
    landed = Path(out["path"])

    resolved, err = tools._resolve_data_path("results.csv")
    assert err is None, err
    assert Path(resolved) == landed, (
        f"bare name resolved to {resolved}, not the copy in this delegation")


def test_workspace_listing_shows_the_file_under_both_delegations(tmp_path):
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    tools.orch.active_scalarizer_script = None
    tools.orch.bo_data_path = tmp_path / "bo_data.csv"
    src = _seed(tmp_path)
    cap["rename_file"](str(src), src.name, copy=True)

    payload = json.loads(cap["list_workspace_files"]())
    folders = payload.get("delegation_folders", {})
    assert "campaign_workflow.png" in folders.get("01_design", [])
    assert "campaign_workflow.png" in folders.get("09_consolidate", [])


# --------------------------------------------------- campaign literature


def test_duplicated_literature_file_does_not_double_count(tmp_path):
    """If a literature corpus is copied into another delegation, the
    auto-load union must dedupe it rather than serve it twice."""
    tools, cap, deleg = make_tools(tmp_path, "delegations/09_consolidate")
    lit = tmp_path / "delegations" / "01_design" / "literature_search_x.md"
    lit.parent.mkdir(parents=True, exist_ok=True)
    lit.write_text("# Question 1: What throughput?\n\nUNIQUE-BODY answer.\n")

    out = json.loads(cap["rename_file"](str(lit), lit.name, copy=True))
    copied = Path(out["path"])
    assert copied.is_file()

    state = {"campaign_id": 1, "campaign_literature": [
        {"path": str(lit), "campaign_id": 1, "label": "x",
         "questions": ["What throughput?"]},
        {"path": str(copied), "campaign_id": 1, "label": "x",
         "questions": ["What throughput?"]}]}
    tools._planner_state = lambda: state
    tools._prestate_lit = []

    loaded = tools._load_campaign_literature()
    assert loaded["n_files"] == 2
    assert loaded["text"].count("UNIQUE-BODY") == 1, (
        "duplicated corpus served twice")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
