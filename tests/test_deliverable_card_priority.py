"""Resume 'Latest deliverable' card picks the headline document (#533).

Selection used to be max(mtime) over marked deliverables, so a late
delegation that refined the ideation report surfaced it over the white
paper the session exists to produce. Now: rank by kind (white paper > other
documents > ideation report / portfolio), mtime only as the tiebreak.
"""

import json
import os
from pathlib import Path

from scilink.agents.planning_agents.user_interface import (
    DELIVERABLES_MANIFEST, deliverable_rank, load_deliverables,
    select_headline_deliverable)
from scilink.server.session_manager import collect_restored_deliverables


def _write(base: Path, rel: str, title: str, mtime: float,
           deliverable: bool = True, manifest_dir: str = "") -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {title}\n")
    os.utime(p, (mtime, mtime))
    m = base / manifest_dir / DELIVERABLES_MANIFEST
    m.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(m.read_text()) if m.exists() else []
    entries.append({"path": str(p.resolve()), "title": title,
                    "deliverable": deliverable})
    m.write_text(json.dumps(entries))
    return p.resolve()


T = 1_700_000_000.0


def test_white_paper_beats_a_later_refined_ideation_report(tmp_path):
    wp = _write(tmp_path, "planning/out/white_paper.md", "White paper", T)
    _write(tmp_path, "planning/out/ideation_report.md",
           "Ideation report — all candidate directions", T + 900)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == wp
    assert collect_restored_deliverables(tmp_path) == ([str(wp)], [])


def test_only_secondary_artifacts_still_surface(tmp_path):
    ir = _write(tmp_path, "ideation_report.md", "Ideation report", T)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == ir


def test_newest_wins_within_a_rank(tmp_path):
    _write(tmp_path, "a/white_paper.md", "White paper", T)
    newer = _write(tmp_path, "b/white_paper.md", "White paper", T + 60)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == newer


def test_other_documents_outrank_working_artifacts(tmp_path):
    memo = _write(tmp_path, "cost_memo.md", "Cost estimate memo", T)
    _write(tmp_path, "portfolio.md", "Ideation portfolio", T + 60)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == memo
    # ...but a white paper outranks the memo regardless of age
    wp = _write(tmp_path, "white_paper.md", "White paper", T - 3600)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == wp


def test_unmarked_missing_and_non_embeddable_are_skipped(tmp_path):
    _write(tmp_path, "white_paper.md", "White paper", T, deliverable=False)
    gone = _write(tmp_path, "gone/white_paper.md", "White paper", T)
    gone.unlink()
    _write(tmp_path, "white_paper.pdf", "White paper PDF", T + 10)
    html = _write(tmp_path, "plan_report.html", "Experimental plan (report)", T)
    assert select_headline_deliverable(load_deliverables(tmp_path)) == html
    assert collect_restored_deliverables(tmp_path) == ([], [str(html)])
    assert collect_restored_deliverables(tmp_path / "empty") == ([], [])


def test_rank_uses_title_when_the_filename_is_user_chosen(tmp_path):
    assert deliverable_rank({"path": "x/top3_brief.md", "title": "White Paper v2"}) == 0
    assert deliverable_rank({"path": "x/directions.md",
                             "title": "Ideation report — all candidate directions"}) == 2
    assert deliverable_rank({"path": "x/roadmap.md", "title": "Roadmap"}) == 1
    assert deliverable_rank({"path": "x/Portfolio_final.md", "title": ""}) == 2


def test_manifests_across_child_orchestrators_are_pooled(tmp_path):
    wp = _write(tmp_path, "planning/white_paper.md", "White paper", T,
                manifest_dir="planning")
    _write(tmp_path, "planning/delegations/02/ideation_report.md",
           "Ideation report", T + 900, manifest_dir="planning/delegations/02")
    assert select_headline_deliverable(load_deliverables(tmp_path)) == wp
