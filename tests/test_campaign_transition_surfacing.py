"""A campaign boundary must reach the decision that grounds the next work.

When an objective opens a NEW campaign while being handed the OUTGOING
campaign's literature, nothing told the agent. The hard guard next door
cannot: it reports corpora from campaigns strictly OLDER than the current
one, and at guard time the transition has not been applied, so the
outgoing campaign's corpus still reads as current. Live, campaign 2's
Pd-colloid corpus silently grounded a campaign-3 portfolio spanning MOFs,
perovskites and high-entropy alloys, and was then re-registered to
campaign 3 by adoption while the console said literature is "NOT carried
forward".

This is REPORTED, not refused — the transition is a lexical heuristic that
has been seen comparing against a placeholder objective, and a hard gate on
it would block legitimate work. So the tests below are as much about what
stays silent as about what speaks.
"""

import types
from pathlib import Path

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools


def _tools(tmp_path, entries, campaign_id=2):
    t = OrchestratorTools.__new__(OrchestratorTools)
    t._prestate_lit = []
    t.orch = types.SimpleNamespace(base_dir=tmp_path, planner=None)
    state = {"campaign_id": campaign_id, "campaign_literature": entries}
    t._planner_state = lambda: state
    return t


def _lit(tmp_path, name="literature_search_x.md", questions=None):
    p = tmp_path / name
    p.write_text("# Question 1: What throughput?\n\nbody\n")
    return p, questions or ["What throughput is typical?", "How does carryover bite?"]


# ------------------------------------------------------- it speaks up


def test_outgoing_campaigns_corpus_is_reported(tmp_path):
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": 2,
                               "questions": qs}], campaign_id=2)

    carried = tools._outgoing_campaign_literature(str(p))

    assert len(carried) == 1, "the outgoing campaign's corpus went unreported"
    assert carried[0]["file"] == p.name
    assert carried[0]["covers"] == qs, "the agent needs WHAT it covers"


def test_the_hard_guard_still_cannot_see_it(tmp_path):
    """Pins the gap this exists to close: the same file, same moment, is
    invisible to _prior_campaign_literature."""
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": 2,
                               "questions": qs}], campaign_id=2)

    assert tools._prior_campaign_literature(str(p)) == []
    assert tools._outgoing_campaign_literature(str(p)) != []


def test_multiple_files_and_section_refs_are_all_reported(tmp_path):
    a, qa = _lit(tmp_path, "literature_search_a.md", ["Q about A"])
    b, qb = _lit(tmp_path, "literature_search_b.md", ["Q about B"])
    tools = _tools(tmp_path, [
        {"path": str(a.resolve()), "campaign_id": 2, "questions": qa},
        {"path": str(b.resolve()), "campaign_id": 2, "questions": qb}])

    carried = tools._outgoing_campaign_literature(f"{a}#q1,{b}")

    assert {c["file"] for c in carried} == {a.name, b.name}


# ------------------------------------------------- it stays quiet (no-op)


def test_silent_when_no_literature_was_passed(tmp_path):
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": 2,
                               "questions": qs}])
    assert tools._outgoing_campaign_literature(None) == []
    assert tools._outgoing_campaign_literature("") == []


def test_silent_for_freshly_searched_literature(tmp_path):
    """THE regression that matters: a search run for THIS topic before any
    plan is pending (campaign_id None) and must never be called carried
    over — that is the search-then-ground flow we want to encourage."""
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": None,
                               "questions": qs}])
    assert tools._outgoing_campaign_literature(str(p)) == []


def test_silent_for_literature_from_older_campaigns(tmp_path):
    """Those are the hard guard's business; reporting them too would
    double-signal the same file."""
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": 1,
                               "questions": qs}], campaign_id=2)
    assert tools._outgoing_campaign_literature(str(p)) == []
    assert tools._prior_campaign_literature(str(p)) == [str(p.resolve())]


def test_silent_for_unregistered_files_and_raw_text(tmp_path):
    p, _ = _lit(tmp_path)
    tools = _tools(tmp_path, [])
    assert tools._outgoing_campaign_literature(str(p)) == []
    assert tools._outgoing_campaign_literature(
        "Some raw literature text pasted inline, not a path") == []


def test_missing_questions_field_still_reports_the_file(tmp_path):
    """Pre-#425 entries have no questions; the file itself is still the
    signal that matters."""
    p, _ = _lit(tmp_path)
    tools = _tools(tmp_path, [{"path": str(p.resolve()), "campaign_id": 2}])
    carried = tools._outgoing_campaign_literature(str(p))
    assert len(carried) == 1 and carried[0]["covers"] == []


def test_a_corpus_carried_twice_is_reported_at_each_boundary(tmp_path):
    """Adoption leaves a file owned by the old campaign AND the new one.
    Passing it across the NEXT boundary is a second carry-over and gets its
    own notice — each transition is a distinct decision the agent is
    making, not a repeat of one it already made."""
    p, qs = _lit(tmp_path)
    tools = _tools(tmp_path, [
        {"path": str(p.resolve()), "campaign_id": 2, "questions": qs},
        {"path": str(p.resolve()), "campaign_id": 3, "questions": qs}],
        campaign_id=3)

    carried = tools._outgoing_campaign_literature(str(p))

    assert len(carried) == 1, "a second carry-over went unreported"
    assert carried[0]["file"] == p.name
    # It is leaving campaign 3, not 2 — the notice reports the id the call
    # is actually departing.
    assert tools._campaign_id() == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
