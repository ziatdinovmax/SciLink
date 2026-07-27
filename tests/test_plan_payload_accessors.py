"""Plan payload accessors — one reader for three historical shapes.

A plan-mode artifact is an EXPERIMENT or a PORTFOLIO. The portfolio rode the
experiment schema for a long time, so a session on disk may hold any of:
top-level `directions` (the contract), `proposed_experiments[*].concepts`
(PR #394), or an ideation plan predating `concepts` entirely. Consumers must
not branch on that themselves, and a checkpoint written before this change
must still restore.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.planning_agents.parser_utils import (
    plan_directions, plan_is_portfolio, plan_thesis,
    portfolio_to_experiment_shim)

TIER1 = {"type": "ideation", "thesis": "Rare sites dominate.",
         "directions": [{"id": "A", "title": "One"},
                        {"id": "B", "title": "Two"}]}
TIER2 = {"type": "ideation",
         "proposed_experiments": [{"hypothesis": "Rare sites dominate.",
                                   "concepts": [{"id": "C", "title": "Three"}]}]}
TIER3 = {"type": "ideation",
         "proposed_experiments": [{"experiment_name": "Old direction",
                                   "hypothesis": "H", "justification": "J"}]}
LAB = {"proposed_experiments": [{"experiment_name": "E", "hypothesis": "H",
                                 "experimental_steps": ["s1"]}]}


@pytest.mark.parametrize("plan,n,portfolio", [
    (TIER1, 2, True), (TIER2, 1, True), (TIER3, 1, True),
    (LAB, 0, False), ({}, 0, False), (None, 0, False),
])
def test_every_shape_resolves(plan, n, portfolio):
    assert len(plan_directions(plan)) == n
    assert plan_is_portfolio(plan) is portfolio


def test_top_level_directions_win_over_the_nested_shape():
    """A plan carrying both (the transition shim does) must not double."""
    both = dict(TIER1, proposed_experiments=[{"concepts": [{"id": "X"}]}])
    assert [d["id"] for d in plan_directions(both)] == ["A", "B"]


def test_the_pre_concepts_tier_is_marked_synthesised():
    """It is reconstructed from experiment fields, not authored — callers
    that care can tell."""
    d = plan_directions(TIER3)[0]
    assert d["_synthesised"] is True and d["title"] == "Old direction"


def test_a_lab_plan_is_never_mistaken_for_a_portfolio():
    """The costly direction: rendering an experiment as a portfolio would
    drop its protocol."""
    assert plan_directions(LAB) == []
    assert plan_is_portfolio(LAB) is False
    # ...even if it somehow carries an empty directions key
    assert plan_is_portfolio(dict(LAB, directions=[])) is False


def test_thesis_falls_back_to_the_hypothesis():
    assert plan_thesis(TIER1) == "Rare sites dominate."
    assert plan_thesis(TIER2) == "Rare sites dominate."
    assert plan_thesis(LAB) == "H"
    assert plan_thesis({}) == ""


# ── the transition shim ──────────────────────────────────────────────

def test_shim_gives_legacy_readers_a_true_experiment_view():
    p = portfolio_to_experiment_shim(dict(TIER1))
    exp = p["proposed_experiments"][0]
    assert exp["hypothesis"] == "Rare sites dominate."
    assert [c["id"] for c in exp["concepts"]] == ["A", "B"]
    assert exp["_portfolio_shim"] is True
    # and the real payload is untouched
    assert len(plan_directions(p)) == 2


def test_shim_never_overwrites_a_real_experiment():
    p = portfolio_to_experiment_shim(dict(LAB))
    assert p["proposed_experiments"][0]["experiment_name"] == "E"
    assert "_portfolio_shim" not in p["proposed_experiments"][0]


def test_shim_is_a_no_op_without_directions():
    assert portfolio_to_experiment_shim({"type": "ideation"}) == {
        "type": "ideation"}


def test_shimmed_portfolio_satisfies_the_legacy_validity_checks():
    """The checks that gate every plan path: `not plan.get(...)` means
    failure, and a portfolio must not read as a failed plan."""
    p = portfolio_to_experiment_shim(dict(TIER1))
    assert p.get("proposed_experiments")
    assert len(p.get("proposed_experiments", [])) == 1


# ── the real sessions on disk ────────────────────────────────────────

@pytest.mark.parametrize("session", [
    "meta_session_20260726_141723",
])
def test_recorded_sessions_still_resolve(session):
    """Restore compatibility, against real plans rather than fixtures."""
    root = Path(session) / "planning" / "delegations"
    if not root.exists():
        pytest.skip(f"{session} not present")
    seen = 0
    for pj in sorted(root.glob("*/plan.json")):
        plan = json.loads(pj.read_text())
        seen += 1
        if plan.get("type") == "ideation":
            # every ideation plan in that session must yield directions —
            # via concepts (01/03/04) or the synthesised tier (05)
            assert plan_directions(plan), pj.parent.name
        else:
            assert plan_directions(plan) == []
    assert seen, "expected recorded plans"


# ── refinement keeps the two copies consistent ───────────────────────

def _refined(nested):
    """What a refine pass returns: it edited the nested copy it was shown."""
    return {"type": "ideation", "directions": [{"id": "OLD", "title": "stale"}],
            "proposed_experiments": [{"concepts": nested, "_portfolio_shim": True}]}


def test_a_refined_portfolio_does_not_serve_stale_directions():
    """The prompts name `proposed_experiments`, so that is the copy an
    editor reliably updates — a stale top-level would silently outrank it."""
    from scilink.agents.planning_agents.parser_utils import resync_portfolio
    p = _refined([{"id": "NEW", "title": "hardened"}])
    resync_portfolio(p)
    assert [d["id"] for d in plan_directions(p)] == ["NEW"]
    assert p["directions"] == p["proposed_experiments"][0]["concepts"]


def test_resync_is_a_no_op_when_they_agree():
    from scilink.agents.planning_agents.parser_utils import resync_portfolio
    same = [{"id": "A", "title": "One"}]
    p = {"type": "ideation", "directions": same,
         "proposed_experiments": [{"concepts": same}]}
    before = json.dumps(p, sort_keys=True)
    resync_portfolio(p)
    assert json.dumps(p, sort_keys=True) == before


def test_resync_never_empties_a_portfolio():
    """A pass that DROPPED concepts must not take the directions with it."""
    from scilink.agents.planning_agents.parser_utils import resync_portfolio
    p = {"type": "ideation", "directions": [{"id": "A"}],
         "proposed_experiments": [{"hypothesis": "h"}]}
    resync_portfolio(p)
    assert [d["id"] for d in plan_directions(p)] == ["A"]


def test_resync_leaves_experiments_alone():
    from scilink.agents.planning_agents.parser_utils import resync_portfolio
    p = dict(LAB)
    resync_portfolio(p)
    assert p == LAB
