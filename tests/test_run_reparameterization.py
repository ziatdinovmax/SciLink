"""The fixer loop: SciLink sources and self-validates a force-field fix.

advise -> search a candidate -> (human confirm) -> apply + re-check. The
re-check is what validates the fix, so a wrong candidate fails and is discarded.
All operations injected — no model, no literature search, no simulation.
"""

from scilink.agents.sim_agents.reference_validation import run_reparameterization

FLAGGED = [{"component": "EIS", "property": "density", "consistent": False,
            "reasoning": "under-dense sulfone"}]


def _advise_add(*a, **k):
    return {"recommended_action": "add_force_field", "requires_human": True}


def _good():
    return {"verdict": {"verdict": "good", "failure_class": None}}


def _poor():
    return {"verdict": {"verdict": "poor", "failure_class": "force_field"}}


def test_first_candidate_revalidates_is_fixed():
    out = run_reparameterization(
        FLAGGED, "aqueous sulfone electrolyte", "openff",
        advise_fn=_advise_add,
        search_fn=lambda rec, tried: {"extra_force_fields": ["sulfone_A.offxml"]},
        apply_and_recheck_fn=lambda cand: _good())
    assert out["status"] == "fixed"
    assert out["candidate"] == {"extra_force_fields": ["sulfone_A.offxml"]}
    assert out["reference_validation"]["verdict"]["verdict"] == "good"


def test_iterates_to_a_second_candidate():
    # First candidate still fails the re-check; a distinct second one passes.
    seq = [{"extra_force_fields": ["A.offxml"]}, {"extra_force_fields": ["B.offxml"]}]

    def search(rec, tried):
        return seq[len(tried)] if len(tried) < len(seq) else None

    results = iter([_poor(), _good()])

    out = run_reparameterization(
        FLAGGED, "sys", "openff", advise_fn=_advise_add,
        search_fn=search, apply_and_recheck_fn=lambda cand: next(results),
        max_attempts=2)
    assert out["status"] == "fixed"
    assert out["candidate"]["extra_force_fields"] == ["B.offxml"]
    assert len(out["attempts"]) == 2


def test_escalates_when_advisor_has_no_action():
    out = run_reparameterization(
        FLAGGED, "sys", "openff",
        advise_fn=lambda *a, **k: {"recommended_action": "escalate"},
        search_fn=lambda rec, tried: {"x": 1},
        apply_and_recheck_fn=lambda cand: _good())
    assert out["status"] == "escalated"
    assert out["attempts"] == []


def test_no_candidate_found():
    out = run_reparameterization(
        FLAGGED, "sys", "openff", advise_fn=_advise_add,
        search_fn=lambda rec, tried: None,
        apply_and_recheck_fn=lambda cand: _good())
    assert out["status"] == "no_candidate"


def test_human_declines_the_candidate():
    out = run_reparameterization(
        FLAGGED, "sys", "openff", advise_fn=_advise_add,
        search_fn=lambda rec, tried: {"extra_force_fields": ["A.offxml"]},
        apply_and_recheck_fn=lambda cand: _good(),
        confirm_fn=lambda cand: False)
    assert out["status"] == "declined"
    assert "candidate" in out


def test_unresolved_when_no_candidate_passes():
    out = run_reparameterization(
        FLAGGED, "sys", "openff", advise_fn=_advise_add,
        search_fn=lambda rec, tried: {"extra_force_fields": [f"c{len(tried)}.offxml"]},
        apply_and_recheck_fn=lambda cand: _poor(),
        max_attempts=2)
    assert out["status"] == "unresolved"
    assert len(out["attempts"]) == 2
