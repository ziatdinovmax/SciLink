"""#598 — completion is judged against the request as stated: both the
analysis orchestrator and the meta carry the one-sentence principle, so a
self-review cannot invent a deliverable and then chase it."""
from scilink.agents.exp_agents import analysis_orchestrator as ao
from scilink.agents.meta_agent import meta_orchestrator as mo


def test_analysis_prompt_anchors_completion_to_the_stated_request():
    body = ao._SYSTEM_PROMPT_BODY_POST
    assert "Judge completion against the request AS STATED" in body
    assert "is not a request for\n  a table of it" in body
    assert "produced only when explicitly requested" in body


def test_meta_prompt_carries_the_same_principle():
    import inspect
    src = inspect.getsource(mo)
    assert "Judge a specialist's result against the user's request AS STATED" in src
    assert "delegate follow-ups to satisfy it" in src
