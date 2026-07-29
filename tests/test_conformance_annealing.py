"""The plan-conformance checker must speak at the same skill temperature as
codegen: PR #294 relaxes the skill with annealing in the codegen prompts, and
the conformance framing must follow the same schedule instead of a frozen
MANDATORY header (which contradicted verifier-directed refits at hot levels).
"""

import inspect

from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cc


def _conformance_src():
    src = inspect.getsource(cc)
    i = src.index("Build skill rules text for conformance checking")
    return src[i - 200: i + 1200]


def test_conformance_framing_follows_annealing_schedule():
    seg = _conformance_src()
    assert "_SKILL_STRICTNESS_SCHEDULE" in seg
    assert "_annealing_level" in seg
    assert '**MANDATORY Domain Skill Rules' not in seg, (
        "conformance still hardcodes the T=0 framing")


def test_refit_prompt_requires_stated_justification_for_departures():
    src = inspect.getsource(cc)
    assert "stated justification are acceptable; silent ones are flagged" in src
