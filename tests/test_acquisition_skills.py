"""Acquisition skills: technique knowledge about STEERING a measurement, read
by the live loop's slow-clock consumers — not by run_analysis.

Knowledge-only domain with its own section vocabulary; selected once per
recommender through the shared selector; rendered into the LLM recommender's
prompt; recorded on the recommendation. No LLM calls here.
"""

import json
from types import SimpleNamespace

import pytest

from scilink.live import LLMRecommender
from scilink.live.acquisition_skills import (DOMAIN, SECTIONS, render_guidance,
                                             select_acquisition_skill)
from scilink.live.recommend import InstrumentSchema, finalize
from scilink.skills.loader import list_skills, load_skill

SCHEMA = InstrumentSchema.from_dict({"laser_power_mw": {"low": 0.5, "high": 20.0, "units": "mW"}})
RAMAN = {"technique": "Raman spectroscopy", "sample": "carbon film, annealed in situ"}


class Model:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate_content(self, prompt, **kw):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.replies.pop(0))


def test_every_skill_is_complete_and_on_vocabulary():
    names = list_skills(DOMAIN)
    assert {"raman", "powder_xrd", "afm_force_curve", "sts_didv"} <= set(names)
    for name in names:
        parsed = load_skill(name, domain=DOMAIN)
        assert all((parsed.get(s) or "").strip() for s in SECTIONS), name
        assert not parsed.get("extras"), f"{name}: off-vocabulary headings {list(parsed['extras'])}"
        meta = parsed["meta"]
        assert meta.get("description") and meta.get("technique"), name


def test_knowledge_only():
    from pathlib import Path
    import scilink.skills as pkg
    root = Path(pkg.__file__).parent / DOMAIN
    assert [p.name for p in root.rglob("*.py")] == ["__init__.py"]


def test_acquisition_skills_stay_out_of_the_run_analysis_menu():
    from scilink.agents.exp_agents.analysis_orchestrator_tools import _build_skill_description
    text = _build_skill_description()
    assert "sts_didv" not in text and "afm_force_curve" not in text
    assert "xrd_profile" in text          # the analysis skills are still there


def test_selection_goes_through_the_shared_selector_and_is_exclusive():
    m = Model(['{"skills": ["raman", "powder_xrd"]}'])
    assert select_acquisition_skill(m, RAMAN, SCHEMA) == "raman"
    [prompt] = m.prompts
    assert "technique: Raman spectroscopy" in prompt and "laser_power_mw" in prompt
    assert "**sts_didv**" in prompt
    assert select_acquisition_skill(Model(['{"skills": []}']), RAMAN) is None
    assert select_acquisition_skill(Model(['{"skills": ["not_a_skill"]}']), RAMAN) is None
    assert select_acquisition_skill(Model(["not json"]), RAMAN) is None     # never raises
    assert select_acquisition_skill(Model([]), None) is None                # nothing to match: no call


def test_the_recommender_selects_once_and_carries_the_guidance():
    m = Model(['{"skills": ["raman"]}', '{"params": {"laser_power_mw": 3.0}}',
               '{"params": {"laser_power_mw": 2.0}}'])
    r = LLMRecommender(m, SCHEMA, "goal", context=RAMAN)
    first, second = r.suggest(), r.suggest()
    assert len(m.prompts) == 3                                   # one selection, two suggestions
    assert first["acquisition_skill"] == second["acquisition_skill"] == "raman"
    assert "the `raman` acquisition skill" in m.prompts[1]
    assert "Limits — never recommend past these" in m.prompts[1]
    rec = finalize(first, SCHEMA, source="llm", based_on_step=1)
    assert rec["acquisition_skill"] == "raman" and rec["params"] == {"laser_power_mw": 3.0}


def test_a_named_skill_is_authoritative_and_none_turns_it_off():
    m = Model(['{"params": {"laser_power_mw": 3.0}}'])
    r = LLMRecommender(m, SCHEMA, "goal", context={"technique": "something else"}, skill="sts_didv")
    assert r.suggest()["acquisition_skill"] == "sts_didv" and len(m.prompts) == 1
    assert "Modulation amplitude" in m.prompts[0]
    off = Model(['{"params": {"laser_power_mw": 3.0}}'])
    out = LLMRecommender(off, SCHEMA, "goal", context=RAMAN, skill=None).suggest()
    assert "acquisition_skill" not in out and "acquisition skill" not in off.prompts[0]


def test_no_match_costs_one_call_once_and_changes_nothing():
    m = Model(['{"skills": []}', '{"params": {"laser_power_mw": 3.0}}',
               '{"params": {"laser_power_mw": 3.0}}'])
    r = LLMRecommender(m, SCHEMA, "goal", context={"technique": "neutron reflectometry"})
    r.suggest(); r.suggest()
    assert len(m.prompts) == 3 and "acquisition skill" not in m.prompts[2]


def test_an_unloadable_skill_degrades_to_no_guidance():
    assert render_guidance("no_such_skill") == ""
