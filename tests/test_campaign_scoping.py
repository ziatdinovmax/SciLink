"""Offline tests for campaign-scoped literature continuity (issue #396).

One session can hold several unrelated brainstorms/campaigns. Literature
continuity heuristics that are correct WITHIN a campaign (max-length
white-paper corpus selection, session-wide refine auto-load, carry-forward,
persistent planner state) must never cross a campaign boundary. Covers:

- the lexical objective-continuity heuristic (disjoint topics vs rewording);
- campaign transition in generate_plan: id bump, current_plan/plan_candidates
  reset, history stamps; explicit new_campaign overrides in both directions;
- white paper: other campaigns' literature excluded; within-campaign
  max-length selection unchanged; plan fallback is same-campaign only;
- the campaign literature registry: record/adopt lifecycle, campaign-scoped
  _latest_literature_file, legacy session-wide fallback only for
  pre-registry states, stale-literature detection for the new-campaign guard;
- resolve_n_candidates treats a new campaign as campaign-first (best-of-3).

All LLM traffic is a scripted mock; no network.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.planning_agent import (
    PlanningAgent,
    objectives_share_campaign,
)
from scilink.agents.planning_agents.base_agent import BaseAgent
from scilink.agents.planning_agents import planning_agent as pa_mod
from scilink.agents.planning_agents.orchestrator_tools import (
    OrchestratorTools,
    resolve_n_candidates,
)


OBJ_A = ("Non-additive materials degradation under simultaneous "
         "irradiation, mechanical stress and high temperature")
OBJ_B = ("Ideate research directions on humidity-driven degradation "
         "of perovskite solar cells")
OBJ_B_FOLLOWUP = ("Add a controlled-humidity aging protocol dimension to "
                  "the perovskite solar cell degradation study")

LIT_A = "IRRADIATION CORPUS " + "coupled extreme conditions. " * 400
LIT_B = "PEROVSKITE CORPUS " + "humidity ingress pathways. " * 40


# ---------------------------------------------------------------- helpers

def plan_json(name):
    return json.dumps({
        "proposed_experiments": [{
            "hypothesis": f"H-{name}",
            "experiment_name": name,
            "experimental_steps": ["Step 1", "Step 2"],
            "expected_outcome": "outcome",
            "justification": "justified",
        }]
    })


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, prompt_parts, generation_config=None):
        if isinstance(prompt_parts, str):
            prompt_parts = [prompt_parts]
        self.calls.append("\n".join(p for p in prompt_parts
                                    if isinstance(p, str)))
        if not self.responses:
            raise AssertionError("ScriptedModel exhausted")
        return SimpleNamespace(text=self.responses.pop(0))


def make_agent(tmp_path, model):
    agent = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    agent.agent_type = "planning"
    agent.model = model
    agent.generation_config = None
    agent.kb_docs = None
    agent.kb_code = None
    agent.lit_agent = None
    agent._ensure_kb_is_ready = lambda *a, **k: False
    return agent


def make_tools(tmp_path, state):
    """OrchestratorTools with a fake orchestrator — registry surface only."""
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(
        planner=SimpleNamespace(state=state),
        base_dir=Path(tmp_path),
        _active_output_subdir=None,
    )
    tools._prestate_lit = []
    return tools


@pytest.fixture
def no_verify_no_critic(monkeypatch):
    monkeypatch.setattr(pa_mod, "verify_plan_relevance",
                        lambda *a, **k: (True, ""))
    monkeypatch.setattr(pa_mod, "critique_plan",
                        lambda *a, **k: {"findings": []})


# ------------------------------------------------- objective heuristic

def test_disjoint_topics_do_not_share_campaign():
    assert not objectives_share_campaign(OBJ_A, OBJ_B)
    assert not objectives_share_campaign(OBJ_B, OBJ_A)


def test_rewording_and_narrowing_share_campaign():
    assert objectives_share_campaign(OBJ_B, OBJ_B_FOLLOWUP)
    assert objectives_share_campaign(OBJ_A, OBJ_A)
    assert objectives_share_campaign(
        OBJ_A, "Refine coupled-extremes degradation plan: separate "
               "irradiation and stress contributions")


def test_empty_or_generic_objectives_tie_toward_continuity():
    assert objectives_share_campaign("", OBJ_B)
    assert objectives_share_campaign(OBJ_A, "")
    assert objectives_share_campaign("investigate novel materials",
                                     "optimize the new approach")


# ------------------------------------------------- generate_plan transition

def test_second_topic_starts_new_campaign(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("PlanA"), plan_json("PlanB")])
    agent = make_agent(tmp_path, model)

    agent.generate_plan(OBJ_A, enable_human_feedback=False,
                        external_context=LIT_A)
    assert agent.state["campaign_id"] == 1
    assert agent.state["current_plan"]["campaign_id"] == 1
    assert agent.state["current_plan"]["literature_search"] == LIT_A

    agent.generate_plan(OBJ_B, enable_human_feedback=False)
    assert agent.state["campaign_id"] == 2
    assert agent.state["objective"] == OBJ_B
    plan_b = agent.state["current_plan"]
    assert plan_b["campaign_id"] == 2
    # topic A's corpus must not ride into topic B's plan
    assert "literature_search" not in plan_b or \
        LIT_A not in str(plan_b.get("literature_search"))
    # history keeps both campaigns, each stamped
    stamps = {p["campaign_id"] for p in agent.state["plan_history"]}
    assert stamps == {1, 2}


def test_same_topic_followup_stays_in_campaign(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("PlanB1"), plan_json("PlanB2")])
    agent = make_agent(tmp_path, model)
    agent.generate_plan(OBJ_B, enable_human_feedback=False,
                        external_context=LIT_B)
    agent.generate_plan(OBJ_B_FOLLOWUP, enable_human_feedback=False)
    assert agent.state["campaign_id"] == 1
    assert all(p["campaign_id"] == 1 for p in agent.state["plan_history"])


def test_explicit_new_campaign_overrides_heuristic(tmp_path,
                                                   no_verify_no_critic):
    model = ScriptedModel([plan_json("P1"), plan_json("P2"),
                           plan_json("P3")])
    agent = make_agent(tmp_path, model)
    agent.generate_plan(OBJ_B, enable_human_feedback=False)
    # same wording, but the caller declares a new campaign -> honored
    agent.generate_plan(OBJ_B, enable_human_feedback=False,
                        new_campaign=True)
    assert agent.state["campaign_id"] == 2
    # disjoint wording, but the caller forces continuation -> honored
    agent.generate_plan(OBJ_A, enable_human_feedback=False,
                        new_campaign=False)
    assert agent.state["campaign_id"] == 2


def test_transition_clears_working_state(tmp_path, no_verify_no_critic):
    model = ScriptedModel([plan_json("PlanA")])
    agent = make_agent(tmp_path, model)
    agent.generate_plan(OBJ_A, enable_human_feedback=False)
    agent.state["plan_candidates"] = {"candidates": [{"x": 1}]}

    assert agent.starts_new_campaign(OBJ_B) is True
    agent._apply_campaign_transition(OBJ_B)
    assert agent.state["campaign_id"] == 2
    assert agent.state["current_plan"] is None
    assert "plan_candidates" not in agent.state
    # refine on the fresh campaign has no plan to refine — honest error,
    # never the previous topic's plan
    with pytest.raises(ValueError):
        agent.refine_plan("results", enable_human_feedback=False)


# ------------------------------------------------- white paper scoping

def _two_campaign_state(current_lit=None):
    hist_a = {"proposed_experiments": [{"experiment_name": "A"}],
              "literature_search": LIT_A, "campaign_id": 1}
    cur_b = {"proposed_experiments": [{"experiment_name": "B",
                                       "hypothesis": "H-B"}],
             "campaign_id": 2}
    if current_lit:
        cur_b["literature_search"] = current_lit
    return {
        "objective": OBJ_B,
        "campaign_id": 2,
        "current_plan": cur_b,
        "plan_history": [hist_a, dict(cur_b)],
    }


def test_white_paper_excludes_other_campaign_literature(tmp_path):
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = _two_campaign_state()
    agent.model = ScriptedModel(["White paper text."])
    agent.generation_config = None
    agent.generate_white_paper()
    prompt = agent.model.calls[0]
    assert "IRRADIATION CORPUS" not in prompt
    assert "Literature Context" not in prompt  # honest no-corpus, no section


def test_white_paper_uses_own_campaign_literature(tmp_path):
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = _two_campaign_state(current_lit=LIT_B)
    agent.model = ScriptedModel(["White paper text."])
    agent.generation_config = None
    agent.generate_white_paper()
    prompt = agent.model.calls[0]
    assert "PEROVSKITE CORPUS" in prompt
    assert "IRRADIATION CORPUS" not in prompt


def test_white_paper_within_campaign_max_length_unchanged(tmp_path):
    # a later same-campaign snapshot carries a model-authored stub; the
    # substantial corpus from an earlier same-campaign entry still wins
    stub = "See prior iteration context (unchanged)."
    state = {
        "objective": OBJ_A,
        "campaign_id": 1,
        "current_plan": {"proposed_experiments": [{"experiment_name": "A2"}],
                         "literature_search": stub, "campaign_id": 1},
        "plan_history": [
            {"proposed_experiments": [{"experiment_name": "A1"}],
             "literature_search": LIT_A, "campaign_id": 1},
        ],
    }
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = state
    agent.model = ScriptedModel(["White paper text."])
    agent.generation_config = None
    agent.generate_white_paper()
    assert "IRRADIATION CORPUS" in agent.model.calls[0]


def test_white_paper_plan_fallback_is_same_campaign_only(tmp_path):
    # no current_plan; the newest history entry is another campaign's plan —
    # the fallback must skip it and honestly report no plan
    state = {
        "objective": OBJ_B,
        "campaign_id": 2,
        "current_plan": None,
        "plan_history": [
            {"proposed_experiments": [{"experiment_name": "A"}],
             "literature_search": LIT_A, "campaign_id": 1},
        ],
    }
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = state
    agent.model = ScriptedModel([])
    agent.generation_config = None
    with pytest.raises(ValueError):
        agent.generate_white_paper()


def test_white_paper_legacy_unstamped_state_still_works(tmp_path):
    # pre-fix states carry no campaign ids anywhere -> everything is
    # campaign 1 and the old behavior is preserved exactly
    state = {
        "objective": OBJ_A,
        "current_plan": {"proposed_experiments": [{"experiment_name": "A"}]},
        "plan_history": [
            {"proposed_experiments": [{"experiment_name": "A0"}],
             "literature_search": LIT_A},
        ],
    }
    agent = PlanningAgent.__new__(PlanningAgent)
    agent.state = state
    agent.model = ScriptedModel(["White paper text."])
    agent.generation_config = None
    agent.generate_white_paper()
    assert "IRRADIATION CORPUS" in agent.model.calls[0]


# ------------------------------------------------- literature registry

def _lit_file(tmp_path, name, text):
    p = Path(tmp_path) / name
    p.write_text(text)
    return p


def test_latest_literature_file_is_campaign_scoped(tmp_path):
    f1 = _lit_file(tmp_path, "literature_search_topicA.md", LIT_A)
    f2 = _lit_file(tmp_path, "literature_search_topicB.md", LIT_B)
    state = {"campaign_id": 2, "current_plan": {"x": 1},
             "campaign_literature": [
                 {"path": str(f1), "campaign_id": 1},
                 {"path": str(f2), "campaign_id": 2}]}
    tools = make_tools(tmp_path, state)
    assert tools._latest_literature_file() == f2


def test_no_literature_for_campaign_yields_none_not_other_topic(tmp_path):
    f1 = _lit_file(tmp_path, "literature_search_topicA.md", LIT_A)
    state = {"campaign_id": 2, "current_plan": {"x": 1},
             "campaign_literature": [{"path": str(f1), "campaign_id": 1}]}
    tools = make_tools(tmp_path, state)
    # campaign 2 supplied no literature: honest None, never topic A's file
    assert tools._latest_literature_file() is None


def test_legacy_fallback_only_for_preregistry_single_campaign(tmp_path):
    f1 = _lit_file(tmp_path, "literature_search_old.md", LIT_A)
    # restored pre-registry session: no entries, no transition -> old glob
    tools = make_tools(tmp_path, {"campaign_id": 1, "current_plan": {"x": 1}})
    assert tools._latest_literature_file() == f1
    # but once a campaign transition happened, no session-wide scraping
    tools2 = make_tools(tmp_path, {"campaign_id": 2, "current_plan": {"x": 1}})
    assert tools2._latest_literature_file() is None


def test_record_before_state_then_adopt_into_campaign(tmp_path):
    f = _lit_file(tmp_path, "literature_search_early.md", LIT_B)
    tools = make_tools(tmp_path, None)  # no session state yet
    tools._record_literature_file(f)
    assert tools._prestate_lit[0]["campaign_id"] is None  # pending

    # first plan call creates state; adoption claims the pending entry
    tools.orch.planner.state = {"campaign_id": 1, "current_plan": {"x": 1}}
    tools._adopt_literature()
    reg = tools.orch.planner.state["campaign_literature"]
    assert reg == [{"path": str(f.resolve()), "campaign_id": 1}]
    assert tools._prestate_lit == []
    assert tools._latest_literature_file() == f


def test_record_during_active_campaign_tags_immediately(tmp_path):
    f = _lit_file(tmp_path, "literature_search_mid.md", LIT_B)
    state = {"campaign_id": 2, "current_plan": {"x": 1}}
    tools = make_tools(tmp_path, state)
    tools._record_literature_file(f)
    assert state["campaign_literature"] == [
        {"path": str(f.resolve()), "campaign_id": 2}]


def test_adopt_explicit_context_files(tmp_path):
    f1 = _lit_file(tmp_path, "literature_search_a.md", LIT_A)
    state = {"campaign_id": 2, "current_plan": {"x": 1},
             "campaign_literature": [{"path": str(f1.resolve()),
                                      "campaign_id": 1}]}
    tools = make_tools(tmp_path, state)
    # deliberate reuse: explicitly passing the file makes it campaign 2's too
    tools._adopt_literature(str(f1))
    cids = {e["campaign_id"] for e in state["campaign_literature"]
            if e["path"] == str(f1.resolve())}
    assert cids == {1, 2}


def test_prior_campaign_literature_flags_stale_files(tmp_path):
    f1 = _lit_file(tmp_path, "literature_search_a.md", LIT_A)
    f2 = _lit_file(tmp_path, "literature_search_b.md", LIT_B)
    state = {"campaign_id": 2, "current_plan": {"x": 1},
             "campaign_literature": [
                 {"path": str(f1.resolve()), "campaign_id": 1},
                 {"path": str(f2.resolve()), "campaign_id": 2}]}
    tools = make_tools(tmp_path, state)
    assert tools._prior_campaign_literature(str(f1)) == [str(f1.resolve())]
    assert tools._prior_campaign_literature(str(f2)) == []
    # unknown files (fresh search results not yet registered) are not stale
    f3 = _lit_file(tmp_path, "literature_search_new.md", "fresh")
    assert tools._prior_campaign_literature(str(f3)) == []
    # raw text is never flagged
    assert tools._prior_campaign_literature("some pasted abstract") == []


# ------------------------------------------------- best-of-N default

def test_new_campaign_is_campaign_first_for_best_of_n():
    state = {"current_plan": {"x": 1}}
    assert resolve_n_candidates(None, state) == 1
    assert resolve_n_candidates(None, state, new_campaign=True) == 3
    assert resolve_n_candidates(2, state, new_campaign=True) == 2
    assert resolve_n_candidates(None, {}) == 3


# ------------------------------------------------- state hygiene fixes

def test_literature_stored_once_and_restored_verbatim():
    """A campaign corpus rides every plan snapshot, so a state file held the
    same text N times (live: 1.73 MB of a 1.98 MB file was five copies of two
    corpora). Store once, reference, and restore byte-identically."""
    from scilink.agents.planning_agents.planning_agent import (
        compact_planner_state, expand_planner_state)

    big_a, big_b = "A" * 50_000, "B" * 30_000
    state = {
        "campaign_id": 2,
        "current_plan": {"literature_search": big_b, "x": 1},
        "plan_history": [
            {"literature_search": big_a}, {"literature_search": big_a},
            {"literature_search": big_a}, {"literature_search": big_b},
        ],
        "plan_candidates": {"candidates": [{"literature_search": big_a}],
                            "selected_index": 1},
    }
    packed = compact_planner_state(state)
    assert len(packed["_literature_store"]) == 2          # deduped
    assert packed["plan_history"][0]["literature_search"].startswith(
        "__scilink_lit_ref__:")
    assert len(json.dumps(packed)) < len(json.dumps(state)) / 2
    assert expand_planner_state(packed) == state          # lossless
    # the live state must never be mutated by serialization
    assert state["plan_history"][0]["literature_search"] == big_a


def test_expand_is_safe_on_uncompacted_and_broken_states():
    from scilink.agents.planning_agents.planning_agent import (
        compact_planner_state, expand_planner_state)

    plain = {"current_plan": {"literature_search": "short corpus"},
             "plan_history": []}
    assert expand_planner_state(plain) == plain           # no-op, idempotent
    assert compact_planner_state(plain) == plain          # below threshold

    # an orphaned reference is kept visible, never silently dropped
    broken = {"current_plan": {"literature_search": "__scilink_lit_ref__:dead"},
              "plan_history": []}
    out = expand_planner_state(broken)
    assert out["current_plan"]["literature_search"] == "__scilink_lit_ref__:dead"


def test_agent_state_file_round_trips_through_compaction(tmp_path):
    from scilink.agents.planning_agents.planning_agent import PlanningAgent
    from scilink.agents.planning_agents.base_agent import BaseAgent

    agent = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    agent.agent_type = "planning"
    corpus = "CORPUS " * 2000
    agent.state = {"session_id": "s", "campaign_id": 1,
                   "current_plan": {"literature_search": corpus},
                   "plan_history": [{"literature_search": corpus}] * 3,
                   "action_history": []}
    agent._save_state()

    on_disk = json.loads((tmp_path / "planning_state.json").read_text())
    assert "_literature_store" in on_disk                  # stored once
    assert on_disk["plan_history"][0]["literature_search"].startswith(
        "__scilink_lit_ref__:")

    restored = PlanningAgent.__new__(PlanningAgent)
    BaseAgent.__init__(restored, str(tmp_path))
    restored.agent_type = "planning"
    assert restored.load_state(str(tmp_path / "planning_state.json"))
    assert restored.state["current_plan"]["literature_search"] == corpus
    assert all(p["literature_search"] == corpus
               for p in restored.state["plan_history"])


def test_adopting_literature_refreshes_the_state_mirror(tmp_path):
    """planning_state.json is written from inside generate_plan, before the
    tool layer adopts literature — live, the mirror showed a file still
    tagged to the previous campaign while the checkpoint was correct."""
    saved = []
    lit = tmp_path / "literature_search_x.md"
    lit.write_text("corpus")
    state = {"campaign_id": 2, "current_plan": {"x": 1}}
    tools = make_tools(tmp_path, state)
    tools.orch.planner._save_state = lambda: saved.append(
        dict(state.get("campaign_literature") or []) or True)

    tools._adopt_literature(str(lit))
    assert saved, "planner state was not re-saved after adoption"
    assert any(e["campaign_id"] == 2 for e in state["campaign_literature"])


def test_saved_files_land_in_the_active_delegation_dir(tmp_path):
    """save_file rooted at base_dir put LLM-written files OUTSIDE the
    delegation directory holding every other artifact of that turn — live, a
    white paper landed in planning/<slug>/ beside delegations/, duplicating
    the copy already inside it."""
    from scilink.agents.planning_agents import orchestrator_tools as ot

    deleg = tmp_path / "delegations" / "03_regenerate_white_paper"
    deleg.mkdir(parents=True)
    tools = OrchestratorTools.__new__(OrchestratorTools)
    tools.orch = SimpleNamespace(planner=SimpleNamespace(state={}, model=None),
                                 base_dir=tmp_path,
                                 _active_output_subdir=deleg,
                                 lit_agent=None)
    captured = {}
    tools._register_tool = lambda func, name, description, parameters, required=None: (
        captured.update({name: func}))
    ot.OrchestratorTools._register_all_tools(tools)

    out = json.loads(captured["save_file"]("white_paper_v2.md", "# paper"))
    assert out["status"] == "success"
    assert Path(out["path"]).parent == deleg          # inside the delegation
    assert not (tmp_path / "white_paper_v2.md").exists()

    # a later turn can still find it by bare name, wherever it was written
    tools.orch._active_output_subdir = tmp_path / "delegations" / "04_next"
    (tools.orch._active_output_subdir).mkdir(parents=True)
    resolved, err = tools._resolve_data_path("white_paper_v2.md")
    assert err is None and Path(resolved) == Path(out["path"])

    # standalone plan mode (no active delegation) is unchanged: base_dir
    tools.orch._active_output_subdir = None
    out2 = json.loads(captured["save_file"]("notes.md", "x"))
    assert Path(out2["path"]).parent == tmp_path
