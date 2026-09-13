"""Offline tests for the TEA grounding / provenance / multi-file work.

Pins three contracts:

1. The WHOLE technoeconomic assessment reaches downstream prompts (plan
   authoring, refinement, technical documents) — cost drivers, risks, the
   comparison to alternatives, the data gaps and a provenance line — not
   only the one-sentence summary.
2. A TEA records which instruction tier produced it (``generation_mode``),
   writes the evidence the author saw to a ``.grounding.md`` sidecar, and
   runs an advisory critic whose findings ride on the result as
   ``critic_findings`` (fail-open).
3. ``primary_data_set`` accepts several tabular files at once — a folder
   contributes every file it holds, a comma list resolves each path — and
   the prompt carries each file under its own name.

All LLM traffic is a scripted mock; no network.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scilink.agents.planning_agents import planning_rag as pr
from scilink.agents.planning_agents.planning_agent import PlanningAgent
from scilink.agents.planning_agents.base_agent import BaseAgent
from scilink.agents.planning_agents import orchestrator_tools as ot
from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.html_generator import HTMLReportGenerator


# ---------------------------------------------------------------- helpers

class ScriptedModel:
    """Returns canned responses in order; records every prompt it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, prompt_parts, generation_config=None):
        if isinstance(prompt_parts, str):
            prompt_parts = [prompt_parts]
        self.calls.append("\n".join(p for p in prompt_parts if isinstance(p, str)))
        if not self.responses:
            raise AssertionError("ScriptedModel exhausted — unexpected extra LLM call")
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
    agent._ensure_kb_is_ready = lambda *a, **k: True
    return agent


def _csv(path: Path, **cols):
    pd.DataFrame(cols).to_csv(path, index=False)
    # a metadata sidecar keeps resolve_primary_data_path from prompting
    path.with_suffix(".json").write_text(json.dumps(
        {"title": path.stem, "objective": f"{path.stem} table"}))
    return path


def _two_tables(d: Path):
    comp = _csv(d / "feedstock.csv",
                Element=["Nd", "Dy", "Fe"], Concentration_Percent=[24.5, 1.2, 58.0])
    price = _csv(d / "prices.csv",
                 Element=["Nd", "Dy"], Market_Value_USD_kg=[120, 350])
    return comp, price


ASSESSMENT = {
    "summary": "Context suggests viability; Dy price volatility is the main risk.",
    "key_cost_drivers": ["(Quantitative) HCl consumption at 4 M",
                         "(Qualitative) Energy for calcination"],
    "potential_benefits_or_revenue": ["(Quantitative) Nd at $120/kg"],
    "economic_risks": ["(Qualitative) Dy price volatility"],
    "comparison_to_alternatives": "Hydrometallurgy cheaper than pyrometallurgy per context.",
    "data_gaps_for_quantitative_analysis": ["Reagent cost per kg feed",
                                            "Plant CAPEX at 1 kt/yr"],
    "source_documents": ["market_report.txt"],
}


def tea_json(assess=ASSESSMENT):
    return json.dumps({"technoeconomic_assessment": assess})


def critic_json(findings):
    return json.dumps({"findings": findings})


FINDINGS = [
    {"dimension": "grounding", "severity": "minor",
     "claim": "HCl consumption at 4 M", "issue": "No HCl price in the evidence."},
    {"dimension": "quantification", "severity": "critical",
     "claim": "Nd at $120/kg", "issue": "Evidence gives a 2023 range, not a point."},
]


# --------------------------------------------- 3. multi-file primary data

def test_single_file_summary_is_unchanged(tmp_path):
    comp, _ = _two_tables(tmp_path)
    one = pr.summarize_primary_data({"file_path": str(comp), "metadata_path": None})
    assert one and "Concentration_Percent" in one
    assert "Data file 1 of" not in one          # bare text, as before


def test_multi_file_summary_labels_each_file(tmp_path):
    comp, price = _two_tables(tmp_path)
    out = pr.summarize_primary_data([
        {"file_path": str(comp), "metadata_path": None},
        {"file_path": str(price), "metadata_path": None},
    ])
    assert "## Data file 1 of 2: feedstock.csv" in out
    assert "## Data file 2 of 2: prices.csv" in out
    assert "Concentration_Percent" in out and "Market_Value_USD_kg" in out
    assert out.index("feedstock.csv") < out.index("prices.csv")


def test_multi_file_summary_skips_unparseable_and_keeps_the_rest(tmp_path):
    comp, _ = _two_tables(tmp_path)
    out = pr.summarize_primary_data([
        {"file_path": str(tmp_path / "missing.csv")},
        {"file_path": str(comp)},
    ])
    assert out and "Concentration_Percent" in out
    assert "missing.csv" not in out


def test_summary_handles_empty_inputs():
    assert pr.summarize_primary_data(None) is None
    assert pr.summarize_primary_data([]) is None
    assert pr.summarize_primary_data([{}]) is None


# ------------------------------------- 2. provenance, sidecar, TEA critic

def test_tea_records_strict_tier_evidence_and_critic(tmp_path):
    comp, price = _two_tables(tmp_path)
    model = ScriptedModel([tea_json(), critic_json(FINDINGS)])
    agent = make_agent(tmp_path, model)
    out = tmp_path / "out" / "tea_analysis.json"
    out.parent.mkdir()

    res = agent.perform_technoeconomic_analysis(
        objective="Recover Nd from NdFeB magnets",
        primary_data_set=[str(comp), str(price)],
        external_context="Nd spot price ~$120/kg (2023 range 100-140).",
        additional_context="Target plant: 1 kt/yr.",
        output_json_path=str(out),
    )

    assert not res.get("error")
    assert res["generation_mode"] == "strict"
    g = res["grounding"]
    assert g["mode"] == "strict"
    assert g["primary_data_files"] == ["feedstock.csv", "prices.csv"]
    assert g["has_primary_data"] and g["has_external_literature"]
    assert g["retrieved_context_chars"] > 0

    # critic ran on the same evidence the author saw, strict framing
    assert len(model.calls) == 2
    author, critic = model.calls
    assert "Data file 1 of 2: feedstock.csv" in author and "Data file 2 of 2: prices.csv" in author
    assert "Target plant: 1 kt/yr." in author
    assert "STRICT mode" in critic
    assert "Nd spot price ~$120/kg" in critic and "Data file 2 of 2: prices.csv" in critic
    # findings recorded, critical first, assessment untouched
    assert [f["severity"] for f in res["critic_findings"]] == ["critical", "minor"]
    assert res["technoeconomic_assessment"] == ASSESSMENT

    # sidecar: auditable evidence record next to the JSON
    rec = (out.parent / "tea_analysis.grounding.md").read_text()
    assert "generation mode: **strict**" in rec
    assert "feedstock.csv, prices.csv" in rec
    assert "Nd spot price ~$120/kg" in rec
    saved = json.loads(out.read_text())
    assert saved["grounding"]["mode"] == "strict"
    assert saved["critic_findings"]
    # history snapshot carries the TEA kind
    assert agent.state["plan_history"][-1]["type"] == "technoeconomic_analysis"


def test_tea_fallback_tier_is_recorded_and_critic_told(tmp_path):
    model = ScriptedModel([
        json.dumps({"error": "Insufficient economic context provided."}),
        tea_json(),
        critic_json([]),
    ])
    agent = make_agent(tmp_path, model)
    out = tmp_path / "tea_analysis.json"
    res = agent.perform_technoeconomic_analysis(
        objective="Recover Nd from NdFeB magnets", output_json_path=str(out))
    assert res["generation_mode"] == "fallback"
    assert res["grounding"]["mode"] == "fallback"
    assert "FALLBACK mode" in model.calls[-1]
    assert "critic_findings" not in res           # clean critic -> no key
    assert "generation mode: **fallback**" in (tmp_path / "tea_analysis.grounding.md").read_text()


def test_tea_critic_failure_fails_open(tmp_path):
    model = ScriptedModel([tea_json(), "this is not json at all"])
    agent = make_agent(tmp_path, model)
    res = agent.perform_technoeconomic_analysis(objective="obj")
    assert not res.get("error")
    assert res["technoeconomic_assessment"] == ASSESSMENT
    assert "critic_findings" not in res


def test_tea_prompt_never_gets_the_plan_constraint_coverage_note(monkeypatch):
    seen = {}
    monkeypatch.setattr(pr, "run_rag", lambda **kw: (seen.update(kw), {})[1])
    pr.perform_science_rag(objective="o", instructions=pr.TEA_INSTRUCTIONS, task_name="t",
                           kb_docs=None, model=None, generation_config=None,
                           additional_context="1 kt/yr plant", external_context="lit")
    assert "MANDATORY CONSTRAINT COVERAGE" not in seen["additional_context"]
    # ...while a plan author with both still gets it (unchanged behaviour)
    pr.perform_science_rag(objective="o", instructions=pr.HYPOTHESIS_GENERATION_INSTRUCTIONS,
                           task_name="t", kb_docs=None, model=None, generation_config=None,
                           additional_context="constraint", external_context="lit")
    assert "MANDATORY CONSTRAINT COVERAGE" in seen["additional_context"]


def test_critique_tea_skips_when_no_assessment():
    model = ScriptedModel([])
    assert pr.critique_tea("o", {"error": "x"}, model, None) == {"findings": []}
    assert model.calls == []


# ---------------------------------- 1. the full TEA reaches downstream prompts

def _tools(base: Path, planner=None, **orch_attrs):
    orch = SimpleNamespace(base_dir=base, planner=planner or SimpleNamespace(),
                           objective="Recover Nd from NdFeB magnets",
                           knowledge_dir=None, latest_tea_results=None,
                           _active_output_subdir=None)
    for k, v in orch_attrs.items():
        setattr(orch, k, v)
    return OrchestratorTools(orch)


def _stored_tea(mode="strict", findings=None, n_gaps=2):
    assess = dict(ASSESSMENT)
    assess["data_gaps_for_quantitative_analysis"] = [f"gap {i}" for i in range(n_gaps)]
    return {"summary": assess["summary"], "full_analysis": assess,
            "generation_mode": mode, "critic_findings": findings or [],
            "source_documents": assess["source_documents"], "timestamp": "t"}


def test_tea_context_block_is_none_without_a_tea(tmp_path):
    assert _tools(tmp_path)._tea_context_block() is None


def test_tea_context_block_carries_the_whole_assessment(tmp_path):
    t = _tools(tmp_path, latest_tea_results=_stored_tea(findings=FINDINGS))
    block = t._tea_context_block()
    assert block.startswith("## Techno-Economic Assessment")
    assert "Provenance: KB/literature-sourced (strict mode)" in block
    assert ASSESSMENT["summary"] in block
    assert "HCl consumption at 4 M" in block                  # cost driver
    assert "Dy price volatility" in block                      # risk
    assert "Hydrometallurgy cheaper" in block                  # comparison
    assert "Data gaps for a quantitative TEA" in block and "gap 1" in block
    assert "[quantification] Evidence gives a 2023 range" in block   # caveat
    assert "Minor: [grounding]" in block


def test_tea_context_block_marks_legacy_records_without_a_tier(tmp_path):
    legacy = {"summary": "s", "full_analysis": ASSESSMENT, "timestamp": "t"}
    block = _tools(tmp_path, latest_tea_results=legacy)._tea_context_block()
    assert "Provenance: not recorded" in block
    assert "strict" not in block.split("Summary:")[0]


def test_tea_context_block_flags_fallback_and_caps_lists(tmp_path):
    t = _tools(tmp_path, latest_tea_results=_stored_tea(mode="fallback", n_gaps=11))
    block = t._tea_context_block()
    assert "GENERAL BENCHMARKS" in block and "fallback mode" in block
    assert "gap 7" in block and "gap 8" not in block
    assert "(+3 more in tea_analysis.json)" in block


def test_generate_initial_plan_threads_the_full_tea_block(tmp_path, monkeypatch):
    seen = {}

    def fake_generate_plan(**kw):
        seen.update(kw)
        return {"proposed_experiments": [{"experiment_name": "E1"}], "iteration": 1}

    planner = SimpleNamespace(generate_plan=fake_generate_plan, state=None,
                              _build_skill_context=lambda *a, **k: None)
    t = _tools(tmp_path, planner=planner,
               latest_tea_results=_stored_tea(findings=FINDINGS))
    # keep the test about context threading, not report rendering / ledgers
    monkeypatch.setattr(t, "_emit_plan_report", lambda *a, **k: None)
    monkeypatch.setattr(t, "_adopt_literature", lambda *a, **k: None)
    monkeypatch.setattr(ot, "resolve_n_candidates", lambda *a, **k: 1, raising=False)

    out = json.loads(t.functions_map["generate_initial_plan"](
        specific_objective="Leach Nd from NdFeB", n_candidates=1,
        additional_context="Only 50 mL reactors available."))
    assert out["status"] == "success", out
    assert out["tea_context_included"] is True
    assert seen["additional_context"]["user_context"] == \
        "User Requirements: Only 50 mL reactors available."
    del seen["additional_context"]["user_context"]
    ctx_dict = seen["additional_context"]
    # the TEA rides under its OWN heading, never inside the user's constraints
    assert "user_context" not in ctx_dict
    ctx = ctx_dict["Techno-Economic Assessment (prior TEA step)"]
    assert not ctx.startswith("## ")
    # the WHOLE assessment, not just the summary sentence
    assert "Data gaps for a quantitative TEA" in ctx
    assert "gap 0" in ctx and "Hydrometallurgy cheaper" in ctx
    assert "Provenance: KB/literature-sourced" in ctx
    assert "Evidence gives a 2023 range" in ctx


def test_refine_plan_threads_tea_as_its_own_block_not_literature(tmp_path, monkeypatch):
    seen = {}

    def fake_refine(**kw):
        seen.update(kw)
        return {"proposed_experiments": [{"experiment_name": "E1"}], "iteration": 2}

    planner = SimpleNamespace(refine_plan=fake_refine, state=None)
    t = _tools(tmp_path, planner=planner, latest_tea_results=_stored_tea())
    monkeypatch.setattr(t, "_emit_plan_report", lambda *a, **k: None)
    monkeypatch.setattr(t, "_adopt_literature", lambda *a, **k: None)
    monkeypatch.setattr(t, "_load_campaign_literature", lambda *a, **k: None)
    monkeypatch.setattr(t, "_collect_scalarizer_context", lambda *a, **k: [])

    out = json.loads(t.functions_map["refine_plan_with_results"](
        result_data="Yield was 12%."))
    assert out["status"] == "success", out
    assert "Data gaps for a quantitative TEA" in seen["external_context"]
    assert seen["literature_text"] is None          # provenance stays honest


def test_technical_document_threads_the_tea(tmp_path, monkeypatch):
    seen = {}

    def fake_author(**kw):
        seen.update(kw)
        return {"sections": [{"heading": "Scope", "content": "Body."}]}

    monkeypatch.setattr(ot, "author_technical_document", fake_author)
    planner = SimpleNamespace(kb_docs=None, model=None, generation_config=None,
                              _build_skill_context=lambda *a, **k: None, state=None)
    t = _tools(tmp_path, planner=planner, latest_tea_results=_stored_tea())
    monkeypatch.setattr(t, "_load_campaign_literature", lambda *a, **k: None)

    out = json.loads(t.functions_map["write_technical_document"](
        request="Write a one-page roadmap", filename="roadmap.md"))
    assert out.get("status") == "success", out
    assert "Data gaps for a quantitative TEA" in (seen["additional_context"] or "")


# ------------------------------------ 3b. run_economic_analysis multi-file

def _planner_recording_tea(result_extra=None):
    seen = {}

    def fake_tea(**kw):
        seen.update(kw)
        res = {"technoeconomic_assessment": ASSESSMENT,
               "grounding": {"mode": "strict", "primary_data_files": []},
               "generation_mode": "strict"}
        res.update(result_extra or {})
        return res

    return SimpleNamespace(perform_technoeconomic_analysis=fake_tea), seen


def test_run_economic_analysis_folder_uses_every_table(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    _two_tables(data)
    planner, seen = _planner_recording_tea()
    t = _tools(tmp_path, planner=planner)
    out = json.loads(t.functions_map["run_economic_analysis"](
        focus_topic="NdFeB recycling", primary_data_set=str(data)))
    assert out["status"] == "success", out
    names = [Path(e["file_path"]).name for e in seen["primary_data_set"]]
    assert names == ["feedstock.csv", "prices.csv"]
    assert seen["additional_context"] is None


def test_run_economic_analysis_comma_list_and_single_file(tmp_path):
    comp, price = _two_tables(tmp_path)
    planner, seen = _planner_recording_tea()
    t = _tools(tmp_path, planner=planner)
    t.functions_map["run_economic_analysis"](
        primary_data_set=f"{comp}, {price}", additional_context="1 kt/yr plant")
    assert [Path(e["file_path"]).name for e in seen["primary_data_set"]] == \
        ["feedstock.csv", "prices.csv"]
    assert seen["additional_context"] == "1 kt/yr plant"
    # a single file keeps the historical dict shape
    t.functions_map["run_economic_analysis"](primary_data_set=str(comp))
    assert isinstance(seen["primary_data_set"], dict)
    assert Path(seen["primary_data_set"]["file_path"]).name == "feedstock.csv"


def test_run_economic_analysis_empty_folder_errors(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    planner, _ = _planner_recording_tea()
    t = _tools(tmp_path, planner=planner)
    out = json.loads(t.functions_map["run_economic_analysis"](primary_data_set=str(empty)))
    assert out["status"] == "error" and "No data files" in out["message"]


def test_run_economic_analysis_result_carries_tier_gaps_and_caveats(tmp_path):
    planner, _ = _planner_recording_tea({
        "grounding": {"mode": "fallback", "primary_data_files": []},
        "generation_mode": "fallback",
        "critic_findings": FINDINGS})
    t = _tools(tmp_path, planner=planner)
    out = json.loads(t.functions_map["run_economic_analysis"](focus_topic="x"))
    assert out["generation_mode"] == "fallback"
    assert "FALLBACK tier" in out["warning"]
    assert out["data_gaps_for_quantitative_analysis"] == ASSESSMENT["data_gaps_for_quantitative_analysis"]
    # caveats are the planner's (already ordered) findings, rendered
    assert set(out["caveats"]) == {"Minor: [grounding] No HCl price in the evidence.",
                                   "[quantification] Evidence gives a 2023 range, not a point."}
    assert out["grounding_record"].endswith("tea_analysis.grounding.md")
    stored = t.orch.latest_tea_results
    assert stored["generation_mode"] == "fallback"
    assert stored["critic_findings"] == FINDINGS
    assert stored["full_analysis"] == ASSESSMENT
    # and the downstream block now says so
    assert "GENERAL BENCHMARKS" in t._tea_context_block()


# ---------------------------------------------------------- HTML card

def test_tea_card_shows_gaps_comparison_provenance_and_caveats():
    gen = HTMLReportGenerator.__new__(HTMLReportGenerator)
    plan = {"technoeconomic_assessment": ASSESSMENT,
            "grounding": {"mode": "strict", "primary_data_files": ["prices.csv"]},
            "critic_findings": FINDINGS}
    card = gen._render_tea(plan)
    assert "Data Gaps for a Quantitative TEA" in card and "Plant CAPEX at 1 kt/yr" in card
    assert "Comparison to Alternatives" in card and "Hydrometallurgy cheaper" in card
    assert "Grounded (strict tier)" in card and "market_report.txt" in card
    assert "Primary data: prices.csv" in card
    assert "Critic Caveats" in card and "Evidence gives a 2023 range" in card

    fb = gen._render_tea({"technoeconomic_assessment": ASSESSMENT,
                          "grounding": {"mode": "fallback"}})
    assert "Fallback tier" in fb and "Grounded (strict tier)" not in fb
    assert "Critic Caveats" not in fb


# =============================================================== stress / edge

def test_critic_output_is_sanitised_and_sorted(tmp_path):
    """Malformed findings (non-dicts, missing issue, unknown severity) are
    dropped or ranked as minor — never a crash, never a blank caveat line."""
    messy = json.dumps({"findings": [
        "just a string", 42, {"severity": "critical"},             # no issue -> dropped
        {"dimension": "scope", "issue": "no severity given"},      # unknown -> after critical
        {"dimension": "grounding", "severity": "critical", "issue": "Unsupported $ figure."},
        {"dimension": "consistency", "severity": "bogus", "issue": "Odd severity."},
    ]})
    model = ScriptedModel([tea_json(), messy])
    agent = make_agent(tmp_path, model)
    res = agent.perform_technoeconomic_analysis(objective="o")
    f = res["critic_findings"]
    assert [x["issue"] for x in f] == ["Unsupported $ figure.", "no severity given", "Odd severity."]
    assert all(isinstance(x, dict) for x in f)
    # findings returned as a dict instead of a list -> treated as clean
    model = ScriptedModel([tea_json(), json.dumps({"findings": {"a": 1}})])
    res = make_agent(tmp_path / "b", model).perform_technoeconomic_analysis(objective="o")
    assert "critic_findings" not in res


def test_non_string_and_null_assessment_fields_do_not_crash(tmp_path):
    weird = {"summary": None, "key_cost_drivers": [{"k": 1}, 3.5, "", None],
             "potential_benefits_or_revenue": None, "economic_risks": "not a list",
             "comparison_to_alternatives": None,
             "data_gaps_for_quantitative_analysis": [], "source_documents": None}
    tea = {"summary": None, "full_analysis": weird, "generation_mode": "strict",
           "critic_findings": [{"issue": "x"}]}           # no dimension / severity
    block = _tools(tmp_path, latest_tea_results=tea)._tea_context_block()
    assert "{'k': 1}" in block and "3.5" in block
    assert "- not a list" in block and "- n\n" not in block  # string field = one item
    assert "[] x" in block                                   # format_caveats tolerates it
    card = HTMLReportGenerator.__new__(HTMLReportGenerator)._render_tea(
        {"technoeconomic_assessment": weird, "grounding": {"mode": "strict"}})
    assert "Data Gaps" not in card                          # empty list -> section omitted
    assert "Grounded (strict tier)" in card
    assert card.count("<li class=''>") == 1                  # 'not a list' -> one risk item
    # assessment that is not even a dict
    assert "oops" in HTMLReportGenerator.__new__(HTMLReportGenerator)._render_tea(
        {"technoeconomic_assessment": "oops"})
    # full_analysis missing entirely
    assert _tools(tmp_path, latest_tea_results={"summary": "s"})._tea_context_block()


def test_legacy_tea_record_renders_neutral_provenance():
    card = HTMLReportGenerator.__new__(HTMLReportGenerator)._render_tea(
        {"technoeconomic_assessment": ASSESSMENT})        # no grounding, no mode
    assert "Provenance not recorded" in card
    assert "Grounded (strict tier)" not in card and "Fallback tier" not in card


def test_resolve_tabular_inputs_edge_cases(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    comp, price = _two_tables(data)
    pd.DataFrame({"a": [1]}).to_excel(data / "book.xlsx", index=False)
    (data / "notes.txt").write_text("not a table")
    (data / "meta.json").write_text("{}")
    t = _tools(tmp_path)
    # folder with trailing slash + mixed types: tabular only, natural order
    entries, err = t._resolve_tabular_inputs(str(data) + "/")
    assert err is None
    assert [Path(e["file_path"]).name for e in entries] == ["feedstock.csv", "prices.csv", "book.xlsx"] \
        or sorted(Path(e["file_path"]).name for e in entries) == ["book.xlsx", "feedstock.csv", "prices.csv"]
    # duplicates (folder + a file inside it + same file twice) collapse
    entries, _ = t._resolve_tabular_inputs(f"{data}, {comp}, {comp}")
    assert len(entries) == 3
    # whitespace-only spec
    entries, err = t._resolve_tabular_inputs("  ,  ")
    assert entries is None and "No data files resolved" in json.loads(err)["message"]
    # a missing path surfaces _resolve_data_path's error (with suggestions)
    entries, err = t._resolve_tabular_inputs(f"{comp}, {data / 'nope.csv'}")
    assert entries is None and json.loads(err)["status"] == "error"
    # folder of only non-tabular files
    other = tmp_path / "other"; other.mkdir(); (other / "a.txt").write_text("x")
    entries, err = t._resolve_tabular_inputs(str(other))
    assert entries is None and "No data files" in json.loads(err)["message"]


def test_too_many_tables_is_refused_with_the_list(tmp_path):
    data = tmp_path / "many"; data.mkdir()
    for i in range(12):
        pd.DataFrame({"x": [i]}).to_csv(data / f"t{i}.csv", index=False)
    t = _tools(tmp_path)
    entries, err = t._resolve_tabular_inputs(str(data))
    assert entries is None
    e = json.loads(err)
    assert "12 data files" in e["message"] and len(e["available_files"]) == 12
    assert "comma-separated" in e["hint"]
    # exactly the cap is fine
    (data / "t11.csv").unlink(); (data / "t10.csv").unlink()
    entries, err = t._resolve_tabular_inputs(str(data))
    assert err is None and len(entries) == 10


def test_header_only_and_garbage_files_are_tolerated(tmp_path):
    good, _ = _two_tables(tmp_path)
    (tmp_path / "empty.csv").write_text("a,b\n")
    (tmp_path / "garbage.csv").write_bytes(b"\xff\xfe\x00\x00binary\x00junk")
    out = pr.summarize_primary_data([
        {"file_path": str(tmp_path / "empty.csv")},
        {"file_path": str(tmp_path / "garbage.csv")},
        {"file_path": str(good)},
    ])
    assert out and "Concentration_Percent" in out
    # the garbage file never poisons the good one
    assert "feedstock.csv" in out


def test_tea_with_only_missing_data_files_runs_without_data(tmp_path):
    model = ScriptedModel([tea_json(), critic_json([])])
    agent = make_agent(tmp_path, model)
    res = agent.perform_technoeconomic_analysis(
        objective="o", primary_data_set=[str(tmp_path / "a.csv"), str(tmp_path / "b.csv")])
    assert not res.get("error")
    assert res["grounding"]["primary_data_files"] == []
    assert res["grounding"]["has_primary_data"] is False
    assert "Primary Experimental Data" not in model.calls[0]


def test_tea_author_error_short_circuits_provenance_and_critic(tmp_path):
    model = ScriptedModel([json.dumps({"error": "Model refused for some other reason"})])
    agent = make_agent(tmp_path, model)
    out = tmp_path / "tea.json"
    res = agent.perform_technoeconomic_analysis(objective="o", output_json_path=str(out))
    assert res.get("error")
    assert "grounding" not in res and "critic_findings" not in res
    assert len(model.calls) == 1                         # no critic call
    assert not (tmp_path / "tea.grounding.md").exists()
    assert not agent.state["plan_history"]


def test_tea_first_then_plan_keeps_iterations_distinct(tmp_path):
    """TEA-first numbering is unchanged: TEA is iteration 0, the first plan 1."""
    model = ScriptedModel([tea_json(), critic_json([])])
    agent = make_agent(tmp_path, model)
    agent.perform_technoeconomic_analysis(objective="o")
    assert agent.state["iteration_index"] == 0
    assert agent.state["plan_history"][-1]["iteration"] == 0
    assert agent.state["plan_history"][-1]["stage"] == "TEA Initial"


def test_tea_mid_campaign_does_not_reset_the_iteration_counter(tmp_path):
    """Regression guard: a TEA run AFTER a plan used to set iteration_index
    back to 0, so the next plan was numbered 1 again and overwrote the
    existing iteration-1 card in the report."""
    model = ScriptedModel([tea_json(), critic_json([])])
    agent = make_agent(tmp_path, model)
    agent.state = agent._initialize_state(objective="o", knowledge_paths=None, code_paths=None,
                                          primary_data_set=None, image_paths=[], image_descriptions=None)
    agent.state["iteration_index"] = 2
    agent.state["plan_history"] = [
        {"type": "lab", "iteration": 1, "proposed_experiments": [{"experiment_name": "E1"}]},
        {"type": "lab", "iteration": 2, "proposed_experiments": [{"experiment_name": "E2"}]},
    ]
    agent.perform_technoeconomic_analysis(objective="o")
    assert agent.state["iteration_index"] == 2
    tea = agent.state["plan_history"][-1]
    assert tea["iteration"] == 2 and tea["stage"] == "TEA Update"


def test_full_report_renders_tea_and_plan_cards_without_collision(tmp_path):
    state = {
        "session_id": "s", "objective": "obj", "start_time": "2026-09-12T00:00:00",
        "plan_history": [
            {"type": "technoeconomic_analysis", "iteration": 0, "stage": "TEA Initial",
             "technoeconomic_assessment": ASSESSMENT,
             "grounding": {"mode": "fallback"}, "critic_findings": FINDINGS},
            {"type": "lab", "iteration": 1, "proposed_experiments": [
                {"experiment_name": "E1", "hypothesis": "H", "experimental_steps": ["s1"],
                 "required_equipment": ["x"], "expected_outcome": "o", "justification": "j"}],
             "critic_findings": [{"dimension": "physics", "severity": "minor", "issue": "plan caveat"}]},
            # a mid-campaign TEA sharing iteration 1 with the plan
            {"type": "technoeconomic_analysis", "iteration": 1, "stage": "TEA Update",
             "technoeconomic_assessment": dict(ASSESSMENT, summary="Updated TEA"),
             "grounding": {"mode": "strict"}},
        ],
        "experimental_results": [],
    }
    out = tmp_path / "plan.html"
    HTMLReportGenerator(state).generate(str(out))
    h = out.read_text()
    assert h.count("TECHNO-ECONOMIC ANALYSIS") == 2 and h.count("EXPERIMENTAL STRATEGY") == 1
    assert "Updated TEA" in h and "Fallback tier" in h and "Grounded (strict tier)" in h
    assert "Critic Caveats on this Assessment" in h          # TEA caveats
    assert "plan caveat" in h                                  # plan caveats still render
    assert h.count("Evidence gives a 2023 range") == 1         # rendered once, not duplicated
    # the TEA precedes the plan it shares an iteration with
    assert h.index("Updated TEA") < h.index("plan caveat")


def test_context_block_with_empty_assessment_still_yields_provenance(tmp_path):
    t = _tools(tmp_path, latest_tea_results={"summary": "", "full_analysis": {},
                                             "generation_mode": "fallback"})
    block = t._tea_context_block()
    assert "GENERAL BENCHMARKS" in block
    assert "Data gaps" not in block and "Key cost drivers" not in block


def test_run_economic_analysis_planner_error_and_exception(tmp_path):
    planner = SimpleNamespace(perform_technoeconomic_analysis=lambda **kw: {"error": "KB Init Failed"})
    t = _tools(tmp_path, planner=planner)
    out = json.loads(t.functions_map["run_economic_analysis"](focus_topic="x"))
    assert out == {"status": "error", "message": "KB Init Failed"}
    assert t.orch.latest_tea_results is None               # nothing stored on failure

    def boom(**kw):
        raise RuntimeError("model exploded")
    t = _tools(tmp_path, planner=SimpleNamespace(perform_technoeconomic_analysis=boom))
    out = json.loads(t.functions_map["run_economic_analysis"](focus_topic="x"))
    assert out["status"] == "error" and "model exploded" in out["message"]


def test_run_economic_analysis_inline_literature_text_and_folder_slash(tmp_path):
    data = tmp_path / "data"; data.mkdir(); _two_tables(data)
    planner, seen = _planner_recording_tea()
    t = _tools(tmp_path, planner=planner)
    out = json.loads(t.functions_map["run_economic_analysis"](
        primary_data_set=str(data) + "/", literature_context="Nd price is $120/kg (inline text)."))
    assert out["status"] == "success"
    assert seen["external_context"] == "Nd price is $120/kg (inline text)."
    assert len(seen["primary_data_set"]) == 2


def test_unicode_and_html_are_escaped_in_card_and_survive_block(tmp_path):
    assess = dict(ASSESSMENT, summary="Coût ≈ 12 €/kg <b>bold</b>",
                  key_cost_drivers=["<script>alert(1)</script>"])
    card = HTMLReportGenerator.__new__(HTMLReportGenerator)._render_tea(
        {"technoeconomic_assessment": assess, "grounding": {"mode": "strict"}})
    assert "&lt;script&gt;" in card and "<script>alert" not in card
    block = _tools(tmp_path, latest_tea_results={"summary": assess["summary"],
                                                 "full_analysis": assess,
                                                 "generation_mode": "strict"})._tea_context_block()
    assert "Coût ≈ 12 €/kg" in block


def test_damaged_tea_state_never_crashes_downstream(tmp_path):
    """A corrupted checkpoint can hand back a string, a list or a dict with
    the wrong shapes; plan generation and the run_task summary must shrug."""
    from scilink.agents.planning_agents.user_interface import format_caveats
    for bad in ("a string", ["list"], 42, {"full_analysis": "str", "critic_findings": {"x": 1}},
                {"full_analysis": {"data_gaps_for_quantitative_analysis": "one gap"},
                 "critic_findings": ["str", None, {"issue": "ok"}]}):
        t = _tools(tmp_path, latest_tea_results=bad)
        block = t._tea_context_block()          # None or a string, never an exception
        assert block is None or isinstance(block, str)
    assert format_caveats({"x": 1}) == [] and format_caveats("nope") == []
    assert format_caveats(["str", None, {"issue": "ok"}]) == ["[] ok"]


def test_run_task_key_findings_surface_tea_tier_summary_and_gaps(tmp_path, monkeypatch):
    """The meta agent reads a planning delegation through run_task; the TEA's
    tier, summary and (up to five) data gaps must be in key_findings, and a
    damaged record must not break the summary."""
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    (tmp_path / "data").mkdir()
    try:
        o = PlanningOrchestratorAgent(objective="x", base_dir=str(tmp_path / "s"), api_key=None,
                                      model_name="bedrock/us.anthropic.claude-opus-4-8",
                                      autonomy_level=AutonomyLevel.AUTONOMOUS,
                                      data_dir=str(tmp_path / "data"))
    except Exception as e:  # noqa: BLE001 - constructor needs an installed LLM stack
        pytest.skip(f"orchestrator not constructible here: {e}")
    monkeypatch.setattr(o, "chat", lambda prompt: "done")
    o.latest_tea_results = {"summary": "Conditionally viable.",
                            "full_analysis": {"data_gaps_for_quantitative_analysis":
                                              [f"gap {i}" for i in range(7)]},
                            "generation_mode": "fallback", "critic_findings": []}
    r = o.run_task("summarise")
    kf = r["key_findings"]
    assert any("FALLBACK tier" in k for k in kf)
    assert "TEA summary: Conditionally viable." in kf
    assert sum(k.startswith("TEA data gap:") for k in kf) == 5
    # legacy record without a tier
    o.latest_tea_results = {"summary": "s", "full_analysis": {}}
    assert any("provenance not recorded" in k for k in o.run_task("x")["key_findings"])
    # damaged record
    o.latest_tea_results = {"full_analysis": "oops", "generation_mode": "strict"}
    assert any("KB/literature-grounded" in k for k in o.run_task("x")["key_findings"])
    o.latest_tea_results = "corrupt"
    assert not any("Techno-economic" in k for k in o.run_task("x")["key_findings"])
