"""Offline tests for issue #323 Channel B (curve-fitting scope).

Covers the two load-bearing behaviors that need no API key:
- the ID-mode literature gates in the curve-fitting controllers (D2), and
- `refine_interpretation` registration + record plumbing (revision storage,
  get_recommendations preferring the latest revision) via source inspection
  and a stubbed orchestrator.
"""

import json
import re
from pathlib import Path

import scilink.agents.exp_agents.controllers.curve_fitting_controllers as cc
import scilink.agents.exp_agents.analysis_orchestrator_tools as tools_mod

CTRL_SRC = Path(cc.__file__).read_text()
TOOLS_SRC = Path(tools_mod.__file__).read_text()


class TestIdModeLiteratureGates:
    def test_all_literature_context_consumers_are_id_gated(self):
        # Every site that injects literature_context into a prompt/context
        # must carry the identification-mode gate (D2). Reads of the field
        # for provenance/reporting are exempt.
        lines = CTRL_SRC.split("\n")
        injectors = [
            i for i, l in enumerate(lines)
            if re.search(r'if state\.get\("literature_context"\)', l)
            # injection sites feed the value into a prompt/context on the next
            # line; the in-pipeline search controller's skip guard does not
            and re.search(r"append|extend", lines[i + 1])
        ]
        assert len(injectors) == 4, "expected planner x2 + script-gen + synthesis sites"
        for i in injectors:
            assert 'task_mode") != "identification"' in lines[i], (
                f"ungated literature injection at line {i + 1}: {lines[i].strip()}"
            )

    def test_agent_docstring_no_longer_claims_stage2_consumption(self):
        import scilink.agents.exp_agents.curve_fitting_agent as agent_mod
        src = Path(agent_mod.__file__).read_text()
        assert "Stage-2 candidate enumeration still consumes it" not in src


class TestRefineInterpretationPlumbing:
    def test_tool_is_registered(self):
        assert 'name="refine_interpretation"' in TOOLS_SRC

    def test_storage_is_append_only(self):
        assert 'setdefault(\n                "interpretation_revisions", []).append' in TOOLS_SRC

    def test_effective_full_result_helper(self):
        from scilink.agents.exp_agents.analysis_orchestrator_tools import _effective_full_result
        rec = {"full_result": {"detailed_analysis": "original", "x": 1},
               "interpretation_revisions": [{"revised_analysis": "v1"},
                                            {"revised_analysis": "v2"}]}
        eff = _effective_full_result(rec)
        assert eff["detailed_analysis"] == "v2" and eff["x"] == 1
        assert rec["full_result"]["detailed_analysis"] == "original"  # no mutation
        assert _effective_full_result({"full_result": {"detailed_analysis": "o"}})[
            "detailed_analysis"] == "o"  # no revisions -> unchanged

    def test_all_record_consumers_prefer_latest_revision(self):
        # every downstream consumer of a record's interpretation must go
        # through the helper: recommendations, DFT recommendations, and the
        # multi-source context builder. (The post-run summary preview reads
        # the fresh analyze() return, where no revision can exist yet.)
        for func in ("def get_recommendations", "def recommend_dft_structures"):
            idx = TOOLS_SRC.index(func)
            assert "_effective_full_result" in TOOLS_SRC[idx: idx + 4000], func
        assert TOOLS_SRC.count("_effective_full_result(record)") >= 3

    def test_series_features_are_trend_conditioned(self):
        idx = TOOLS_SRC.index("def refine_interpretation")
        body = TOOLS_SRC[idx: TOOLS_SRC.index("def assess_novelty")]
        # series: trends preferred; single: fitting_parameters fallback
        assert "parameter_trends" in body
        assert "fitting_parameters" in body
        assert body.index("parameter_trends") < body.index('elif full_result.get("fitting_parameters")')


    def test_revision_is_persisted_human_readably(self):
        idx = TOOLS_SRC.index("def refine_interpretation")
        body = TOOLS_SRC[idx: TOOLS_SRC.index("def assess_novelty")]
        # revised text appended to the on-disk revision document...
        assert "Literature-Refined Interpretation" in body
        # ...and a companion HTML lands next to the original report when the
        # record carries its output_directory (append-only: separate file)
        assert "Interpretation_Revision_" in body
        assert 'record.get("output_directory")' in body


class TestOrchestratorSurface:
    def test_followups_suggest_refinement(self):
        import scilink.agents.exp_agents.analysis_orchestrator as orch_mod
        src = Path(orch_mod.__file__).read_text()
        assert "refine_interpretation" in src
        assert "interpretation_revisions" in src

    def test_id_mode_exception_in_both_literature_offers(self):
        import scilink.agents.exp_agents.analysis_orchestrator as orch_mod
        src = Path(orch_mod.__file__).read_text()
        assert src.count("refine_interpretation") >= 3  # copilot + autonomous offers + followup


class TestNoveltyRipple:
    def _refine_body(self):
        idx = TOOLS_SRC.index("def refine_interpretation")
        return TOOLS_SRC[idx: TOOLS_SRC.index("def assess_novelty")]

    def test_revision_stales_prior_novelty_assessment(self):
        body = self._refine_body()
        assert 'prior_novelty["stale"] = True' in body
        assert "staled_by_revision" in body

    def test_revision_optionally_revises_claims(self):
        body = self._refine_body()
        assert "revised_claims" in body
        assert "has_anyone_question" in body  # same claim schema demanded

    def test_assess_novelty_prefers_revised_claims_and_stamps_coverage(self):
        idx = TOOLS_SRC.index("def assess_novelty")
        body = TOOLS_SRC[idx: idx + 6000]
        assert 'rev.get("revised_claims")' in body
        assert "assessed_at_revision" in body
        assert "claims_source" in body

    def test_staleness_is_visible_to_the_agent(self):
        # list_results + get_recommendations both surface the stale flag
        assert TOOLS_SRC.count("novelty_assessment_stale") >= 2
