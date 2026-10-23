"""Fusion's independence count is a number computed from the ledger's own
edges (``fanout.independent_support_of``): a re-analysis that cites a fusion
(fusion feedback), a check given a peer's finding through ``context_from``,
a branch steered by another, and a mesh of co-registered operands, which is
a shared dataset and not a coupling of findings. Keyed by index, so a
repeated label and a fusion of fusions count right.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    monkeypatch.setattr(fo, "_llm_json", _fusion_llm)
    _fusion_llm.prompts.clear()
    return m


def _fusion_llm(orch, prompt, extra_parts=None):
    _fusion_llm.prompts.append(prompt)
    return {"detailed_analysis": "fused narrative",
            "scientific_claims": [{"claim": "anatase throughout"}]}


_fusion_llm.prompts = []


def _branch(meta, index, label, context_from=None, context=None):
    e = meta._open_delegation("analysis", f"task {label}", context, context_from, label)
    with meta._fanout_lock:
        e["fanout"] = True
        e["parallel_group"] = "fanout_1"
    meta._close_delegation(e, {"status": "success", "summary": f"{label} summary",
                               "key_findings": [f"[analysis_{index}] anatase ({label})"],
                               "analyses": [{"analysis_id": f"analysis_{index}", "status": "success",
                                             "verified": True}],
                               "files_produced": [], "warnings": []})
    return e


def _support(meta, indices):
    out = json.loads(fo.fuse_delegations(meta, indices))
    assert out["status"] == "success", out
    return out


def test_two_independent_branches_count_two_and_a_fusions_reader_inherits_its_inputs(meta):
    _branch(meta, 1, "Raman A7")
    _branch(meta, 2, "XRD A7")
    out = _support(meta, [1, 2])
    assert out["independent_support"] == {"count": 2, "raw": 2, "dependent": {}, "by_index": {}, "exact": True}
    assert "INDEPENDENT SUPPORT (computed, not judged): 2 of 2 — the largest set of branches" in _fusion_llm.prompts[-1]
    fusion = meta._delegation_ledger[-1]
    assert fusion["mode"] == "fusion"
    # a re-analysis citing the fusion rests on everything the fusion fused
    e3 = meta._open_delegation("analysis", "re-analyze Raman", None, [fusion["index"]], "Raman A7 again")
    assert e3["informed_via"] == "fusion_feedback"
    with meta._fanout_lock:
        e3["fanout"] = True
        e3["parallel_group"] = "fanout_1"
    meta._close_delegation(e3, {"status": "success", "summary": "again",
                                "key_findings": ["[analysis_3] anatase (again)"],
                                "analyses": [{"analysis_id": "analysis_3", "status": "success", "verified": True}],
                                "files_produced": [], "warnings": []})
    out = _support(meta, [1, 2, e3["index"]])
    assert out["independent_support"] == {"count": 2, "raw": 3, "exact": True,
                                          "dependent": {"'Raman A7 again' (#4)": ["'Raman A7' (#1)", "'XRD A7' (#2)"]},
                                          "by_index": {"4": [1, 2]}}
    assert any("had read or been given findings of" in c for c in out["caveats"])
    assert "not judged): 2 of 3 — the largest set" in _fusion_llm.prompts[-1]


def test_one_branch_steered_by_the_other_counts_one(meta):
    _branch(meta, 1, "Raman A7")
    b = _branch(meta, 2, "XRD A7")
    with meta._fanout_lock:
        b["informed_by"] = ["Raman A7"]
        b["informed_via"] = "steering"
    out = _support(meta, [1, 2])
    assert out["independent_support"] == {"count": 1, "raw": 2, "dependent": {"'XRD A7' (#2)": ["'Raman A7' (#1)"]},
                                          "by_index": {"2": [1]}, "exact": True}
    assert any("not an independent confirmation" in c for c in out["caveats"])


def test_independence_counts_the_ledgers_own_edges(meta):
    """Review of #702: a check given a peer's finding in `context`, a
    re-analysis citing an analysis (not a fusion), and an entry stamped
    informed_by were all counted independent. Keyed by index, so duplicate
    labels and a fusion of fusions count right."""
    a = _branch(meta, 1, "fit")
    b = _branch(meta, 2, "fit")                                      # a duplicate label
    # a check told a peer's finding through context (declared context_from)
    _branch(meta, 3, "fit", context_from=[1], context={"peer": a["key_findings"]})
    out = _support(meta, [1, 2, 3])
    assert out["independent_support"] == {"count": 2, "raw": 3, "dependent": {"'fit' (#3)": ["'fit' (#1)"]},
                                          "by_index": {"3": [1]}, "exact": True}
    # an informed_by stamp (the fan-out's steering) is an edge too
    with meta._fanout_lock:
        b["informed_by"] = ["fit"]; b["informed_via"] = "steering"
    out = _support(meta, [1, 2])
    assert out["independent_support"]["by_index"] == {"2": [1]}
    # a fusion of two fusions: raw 2, both labelled "cross-dataset fusion"
    f1 = meta._delegation_ledger[-2]["index"]; f2 = meta._delegation_ledger[-1]["index"]
    out = _support(meta, [f1, f2])
    # (they share inputs but neither cites the other: raw 2, and the count is
    # about edges, not common ancestry — a caveat the prompt states)
    assert out["independent_support"]["raw"] == 2
    assert set(out["independent_support"]["dependent"]) <= {f"'cross-dataset fusion' (#{f1})", f"'cross-dataset fusion' (#{f2})"}
    # the prompt says what the count leaves out
    assert "does NOT see a finding pasted into a task" in _fusion_llm.prompts[-1]


def test_meshed_branches_count_once_and_labels_bind_only_earlier_entries(meta):
    """Round 2 of #702: three co-registered branches reported 0 of 3 (a shared
    dataset is not a finding); a later independent run with the same label
    made an earlier branch dependent; a mention of analysis_results.json
    inferred a dependency on every analysis."""
    a, b, c = _branch(meta, 1, "EELS"), _branch(meta, 2, "HAADF"), _branch(meta, 3, "EDS")
    for e in (a, b, c):
        with meta._fanout_lock:
            e["informed_by"] = [x["label"] for x in (a, b, c) if x is not e]
            e["informed_via"] = "co_registered_operands"
    out = _support(meta, [1, 2, 3])
    assert out["independent_support"] == {"count": 3, "raw": 3, "dependent": {}, "by_index": {}, "exact": True}
    assert "A shared dataset (co-registered operands) is NOT a coupling" in _fusion_llm.prompts[-1]
    # mesh PLUS steering (round 3): b and c steered by a keep the steering edge; a is the one
    # independent root, so 2 of 3 — the mesh must not swallow the steering
    with meta._fanout_lock:
        b["steered_by"] = ["EELS"]; c["steered_by"] = ["EELS"]
        b["informed_via"] = c["informed_via"] = "co_registered_operands+steering"
    out = _support(meta, [1, 2, 3])
    assert out["independent_support"]["count"] == 2 and out["independent_support"]["by_index"] == {"2": [1], "3": [1]}
    # steering from a HIGHER slot inside one fan-out: the slots are created together
    with meta._fanout_lock:
        b.pop("steered_by"); c.pop("steered_by")
        b["informed_via"] = c["informed_via"] = "co_registered_operands"
        a["steered_by"] = ["HAADF"]                                 # #1 steered by #2
    out = _support(meta, [1, 2])
    assert out["independent_support"]["count"] == 1 and out["independent_support"]["by_index"] == {"1": [2]}
    # the sibling INDEX wins over the label when both are stamped (labels can repeat in a group)
    with meta._fanout_lock:
        a["steered_by_index"] = [3]
    assert _support(meta, [1, 2, 3])["independent_support"]["by_index"] == {"1": [3]}
    with meta._fanout_lock:
        a.pop("steered_by"); a.pop("steered_by_index")
    # a stamp from before steered_by that says "+steering": every label is an edge (errs low)
    with meta._fanout_lock:
        b["informed_via"] = "co_registered_operands+steering"
    assert _support(meta, [1, 2, 3])["independent_support"]["by_index"] == {"2": [1, 3]}
    with meta._fanout_lock:
        b["informed_via"] = "co_registered_operands"
    # three mutually informed (steered) branches, legacy stamp: one observation, not none
    for e in (a, b, c):
        with meta._fanout_lock:
            e["informed_via"] = "steering"
    out = _support(meta, [1, 2, 3])
    assert out["independent_support"]["count"] == 1 and out["independent_support"]["by_index"] == {
        "1": [2, 3], "2": [1, 3], "3": [1, 2]}
    # a LATER independent run that reuses a label does not couple the earlier branch
    with meta._fanout_lock:
        for e in (a, b, c):
            e.pop("informed_by"); e.pop("informed_via")
    later = _branch(meta, 9, "EELS")                                  # same label, later, independent
    with meta._fanout_lock:
        later["parallel_group"] = "fanout_9"                            # another fan-out
        a["informed_by"] = ["EELS"]; a["informed_via"] = "steering"     # can only mean its OWN group's "EELS"
    out = _support(meta, [1, later["index"]])
    assert out["independent_support"]["count"] == 2 and out["independent_support"]["by_index"] == {}
    # a task mentioning analysis_results.json infers no analysis id
    assert meta._analysis_ids_of({"files_produced": ["/s/results/analysis_results.json",
                                                     "/s/results/analysis_x_CurveFit_20260930_143600_001/a.json"]}) == [
        "analysis_x_CurveFit_20260930_143600_001"]


def test_the_counts_exactness_travels():
    """Exact up to twelve branches; a greedy count beyond says "at least"."""
    assert fo.independent_set_size([1, 2, 3], {1: {2}, 2: {3}, 3: set()}) == (2, True)
    big = list(range(13))
    assert fo.independent_set_size(big, {k: set() for k in big}) == (13, False)
