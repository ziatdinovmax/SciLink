"""Stage 3 of the swarm: reactions.

The harness drives the REAL coordinator (``run_swarm``), the real meta ledger
and the real board; only the workers are stand-ins. Each stand-in returns a
``run_task`` result shaped as the mode orchestrators return it, so what gets
posted goes through ``records_for`` unchanged: an analysis row with a
``verified`` verdict posts a verified claim, a simulation structure with
``validation_status`` posts a structure, a worker's ``suggested_followups``
post ``task_request`` records.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.meta_agent import board as board_mod
from scilink.agents.meta_agent import reactions, swarm
from scilink.agents.meta_agent.board import Board
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent


class ScriptedWorker:
    """A worker whose result is a function of (mode, task): the scenario
    decides what each item posts."""

    def __init__(self, mode, base_dir, script):
        self.mode, self.base_dir, self.script = mode, Path(base_dir), script

    def run_task(self, task, context=None, autonomy=None):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        spec = self.script(self.mode, task, context) or {}
        res = {"status": "success", "summary": f"{self.mode}: {task[:60]}", "key_findings": [],
               "files_produced": [], "suggested_followups": [], "warnings": []}
        if self.mode == "analysis":
            aid = spec.get("analysis_id", "a1")
            res["key_findings"] = [f"[{aid}] {t}" for t in spec.get("claims", [])]
            res["analyses"] = [{"analysis_id": aid, "status": "success",
                                "verified": spec.get("verified", True),
                                "reason": "approved by the analysis verifier" if spec.get("verified", True) else "salvaged",
                                "output_directory": str(self.base_dir)}]
        elif self.mode == "simulation":
            res["structures"] = [{"structure_path": str(self.base_dir / "POSCAR"), "slug": "cell",
                                  "description": spec.get("structure", "a cell"),
                                  "validation_status": spec.get("validation", "success")}]
        elif self.mode == "planning":
            res["plan_review"] = {"human_review": spec.get("approved", False), "written_here": True,
                                  "hypotheses": spec.get("hypotheses", [])}
        res["suggested_followups"] = list(spec.get("followups", []))
        return res


@pytest.fixture(scope="module", autouse=True)
def _mode_modules():
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    import scilink.agents.planning_agents.planning_orchestrator  # noqa: F401
    import scilink.agents.sim_agents.simulation_orchestrator  # noqa: F401


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    return m


def _script(monkeypatch, script):
    built = []

    def build(orch, mode, base_dir, **kw):
        w = ScriptedWorker(mode, base_dir, script)
        built.append(w)
        return w
    monkeypatch.setattr(swarm, "build_child", build)
    return built


S = "TiO2 A7"
ITEMS = [{"mode": "analysis", "task": "fit the Raman spectrum", "label": "Raman A7", "subject": S},
         {"mode": "planning", "task": "plan the purity check", "label": "purity plan", "subject": "other"}]
SIM_ON_CLAIM = {"on": {"kind": "claim", "subject": S},
                "enqueue": {"mode": "simulation", "label": "cell for {from.label}",
                            "task": "Build the cell that tests: {finding.text} (from {from.label}, {analysis_id})"}}


def _by_label(meta):
    return {e["label"]: e for e in meta._delegation_ledger}


# ------------------------------------------------------------ pure functions
def test_subscriptions_are_normalised_matched_and_filled_without_a_model():
    ok, bad = reactions.normalize_subscriptions([
        SIM_ON_CLAIM, {"on": {"kind": "verdict"}, "enqueue": {"mode": "analysis", "task": "x"}},
        {"on": {"kind": "claim"}, "enqueue": {"mode": "chat", "task": "x"}},
        {"on": {"kind": "claim"}, "enqueue": {"mode": "analysis", "task": "  "}},
        "nope", {"on": {"kind": "claim", "status": "maybe"}, "enqueue": {"mode": "analysis", "task": "x"}},
        {"on": {"kind": "task_request"}, "enqueue": {"mode": "analysis", "task": "do: {finding.text}"}, "max_fires": "3"}])
    assert [b["subscription"] for b in bad] == [2, 3, 4, 5, 6]
    assert ok[0]["on"] == {"kind": "claim", "status": "verified", "subject": S} and ok[0]["max_fires"] == 1
    assert ok[1]["max_fires"] == 3 and ok[1]["enqueue"]["label"] == "reaction to a task_request"
    rec = {"finding_id": "f0001-abc", "kind": "claim", "status": "verified", "subject": "tio2  a7",
           "payload": {"text": "anatase, {finding.path} is data"}, "evidence": {"analysis_ids": ["a9"]}}
    assert reactions.matches(ok[0], rec)                                   # NFKC/casefold/whitespace subject
    assert not reactions.matches(ok[0], {**rec, "status": "provisional"})
    assert not reactions.matches(ok[0], {**rec, "subject": "TiO2 B1"})
    assert reactions.matches({**ok[0], "on": {**ok[0]["on"], "status": "any"}}, {**rec, "status": "provisional"})
    frm = {"label": "Raman A7", "index": 3, "mode": "analysis"}
    assert reactions.fill(ok[0]["enqueue"]["task"], rec, frm) == (
        "Build the cell that tests: anatase, {finding.path} is data (from Raman A7, a9)")   # one pass, no code
    assert reactions.fill("{unknown} {finding.id} {subject} {from.index}", rec, frm) == "{unknown} f0001-abc tio2  a7 3"
    assert reactions.hop(frm, rec) == {"mode": "analysis", "subject": "tio2  a7", "kind": "claim",
                                       "finding_id": "f0001-abc", "index": 3}


# --------------------------------------------------------- through the swarm
def test_a_verified_claim_fires_a_subscription_once_with_its_cause_stamped(meta, monkeypatch):
    def script(mode, task, context):
        if mode == "analysis":
            return {"claims": ["anatase 144 cm-1 dominant", "rutile trace"], "analysis_id": "a1"}
        if mode == "simulation":
            return {"structure": "anatase cell"}
        return {}
    built = _script(monkeypatch, script)
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[SIM_ON_CLAIM]))
    assert res["status"] == "success" and res["subscriptions"] == {"accepted": 1, "refused": []}
    # two verified claims, one subscription with max_fires 1: ONE reaction, the second refused and said so
    assert len(res["fired"]) == 1 and [r["reason"] for r in res["refused_reactions"]] == ["max_fires (1) reached"]
    fired = res["fired"][0]
    ledger = _by_label(meta)
    react_entry = ledger["cell for Raman A7"]
    claim_ids = [f for f in ledger["Raman A7"]["posted"] if meta.board.get(f)["kind"] == "claim"]
    assert fired["caused_by"] == [claim_ids[0]] == react_entry["caused_by"]
    assert react_entry["chain"] == [{"mode": "analysis", "subject": S, "kind": "claim",
                                     "finding_id": claim_ids[0], "index": ledger["Raman A7"]["index"]}]
    assert react_entry["subscription"] == 1 and react_entry["subject"] == S
    assert react_entry["swarm"] == res["swarm_id"] and react_entry["status"] == "success"
    # the task was filled from the finding, by the coordinator
    sim = next(w for w in built if w.mode == "simulation")
    assert react_entry["task"] == f"Build the cell that tests: anatase 144 cm-1 dominant (from Raman A7, a1)"
    assert sim.base_dir.name.startswith("03_cell_for_raman_a7")
    # the reaction's own records are on the board, authored by it, and it ran as an item of this swarm
    assert meta.board.get(react_entry["posted"][0])["kind"] == "structure"
    assert [r["label"] for r in res["results"]] == ["Raman A7", "purity plan", "cell for Raman A7"]
    assert res["results"][2]["caused_by"] == [claim_ids[0]] and "caused_by" not in res["results"][0]
    # the refusal is also on the triggering entry
    assert ledger["Raman A7"]["refused_reactions"][0]["finding_id"] == claim_ids[1]
    # max_fires 2 fires on both claims, in a fresh swarm
    res2 = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[{**SIM_ON_CLAIM, "max_fires": 2}]))
    assert len(res2["fired"]) == 2 and res2["refused_reactions"] == []


def test_a_two_item_cycle_is_refused_and_recorded(meta, monkeypatch):
    """analysis(S) posts a claim → simulation(S) posts a structure → analysis(S)
    posts a claim → the simulation would fire again: the hop (analysis, S,
    claim) is already in the chain, so it is refused, on the record."""
    def script(mode, task, context):
        if mode == "analysis":
            return {"claims": ["a claim on S"]}
        if mode == "simulation":
            return {"structure": "a cell"}
        return {}
    _script(monkeypatch, script)
    subs = [{**SIM_ON_CLAIM, "max_fires": 3},
            {"on": {"kind": "structure", "subject": S},
             "enqueue": {"mode": "analysis", "label": "re-analyse after {from.label}",
                         "task": "Analyse again given {finding.name} at {finding.path}"}}]
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=subs, budget={"max_triggers_per_subject": 5}))
    assert [f["label"] for f in res["fired"]] == ["cell for Raman A7", "re-analyse after cell for Raman A7"]
    assert len(res["refused_reactions"]) == 1
    ref = res["refused_reactions"][0]
    assert ref["reason"].startswith("cycle: (analysis, 'TiO2 A7', claim) is already in this item's chain")
    assert [(h["mode"], h["kind"]) for h in ref["chain"]] == [("analysis", "claim"), ("simulation", "structure")]
    assert ref["from_index"] == _by_label(meta)["re-analyse after cell for Raman A7"]["index"]
    # the chain on the second reaction has two hops, oldest first
    chain = _by_label(meta)["re-analyse after cell for Raman A7"]["chain"]
    assert [h["index"] for h in chain] == [1, 3] and chain[1]["kind"] == "structure"


def test_the_subject_cap_and_the_item_limit_bound_reactions(meta, monkeypatch):
    def script(mode, task, context):
        if mode == "analysis":
            return {"claims": [f"claim {i}" for i in range(4)]}
        return {"structure": "a cell"}
    _script(monkeypatch, script)
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[{**SIM_ON_CLAIM, "max_fires": 4}]))
    assert len(res["fired"]) == 2                                   # the default cap: 2 per subject
    reasons = [r["reason"] for r in res["refused_reactions"]]
    assert len(reasons) == 2 and all("re-triggered 2 time(s) already" in r for r in reasons)
    # the item limit counts fired items: 7 items + reactions → one fires
    many = [{"mode": "planning", "task": f"plan {i}", "label": f"plan {i}"} for i in range(6)] + ITEMS[:1]
    res = json.loads(swarm.run_swarm(meta, many, subscriptions=[{**SIM_ON_CLAIM, "max_fires": 4}],
                                     budget={"max_triggers_per_subject": 9}))
    assert len(res["results"]) == 8 and len(res["fired"]) == 1
    assert any("over the limit of 8 items" in r["reason"] for r in res["refused_reactions"])
    # a budget tighter than the limit
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[{**SIM_ON_CLAIM, "max_fires": 4}],
                                     budget={"max_reactions": 1, "max_triggers_per_subject": 9}))
    assert len(res["fired"]) == 1 and all("max_reactions (1)" in r["reason"] for r in res["refused_reactions"])


def test_a_task_request_becomes_an_item_only_through_a_subscription(meta, monkeypatch):
    def script(mode, task, context):
        if mode == "analysis" and "Raman" in task:
            return {"claims": ["anatase"], "followups": ["Measure XRD on the same pellet", "Run EELS at the edge"]}
        if mode == "analysis":
            return {"claims": ["did it: " + task[:30]]}
        return {}
    _script(monkeypatch, script)
    # no subscription: asked and not done, never read by default
    res = json.loads(swarm.run_swarm(meta, ITEMS))
    assert res["fired"] == [] and [t["text"] for t in res["task_requests"]] == [
        "Measure XRD on the same pellet", "Run EELS at the edge"]
    assert all(t["item"] is None and t["from"] == "Raman A7" for t in res["task_requests"])
    req = meta.board.get(res["task_requests"][0]["finding_id"])
    assert req["kind"] == "task_request" and req["status"] == "provisional" and req["subject"] == S
    assert all(r["kind"] != "task_request" for r in meta.board.snapshot(subject=S, include_provisional=True).records)
    with pytest.raises(ValueError):
        meta.board.snapshot(kind="task_request")
    # a subscription on the kind makes an item of it, filled from the request
    sub = {"on": {"kind": "task_request", "status": "provisional"},
           "enqueue": {"mode": "analysis", "label": "asked: {finding.text}", "task": "{finding.text} ({subject})"},
           "max_fires": 2}
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[sub]))
    assert [f["label"] for f in res["fired"]] == ["asked: Measure XRD on the same pellet", "asked: Run EELS at the edge"]
    assert [t["item"] for t in res["task_requests"] if t["from"] == "Raman A7"] == [f["delegation_index"] for f in res["fired"]]
    done = _by_label(meta)["asked: Run EELS at the edge"]
    assert done["task"] == f"Run EELS at the edge ({S})" and done["caused_by"] == [res["task_requests"][1]["finding_id"]]


def test_a_retraction_taints_exactly_its_dependents(meta, tmp_path):
    board = meta.board
    a = board.post(kind="claim", author={"worker": "w1", "delegation_index": 1, "mode": "analysis"},
                   subject=S, payload={"text": "A"}, status="verified")
    b = board.post(kind="claim", author={"worker": "w2", "delegation_index": 2, "mode": "analysis"},
                   subject=S, payload={"text": "B rests on A"}, status="verified", reads=[a["finding_id"]])
    c = board.post(kind="claim", author={"worker": "w3", "delegation_index": 3, "mode": "planning"},
                   subject=S, payload={"text": "C independent"}, status="verified")
    d = board.post(kind="measurement", author={"worker": "w4", "delegation_index": 4, "mode": "analysis"},
                   subject=S, payload={"name": "d", "value": 1}, status="verified", reads=[b["finding_id"]])
    meta._delegation_ledger.extend([
        {"index": 2, "label": "B item", "mode": "analysis", "subject": S, "task": "do B", "status": "success"},
        {"index": 4, "label": "D item", "mode": "analysis", "subject": S, "task": "do D", "status": "success"}])
    out = board_mod.retract_and_report(meta, a["finding_id"], "the peak was a cosmic ray")
    assert out["status"] == "success" and out["retracted"] == a["finding_id"]
    assert [(t["finding_id"], t["tainted_by"]) for t in out["tainted"]] == [
        (b["finding_id"], [a["finding_id"]]), (d["finding_id"], [a["finding_id"]])]    # transitive, C untouched
    assert [s["delegation_index"] for s in out["sources"]] == [2, 4]
    assert [(i["label"], i["task"], i["context"]["reruns_delegation"]) for i in out["rerun_items"]] == [
        ("rerun: B item", "do B", 2), ("rerun: D item", "do D", 4)]
    status = {r["finding_id"]: r["status"] for r in board.fold()}
    assert status[a["finding_id"]] == "retracted" and status[b["finding_id"]] == "tainted"
    assert status[c["finding_id"]] == "verified" and status[d["finding_id"]] == "tainted"
    assert set(board.snapshot(subject=S).ids) == {c["finding_id"]}            # out of the default read
    assert set(board.snapshot(subject=S, include_provisional=True).ids) == {c["finding_id"]}
    # a record posted AFTER the retraction by a worker that had read A before is caught too
    late = board.post(kind="claim", author={"worker": "w5", "delegation_index": 5, "mode": "analysis"},
                      subject=S, payload={"text": "late, read A"}, status="verified", reads=[a["finding_id"]])
    assert {r["finding_id"]: r["status"] for r in board.fold()}[late["finding_id"]] == "tainted"
    # a correction rests on what it corrects by design: superseding a tainted record is not tainted
    fix = board.post(kind="claim", author={"worker": "w2", "delegation_index": 2, "mode": "analysis"},
                     subject=S, payload={"text": "B redone without A"}, status="verified", supersedes=b["finding_id"])
    assert {r["finding_id"]: r["status"] for r in board.fold()}[fix["finding_id"]] == "verified"
    # nothing to re-run twice; a retraction of a retraction is refused; unknown ids are a KeyError
    with pytest.raises(ValueError):
        board_mod.retract_and_report(meta, out["retraction"], "x")
    with pytest.raises(KeyError):
        board_mod.retract_and_report(meta, "f9999-nope", "x")
    with pytest.raises(ValueError):
        board_mod.retract_and_report(meta, c["finding_id"], "  ")
    assert board.dependents(a["finding_id"]) == [b["finding_id"], d["finding_id"], late["finding_id"]]


def test_the_retract_tool_and_get_board_show_withdrawn_records(meta):
    board = meta.board
    a = board.post(kind="claim", author={"worker": "w1", "delegation_index": 1, "mode": "analysis"},
                   subject=S, payload={"text": "A"}, status="verified")
    board.post(kind="claim", author={"worker": "w2", "delegation_index": 2, "mode": "analysis"},
               subject=S, payload={"text": "B"}, status="verified", reads=[a["finding_id"]])
    board.post(kind="task_request", author={"worker": "w2", "delegation_index": 2, "mode": "analysis"},
               subject=S, payload={"text": "measure more"}, status="provisional")
    retract = meta.tools.functions_map["retract_finding"]
    out = json.loads(retract(finding_id=a["finding_id"], reason="wrong peak"))
    assert out["status"] == "success" and len(out["tainted"]) == 1
    assert json.loads(retract(finding_id="f0000-none", reason="x"))["status"] == "error"
    get_board = meta.tools.functions_map["get_board"]
    shown = json.loads(get_board(subject=S))
    assert shown["count"] == 0
    assert [(w["finding_id"], w["status"]) for w in shown["withdrawn"]] == [
        (a["finding_id"], "retracted"), (shown["withdrawn"][1]["finding_id"], "tainted")]
    assert [t["payload"]["text"] for t in shown["task_requests"]] == ["measure more"]


def test_a_supersede_chain_of_three_stops_the_coordinator(meta, monkeypatch):
    board = meta.board
    author = {"worker": "coordinator", "mode": "coordinator"}
    c1 = board.post(kind="claim", author=author, subject=S, payload={"text": "x = 1"}, status="verified")
    c2 = board.post(kind="claim", author=author, subject=S, payload={"text": "x = 2"}, status="verified",
                    supersedes=c1["finding_id"])
    c3 = board.post(kind="claim", author=author, subject=S, payload={"text": "x = 1 after all"},
                    status="verified", supersedes=c2["finding_id"])
    entry = {"index": 7, "label": "w", "mode": "analysis", "chain": []}
    sub = reactions.normalize_subscriptions([SIM_ON_CLAIM])[0][0]
    assert reactions.supersede_depth(board, c3) == 2 and reactions.supersede_depth(board, c2) == 1
    item, why = reactions.decide(sub, c2, entry, board=board, fired_by_sub={}, fired_pairs=set(),
                                 triggers_by_subject={}, items_so_far=2, max_items=8)
    assert item is not None and why is None
    item, why = reactions.decide(sub, c3, entry, board=board, fired_by_sub={}, fired_pairs=set(),
                                 triggers_by_subject={}, items_so_far=2, max_items=8)
    assert item is None and why.startswith("supersede chain of 3 on 'TiO2 A7': a disagreement to report")


def test_a_hazard_reaches_a_reader_that_filtered_it_out(meta, monkeypatch):
    board = meta.board
    board.post(kind="claim", author={"worker": "w1", "delegation_index": 1, "mode": "analysis"},
               subject=S, payload={"text": "a verified claim"}, status="verified")
    hz = board.post(kind="hazard", author={"worker": "critic", "delegation_index": 1, "mode": "planning"},
                    subject=S, payload={"issue": "anneal above 900 C exceeds the furnace limit"},
                    status="provisional")
    board.post(kind="hazard", author={"worker": "critic", "delegation_index": 1, "mode": "planning"},
               subject="another sample", payload={"issue": "elsewhere"}, status="provisional")
    seen = {}

    def script(mode, task, context):
        seen[mode] = task
        return {}
    _script(monkeypatch, script)
    items = [{"mode": "planning", "task": "plan", "label": "reader", "subject": S,
              "reads_board": {"kinds": ["claim"]}},
             {"mode": "simulation", "task": "cell", "label": "other", "subject": S,
              "reads_board": {"kinds": ["measurement"]}}]
    res = json.loads(swarm.run_swarm(meta, items))
    ledger = _by_label(meta)
    assert hz["finding_id"] in ledger["reader"]["reads"] and hz["finding_id"] in ledger["other"]["reads"]
    assert "anneal above 900 C" in seen["planning"] and "[provisional]" in seen["planning"]
    assert "elsewhere" not in seen["planning"]                      # another subject's hazard stays there
    assert "a verified claim" in seen["planning"] and "a verified claim" not in seen["simulation"]
    # a check is still refused everything, hazards included
    res = json.loads(swarm.run_swarm(meta, [{**items[0], "check": True}, items[1]]))
    assert res["results"][0]["board_read_refused"] and res["results"][0]["reads"] == []


def test_chains_and_causes_survive_a_checkpoint(meta, monkeypatch, tmp_path):
    def script(mode, task, context):
        return {"claims": ["anatase"]} if mode == "analysis" else {"structure": "cell"}
    _script(monkeypatch, script)
    res = json.loads(swarm.run_swarm(meta, ITEMS, subscriptions=[SIM_ON_CLAIM]))
    assert len(res["fired"]) == 1
    meta._auto_checkpoint(verbose=False)
    again = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                                  meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path), restore_checkpoint=True)
    e = _by_label(again)["cell for Raman A7"]
    assert e["caused_by"] == res["fired"][0]["caused_by"] and e["chain"] == res["fired"][0]["chain"]
    assert e["subscription"] == 1 and "refused_reactions" not in _by_label(again)["Raman A7"]


def test_a_swarm_without_subscriptions_is_stage_two(meta, monkeypatch):
    """Same ledger and board as before this stage: no reaction fields on the
    entries, nothing fired, nothing refused; the result only gains empty
    lists. A caller cannot smuggle a chain onto an initial item."""
    _script(monkeypatch, lambda mode, task, context: {"claims": ["anatase"]} if mode == "analysis" else {})
    res = json.loads(swarm.run_swarm(meta, [{**ITEMS[0], "chain": [{"mode": "x"}], "caused_by": ["f1"]}, ITEMS[1]]))
    assert res["fired"] == [] and res["refused_reactions"] == [] and res["task_requests"] == []
    assert res["subscriptions"] == {"accepted": 0, "refused": []}
    for e in meta._delegation_ledger:
        assert not any(k in e for k in ("caused_by", "chain", "subscription", "refused_reactions"))
    assert "caused_by" not in res["results"][0]
