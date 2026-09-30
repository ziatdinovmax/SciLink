"""The board (swarm stage 2): one writer, typed records, verified-only reads,
checks refused a read, a fold over supersedes / retractions, independence
counted on the read graph, and the board through the meta (posting at close,
reads on swarm items, fusion's independent_support, checkpoint and restore).
"""

import json
import threading
from pathlib import Path

import pytest

from scilink.agents.meta_agent import board as board_mod
from scilink.agents.meta_agent import fanout as fo
from scilink.agents.meta_agent import swarm
from scilink.agents.meta_agent.board import Board, BoardReadRefused, render
from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent

A = {"worker": "Raman A7", "delegation_index": 1, "mode": "analysis"}
B = {"worker": "XRD A7", "delegation_index": 2, "mode": "analysis"}


def _claim(board, author, text, **kw):
    return board.post(kind="claim", author=author, payload={"text": text},
                      subject="TiO2 A7", status=kw.pop("status", "verified"), **kw)


# ----------------------------------------------------------------- the record
def test_a_record_is_written_flushed_and_reloaded(tmp_path):
    b = Board(tmp_path)
    rec = _claim(b, A, "anatase, Eg at 144", evidence={"analysis_ids": ["analysis_1"]})
    assert rec["finding_id"].startswith("f0001-") and rec["status"] == "verified"
    assert rec["board_version"] == 0 and rec["reads"] == [] and rec["created_at"]
    lines = (tmp_path / "swarm" / "board.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == rec
    again = Board(tmp_path)
    assert len(again) == 1 and again.get(rec["finding_id"]) == rec


def test_a_torn_last_line_is_skipped_on_load(tmp_path):
    b = Board(tmp_path)
    r1 = _claim(b, A, "one")
    _claim(b, A, "two")
    p = tmp_path / "swarm" / "board.jsonl"
    text = p.read_text()
    p.write_text(text[: len(text) - 20])          # the second record torn mid-write
    again = Board(tmp_path)
    assert [r["finding_id"] for r in again.records()] == [r1["finding_id"]]
    # the next post continues the log on a fresh line
    r3 = _claim(again, B, "three")
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert json.loads(lines[-1])["finding_id"] == r3["finding_id"] and len(lines) == 3
    assert len(Board(tmp_path)) == 2                # the torn line is skipped wherever it sits


def test_the_record_is_typed_and_small(tmp_path):
    b = Board(tmp_path)
    with pytest.raises(ValueError, match="kind must be one of"):
        b.post(kind="hunch", author=A, payload={})
    with pytest.raises(ValueError, match="inline numeric array"):
        b.post(kind="measurement", author=A, payload={"spectrum": list(range(200))})
    with pytest.raises(ValueError, match="small typed"):
        b.post(kind="claim", author=A, payload={"text": "x" * 5000})
    with pytest.raises(ValueError, match="provisional or verified"):
        b.post(kind="claim", author=A, payload={"text": "t"}, status="retracted")
    with pytest.raises(ValueError, match="author"):
        b.post(kind="claim", author={}, payload={"text": "t"})
    with pytest.raises(KeyError):
        b.post(kind="claim", author=A, payload={"text": "t"}, supersedes="f9999-nope")
    assert len(b) == 0


# ------------------------------------------------------------------- reading
def test_only_verified_findings_propagate_by_default(tmp_path):
    b = Board(tmp_path)
    v = _claim(b, A, "verified one")
    p = _claim(b, B, "provisional one", status="provisional")
    view = b.snapshot(subject="tio2 a7")               # subject match is case-insensitive
    assert view.ids == (v["finding_id"],) and view.version == 2
    both = b.snapshot(subject="TiO2 A7", include_provisional=True)
    assert set(both.ids) == {v["finding_id"], p["finding_id"]} and both.include_provisional
    assert b.snapshot(subject="another sample").ids == ()
    assert b.snapshot(kind="hazard").ids == ()
    with pytest.raises(ValueError, match="unknown kind"):
        b.snapshot(kind="retraction")                  # bookkeeping kinds are not readable
    # the view is immutable
    with pytest.raises(Exception):
        view.records = ()


def test_a_check_is_refused_a_read_by_the_api(tmp_path):
    b = Board(tmp_path)
    _claim(b, A, "x")
    with pytest.raises(BoardReadRefused):
        b.snapshot(subject="TiO2 A7", check=True)


def test_superseded_and_retracted_records_fold_out(tmp_path):
    b = Board(tmp_path)
    old = _claim(b, A, "Eg at 141 cm-1")
    new = _claim(b, A, "Eg at 144 cm-1 (recalibrated)", supersedes=old["finding_id"])
    with pytest.raises(ValueError, match="own kind"):
        b.post(kind="measurement", author=A, payload={"name": "Eg", "value": 144},
               supersedes=old["finding_id"])
    gone = _claim(b, B, "rutile present")
    b.retract(gone["finding_id"], B, reason="the 447 band was a cosmic ray")
    status = {r["finding_id"]: r["status"] for r in b.fold()}
    assert status[old["finding_id"]] == "superseded" and status[gone["finding_id"]] == "retracted"
    assert status[new["finding_id"]] == "verified"
    assert b.snapshot(subject="TiO2 A7").ids == (new["finding_id"],)
    # a correction rests on what it corrects: the closure sees through it
    assert b.read_closure([new["finding_id"]]) == {old["finding_id"]}
    # nothing was edited in place
    assert b.get(old["finding_id"])["status"] == "verified"


def test_render_is_hints_under_the_additive_rule(tmp_path):
    b = Board(tmp_path)
    _claim(b, A, "anatase")
    b.post(kind="hazard", author=B, subject="TiO2 A7", status="verified",
           payload={"issue": "laser power above 5 mW converts anatase", "conflict": "5 mW vs 10 mW"})
    b.post(kind="parameter_point", author=B, subject="TiO2 A7", status="verified",
           payload={"point": {"T_C": 450}})
    text = "\n".join(render(b.snapshot(subject="TiO2 A7")))
    assert "BOARD — findings earlier work posted on 'TiO2 A7'" in text
    assert "claim: anatase" in text and "hazard: laser power" in text and '"T_C": 450' in text
    assert "never sets a fit window, a threshold, a target" in text
    assert render(b.snapshot(subject="nothing here")) == []


# -------------------------------------------------------------- independence
def test_independence_is_counted_on_the_read_graph(tmp_path):
    b = Board(tmp_path)
    a = _claim(b, A, "anatase")                                  # A read nothing
    bb = _claim(b, B, "anatase too")                             # B read nothing
    C = {"worker": "C", "delegation_index": 3, "mode": "analysis"}
    c = _claim(b, C, "anatase, agreeing", reads=[a["finding_id"]])   # C read A
    D = {"worker": "D", "delegation_index": 4, "mode": "analysis"}
    d = _claim(b, D, "anatase as well", reads=[c["finding_id"]])     # D read C (so A, transitively)
    E = {"worker": "E", "delegation_index": 5, "mode": "analysis"}
    e = _claim(b, E, "anatase, again", reads=[])
    sup = b.independent_support({"A": [a["finding_id"]], "B": [bb["finding_id"]],
                                 "C": [c["finding_id"]], "D": [d["finding_id"]]})
    assert sup == {"count": 2, "raw": 4, "dependent": {"C": ["A"], "D": ["A", "C"]}}
    # a supporter that posted nothing but read at launch is dependent through its reads
    sup = b.independent_support({"A": [a["finding_id"]], "E": []},
                                reads={"E": [bb["finding_id"], a["finding_id"]]})
    assert sup["count"] == 1 and sup["dependent"] == {"E": ["A"]}
    # reading something outside the agreeing set costs nothing
    sup = b.independent_support({"B": [bb["finding_id"]], "E": [e["finding_id"]]},
                                reads={"E": [a["finding_id"]]})
    assert sup == {"count": 2, "raw": 2, "dependent": {}}


# --------------------------------------------------------------- the writer
def test_the_writer_is_serialised_under_concurrent_posts(tmp_path):
    b = Board(tmp_path)
    n_threads, per = 8, 40

    def poster(k):
        for i in range(per):
            b.post(kind="measurement", author={"worker": f"w{k}", "mode": "analysis"},
                   payload={"name": "x", "value": i}, status="verified")
    ts = [threading.Thread(target=poster, args=(k,)) for k in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = [l for l in (tmp_path / "swarm" / "board.jsonl").read_text().splitlines() if l.strip()]
    recs = [json.loads(l) for l in lines]                         # every line whole
    assert len(recs) == n_threads * per == len(b)
    assert len({r["finding_id"] for r in recs}) == len(recs)
    assert [r["finding_id"][:5] for r in recs] == [f"f{i:04d}" for i in range(1, len(recs) + 1)]
    assert len(Board(tmp_path)) == len(recs)


# ------------------------------------------------------- through the meta
class AnalysisWorker:
    """A stand-in analysis specialist: one successful analysis with a claim,
    plus one that failed (its claim must stay provisional)."""

    def __init__(self, mode, base_dir):
        self.mode, self.base_dir = mode, Path(base_dir)
        self.tasks = []

    def run_task(self, task, context=None, autonomy=None):
        self.tasks.append(task)
        out = self.base_dir / "results" / "analysis_1"
        (out / "scripts").mkdir(parents=True)
        (out / "scripts" / "analysis_script.py").write_text("print('fit')\n")
        return {"status": "success", "summary": f"{self.mode}: done",
                "key_findings": [f"[analysis_1] anatase from {task[:12]}",
                                 "[analysis_2] rutile trace (unverified run)",
                                 "an unattributed remark"],
                "analyses": [{"analysis_id": "analysis_1", "status": "success",
                              "output_directory": str(out), "agent_name": "CurveFittingAgent"},
                             {"analysis_id": "analysis_2", "status": "error"}],
                "files_produced": [], "suggested_followups": [], "warnings": []}


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(swarm, "_POLL_S", 0.05)
    monkeypatch.setattr(swarm.fo, "_available_memory", lambda: 8e9)
    monkeypatch.setattr(swarm, "_memory", lambda: {"total": 16e9, "available": 8e9})
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401  (import once, see test_run_swarm)
    import scilink.agents.planning_agents.planning_orchestrator  # noqa: F401
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"), model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    m._enable_human_feedback = False
    built = []

    def build(orch, mode, base_dir, **kw):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        w = AnalysisWorker(mode, base_dir)
        built.append(w)
        return w
    monkeypatch.setattr(swarm, "build_child", build)
    m._built = built
    return m


ITEMS = [{"mode": "analysis", "task": "fit the Raman spectrum", "label": "Raman A7", "subject": "TiO2 A7"},
         {"mode": "analysis", "task": "index the XRD pattern", "label": "XRD A7", "subject": "TiO2 A7"}]


def test_a_finished_item_posts_what_its_mode_verified(meta):
    res = json.loads(swarm.run_swarm(meta, ITEMS))
    assert res["status"] == "success" and res["board_version"] == 8
    recs = meta.board.records()
    by_worker = {}
    for r in recs:
        by_worker.setdefault(r["author"]["worker"], []).append(r)
    for label in ("Raman A7", "XRD A7"):
        mine = by_worker[label]
        assert [(r["kind"], r["status"]) for r in mine] == [
            ("claim", "verified"), ("claim", "provisional"), ("claim", "provisional"), ("recipe", "verified")]
        assert mine[0]["evidence"]["analysis_ids"] == ["analysis_1"] and mine[0]["subject"] == "TiO2 A7"
        assert mine[0]["payload"]["text"].startswith("anatase from")      # the id prefix is evidence, not text
        assert mine[3]["payload"]["path"].endswith("scripts/analysis_script.py")
        assert all(r["reads"] == [] for r in mine)
    ledger = {e["label"]: e for e in meta._delegation_ledger}
    assert len(ledger["Raman A7"]["posted"]) == 4 and "reads" not in ledger["Raman A7"]
    assert [r["posted"] for r in res["results"]] == [ledger["Raman A7"]["posted"], ledger["XRD A7"]["posted"]]
    # only the verified records reach a default read
    assert len(meta.board.snapshot(subject="TiO2 A7")) == 4


def test_a_later_item_reads_the_board_and_a_check_is_refused(meta):
    swarm.run_swarm(meta, ITEMS)
    first_ids = set(meta.board.snapshot(subject="TiO2 A7").ids)
    res = json.loads(swarm.run_swarm(meta, [
        {"mode": "analysis", "task": "fit the EELS edge", "label": "EELS A7", "subject": "TiO2 A7",
         "reads_board": True},
        {"mode": "analysis", "task": "re-fit the Raman to confirm", "label": "Raman audit",
         "subject": "TiO2 A7", "reads_board": {"include_provisional": True}, "check": True},
        {"mode": "analysis", "task": "look at another sample", "label": "Raman B2", "subject": "TiO2 B2",
         "reads_board": {"kinds": ["claim"]}},
    ]))
    by_label = {r["label"]: r for r in res["results"]}
    assert set(by_label["EELS A7"]["reads"]) == first_ids and by_label["EELS A7"]["board_read_refused"] is None
    assert by_label["Raman audit"]["reads"] == [] and "check" in by_label["Raman audit"]["board_read_refused"]
    assert by_label["Raman B2"]["reads"] == []                 # nothing on its subject yet
    tasks = {w.tasks[0][:30]: w.tasks[0] for w in meta._built}
    eels = next(t for k, t in tasks.items() if k.startswith("fit the EELS"))
    assert "BOARD — findings earlier work posted on 'TiO2 A7'" in eels
    assert "claim: anatase from fit the Ram" in eels and "recipe (an approved analysis script" in eels
    assert "rutile trace" not in eels                          # provisional, not asked for
    assert "BOARD" not in next(t for k, t in tasks.items() if k.startswith("re-fit the Raman"))
    ledger = {e["label"]: e for e in meta._delegation_ledger}
    assert set(ledger["EELS A7"]["reads"]) == first_ids and ledger["EELS A7"]["board_version"] == 8
    assert ledger["EELS A7"]["task"] == eels                   # the ledger keeps the task as sent
    # what the reader posts carries its reads, so the fold sees the dependency
    posted = [meta.board.get(f) for f in ledger["EELS A7"]["posted"]]
    assert all(set(r["reads"]) == first_ids and r["board_version"] == 8 for r in posted)
    sup = meta.board.independent_support({l: ledger[l]["posted"] for l in ("Raman A7", "XRD A7", "EELS A7")})
    assert sup["count"] == 2 and sup["dependent"] == {"EELS A7": ["Raman A7", "XRD A7"]}
    # a check is refused a read even when the caller reads through the tool path
    with pytest.raises(BoardReadRefused):
        meta.board.snapshot(subject="TiO2 A7", check=True)


def test_reads_board_opt_in_forms():
    """``{}`` is the tool schema's plain opt-in (found live: it read nothing)."""
    assert swarm._read_spec({}) == {}
    assert swarm._read_spec(True) == {} and swarm._read_spec("yes") == {}
    assert swarm._read_spec(None) is None and swarm._read_spec(False) is None
    assert swarm._read_spec({"kinds": "claim", "include_provisional": True, "subject": " S "}) == {
        "kinds": ["claim"], "include_provisional": True, "subject": "S"}
    items, _ = swarm.normalize_items([{"mode": "analysis", "task": "t", "label": "a", "reads_board": {}},
                                      {"mode": "analysis", "task": "t", "label": "b"}])
    assert items[0]["reads_board"] == {} and items[1]["reads_board"] is None


def test_the_board_survives_the_checkpoint_and_restore(meta, tmp_path):
    swarm.run_swarm(meta, ITEMS)
    ck = json.loads((Path(meta.base_dir) / "checkpoint.json").read_text())
    assert ck["board_version"] == 8
    again = MetaOrchestratorAgent(base_dir=str(meta.base_dir), model_name="anthropic/claude-sonnet-4-5",
                                  meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path),
                                  restore_checkpoint=True)
    assert len(again.board) == 8
    assert again.board.snapshot(subject="TiO2 A7").ids == meta.board.snapshot(subject="TiO2 A7").ids
    assert [e.get("posted") for e in again._delegation_ledger] == [e.get("posted") for e in meta._delegation_ledger]


def test_get_board_tool_shows_public_records(meta):
    swarm.run_swarm(meta, ITEMS)
    fn = meta.tools.functions_map["get_board"]
    out = json.loads(fn(subject="TiO2 A7"))
    assert out["status"] == "success" and out["count"] == 4 and out["board_version"] == 8
    assert {r["kind"] for r in out["findings"]} == {"claim", "recipe"}
    assert all(r["status"] == "verified" for r in out["findings"])
    out = json.loads(fn(subject="TiO2 A7", include_provisional=True, kind="claim"))
    assert out["count"] == 6 and {r["status"] for r in out["findings"]} == {"verified", "provisional"}
    assert json.loads(fn(kind="hunch"))["status"] == "error"


# ---------------------------------------------------------------- fusion
def _fusion_llm(orch, prompt, extra_parts=None):
    _fusion_llm.prompts.append(prompt)
    return {"detailed_analysis": "fused narrative",
            "scientific_claims": [{"claim": "anatase throughout"}]}


_fusion_llm.prompts = []


def _branch(meta, index, label, reads=None):
    e = meta._open_delegation("analysis", f"task {label}", None, None, label)
    with meta._fanout_lock:
        e["fanout"] = True
        e["parallel_group"] = "fanout_1"
        e["subject"] = "TiO2 A7"
        if reads:
            e["reads"] = list(reads)
    meta._close_delegation(e, {"status": "success", "summary": f"{label} summary",
                               "key_findings": [f"[analysis_{index}] anatase ({label})"],
                               "analyses": [{"analysis_id": f"analysis_{index}", "status": "success"}],
                               "files_produced": [], "warnings": []})
    return e


def test_fusion_reports_independent_support_and_posts_its_claims(meta, monkeypatch):
    monkeypatch.setattr(fo, "_llm_json", _fusion_llm)
    _fusion_llm.prompts.clear()
    a = _branch(meta, 1, "Raman A7")
    b = _branch(meta, 2, "XRD A7")
    out = json.loads(fo.fuse_delegations(meta, [1, 2]))
    assert out["status"] == "success"
    assert out["independent_support"] == {"count": 2, "raw": 2, "dependent": {}}
    assert "INDEPENDENT SUPPORT (computed from the board's read graph, not judged): 2 of 2" in _fusion_llm.prompts[-1]
    fusion = meta._delegation_ledger[-1]
    assert fusion["mode"] == "fusion" and set(fusion["reads"]) == set(a["posted"]) | set(b["posted"])
    rec = meta.board.get(fusion["posted"][0])
    assert rec["kind"] == "claim" and rec["status"] == "provisional" and rec["payload"]["text"] == "anatase throughout"
    assert set(rec["reads"]) == set(fusion["reads"]) and rec["author"]["mode"] == "fusion"
    # a re-analysis citing the fusion inherits its reads, and the next fusion counts 1 + ...
    e3 = meta._open_delegation("analysis", "re-analyze Raman", None, [fusion["index"]], "Raman A7 again")
    assert e3["informed_via"] == "fusion_feedback" and e3["reads"] == fusion["posted"]
    with meta._fanout_lock:
        e3["fanout"] = True
        e3["parallel_group"] = "fanout_1"
    meta._close_delegation(e3, {"status": "success", "summary": "again",
                                "key_findings": ["[analysis_3] anatase (again)"],
                                "analyses": [{"analysis_id": "analysis_3", "status": "success"}],
                                "files_produced": [], "warnings": []})
    out = json.loads(fo.fuse_delegations(meta, [1, 2, e3["index"]]))
    assert out["independent_support"] == {"count": 2, "raw": 3,
                                          "dependent": {"Raman A7 again": ["Raman A7", "XRD A7"]}}
    assert any("had read findings of" in c for c in out["caveats"])
    assert "2 of 3 branches" in _fusion_llm.prompts[-1]


def test_one_branch_informed_by_the_other_counts_one(meta, monkeypatch):
    monkeypatch.setattr(fo, "_llm_json", _fusion_llm)
    a = _branch(meta, 1, "Raman A7")
    _branch(meta, 2, "XRD A7", reads=a["posted"])
    out = json.loads(fo.fuse_delegations(meta, [1, 2]))
    assert out["independent_support"] == {"count": 1, "raw": 2, "dependent": {"XRD A7": ["Raman A7"]}}
    assert out["caveats"] and "not an independent confirmation" in out["caveats"][0]


# ------------------------------------------------------- what the modes post
def test_planning_and_simulation_records_follow_their_gates():
    entry = {"index": 5, "label": "purity plan", "mode": "planning", "status": "success",
             "recommended_parameters": [{"T_C": 450, "t_min": 30}]}
    approved = {"key_findings": ["Optimization target: purity (maximize)."],
                "plan_review": {"human_review": {"status": "accepted"}, "blocking_findings": []}}
    recs = board_mod.records_for(entry, approved)
    assert [(r["kind"], r["status"]) for r in recs] == [("claim", "verified"), ("parameter_point", "verified")]
    unattended = {"key_findings": ["Optimization target: purity (maximize)."],
                  "plan_review": {"human_review": None, "unattended_gate": {"would_have_been": "accepted"},
                                  "blocking_findings": [{"issue": "650 C exceeds the furnace limit",
                                                         "conflict": "650 C vs 600 C"}]}}
    recs = board_mod.records_for(entry, unattended)
    assert [(r["kind"], r["status"]) for r in recs] == [
        ("claim", "provisional"), ("parameter_point", "verified"), ("hazard", "verified")]
    assert "unattended" in recs[0]["evidence"]["gate"] and recs[2]["payload"]["conflict"] == "650 C vs 600 C"
    sim = {"structures": [{"slug": "anatase", "structure_path": "/s/POSCAR", "description": "anatase 2x2x1",
                           "input_files": {"INCAR": "/s/INCAR"}, "validation_status": "success"},
                          {"slug": "rutile", "structure_path": "/r/POSCAR", "validation_status": "needs_correction"},
                          {"slug": "brookite", "structure_path": "/b/POSCAR"}]}
    recs = board_mod.records_for({"index": 6, "mode": "simulation", "status": "success", "label": "cells"}, sim)
    assert [(r["payload"]["slug"], r["status"]) for r in recs] == [
        ("anatase", "verified"), ("rutile", "provisional"), ("brookite", "provisional")]
    assert recs[0]["payload"]["inputs"] == ["INCAR"]
    # a failed delegation posts nothing
    assert board_mod.records_for({"mode": "analysis", "status": "error"}, {"key_findings": ["[a] x"]}) == []


def test_planning_run_task_reports_how_the_plan_was_settled(tmp_path, monkeypatch):
    """The board's planning rule needs the review stamps on the result: a
    human-approved plan, an unattended one, and any standing blocking
    finding travel as ``plan_review``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-dummy")
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent
    orch = PlanningOrchestratorAgent(api_key="sk-dummy", data_dir=str(tmp_path),
                                     base_dir=str(tmp_path / "pl"))

    def fake_chat(_prompt):
        orch._last_chat_hit_iter_cap = False
        orch._last_chat_error = None
        return "Plan ready."
    orch.chat = fake_chat
    orch.planner.state["current_plan"] = {
        "iteration": 1, "human_review": {"status": "accepted", "iteration": 1},
        "critic_findings": [{"severity": "blocking", "issue": "650 C exceeds the furnace limit",
                             "conflict": "650 C vs 600 C"}]}
    r = orch.run_task("plan it")
    assert r["plan_review"]["human_review"] == {"status": "accepted", "iteration": 1}
    assert r["plan_review"]["unattended_gate"] is None
    assert r["plan_review"]["blocking_findings"] == [{"issue": "650 C exceeds the furnace limit",
                                                      "conflict": "650 C vs 600 C"}]
    orch.planner.state["current_plan"] = {"iteration": 2, "unattended_gate": {"would_have_been": "accepted"}}
    r = orch.run_task("plan again")
    assert r["plan_review"]["human_review"] is None and r["plan_review"]["unattended_gate"]["would_have_been"] == "accepted"
    assert r["plan_review"]["blocking_findings"] == []


def test_the_recipe_is_the_agents_approved_script(tmp_path):
    from scilink.agents.meta_agent.board import _recipe_script
    assert _recipe_script(None) is None and _recipe_script(tmp_path) is None
    (tmp_path / "scripts").mkdir()
    assert _recipe_script(tmp_path) is None
    (tmp_path / "scripts" / "spectrum_0001.py").write_text("")       # a series: one per spectrum
    (tmp_path / "scripts" / "spectrum_0000.py").write_text("")
    assert _recipe_script(tmp_path).name == "spectrum_0000.py"
    (tmp_path / "scripts" / "fitting_script.py").write_text("")      # the curve agent's single fit
    assert _recipe_script(tmp_path).name == "fitting_script.py"
    (tmp_path / "scripts" / "analysis_script.py").write_text("")     # the image agent's
    assert _recipe_script(tmp_path).name == "analysis_script.py"
