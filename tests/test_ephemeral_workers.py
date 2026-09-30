"""Ephemeral workers of every mode: built like the persistent specialists, and
isolated from each other when they run at once.

A persistent child's ``run_task`` reports what one call produced as a window
over shared state (``analysis_results[n_before:]``, a file snapshot of its
directory, ``generated_structures[n_before:]``), so two concurrent calls on one
child report each other's output. A swarm runs a worker per item instead; these
tests hold every mode pair to that, with the LLM turn replaced by a stand-in
that produces the mode's output slowly enough for the calls to overlap.
"""

import itertools
import threading
import time
from pathlib import Path

import pytest

from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
from scilink.agents.meta_agent.workers import MODES, build_child
from scilink.agents.meta_agent import fanout


@pytest.fixture()
def meta(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"),
                                 model_name="anthropic/claude-sonnet-4-5",
                                 meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))


def _needs(mode):
    if mode == "simulation":
        pytest.importorskip("ase")


def _persistent(meta, mode):
    return {"analysis": meta._get_analysis_child, "planning": meta._get_planning_child,
            "simulation": meta._get_simulation_child}[mode]()


def _resting(child):
    for attr in ("analysis_mode", "autonomy_level", "simulation_mode"):
        if hasattr(child, attr):
            return getattr(child, attr).name
    raise AssertionError("no resting mode")


@pytest.mark.parametrize("mode", MODES)
def test_a_worker_is_built_like_the_specialist(meta, tmp_path, mode):
    _needs(mode)
    skill = tmp_path / "my_skill.md"
    skill.write_text("---\ndescription: a user skill\n---\n\n## overview\n\nA user skill.\n")
    meta._shared_extensions.append({"kind": "skill", "skill_path": str(skill)})

    specialist = _persistent(meta, mode)
    worker = build_child(meta, mode, tmp_path / "swarm" / f"01_{mode}")

    assert type(worker) is type(specialist)
    assert worker is not specialist and worker not in meta._children.values()
    assert Path(worker.base_dir) == tmp_path / "swarm" / f"01_{mode}"
    assert getattr(worker, "model_name", None) == getattr(specialist, "model_name", None)
    assert worker._agent_label == f"{mode.capitalize()} worker"
    assert specialist._agent_label == f"{mode.capitalize()} specialist"
    assert str(skill) in [str(p) for p in (getattr(worker, "_custom_skills", {}) or {}).values()]
    # Workers rest in AUTONOMOUS, except planning, whose constructor needs a
    # data_dir there; run_task pins the autonomy per call in every case.
    assert _resting(specialist) == "CO_PILOT"
    assert _resting(worker) == ("CO_PILOT" if mode == "planning" else "AUTONOMOUS")


def test_a_planning_worker_uses_the_attached_kb_or_its_own(meta, tmp_path):
    own = build_child(meta, "planning", tmp_path / "w1")
    assert Path(own.knowledge_dir) == tmp_path / "w1" / "knowledge"
    store = tmp_path / "kb"                       # a store KB: it has a manifest
    store.mkdir()
    (store / "manifest.json").write_text('{"name": "kb", "embedding_model": "m", "sources": []}')
    (store / "default_kb_docs.faiss").write_bytes(b"INDEX")
    (store / "default_kb_docs.json").write_text("[]")
    meta.knowledge_dir = store
    attached = build_child(meta, "planning", tmp_path / "w2")
    assert Path(attached.knowledge_dir) == store   # the planner copies a store KB into its kb_cache


def test_the_fanout_branch_is_an_analysis_worker(meta, tmp_path):
    child = fanout._make_ephemeral_analysis_child(meta, tmp_path / "branch")
    assert type(child).__name__ == "AnalysisOrchestratorAgent"
    assert child._agent_label == "Analysis branch"
    assert _resting(child) == "AUTONOMOUS" and child not in meta._children.values()


def test_an_unknown_mode_is_refused(meta, tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        build_child(meta, "optimization", tmp_path / "w")


# ------------------------------------------------------------------ isolation

def _fake_turn(child, mode, tag, started, delay=0.3):
    """Stand in for the LLM turn: produce this mode's output, slowly."""
    base = Path(child.base_dir)

    def chat(_prompt):
        started.wait(5)                  # both calls are inside chat() at once
        time.sleep(delay)
        if mode == "analysis":
            out = base / f"analysis_{tag}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.json").write_text("{}")
            child.analysis_results.append({
                "analysis_id": tag, "status": "success", "output_directory": str(out),
                "full_result": {"scientific_claims": [{"claim": f"claim from {tag}"}]}})
        elif mode == "planning":
            (base / f"plan_{tag}.md").write_text(f"# plan {tag}\n")
        else:
            path = base / "structures" / tag / "POSCAR"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(tag)
            child.generated_structures.append({"slug": tag, "description": tag,
                                               "structure_path": str(path)})
        return f"done {tag}"

    child.chat = chat


def _own(result, tag):
    return all(tag in f for f in result["files_produced"]) and result["files_produced"]


@pytest.mark.parametrize("pair", list(itertools.combinations_with_replacement(MODES, 2)),
                         ids=lambda p: "+".join(p))
def test_two_workers_at_once_each_report_only_their_own_output(meta, tmp_path, pair):
    for mode in pair:
        _needs(mode)
    barrier = threading.Barrier(2)
    started = threading.Event()
    jobs = []
    for i, mode in enumerate(pair):
        tag = f"item{i}_{mode}"
        child = build_child(meta, mode, tmp_path / "swarm" / f"{i:02d}_{mode}")
        _fake_turn(child, mode, tag, started)
        jobs.append((mode, tag, child))

    results, errors = {}, []

    def run(mode, tag, child):
        try:
            barrier.wait(5)
            started.set()
            results[tag] = child.run_task(f"task for {tag}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=job) for job in jobs]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert not errors, errors
    for mode, tag, _ in jobs:
        res = results[tag]
        assert res["status"] == "success", res
        assert _own(res, tag), (tag, res["files_produced"])
        if mode == "analysis":
            assert [a["analysis_id"] for a in res["analyses"]] == [tag]
            assert res["key_findings"] == [f"[{tag}] claim from {tag}"]
        if mode == "simulation":
            assert [s["slug"] for s in res["structures"]] == [tag]


def test_a_planning_worker_gets_its_own_copy_of_a_plain_folder_kb(meta, tmp_path):
    """A store KB is copied by the planner itself; a plain folder KB is an
    index the planner appends to in place, so a worker gets a copy too."""
    plain = tmp_path / "kb_storage"
    plain.mkdir()
    (plain / "default_kb_docs.faiss").write_bytes(b"INDEX")
    (plain / "default_kb_docs.json").write_text("[]")
    meta.knowledge_dir = plain
    w1 = build_child(meta, "planning", tmp_path / "w1")
    w2 = build_child(meta, "planning", tmp_path / "w2")
    assert Path(w1.knowledge_dir) == tmp_path / "w1" / "knowledge"
    assert Path(w2.knowledge_dir) == tmp_path / "w2" / "knowledge"
    assert (tmp_path / "w1" / "knowledge" / "default_kb_docs.faiss").read_bytes() == b"INDEX"
    specialist = meta._get_planning_child()
    assert Path(specialist.knowledge_dir) == plain                     # the specialist keeps the folder
