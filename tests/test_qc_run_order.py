"""Multi-deck molecular QC runs: which deck runs, in what order, and the logs.

A reply with several decks (e.g. hcn.nw, hnc.nw, ts.nw) used to set no
entry file, so the run command's ``{script}`` was filled with nothing and the
engine started with no input file. No LLM, engine or API key needed.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.molecular_qc_agent import resolve_run_order  # noqa: E402
from scilink.agents.sim_agents.refinement import (  # noqa: E402
    AutonomousPolicy, Executor, LocalExecutor, Phase, RefinementContext, Stage,
    run_campaign,
)
from scilink.agents.sim_agents.simulation_pipeline import (  # noqa: E402
    _collect_phases, _collect_stages, _fill_script,
)


# ── resolve_run_order ────────────────────────────────────────────────────

def test_single_deck_needs_no_run_order():
    r = {"input_files": {"job.nw": "task scf"}}
    assert resolve_run_order(r) is None
    assert r["entry_file"] == "job.nw"
    assert "phases" not in r


def test_several_decks_become_phases_in_run_order():
    r = {"input_files": {"ts.nw": "T", "hcn.nw": "A", "hnc.nw": "B",
                         "basis.lib": "L"},
         "run_order": ["hcn.nw", "hnc.nw", "ts.nw"]}
    assert resolve_run_order(r) is None
    assert "entry_file" not in r
    assert [p["entry_file"] for p in r["phases"]] == ["hcn.nw", "hnc.nw", "ts.nw"]
    # each phase gets its own deck plus the unlisted support file
    assert r["phases"][0]["input_files"] == {"hcn.nw": "A", "basis.lib": "L"}
    assert r["phases"][2]["input_files"] == {"ts.nw": "T", "basis.lib": "L"}


def test_run_order_of_one_with_support_file_sets_entry():
    r = {"input_files": {"job.nw": "J", "basis.lib": "L"}, "run_order": ["job.nw"]}
    assert resolve_run_order(r) is None
    assert r["entry_file"] == "job.nw"


def test_explicit_entry_file_still_honoured():
    r = {"input_files": {"job.nw": "J", "basis.lib": "L"}, "entry_file": "job.nw"}
    assert resolve_run_order(r) is None
    assert r["entry_file"] == "job.nw"


def test_run_order_as_a_bare_string():
    r = {"input_files": {"job.nw": "J", "x.lib": "L"}, "run_order": "job.nw"}
    assert resolve_run_order(r) is None
    assert r["entry_file"] == "job.nw"


@pytest.mark.parametrize("reply, needle", [
    ({"input_files": {"a.nw": "1", "b.nw": "2"}}, "no run_order"),
    ({"input_files": {"a.nw": "1"}, "run_order": ["b.nw"]}, "b.nw"),
    ({"input_files": {"a.nw": "1"}, "run_order": []}, "non-empty"),
    ({"input_files": {"a.nw": "1"}, "run_order": [3]}, "non-empty"),
    ({"input_files": {"a.nw": "1", "b.nw": "2"}, "run_order": ["a.nw", "a.nw"]},
     "more than once"),
])
def test_unusable_replies_are_refused(reply, needle):
    problem = resolve_run_order(reply)
    assert problem and needle in problem


# ── the run command never gets an empty file name ────────────────────────

def test_fill_script_refuses_empty_entry():
    assert _fill_script("nwchem {script}", "job.nw", "p") == "nwchem job.nw"
    assert _fill_script("mpirun vasp_std", "", "p") == "mpirun vasp_std"
    with pytest.raises(ValueError, match="no entry file"):
        _fill_script("nwchem {script}", "", "p")


def test_phases_get_one_command_per_deck():
    r = {"input_files": {"hcn.nw": "A", "hnc.nw": "B"},
         "run_order": ["hcn.nw", "hnc.nw"]}
    resolve_run_order(r)
    phases = _collect_phases(r, "/tmp/x", "nwchem {script}")
    assert [p.run_command for p in phases] == ["nwchem hcn.nw", "nwchem hnc.nw"]
    assert all(p.run_dir == "/tmp/x" for p in phases)


def test_old_multi_file_reply_is_refused_not_run_empty():
    # The pre-fix shape: two decks, no entry_file, no phases.
    r = {"input_files": {"a.nw": "1", "b.nw": "2"}}
    with pytest.raises(ValueError, match="no entry file"):
        _collect_stages(r, "/tmp/x", "nwchem {script}")


def _pipeline(tmp_path, monkeypatch, gen):
    from scilink.agents.sim_agents import critics, refinement, simulation_pipeline as sp
    monkeypatch.setattr(sp, "_generate_inputs", lambda **kw: gen)
    monkeypatch.setattr(critics, "RunCritic", lambda **kw: object())
    seen = {}

    def fake_run_campaign(stages, executor, run_critic, policy, ctx,
                          pre_run_verdict=None):
        seen["commands"] = [p.run_command for s in stages for p in s.phases]
        return {"status": "success"}

    monkeypatch.setattr(refinement, "run_campaign", fake_run_campaign)
    (tmp_path / "mol.xyz").write_text("1\n\nH 0 0 0\n")
    out = sp._run_workflow_once(
        "water", scale="molecular_qc", software="nwchem",
        structure_file=str(tmp_path / "mol.xyz"), output_dir=str(tmp_path),
        validate=False, executor=_LogWriter(), run_command="nwchem {script}")
    return out, seen


def test_pipeline_fails_a_reply_with_no_entry_instead_of_launching(tmp_path, monkeypatch):
    gen = {"status": "success", "input_files": {"a.nw": "1", "b.nw": "2"}}
    out, seen = _pipeline(tmp_path, monkeypatch, gen)
    assert "commands" not in seen          # the campaign never started
    assert out["final_status"] == "refinement_failed"
    assert "no entry file" in out["refinement"]["error"]


def test_pipeline_runs_each_deck_of_a_multi_deck_reply(tmp_path, monkeypatch):
    gen = {"status": "success", "input_files": {"a.nw": "1", "b.nw": "2"},
           "run_order": ["b.nw", "a.nw"]}
    resolve_run_order(gen)
    out, seen = _pipeline(tmp_path, monkeypatch, gen)
    assert seen["commands"] == ["nwchem b.nw", "nwchem a.nw"]
    assert out["final_status"] == "success"


# ── each phase's log survives the next phase ─────────────────────────────

class _LogWriter(Executor):
    """Writes run_stdout.log as the engine would: the deck name it ran."""

    def run(self, input_files, run_command, run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / LocalExecutor.STDOUT_FILE).write_text(
            f"ran {run_command}\n")
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


class _Good:
    def assess(self, *a, **kw):
        return {"status": "success", "run_status": "succeeded",
                "verdict": "good", "suggested_fixes": None}


def _ctx():
    return RefinementContext(
        research_goal="g", scale="molecular_qc", engine="nwchem",
        skill="nwchem", domain="molecular_qc", autonomy="autonomous",
        max_cycles=1)


def _decks(d, names):
    return [Phase(name=n, input_files={n: "x"}, run_command=f"nwchem {n}",
                  run_dir=d, entry_file=n) for n in names]


def _kept(d):
    return sorted(p.name for p in (Path(d) / "phase_logs").glob("*")) \
        if (Path(d) / "phase_logs").is_dir() else []


def test_each_phase_log_is_kept_in_run_order(tmp_path):
    names = ("hcn.nw", "hnc.nw", "ts.nw")
    out = run_campaign([Stage(name="run", phases=_decks(str(tmp_path), names))],
                       _LogWriter(), _Good(), AutonomousPolicy(), _ctx())
    assert out["status"] == "success"
    kept = _kept(tmp_path)
    assert kept == ["01_hcn.nw.run_stdout.log", "02_hnc.nw.run_stdout.log",
                    "03_ts.nw.run_stdout.log"]
    for k, n in zip(kept, names):
        assert (tmp_path / "phase_logs" / k).read_text() == f"ran nwchem {n}\n"
    assert (tmp_path / "run_stdout.log").read_text() == "ran nwchem ts.nw\n"


def test_staged_steps_sharing_a_directory_keep_their_logs_too(tmp_path):
    # MD-style campaign: each step is its own one-phase stage, one directory.
    d = str(tmp_path)
    stages = [Stage(name=n, phases=_decks(d, [n])) for n in ("equil.in", "prod.in")]
    run_campaign(stages, _LogWriter(), _Good(), AutonomousPolicy(), _ctx())
    assert _kept(d) == ["01_equil.in.run_stdout.log", "02_prod.in.run_stdout.log"]


def test_a_directory_used_once_gets_no_copy(tmp_path):
    run_campaign([Stage(name="run", phases=_decks(str(tmp_path), ["job.nw"]))],
                 _LogWriter(), _Good(), AutonomousPolicy(), _ctx())
    assert sorted(os.listdir(tmp_path)) == ["run_stdout.log"]


class _NeverStarts(_LogWriter):
    """The second deck produces nothing (e.g. a cluster job that never ran):
    the executor clears the previous run's files and writes none."""

    def __init__(self):
        self.n = 0

    def run(self, input_files, run_command, run_dir):
        self.n += 1
        for name in ("run_stdout.log", "run_stderr.log", "run_returncode.txt"):
            (Path(run_dir) / name).unlink(missing_ok=True)
        if self.n == 1:
            return super().run(input_files, run_command, run_dir)
        return {"status": "completed", "output_dir": run_dir, "returncode": None}


def test_a_phase_with_no_output_gets_no_copy_of_the_previous_log(tmp_path):
    run_campaign([Stage(name="run", phases=_decks(str(tmp_path), ["a.nw", "b.nw"]))],
                 _NeverStarts(), _Good(), AutonomousPolicy(), _ctx())
    assert _kept(tmp_path) == ["01_a.nw.run_stdout.log"]


def test_the_engine_reader_never_reads_a_kept_copy_as_the_current_run(tmp_path):
    # Deck 2 produced nothing: the only log left is deck 1's kept copy, and
    # NWChem's reader must not parse it as deck 2's output.
    from scilink.skills.molecular_qc.nwchem.nwchem_output import snapshot_run
    run_campaign([Stage(name="run", phases=_decks(str(tmp_path), ["a.nw", "b.nw"]))],
                 _NeverStarts(), _Good(), AutonomousPolicy(), _ctx())
    snap = snapshot_run(str(tmp_path))
    assert not any("a.nw" in f for f in snap["files_found"])
    assert "No NWChem output" in snap["headline"]


def test_logs_left_by_an_earlier_attempt_are_cleared(tmp_path):
    old = tmp_path / "phase_logs"
    old.mkdir()
    (old / "01_old.nw.run_stdout.log").write_text("stale\n")
    run_campaign([Stage(name="run", phases=_decks(str(tmp_path), ["job.nw"]))],
                 _LogWriter(), _Good(), AutonomousPolicy(), _ctx())
    assert not old.exists()


def test_launch_failure_leaves_no_earlier_output(tmp_path, monkeypatch):
    from scilink.agents.sim_agents import refinement
    ex = LocalExecutor(timeout=30)
    ex.run({}, "echo first", str(tmp_path))
    assert (tmp_path / "run_stdout.log").read_text().strip() == "first"

    def cannot_launch(*a, **k):
        raise OSError("no such engine")

    monkeypatch.setattr(refinement, "run_engine", cannot_launch)
    out = ex.run({}, "nwchem job.nw", str(tmp_path))
    assert out["status"] == "error"
    assert not (tmp_path / "run_stdout.log").exists()
    assert (tmp_path / "run_returncode.txt").read_text() == "launch_error"


def test_timeout_leaves_no_earlier_output(tmp_path):
    ex = LocalExecutor(timeout=30)
    ex.run({}, "echo first", str(tmp_path))
    ex.timeout = 1
    out = ex.run({}, "sleep 10", str(tmp_path))
    assert out["status"] == "error"
    assert not (tmp_path / "run_stdout.log").exists()
    assert (tmp_path / "run_returncode.txt").read_text() == "timeout"


# ── the agent itself, with a canned LLM reply ────────────────────────────

_WATER = "3\nwater\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
_DECK = ("start {n}\ngeometry\n O 0 0 0.117\n H 0 0.757 -0.469\n"
         " H 0 -0.757 -0.469\nend\nbasis\n * library 6-31g*\nend\n"
         "task dft energy\n")


def _agent_reply(tmp_path, monkeypatch, *replies, prompts=None):
    """Run generate_inputs with canned LLM replies, one per call (the last
    repeats); ``prompts`` collects the prompts sent."""
    from scilink.agents.sim_agents.molecular_qc_agent import MolecularQCAgent
    agent = MolecularQCAgent(api_key="dummy", base_url="http://localhost:0")
    sent = prompts if prompts is not None else []

    class _Resp:
        def __init__(self, k):
            self.text = str(k)

    def generate(prompt, **kw):
        sent.append(prompt)
        return _Resp(min(len(sent), len(replies)) - 1)

    monkeypatch.setattr(agent.model, "generate_content", generate)
    monkeypatch.setattr(agent, "_parse_response", lambda text: dict(replies[int(text)]))
    xyz = tmp_path / "water.xyz"
    xyz.write_text(_WATER)
    return agent.generate_inputs(structure_file=str(xyz), request="energies")


def test_agent_builds_phases_from_run_order(tmp_path, monkeypatch):
    r = _agent_reply(tmp_path, monkeypatch, {
        "input_files": {"gas.nw": _DECK.format(n="gas"), "solv.nw": _DECK.format(n="solv")},
        "run_order": ["gas.nw", "solv.nw"]})
    assert r["status"] == "success", r.get("message")
    assert [p["entry_file"] for p in r["phases"]] == ["gas.nw", "solv.nw"]


def test_agent_refuses_several_decks_without_run_order(tmp_path, monkeypatch):
    r = _agent_reply(tmp_path, monkeypatch, {
        "input_files": {"gas.nw": _DECK.format(n="gas"), "solv.nw": _DECK.format(n="solv")}})
    assert r["status"] == "error"
    assert "run_order" in r["message"]


_TWO = {"gas.nw": _DECK.format(n="gas"), "solv.nw": _DECK.format(n="solv")}


def test_agent_asks_only_for_the_order_and_keeps_the_decks(tmp_path, monkeypatch):
    prompts = []
    r = _agent_reply(tmp_path, monkeypatch,
                     {"input_files": _TWO},
                     {"run_order": ["solv.nw", "gas.nw"]},
                     prompts=prompts)
    assert r["status"] == "success", r.get("message")
    assert [p["entry_file"] for p in r["phases"]] == ["solv.nw", "gas.nw"]
    assert r["input_files"] == _TWO                      # decks kept as returned
    assert len(prompts) == 2
    assert "### gas.nw" in prompts[1] and "start solv" in prompts[1]   # decks shown
    assert "Do not change" in prompts[1]
    assert r["run_order_repair"]["outcome"] == "fixed"


def test_a_failed_repair_reports_its_own_reason(tmp_path, monkeypatch):
    prompts = []
    r = _agent_reply(tmp_path, monkeypatch, {"input_files": _TWO},
                     {"run_order": ["gas.nw", "missing.nw"]}, prompts=prompts)
    assert r["status"] == "error"
    assert "no run_order" in r["message"] and "missing.nw" in r["message"]
    assert r["run_order_repair"]["outcome"] == "failed"
    assert len(prompts) == 2


def test_a_repair_answer_without_run_order_is_reported(tmp_path, monkeypatch):
    r = _agent_reply(tmp_path, monkeypatch, {"input_files": _TWO}, {"notes": "?"})
    assert r["status"] == "error" and "the answer had no run_order" in r["message"]


def test_a_usable_reply_costs_one_call(tmp_path, monkeypatch):
    prompts = []
    r = _agent_reply(tmp_path, monkeypatch,
                     {"input_files": {"job.nw": _DECK.format(n="job")}},
                     prompts=prompts)
    assert r["status"] == "success" and r["entry_file"] == "job.nw"
    assert len(prompts) == 1


# ── caveats on the result, and one single-file rule ─────────────────────

def test_unrun_files_are_reported():
    r = {"input_files": {"a.nw": "1", "b.nw": "2"}, "run_order": ["a.nw"]}
    assert resolve_run_order(r) is None
    assert r["entry_file"] == "a.nw"
    assert any("never run: b.nw" in w for w in r["warnings"])


def test_an_overridden_entry_file_is_reported():
    r = {"input_files": {"a.nw": "1", "b.nw": "2"}, "entry_file": "b.nw",
         "run_order": ["a.nw", "b.nw"]}
    assert resolve_run_order(r) is None
    assert any("entry_file b.nw was replaced" in w for w in r["warnings"])


def test_a_checking_copy_does_not_touch_the_replys_warnings():
    r = {"input_files": {"a.nw": "1", "b.nw": "2"}, "run_order": ["a.nw"],
         "warnings": ["from the model"]}
    resolve_run_order(dict(r))
    assert r["warnings"] == ["from the model"]


def test_an_empty_entry_file_means_undeclared_everywhere():
    from scilink.agents.sim_agents.refinement import single_entry
    one = {"job.nw": "x"}
    assert single_entry(one, "") == "job.nw"
    r = {"input_files": dict(one), "entry_file": ""}
    assert resolve_run_order(r) is None and r["entry_file"] == "job.nw"
    phases = _collect_phases({"input_files": one, "entry_file": ""}, "/tmp/x",
                             "nwchem {script}")
    assert phases[0].run_command == "nwchem job.nw"


def test_convergence_skip_is_recorded_as_skipped(tmp_path, monkeypatch):
    from scilink.agents.sim_agents import critics, refinement, simulation_pipeline as sp
    gen = {"status": "success", "input_files": {"a.nw": "1", "b.nw": "2"},
           "run_order": ["a.nw", "b.nw"]}
    resolve_run_order(gen)
    monkeypatch.setattr(sp, "_generate_inputs", lambda **kw: gen)
    monkeypatch.setattr(sp, "_convergence_specs", lambda *a, **k: [{"param": "x"}])
    monkeypatch.setattr(critics, "RunCritic", lambda **kw: object())
    monkeypatch.setattr(refinement, "run_campaign",
                        lambda *a, **k: {"status": "success"})
    (tmp_path / "mol.xyz").write_text("1\n\nH 0 0 0\n")
    out = sp._run_workflow_once(
        "water", scale="molecular_qc", software="nwchem",
        structure_file=str(tmp_path / "mol.xyz"), output_dir=str(tmp_path),
        validate=False, executor=_LogWriter(), run_command="nwchem {script}",
        converge_parameters=True)
    assert out["parameter_convergence"]["status"] == "skipped"
    assert "several decks" in out["parameter_convergence"]["message"]
    assert any("convergence was requested but skipped" in w for w in out["warnings"])


def test_generation_caveats_reach_the_runs_warnings(tmp_path, monkeypatch):
    gen = {"status": "success", "input_files": {"a.nw": "1", "b.nw": "2"},
           "run_order": ["a.nw"],
           "run_order_repair": {"outcome": "fixed", "run_order": ["a.nw"]}}
    resolve_run_order(gen)
    out, _ = _pipeline(tmp_path, monkeypatch, gen)
    assert any("never run: b.nw" in w for w in out["warnings"])
    assert any("one follow-up request" in w for w in out["warnings"])


def test_a_staged_step_with_one_file_needs_no_entry_file():
    stages = _collect_stages(
        {"stages": [{"name": "s1", "input_files": {"in.x": "1"}}]},
        "/tmp/x", "engine {script}")
    assert stages[0].phases[0].run_command == "engine in.x"
