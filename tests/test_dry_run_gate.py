"""Engine-neutral dry-run setup-validation gate (refinement.py).

The gate runs a cheap "dry-run" twin of the deck (engine setup only) before the
expensive production run, fixing setup errors on the REAL deck until it starts
clean. These tests stub the executor and critic (no lmp, no LLM) and use the
real LAMMPS prepare_dry_run via the engine-neutral resolver.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from scilink.agents.sim_agents.refinement import (
    Phase,
    _dry_run_gate,
    _resolve_skill_callable,
)

REAL_DECK = (
    "units real\natom_style full\nboundary p p p\n"
    "pair_style lj/cut/coul/long 9.0\nbond_style harmonic\n"
    "angle_style harmonic\ndihedral_style fourier\n"
    "read_data system.data\nkspace_style pppm 1e-4\n"
    "run 5000\n"
)
FIXED_DECK = REAL_DECK.replace("dihedral_style fourier", "dihedral_style harmonic")


class StubExecutor:
    """Materializes input files into run_dir and records each call."""

    def __init__(self, raise_on_run=False):
        self.calls = []
        self.raise_on_run = raise_on_run

    def run(self, input_files, run_command, run_dir):
        if self.raise_on_run:
            raise RuntimeError("boom")
        os.makedirs(run_dir, exist_ok=True)
        for name, contents in (input_files or {}).items():
            with open(os.path.join(run_dir, name), "w") as fh:
                fh.write(contents)
        self.calls.append({"input_files": dict(input_files),
                           "run_dir": run_dir})
        return {"status": "ok", "output_dir": run_dir}


class StubCritic:
    """Returns queued verdicts; records the deck text present at assess time."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen_decks = []

    def assess(self, output_dir, research_goal, skill=None, domain=None,
               fixes_mode="auto"):
        deck_path = os.path.join(output_dir, "run.lammps")
        if os.path.isfile(deck_path):
            with open(deck_path) as fh:
                self.seen_decks.append(fh.read())
        return self.verdicts.pop(0)


def _ctx():
    return SimpleNamespace(skill="lammps", domain="molecular_dynamics",
                           research_goal="aqueous electrolyte MD")


def _phase(tmp_path):
    # a dependency the dry-run must carry into its staging dir
    (tmp_path / "system.data").write_text("LAMMPS data file\n")
    return Phase(
        name="production",
        input_files={"run.lammps": REAL_DECK},
        run_command="lmp -in run.lammps",
        run_dir=str(tmp_path),
        entry_file="run.lammps",
    )


# ── resolver ──

def test_resolver_finds_lammps_prepare_dry_run():
    fn = _resolve_skill_callable("lammps", "molecular_dynamics", "prepare_dry_run")
    assert callable(fn)
    assert "run 0" in fn("run 100\n")


def test_resolver_returns_none_for_unknown():
    assert _resolve_skill_callable("nope", "molecular_dynamics", "prepare_dry_run") is None
    assert _resolve_skill_callable("lammps", "molecular_dynamics", "does_not_exist") is None


# ── skip conditions ──

def test_gate_skips_when_engine_has_no_dry_run(tmp_path):
    # VASP provides no prepare_dry_run -> gate is skipped (returns None).
    ph = _phase(tmp_path)
    ctx = SimpleNamespace(skill="vasp", domain="periodic_dft",
                          research_goal="x")
    assert _dry_run_gate(ph, StubExecutor(), StubCritic([]), ctx) is None


def test_gate_skips_when_no_entry_file(tmp_path):
    ph = _phase(tmp_path)
    ph.entry_file = ""
    assert _dry_run_gate(ph, StubExecutor(), StubCritic([]), _ctx()) is None


# ── convergence + fix targets the real deck ──

def test_gate_converges_and_fix_targets_full_deck(tmp_path):
    ph = _phase(tmp_path)
    ex = StubExecutor()
    critic = StubCritic([
        {"run_status": "failed",
         "suggested_fixes": {"run.lammps": FIXED_DECK}},
        {"run_status": "succeeded"},
    ])
    rec = _dry_run_gate(ph, ex, critic, _ctx())

    assert rec["status"] == "passed"
    assert rec["cycles"] == 2
    # the fix was applied to the REAL deck (full run length preserved)
    assert ph.input_files["run.lammps"] == FIXED_DECK
    assert "run 5000" in ph.input_files["run.lammps"]
    # the executor ran the trimmed TWIN, not the production deck
    twin = ex.calls[0]["input_files"]["run.lammps"]
    assert "run 0" in twin and "5000" not in twin
    # the critic judged against the REAL deck (so fixes patch the full file)
    assert critic.seen_decks[0] == REAL_DECK


def test_gate_exhausts_budget(tmp_path):
    ph = _phase(tmp_path)
    critic = StubCritic([
        {"run_status": "failed", "suggested_fixes": {"run.lammps": FIXED_DECK}}
    ] * 10)
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx(), max_cycles=3)
    assert rec["status"] == "exhausted"
    assert rec["cycles"] == 3


def test_gate_unfixed_when_no_fix_proposed(tmp_path):
    ph = _phase(tmp_path)
    before = ph.input_files["run.lammps"]
    critic = StubCritic([{"run_status": "failed", "suggested_fixes": None}])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "unfixed"
    assert ph.input_files["run.lammps"] == before   # deck untouched


def test_gate_fails_open_on_executor_error(tmp_path):
    ph = _phase(tmp_path)
    before = ph.input_files["run.lammps"]
    rec = _dry_run_gate(ph, StubExecutor(raise_on_run=True),
                        StubCritic([]), _ctx())
    assert rec["status"] == "skipped"          # fail-open, did not raise
    assert ph.input_files["run.lammps"] == before
