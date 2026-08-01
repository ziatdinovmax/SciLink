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
    _select_deck_fix,
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
               fixes_mode="auto", input_files=None, check_observables=False,
               required_observables=None, deterministic_findings=None):
        self.seen_input_files = input_files
        self.seen_check_observables = check_observables
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


# ── critic sees the real deck ──

def test_critic_receives_the_real_deck_as_input(tmp_path):
    ph = _phase(tmp_path)
    critic = StubCritic([{"run_status": "succeeded"}])
    _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    # the gate hands the critic the full deck to patch, keyed by entry name
    assert critic.seen_input_files == {"run.lammps": REAL_DECK}


# ── mis-keyed fix backstop (the S2 run.lammps_header_patch failure) ──

def test_gate_remaps_miskeyed_full_deck_fix(tmp_path):
    # The critic keys a COMPLETE corrected deck under a near-miss filename
    # instead of the entry; the length-gated backstop still applies it.
    ph = _phase(tmp_path)
    critic = StubCritic([
        {"run_status": "failed",
         "suggested_fixes": {"run.lammps_header_patch": FIXED_DECK}},
        {"run_status": "succeeded"},
    ])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "passed"
    assert ph.input_files["run.lammps"] == FIXED_DECK


def test_gate_rejects_miskeyed_fragment_fix(tmp_path):
    # A mis-keyed *fragment* (shorter than the deck) is rejected — applying it
    # would drop the run commands — so the gate reports unfixed, deck untouched.
    ph = _phase(tmp_path)
    before = ph.input_files["run.lammps"]
    fragment = "pair_style lj/cut/coul/long 9.0\nread_data system.data\n"
    assert len(fragment) < len(before)
    critic = StubCritic([
        {"run_status": "failed",
         "suggested_fixes": {"run.lammps_header_patch": fragment}},
    ])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "unfixed"
    assert ph.input_files["run.lammps"] == before


def test_select_deck_fix_prefers_direct_key():
    fixes = {"run.lammps": FIXED_DECK, "run.lammps_header_patch": "x" * 9999}
    assert _select_deck_fix(fixes, "run.lammps", REAL_DECK) == FIXED_DECK


def test_select_deck_fix_remaps_full_and_rejects_fragment():
    full = {"other_name": FIXED_DECK}
    assert _select_deck_fix(full, "run.lammps", REAL_DECK) == FIXED_DECK
    frag = {"other_name": "short\n"}
    assert _select_deck_fix(frag, "run.lammps", REAL_DECK) is None
    assert _select_deck_fix(None, "run.lammps", REAL_DECK) is None


# ── observable-coverage check (deck starts clean but omits a required output) ──

# The real deck plus a required observable output; longer than REAL_DECK, so it
# survives the length-gated fix backstop. The observable is intentionally
# generic (a placeholder "compute/fix" pair) so the fixture does not encode any
# particular property recipe.
COVERAGE_DECK = REAL_DECK + (
    "compute obs all property/atom foo\n"
    "fix obs_out all ave/time 100 1 100 c_obs file obs.dat\n"
)


def test_gate_requests_observable_coverage(tmp_path):
    # The gate must ask the critic to run the coverage check (not just setup).
    ph = _phase(tmp_path)
    critic = StubCritic([{"run_status": "succeeded"}])
    _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert critic.seen_check_observables is True


def test_gate_fixes_missing_observable_when_setup_starts_clean(tmp_path):
    # Setup succeeds (twin runs), but a required output is absent. The gate must
    # NOT pass on that cycle — it applies the coverage fix and re-validates.
    ph = _phase(tmp_path)
    critic = StubCritic([
        {"run_status": "succeeded",
         "missing_observables": [{"property": "p", "required_output": "obs.dat",
                                  "reason": "not emitted"}],
         "suggested_fixes": {"run.lammps": COVERAGE_DECK}},
        {"run_status": "succeeded", "missing_observables": []},
    ])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "passed"
    assert rec["cycles"] == 2
    assert ph.input_files["run.lammps"] == COVERAGE_DECK


def test_gate_passes_when_setup_ok_and_no_missing_observables(tmp_path):
    # Regression: a clean deck with full coverage still passes on the first cycle.
    ph = _phase(tmp_path)
    critic = StubCritic([{"run_status": "succeeded", "missing_observables": []}])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "passed"
    assert rec["cycles"] == 1


def test_gate_unfixed_when_missing_observable_and_no_fix(tmp_path):
    # Coverage gap flagged but no corrected deck offered -> unfixed, deck untouched.
    ph = _phase(tmp_path)
    before = ph.input_files["run.lammps"]
    critic = StubCritic([
        {"run_status": "succeeded",
         "missing_observables": [{"property": "p", "required_output": "obs.dat"}],
         "suggested_fixes": None},
    ])
    rec = _dry_run_gate(ph, StubExecutor(), critic, _ctx())
    assert rec["status"] == "unfixed"
    assert ph.input_files["run.lammps"] == before
    assert _select_deck_fix({}, "run.lammps", REAL_DECK) is None
