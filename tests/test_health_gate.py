"""Integration tests for the health gate inside the refinement phase runners.

The gate is engine-neutral; these drive the real control flow with a fake
executor and fake per-engine hooks (no real skill, no engine). They cover the
hybrid design: the gate judges only a cleanly-finished run (returncode 0),
hands a physical violation to the critic as POST-run physical-sanity evidence
(``physical_findings``), and deterministically holds the run below "acceptable"
so a fail-open verdict can't wave it through — while the normal repair loop
still drives the fix + re-run. A crash (returncode != 0) skips the gate and
reaches the critic. `_run_once_phase` is the combine path: gated, never retried.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents import refinement  # noqa: E402
from scilink.agents.sim_agents.refinement import (  # noqa: E402
    Executor,
    Phase,
    RefinementContext,
    Stage,
    _refine_phase,
    _run_once_phase,
    _run_phase_health_gated,
    policy_for,
    run_campaign,
)

DENSITY_SPEC = [{"observable": "density", "min": 0.02, "max": 30.0}]


class FakeExecutor(Executor):
    """Reports completion with a fixed return code, running nothing."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = 0

    def run(self, input_files, run_command, run_dir):
        self.calls += 1
        return {"status": "completed", "output_dir": run_dir,
                "returncode": self.returncode}


class RecordingCritic:
    """Returns scripted verdicts and records how each assess() was called."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def assess(self, output_dir, research_goal, skill=None, domain=None,
               input_files=None, check_observables=False,
               deterministic_findings=None, physical_findings=None, **kw):
        self.calls.append({
            "input_files": input_files,
            "check_observables": check_observables,
            "deterministic_findings": deterministic_findings,
            "physical_findings": physical_findings,
        })
        idx = min(len(self.calls) - 1, len(self._verdicts) - 1)
        return dict(self._verdicts[idx])


def _good():
    return {"verdict": "good", "run_status": "succeeded", "suggested_fixes": None}


def _needs_fix(files=None, failure_class=None):
    v = {"verdict": "needs_fixes", "run_status": "failed",
         "suggested_fixes": files or {"in.lmp": "patched deck"}}
    if failure_class:
        v["failure_class"] = failure_class
    return v


def _ctx(**kw):
    return RefinementContext(research_goal="g", skill="lammps",
                             domain="molecular_dynamics", **kw)


def _phase():
    return Phase(name="production",
                 input_files={"in.lmp": "velocity all create 300 12345"},
                 run_command="lmp -in in.lmp", run_dir="/tmp/run")


def _patch(monkeypatch, *, density, specs=DENSITY_SPEC, perturb=None):
    """Wire health specs + a density reader (and optional perturb) into refinement.

    ``density`` may be a scalar, None (reader can't read it), or a
    callable(call_index)->value for a reader whose answer changes across cycles.
    """
    monkeypatch.setattr(refinement, "_health_specs", lambda skill, domain: specs)
    state = {"n": 0}

    def _reader(output_dir, observable, since=None):
        if observable != "density":
            return None
        state["n"] += 1
        return density(state["n"] - 1) if callable(density) else density

    def _resolve(skill, domain, fn_name):
        if fn_name == "read_health_observable":
            return _reader
        if fn_name == "perturb_for_retry":
            return perturb
        return None

    monkeypatch.setattr(refinement, "_resolve_skill_callable", _resolve)


# ── _run_phase_health_gated: the gate/retry primitive (4-tuple) ──────────────

def test_gate_healthy_passes(monkeypatch):
    _patch(monkeypatch, density=0.83)
    ex = FakeExecutor()
    result, violations, inputs, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), ex, _ctx())
    assert violations == [] and checked is True and ex.calls == 1


def test_gate_nonzero_returncode_is_not_gated(monkeypatch):
    # A crash (rc != 0) is not a non-physical result — hand it to the critic.
    _patch(monkeypatch, density=0.003)
    ex = FakeExecutor(returncode=127)
    result, violations, inputs, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), ex, _ctx())
    assert violations is None and checked is None and ex.calls == 1


def test_gate_disabled_without_specs(monkeypatch):
    _patch(monkeypatch, density=0.003, specs=[])
    result, violations, _, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), FakeExecutor(), _ctx())
    assert violations is None and checked is None


def test_gate_unreadable_is_checked_false_not_a_pass(monkeypatch):
    # Reader returns None (deck doesn't log it / unsupported units): the gate
    # could not judge — violations empty, but checked is False, not True.
    _patch(monkeypatch, density=None)
    result, violations, _, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), FakeExecutor(), _ctx())
    assert violations == [] and checked is False


def test_gate_violation_no_perturb_fails_fast(monkeypatch):
    _patch(monkeypatch, density=0.003, perturb=None)
    ex = FakeExecutor()
    result, violations, _, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), ex, _ctx(max_health_retries=2))
    assert violations and "density" in violations[0].reason
    assert checked is True and ex.calls == 1       # no pointless re-runs


def test_gate_retries_with_perturb_then_fails(monkeypatch):
    seeds = []

    def perturb(input_files, attempt):
        seeds.append(attempt)
        return {"in.lmp": f"velocity all create 300 {1000 + attempt}"}

    _patch(monkeypatch, density=0.003, perturb=perturb)
    ex = FakeExecutor()
    result, violations, inputs, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), ex, _ctx(max_health_retries=2))
    assert violations and ex.calls == 3 and seeds == [1, 2]
    assert inputs == {"in.lmp": "velocity all create 300 1002"}  # last deck run


def test_gate_retry_recovers_and_returns_run_inputs(monkeypatch):
    # bad, bad, good -> succeeds on the 3rd attempt; returns the deck that ran.
    _patch(monkeypatch, density=lambda n: 0.84 if n >= 2 else 0.003,
           perturb=lambda input_files, attempt: {"in.lmp": f"reseed {attempt}"})
    ex = FakeExecutor()
    result, violations, inputs, checked = _run_phase_health_gated(
        _phase().input_files, _phase(), ex, _ctx(max_health_retries=3))
    assert violations == [] and ex.calls == 3
    assert inputs == {"in.lmp": "reseed 2"}


# ── _refine_phase: hybrid ③ ───────────────────────────────────────────────────

def test_refine_healthy_run_succeeds(monkeypatch):
    _patch(monkeypatch, density=0.83)
    critic = RecordingCritic([_good()])
    rec = _refine_phase(_phase(), FakeExecutor(), critic, policy_for("autonomous"),
                        _ctx())
    assert rec["status"] == "success"
    assert critic.calls[0]["physical_findings"] is None   # nothing to flag
    assert "health_checked" not in rec


def test_refine_violation_goes_through_postrun_channel_and_recovers(monkeypatch):
    # cycle 0: density bad -> fed to critic as POST-run physical evidence (NOT
    # the pre-run coverage channel), critic proposes a deck fix; cycle 1: good.
    _patch(monkeypatch, density=lambda n: 0.83 if n >= 1 else 0.003)
    critic = RecordingCritic([_needs_fix({"in.lmp": "fixed barostat"}), _good()])
    rec = _refine_phase(_phase(), FakeExecutor(), critic, policy_for("autonomous"),
                        _ctx(max_cycles=3))
    assert rec["status"] == "success"
    # physical evidence routed through physical_findings, not coverage flags
    assert critic.calls[0]["physical_findings"]
    assert critic.calls[0]["check_observables"] is False
    assert critic.calls[0]["deterministic_findings"] is None
    # the critic reasoned about the deck that actually ran
    assert critic.calls[1]["input_files"] == {"in.lmp": "fixed barostat"}


def test_refine_violation_vetoes_fail_open_good_verdict(monkeypatch):
    # The critic fails open ("good", no fix), but a measured violation must not
    # be accepted — the phase fails as health_gate with the reasons.
    _patch(monkeypatch, density=0.003)
    critic = RecordingCritic([_good()])
    rec = _refine_phase(_phase(), FakeExecutor(), critic, policy_for("autonomous"),
                        _ctx(max_cycles=2))
    assert rec["status"] != "success"
    assert rec["failure_class"] == "health_gate"
    assert any("density" in r for r in rec["health_violations"])


def test_refine_reasons_kept_even_when_critic_names_its_own_class(monkeypatch):
    # Critic labels the violating run "deck"; the measured violation still wins
    # the terminal class and the reasons are not dropped.
    _patch(monkeypatch, density=0.003)
    critic = RecordingCritic([_needs_fix(failure_class="deck")])
    rec = _refine_phase(_phase(), FakeExecutor(), critic, policy_for("autonomous"),
                        _ctx(max_cycles=1))
    assert rec["failure_class"] == "health_gate"
    assert rec["health_violations"]


def test_refine_records_health_checked_false_when_unreadable(monkeypatch):
    # Gate on, but the reader can't read density -> pass is unvouched; the
    # record says so, and (stub critic accepts) the phase still succeeds.
    _patch(monkeypatch, density=None)
    rec = _refine_phase(_phase(), FakeExecutor(), RecordingCritic([_good()]),
                        policy_for("autonomous"), _ctx())
    assert rec["status"] == "success"
    assert rec["health_checked"] is False


def test_refine_keeps_structure_class_beside_a_violation(monkeypatch):
    # A violating run the critic diagnoses as a bad structure keeps the
    # structure class (its own campaign routing) — not overwritten to
    # health_gate — while the measured reasons are still carried.
    _patch(monkeypatch, density=0.003)
    critic = RecordingCritic([_needs_fix(failure_class="structure")])
    rec = _refine_phase(_phase(), FakeExecutor(), critic, policy_for("autonomous"),
                        _ctx(max_cycles=1))
    assert rec["failure_class"] == "structure"
    assert rec["health_violations"]


def test_refine_crash_reaches_critic_not_health_gate(monkeypatch):
    # rc != 0 with an out-of-band reading: the gate steps aside, the critic owns it.
    _patch(monkeypatch, density=0.003)
    critic = RecordingCritic([_needs_fix(), _good()])
    rec = _refine_phase(_phase(), FakeExecutor(returncode=1), critic,
                        policy_for("autonomous"), _ctx(max_cycles=3))
    assert rec.get("failure_class") != "health_gate"
    assert critic.calls[0]["physical_findings"] is None


# ── _run_once_phase: the combine path ────────────────────────────────────────

def test_combine_healthy_passes(monkeypatch):
    _patch(monkeypatch, density=0.83)
    rec = _run_once_phase(_phase(), FakeExecutor(), RecordingCritic([_good()]), _ctx())
    assert rec["status"] == "success"


def test_combine_nonphysical_fails_health_gate_without_retry(monkeypatch):
    _patch(monkeypatch, density=0.003,
           perturb=lambda input_files, attempt: {"in.lmp": "x"})
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, RecordingCritic([_good()]),
                          _ctx(max_health_retries=2))
    assert rec["status"] == "failed" and rec["failure_class"] == "health_gate"
    assert ex.calls == 1                      # combine never retries


# ── campaign surfacing ───────────────────────────────────────────────────────

def test_campaign_surfaces_health_gate(monkeypatch):
    _patch(monkeypatch, density=0.003)       # every run is non-physical
    critic = RecordingCritic([_good()])       # critic fails open
    stage = Stage(name="prod", phases=[_phase()])
    out = run_campaign([stage], FakeExecutor(), critic, policy_for("autonomous"),
                       _ctx(max_cycles=1), pre_run_verdict={"verdict": "good"})
    assert out["status"] != "success"
    assert out["failure_class"] == "health_gate"
    assert out["health_violations"]


def test_campaign_surfaces_health_checked_false(monkeypatch):
    # The gate couldn't read its observable: the run passes, but the campaign
    # result flags that the pass was unvouched.
    _patch(monkeypatch, density=None)
    stage = Stage(name="prod", phases=[_phase()])
    out = run_campaign([stage], FakeExecutor(), RecordingCritic([_good()]),
                       policy_for("autonomous"), _ctx(max_cycles=1),
                       pre_run_verdict={"verdict": "good"})
    assert out["status"] == "success"
    assert out["health_checked"] is False
