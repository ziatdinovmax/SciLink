"""Integration tests for the health gate inside the refinement phase runners.

Drives ``_run_once_phase`` / ``_refine_phase`` with a fake executor, injecting
the per-engine ``health:`` specs and hooks by monkeypatch (no real skill, no
engine). Verifies: a healthy run passes; a non-physical run fails fast when the
engine can't perturb a retry; a non-physical run is retried-then-failed when a
``perturb_for_retry`` hook exists; and an engine with no ``health:`` block is
ungated.
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
    _refine_phase,
    _run_once_phase,
    policy_for,
)

DENSITY_SPEC = [{"observable": "density", "min": 0.02, "max": 30.0}]


class FakeExecutor(Executor):
    def __init__(self):
        self.calls = 0

    def run(self, input_files, run_command, run_dir):
        self.calls += 1
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


class StubCritic:
    """A critic that always accepts — isolates the health gate from the critic."""

    def assess(self, output_dir, research_goal, skill=None, domain=None,
               input_files=None, **kw):
        return {"run_status": "succeeded", "verdict": "good",
                "suggested_fixes": None}


def _ctx(**kw):
    return RefinementContext(research_goal="g", skill="lammps",
                             domain="molecular_dynamics", **kw)


def _phase():
    return Phase(name="production", input_files={"in.lmp": "velocity all create 300 12345"},
                 run_command="lmp -in in.lmp", run_dir="/tmp/run")


def _patch(monkeypatch, *, density, specs=DENSITY_SPEC, perturb=None):
    """Wire the health specs + a reader (and optional perturb) into refinement."""
    monkeypatch.setattr(refinement, "_health_specs",
                        lambda skill, domain: specs)

    def _reader(output_dir, observable):
        return density if observable == "density" else None

    def _resolve(skill, domain, fn_name):
        if fn_name == "read_health_observable":
            return _reader
        if fn_name == "perturb_for_retry":
            return perturb
        return None

    monkeypatch.setattr(refinement, "_resolve_skill_callable", _resolve)


# ── _run_once_phase: the replica-ensemble path (refine_members=False) ─────────

def test_healthy_run_passes(monkeypatch):
    _patch(monkeypatch, density=0.83)
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx())
    assert rec["status"] == "success"
    assert ex.calls == 1


def test_nonphysical_run_fails_fast_without_perturb(monkeypatch):
    # The box-explosion case with no perturb hook: re-running the identical
    # pinned-seed deck can't help, so the gate fails on the first run.
    _patch(monkeypatch, density=0.003, perturb=None)
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx(max_health_retries=2))
    assert rec["status"] == "failed"
    assert rec["failure_class"] == "health_gate"
    assert any("density" in r for r in rec["health_violations"])
    assert ex.calls == 1  # no wasted re-runs


def test_nonphysical_run_retries_then_fails_with_perturb(monkeypatch):
    seeds = []

    def perturb(input_files, attempt):
        seeds.append(attempt)
        return {"in.lmp": f"velocity all create 300 {1000 + attempt}"}

    _patch(monkeypatch, density=0.003, perturb=perturb)
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx(max_health_retries=2))
    assert rec["status"] == "failed"
    assert rec["failure_class"] == "health_gate"
    assert ex.calls == 3          # initial + 2 retries
    assert seeds == [1, 2]        # perturb called once per retry


def test_retry_recovers_when_a_reseed_lands_in_band(monkeypatch):
    # Reader reports a bad density until the second attempt, then a good one.
    results = iter([0.003, 0.003, 0.84])

    def reader(output_dir, observable):
        return next(results) if observable == "density" else None

    def resolve(skill, domain, fn_name):
        if fn_name == "read_health_observable":
            return reader
        if fn_name == "perturb_for_retry":
            return lambda input_files, attempt: {"in.lmp": "reseeded"}
        return None

    monkeypatch.setattr(refinement, "_health_specs", lambda s, d: DENSITY_SPEC)
    monkeypatch.setattr(refinement, "_resolve_skill_callable", resolve)

    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx(max_health_retries=3))
    assert rec["status"] == "success"
    assert ex.calls == 3


def test_combine_stage_does_not_retry(monkeypatch):
    # allow_retry=False (a combine stage) never re-runs even with a perturb hook.
    _patch(monkeypatch, density=0.003,
           perturb=lambda input_files, attempt: {"in.lmp": "x"})
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx(max_health_retries=2),
                          allow_retry=False)
    assert rec["status"] == "failed"
    assert ex.calls == 1


# ── gate disabled for engines with no health: block ──────────────────────────

def test_no_health_block_is_ungated(monkeypatch):
    _patch(monkeypatch, density=0.003, specs=[])  # empty specs == no block
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx())
    assert rec["status"] == "success"  # ungated: the stub critic accepts it
    assert ex.calls == 1


def test_no_reader_hook_is_ungated(monkeypatch):
    monkeypatch.setattr(refinement, "_health_specs", lambda s, d: DENSITY_SPEC)
    monkeypatch.setattr(refinement, "_resolve_skill_callable",
                        lambda skill, domain, fn_name: None)  # no hooks at all
    ex = FakeExecutor()
    rec = _run_once_phase(_phase(), ex, StubCritic(), _ctx())
    assert rec["status"] == "success"
    assert ex.calls == 1


# ── _refine_phase path (sequential steps / refined members) ──────────────────

def test_refine_phase_fails_on_nonphysical_run(monkeypatch):
    _patch(monkeypatch, density=0.003, perturb=None)
    ex = FakeExecutor()
    rec = _refine_phase(_phase(), ex, StubCritic(), policy_for("autonomous"),
                        _ctx(max_health_retries=1))
    assert rec["status"] == "failed"
    assert rec["failure_class"] == "health_gate"
    assert ex.calls == 1


def test_refine_phase_healthy_run_proceeds(monkeypatch):
    _patch(monkeypatch, density=0.83)
    ex = FakeExecutor()
    rec = _refine_phase(_phase(), ex, StubCritic(), policy_for("autonomous"),
                        _ctx())
    assert rec["status"] == "success"
