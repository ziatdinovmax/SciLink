"""#699: a hyperspectral script that is merely SLOW gets more time — the
same code, a doubled limit, up to the cap — before it is treated as broken,
through the policy the curve and image agents already run under
(`_locked_exec.escalate_timeouts`). The whole analyze() of a strict replay
(no model call) with a script that sleeps past the base limit: it times out
at the base, succeeds at twice the base, the record says what it needed, and
a replay of that record starts from it."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_hs_locked_replay as hs  # noqa: E402

from scilink.agents.exp_agents import _locked_exec  # noqa: E402

SLOW_SCRIPT = '''
def analyze_feature(data, axis):
    import time
    time.sleep(1.6)                       # slower than a 1 s limit, faster than 2 s
    m = data.mean(axis=2)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d"}
'''

OWN_TIMEOUT_SCRIPT = '''
def analyze_feature(data, axis):
    raise TimeoutError("the instrument did not answer")      # the script's own, not the sandbox's
'''


def test_escalate_timeouts_is_the_shared_policy():
    calls = []

    def attempt(t):
        calls.append(t)
        return "slow" if t < 4 else "done"
    out, used = _locked_exec.escalate_timeouts(attempt, base_timeout=1, timed_out=lambda o: o == "slow")
    assert (out, used, calls) == ("done", 4, [1, 2, 4])
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=1, timed_out=lambda o: True)
    assert out == "slow" and used == 4 and calls == [1, 2, 4]            # the budget: TIMEOUT_ESCALATIONS retries
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(attempt, base_timeout=1000, timed_out=lambda o: True)
    assert calls == [1000, 1800] and used == 1800                       # the hard cap ends it early
    assert _locked_exec.escalate_timeouts(lambda t: "ok", base_timeout=7, timed_out=lambda o: False) == ("ok", 7)
    # the run's deadline clamps every limit: 1 s, then only what is left (3 s), and no retry past it
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=1,
                                               timed_out=lambda o: True, remaining_s=lambda: 3)
    assert calls == [1, 2, 3] and used == 3
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=10,
                                               timed_out=lambda o: True, remaining_s=4)
    assert calls == [4] and used == 4                                   # nothing left to escalate into
    # no escalation at all when asked (a strict replay fails fast)
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=1,
                                               timed_out=lambda o: True, escalations=0)
    assert calls == [1] and used == 1


def _dynamic(tmp_path, monkeypatch, records, *, seconds=1, strict=False, deadline_s=None):
    """The real dynamic-analysis controller on a supplied-script replay (no
    model call), under a short execution limit."""
    import time
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    plan = hc.SelectRefinementTargetController(hs._ExplodingModel(), hs.LOGGER, generation_config=None,
                                               safety_settings=None, parse_fn=lambda r: ({}, None))
    state = plan.execute(hs._replay_state(tmp_path, records))
    if strict:
        state["_strict_replay"] = True
    if deadline_s is not None:
        state["_run_deadline"] = time.monotonic() + deadline_s
    ctrl = hc.RunDynamicAnalysisController(hs._ExplodingModel(), hs.LOGGER, generation_config=None,
                                           safety_settings=None, parse_fn=lambda r: ({}, None), executor_timeout=seconds)
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    return ctrl.execute(state)


def test_a_slow_hyperspectral_script_gets_more_time_and_a_replay_starts_from_it(tmp_path, monkeypatch):
    state = _dynamic(tmp_path / "a", monkeypatch, hs._records(SLOW_SCRIPT), seconds=1)
    [rec] = state["dynamic_analysis_records"]
    assert rec["task_success"] and rec["timeout_used_s"] == 2             # timed out at 1 s, ran at 2 s; no model call
    # a replay of this record starts from the limit it needed: one run, no escalation
    seen = []
    orig = _locked_exec.escalate_timeouts

    def spy(attempt, *, base_timeout, timed_out, logger=None, **kw):
        seen.append(base_timeout)
        return orig(attempt, base_timeout=base_timeout, timed_out=timed_out, logger=logger, **kw)
    monkeypatch.setattr(_locked_exec, "escalate_timeouts", spy)
    state2 = _dynamic(tmp_path / "b", monkeypatch, [dict(rec)], seconds=1)
    assert state2["dynamic_analysis_records"][0]["task_success"] and seen == [2]
    assert state2["dynamic_analysis_records"][0]["timeout_used_s"] == 2


def test_a_strict_replay_fails_fast_and_a_deadline_is_never_run_past(tmp_path, monkeypatch):
    import time
    # strict (a live frame): the limit is the limit — one attempt, failed, no escalation
    t0 = time.monotonic()
    state = _dynamic(tmp_path / "s", monkeypatch, hs._records(SLOW_SCRIPT), seconds=1, strict=True)
    [rec] = state["dynamic_analysis_records"]
    assert not rec["task_success"] and "timeout_used_s" not in rec and time.monotonic() - t0 < 4
    # a deadline with 1.5 s left: the retry is clamped to it (1 s, then ~1 s), the script still too slow → failed,
    # and nothing ran for the 2 s the policy would otherwise have granted
    t0 = time.monotonic()
    state = _dynamic(tmp_path / "d", monkeypatch, hs._records(SLOW_SCRIPT), seconds=1, deadline_s=1.5)
    [rec] = state["dynamic_analysis_records"]
    assert not rec["task_success"] and time.monotonic() - t0 < 5
    # a script's OWN TimeoutError is an error to repair/ladder, not "merely slow": no escalation, no timeout_used_s
    seen = []
    orig = _locked_exec.escalate_timeouts

    def spy(attempt, *, base_timeout, timed_out, logger=None, **kw):
        out = orig(attempt, base_timeout=base_timeout, timed_out=timed_out, logger=logger, **kw)
        seen.append(out[1])
        return out
    monkeypatch.setattr(_locked_exec, "escalate_timeouts", spy)
    state = _dynamic(tmp_path / "o", monkeypatch, hs._records(OWN_TIMEOUT_SCRIPT), seconds=5, strict=True)
    [rec] = state["dynamic_analysis_records"]
    assert not rec["task_success"] and seen == [5] and "timeout_used_s" not in rec
