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
    import numpy as np, time
    time.sleep(1.6)                       # slower than a 1 s limit, faster than 2 s
    w = data - data.min(axis=2, keepdims=True)
    pos = (w * axis).sum(axis=2) / np.maximum(w.sum(axis=2), 1e-12)
    return {"maps": {"Peak_Position": pos}, "units": "nm", "description": "centroid"}
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


def _strict_agent_with_limit(tmp_path, name, seconds):
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    ag = HyperspectralAnalysisAgent(api_key="sk-dummy", model_name="claude-opus-4-6",
                                    output_dir=str(tmp_path / name), enable_human_feedback=False,
                                    executor_timeout=seconds)
    ag.model = hs._ExplodingModel()
    for stage in list(getattr(ag, "pipeline", [])) + list(getattr(ag, "synthesis_pipeline", [])):
        if hasattr(stage, "model"):
            stage.model = hs._ExplodingModel()
    return ag


def test_a_slow_hyperspectral_script_gets_more_time_and_a_replay_starts_from_it(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    np.save(tmp_path / "cube.npy", hs._peak_cube(center=660.0, seed=1))
    anchor = hs._anchor_dir(tmp_path, SLOW_SCRIPT)
    ag = _strict_agent_with_limit(tmp_path, "frame", 1)
    res = ag.analyze(str(tmp_path / "cube.npy"), system_info=dict(hs.AXIS_OK),
                     prior_analysis_paths=[str(anchor)], reuse_locked_script=True, strict_replay=True,
                     replay_reference={"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0, "coverage": 1.0}})
    assert res["status"] == "success", res.get("error")                 # on main: a TimeoutError fails the frame
    [rec] = res["dynamic_analysis_records"]
    assert rec["task_success"] and rec["timeout_used_s"] == 2             # timed out at 1 s, ran at 2 s
    assert (res.get("stage_timings") or {}).get("llm_calls", 0) == 0     # no model: the policy is mechanical
    # a replay of this record starts from the limit it needed: one run, no escalation
    (tmp_path / "donor").mkdir()
    (tmp_path / "donor" / "dynamic_analysis_records.json").write_text(json.dumps([rec]))
    seen = []
    orig = _locked_exec.escalate_timeouts

    def spy(attempt, *, base_timeout, timed_out, logger=None):
        seen.append(base_timeout)
        return orig(attempt, base_timeout=base_timeout, timed_out=timed_out, logger=logger)
    monkeypatch.setattr(_locked_exec, "escalate_timeouts", spy)
    ag2 = _strict_agent_with_limit(tmp_path, "frame2", 1)
    res2 = ag2.analyze(str(tmp_path / "cube.npy"), system_info=dict(hs.AXIS_OK),
                       prior_analysis_paths=[str(tmp_path / "donor")], reuse_locked_script=True, strict_replay=True,
                       replay_reference={"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0, "coverage": 1.0}})
    assert res2["status"] == "success" and seen == [2]
    assert res2["dynamic_analysis_records"][0]["timeout_used_s"] == 2
