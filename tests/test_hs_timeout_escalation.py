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
    # the run's remaining time clamps the RETRIES: 1 s, then only what is left (3 s), and no retry past it
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=1,
                                               timed_out=lambda o: True, remaining_s=lambda: 3)
    assert calls == [1, 2, 3] and used == 3
    # ...but never the FIRST limit: the run budget is soft (a script that started finishes), so a 10 s
    # script with 4 s left runs its 10 s — as on main — and is not retried
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=10,
                                               timed_out=lambda o: True, remaining_s=4)
    assert calls == [10] and used == 10
    # no escalation at all when asked (a strict replay fails fast)
    calls.clear()
    out, used = _locked_exec.escalate_timeouts(lambda t: calls.append(t) or "slow", base_timeout=1,
                                               timed_out=lambda o: True, escalations=0)
    assert calls == [1] and used == 1


def _dynamic(tmp_path, monkeypatch, records, *, seconds=1, strict=False, deadline_s=None, model=None):
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
    ctrl = hc.RunDynamicAnalysisController(model if model is not None else hs._ExplodingModel(), hs.LOGGER,
                                           generation_config=None, safety_settings=None, parse_fn=lambda r: ({}, None),
                                           executor_timeout=seconds)
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


# a handle the exec'd script can hang its arrays on, so the test can see them go
# (on builtins: pytest may import this file under another module name)
_HELD = []


HEAVY_ERROR_SCRIPT = '''
def analyze_feature(data, axis):
    import numpy as np
    import weakref
    import builtins
    _HELD = builtins._HS_HELD                 # one list whatever module name pytest imported the test under
    class Holder:
        pass
    big = np.zeros((2_000_000,), dtype=np.float64)       # 16 MB the script's frame holds
    h = Holder()
    h.big = big
    _HELD.append(weakref.ref(h))
    raise ValueError("the script's own error")
'''

# The same arrays, but held only by a frame reachable through the error's
# __cause__ / __context__: a script that raises while handling another error.
_CHAINED_TEMPLATE = '''
def analyze_feature(data, axis):
    import numpy as np
    import weakref
    import builtins
    _HELD = builtins._HS_HELD
    class Holder:
        pass
    def inner():
        h = Holder()
        h.big = np.zeros((2_000_000,), dtype=np.float64)
        _HELD.append(weakref.ref(h))
        raise KeyError("missing band")
    try:
        inner()
    except KeyError as e:
        raise ValueError("the script's own error"){link}
'''
CAUSE_ERROR_SCRIPT = _CHAINED_TEMPLATE.format(link=" from e")      # __cause__
CONTEXT_ERROR_SCRIPT = _CHAINED_TEMPLATE.format(link="")           # __context__


def test_the_escalation_happens_in_a_worker_thread_too(tmp_path, monkeypatch):
    """#715 review, item 1: off the main thread ExecutionTimeout injected a
    bare TimeoutError, which the policy did not recognise — so the CLI shell,
    the web runner, the MCP server, fan-out and the swarm never escalated.
    One subclass (SandboxTimeout) on both paths, tested by type."""
    import threading
    from scilink.executors import ExecutionTimeout, SandboxTimeout
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import _sandbox_timeout
    # the main-thread handler and the injected class are both the sandbox's
    assert issubclass(SandboxTimeout, TimeoutError) and _sandbox_timeout(SandboxTimeout()) and _sandbox_timeout(SandboxTimeout("x"))
    assert not _sandbox_timeout(TimeoutError("Code execution timed out after 1s")) and not _sandbox_timeout(TimeoutError())
    out = {}

    def worker():
        try:
            with ExecutionTimeout(seconds=1) as et:
                import time
                time.sleep(3)
        except BaseException as exc:  # noqa: BLE001
            out["exc"], out["fired"] = exc, et.fired
    t = threading.Thread(target=worker)
    t.start()
    t.join(10)
    assert isinstance(out.get("exc"), SandboxTimeout) and out["fired"] is True
    # the real controller in a worker thread: the slow script times out at 1 s and succeeds at 2 s
    result = {}

    def run():
        result["state"] = _dynamic(tmp_path / "w", monkeypatch, hs._records(SLOW_SCRIPT), seconds=1)
    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    [rec] = result["state"]["dynamic_analysis_records"]
    assert rec["task_success"] and rec["timeout_used_s"] == 2


def _error_releases_before_repair(tmp_path, monkeypatch, script=HEAVY_ERROR_SCRIPT):
    """Runs the heavy-error script and reports what the repair call saw:
    (the script's holder already gone, the number of repair calls)."""
    import builtins, gc
    builtins._HS_HELD = _HELD
    _HELD.clear()
    seen = []

    class ProbeModel:
        """The repair call: at THIS moment the failed attempt's arrays must be gone —
        with the cyclic GC OFF, so only reference counting may have freed them."""

        def generate_content(self, *a, **k):
            seen.append(bool(_HELD) and all(ref() is None for ref in _HELD))
            raise AssertionError("no repair here")                     # the attempt then fails into the ladder
    gc.disable()
    try:
        state = _dynamic(tmp_path, monkeypatch, hs._records(script), seconds=5, model=ProbeModel())
    finally:
        gc.enable()
    [rec] = state["dynamic_analysis_records"]
    assert not rec["task_success"]
    return seen


def test_a_scripts_own_error_releases_its_arrays_before_the_repair(tmp_path, monkeypatch):
    """#715 review, item 2 then A: the attempt returned the exception as a
    value and the controller kept it — and after the direct references were
    dropped, the traceback's frame chain still reached the policy's frame
    whose ``out`` IS the exception, a cycle only the GC would break, so the
    failed 800 MB was alive during the repaired exec (+1306 MB on the main
    thread, +1765 in a worker, against main's +990). The scopes are cleared
    and the traceback's frames too (``clear_frames``: locals go, file, line
    and code stay for the prompt). Checked with the cyclic GC disabled, on
    the main thread and in a worker thread, and the repair is called once."""
    import threading
    seen = _error_releases_before_repair(tmp_path / "e", monkeypatch)
    assert seen == [True]                                               # released before the ONE repair call
    out = {}

    def run():
        out["seen"] = _error_releases_before_repair(tmp_path / "w", monkeypatch)
    t = threading.Thread(target=run)
    t.start()
    t.join(60)
    assert out.get("seen") == [True]


def test_a_chained_errors_arrays_are_released_too(tmp_path, monkeypatch):
    """A script that raises while handling another error: the arrays are held
    only by frames reachable through ``__cause__`` / ``__context__``, which
    clearing the outer traceback alone leaves alive. Cyclic GC off, main
    thread and a worker thread, both links."""
    import threading
    for i, script in enumerate((CAUSE_ERROR_SCRIPT, CONTEXT_ERROR_SCRIPT)):
        assert _error_releases_before_repair(tmp_path / f"m{i}", monkeypatch, script) == [True]
        out = {}

        def run():
            out["seen"] = _error_releases_before_repair(tmp_path / f"w{i}", monkeypatch, script)
        t = threading.Thread(target=run)
        t.start()
        t.join(60)
        assert out.get("seen") == [True]


def test_the_first_limit_is_never_clamped_and_the_loop_budget_bounds_retries(tmp_path, monkeypatch):
    """#715 review, item 3 and the 'also': the deadline clamped the FIRST limit
    too (a 1.6 s script with 3 s left and a 5 s limit failed at 1 s, where
    main succeeded); the retry clamp ignored the verification loop's own
    budget (1 → 2 → 4 s inside a 1.5 s budget)."""
    import time
    # item 3: 3 s left on the run, a 5 s limit, a 1.6 s script: it runs its 5 s and succeeds, as on main
    state = _dynamic(tmp_path / "f", monkeypatch, hs._records(SLOW_SCRIPT), seconds=5, deadline_s=3)
    [rec] = state["dynamic_analysis_records"]
    assert rec["task_success"] and rec["timeout_used_s"] == 5
    # the loop's budget: a 1 s limit, a 1.5 s qc_time_budget_s, the 1.6 s script — the retry is clamped to
    # what the budget leaves (~1 s), so it does not run at 2 s; failed, and quickly
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    orig_init = hc.RunDynamicAnalysisController.__init__

    def tight(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.qc_time_budget_s = 1.5
    monkeypatch.setattr(hc.RunDynamicAnalysisController, "__init__", tight)
    t0 = time.monotonic()
    state = _dynamic(tmp_path / "g", monkeypatch, hs._records(SLOW_SCRIPT), seconds=1)
    [rec] = state["dynamic_analysis_records"]
    assert not rec["task_success"] and time.monotonic() - t0 < 6


def test_a_fan_out_child_gets_no_run_deadline_from_the_branch_budget(tmp_path):
    """Pins main: a fan-out child's run_task carries no time_budget_s. The
    branch's wall-clock budget is the fan-out's hard cancel, not the child's
    run deadline (which counts human wait where the branch budget does not,
    and would expire a branch main completes); a stamp of the budget into the
    child was asked for and withdrawn in #715's review."""
    import json, os
    import numpy as np
    from scilink.agents.meta_agent import fanout as fo
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401  (worker-side lazy import)
    paths = [str(tmp_path / f"{n}.npy") for n in "AB"]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    calls = []

    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None, **kw):
                calls.append(dict(kw))
                return {"status": "success", "summary": "ok", "key_findings": ["finding"], "files_produced": []}
        return C()

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [], "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r", "join_axis": "T",
                "join_type": "shared_parameter_axis", "fanout_set": list(paths), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    orig_child, orig_llm = fo._make_ephemeral_analysis_child, fo._llm_json
    fo._make_ephemeral_analysis_child, fo._llm_json = fake_child, fake_llm
    try:
        ag = MetaOrchestratorAgent(base_dir=str(tmp_path / "s1"), api_key="sk-dummy", meta_mode=MetaMode.AUTONOMOUS)
        out = json.loads(ag._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p)} for p in paths],
                                        branch_time_budget_s=30))
        assert out.get("status") == "success" and len(calls) == 2
        assert all("time_budget_s" not in kw for kw in calls)             # the branch budget is the hard cancel only
        calls.clear()
        ag2 = MetaOrchestratorAgent(base_dir=str(tmp_path / "s2"), api_key="sk-dummy", meta_mode=MetaMode.AUTONOMOUS)
        out = json.loads(ag2._run_fanout([{"data_path": p, "task": f"Analyze {p}", "label": os.path.basename(p), "time_budget_s": 7}
                                          for p in paths], branch_time_budget_s=30))
        # main's normalisation (run_fanout → norm) carries no depth keys from the tool's branch dicts either:
        # the children's kwargs are the same with or without a branch time_budget_s — pinned as is
        assert out.get("status") == "success" and [kw.get("time_budget_s") for kw in calls] == [None, None]
    finally:
        fo._make_ephemeral_analysis_child, fo._llm_json = orig_child, orig_llm
