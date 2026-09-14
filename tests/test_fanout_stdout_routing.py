"""Fan-out branch threads must ride the coordinator's session route (#627).

Branch workers run whole child orchestrators and print their narration from
pool threads. Under the Streamlit-era global capture every thread's output
reached the UI and every thread's next print raised on a user Stop; the
web app's per-thread router forwards (and stops) only ATTRIBUTED threads,
so branch narration vanished from the browser and Stop no longer reached
the branches. Both are restored by attributing each branch to the
coordinator thread for its run.

  conda run -n scilink python -m pytest tests/test_fanout_stdout_routing.py -v
"""
import contextlib
import io
import json
import os
import sys
import threading
import time

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import numpy as np
import pytest

import scilink.agents.meta_agent.fanout as fo
from scilink.server import stdout_router as sr
from scilink.ui.output_capture import AgentStoppedError

MARKER = "BRANCH-NARRATION-0627"


def _install_fakes(behaviors):
    """Fake child + gate/fusion LLM, mirroring
    tests/test_fanout_branch_cancellation.py."""
    def fake_child(orch, base_dir):
        class C:
            def run_task(self, task, context=None, autonomy=None):
                import re
                m = re.search(r"PRIMARY dataset for THIS analysis: (\S+)", task)
                beh = behaviors.get(m.group(1) if m else "", "good")
                if beh == "chatty":
                    for _ in range(400):
                        print(f"{MARKER} still working...")
                        time.sleep(0.05)
                elif beh == "late":
                    # Silent 'model call' that outlives the coordinator's
                    # exit after a Stop, THEN chatty — the live shape.
                    time.sleep(2.5)
                    for _ in range(400):
                        print(f"{MARKER} late branch still working...")
                        time.sleep(0.05)
                else:
                    print(f"{MARKER} quick branch")
                return {"status": "success", "summary": f"{beh} ok",
                        "key_findings": ["finding"], "files_produced": []}
        return C()
    fo._make_ephemeral_analysis_child = fake_child

    def fake_llm(orch, prompt, extra_parts=None):
        if "SCRIPT CONTRACT" in prompt or "AUDITING a computed" in prompt:
            return {"verdict": "accept", "issues": [],
                    "refinement_instructions": "", "method": "qualitative",
                    "rationale": "r", "script": ""}
        if "complementary measurements of ONE system" in prompt:
            return {"detailed_analysis": "fused", "scientific_claims": []}
        return {"verdict": "complementary", "confidence": 0.9, "rationale": "r",
                "join_axis": "T", "join_type": "shared_parameter_axis",
                "fanout_set": list(behaviors), "redundant_clusters": [],
                "unrelated": [], "excluded_notes": ""}
    fo._llm_json = fake_llm


@contextlib.contextmanager
def routed_console():
    """Install the web app's router over a fake console.

    Entered INSIDE the test body, not from a fixture: pytest re-installs its
    own ``sys.stdout`` at the start of the call phase, which would silently
    unwind a router installed during setup."""
    console = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sr._RoutingStream(console)
    sys.stderr = sr._RoutingStream(io.StringIO())
    try:
        yield console
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@pytest.fixture
def meta(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    import scilink.agents.exp_agents.analysis_orchestrator  # noqa: F401
    ag = MetaOrchestratorAgent(base_dir=str(tmp_path), api_key="sk-dummy",
                               meta_mode=MetaMode.AUTONOMOUS)
    orig = (fo._make_ephemeral_analysis_child, fo._llm_json, fo._FANOUT_POLL_S,
            fo._FANOUT_HEARTBEAT_S)
    fo._FANOUT_POLL_S = 0.2
    fo._FANOUT_HEARTBEAT_S = 0.5   # the coordinator prints (and so raises) soon
    yield ag
    (fo._make_ephemeral_analysis_child, fo._llm_json, fo._FANOUT_POLL_S,
     fo._FANOUT_HEARTBEAT_S) = orig


def _datasets(tmp_path, names):
    paths = [str(tmp_path / f"{n}.npy") for n in names]
    for p in paths:
        np.save(p, np.zeros((8, 8)))
    return paths


def test_branch_prints_reach_the_coordinator_session(meta, tmp_path):
    from scilink.utils import log_context as lc
    workers_before = set(lc._WORKERS)
    A, B = _datasets(tmp_path, "AB")
    _install_fakes({A: "good", B: "good"})
    cap = sr.RoutedCapture(tag="")
    with routed_console() as console, cap:
        out = json.loads(meta._run_fanout(
            [{"data_path": p, "task": f"Analyze {p}",
              "label": os.path.basename(p)} for p in (A, B)]))
    assert out.get("status") == "success"
    captured = cap.getvalue()
    assert captured.count(MARKER) == 2, (
        "branch narration did not reach the session stream")
    assert "analysis branch finished" in captured   # coordinator lines too
    assert console.getvalue().count(MARKER) == 2    # console still tees
    # No branch stays attributed once the fan-out has returned (relative to
    # whatever an earlier test may have left running).
    from scilink.utils import log_context as lc
    assert set(lc._WORKERS) <= workers_before


@pytest.mark.parametrize("shape", ["chatty", "late"])
def test_user_stop_reaches_the_branch_threads(meta, tmp_path, shape):
    """Stop during a fan-out must abort the branch workers, not only the
    coordinator: a branch would otherwise run ~20 s on. 'chatty' raises
    while the coordinator (and its route) is still there; 'late' is the
    live shape — the branch is silent inside a model call until after the
    coordinator has exited, and must still be stopped by its next print."""
    from scilink.utils import log_context as lc
    workers_before, stopped_before = set(lc._WORKERS), set(lc._STOPPED_PARENTS)
    A, B = _datasets(tmp_path, "AB")
    _install_fakes({A: shape, B: shape})
    cap = sr.RoutedCapture(tag="")
    threading.Timer(0.8, cap.request_stop).start()
    t0 = time.monotonic()
    # The router stays installed for the whole test, as it does for the
    # server process: the stragglers' prints must still pass through it
    # after the coordinator's capture is gone.
    with routed_console():
        with cap:
            with pytest.raises(AgentStoppedError):
                meta._run_fanout(
                    [{"data_path": p, "task": f"Analyze {p}",
                      "label": os.path.basename(p)} for p in (A, B)])
        assert not sr._ROUTES   # coordinator turn is over; route removed
        # Both branch threads released their admission slots well before
        # the ~20 s they would have run to completion.
        deadline = time.monotonic() + 8.0
        while fo._mem_running and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not fo._mem_running, "a branch thread outlived the user Stop"
        assert time.monotonic() - t0 < 12.0
        if shape == "chatty":
            assert MARKER in cap.getvalue()
        deadline = time.monotonic() + 5.0
        while (set(lc._WORKERS) - workers_before
               or set(lc._STOPPED_PARENTS) - stopped_before) \
                and time.monotonic() < deadline:
            time.sleep(0.05)
        assert set(lc._WORKERS) <= workers_before
        assert set(lc._STOPPED_PARENTS) <= stopped_before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
