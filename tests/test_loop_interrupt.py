"""A Stop ends the frame that is running, not only the ones after it.

The Live tab's Stop set an event the loop checks between frames, so a replay
already running went on to its end or its timeout (up to the executor's 600 s)
before the run stopped. ``MeasurementLoop.interrupt()`` ends the running
frame's script from any thread; the frame is logged as ``interrupted``, not
as a failed fit, so it does not push the loop toward a rebuild.
"""

import threading
import time
from pathlib import Path

import pytest

from scilink import executors as ex
from scilink.live import MeasurementLoop

from test_measurement_loop import make_anchor


class SlowReplayAgent:
    """Runs its 'replay' as a real script through a ScriptExecutor (which the
    loop swaps for its warm interpreter when warm_replay is on)."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.executor = ex.ScriptExecutor(timeout=120)

    def analyze(self, data, **kw):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        r = self.executor.execute_script("import time\ntime.sleep(60)\n", working_dir=self.output_dir)
        return {"status": "error" if r["status"] != "success" else "success",
                "output_directory": self.output_dir}


@pytest.mark.parametrize("warm", [True, False], ids=["warm", "cold"])
def test_interrupt_ends_the_running_frame_quickly(tmp_path, warm):
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=SlowReplayAgent, warm_replay=warm)
    loop.setup(anchor=str(make_anchor(tmp_path)))
    out = {}
    t = threading.Thread(target=lambda: out.update(rec=loop.step(str(tmp_path / "frame.csv"))))
    t.start()
    deadline = time.time() + 30
    while time.time() < deadline and not ex._active_subprocesses:
        time.sleep(0.05)
    assert ex._active_subprocesses, "the replay never started"
    time.sleep(0.3)
    t0 = time.time()
    assert loop.interrupt() is True
    t.join(timeout=20)
    try:
        assert not t.is_alive() and time.time() - t0 < 10
        rec = out["rec"]
        assert rec["flags"] == ["interrupted"]
        assert loop._consecutive_breaches == 0            # not counted against the recipe
    finally:
        loop.close()


def test_interrupt_with_no_frame_running_is_a_no_op(tmp_path):
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=SlowReplayAgent, warm_replay=False)
    assert loop.interrupt() is False
