"""log_tail streaming in scilink_job_status — offline tests."""

import json
import logging
import threading
import time

import pytest

pytest.importorskip("mcp")

from scilink.mcp_server import (  # noqa: E402
    _TeeIO,
    _execute_run_task_captured,
    _handle_job_status,
    _new_job_lines,
)


def _state():
    return {"pending": {}, "jobs": {}, "job_counter": 0,
            "config": {"analysis_mode": "autonomous", "hitl_timeout_s": 5}}


def test_teeio_splits_lines_and_keeps_stream():
    lines = _new_job_lines()
    t = _TeeIO(lines)
    t.write("hello ")
    t.write("world\npartial")
    assert list(lines) == ["hello world"]
    t.write(" done\n")
    assert list(lines) == ["hello world", "partial done"]
    assert t.getvalue() == "hello world\npartial done\n"


def test_run_task_narration_lands_in_buffer_thread_scoped():
    lines = _new_job_lines()
    other_thread_noise = []

    class OrchStub:
        def run_task(self, prompt, autonomy=None):
            logging.getLogger("SomeAgent").info("Fitting spectrum ...")
            logging.getLogger("SomeAgent").info("Fit approved (R2=0.99)")
            print("plan accepted")
            return {"status": "success", "summary": "ok",
                    "files_produced": [], "key_findings": [],
                    "suggested_followups": [], "warnings": []}

    def noise():
        # Concurrent logging from an unrelated thread must NOT leak in.
        for _ in range(20):
            logging.getLogger("Other").info("unrelated narration")
            other_thread_noise.append(1)
            time.sleep(0.005)

    prev_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.INFO)
    try:
        t = threading.Thread(target=noise)
        t.start()
        out = _execute_run_task_captured(OrchStub(), "task", "autonomous",
                                         _state(), "job_x", lines)
        t.join()
    finally:
        logging.getLogger().setLevel(prev_level)

    assert json.loads(out)["status"] == "success"
    joined = "\n".join(lines)
    assert "Fitting spectrum" in joined
    assert "Fit approved" in joined
    assert "plan accepted" in joined
    assert "unrelated narration" not in joined


def test_job_status_returns_bounded_log_tail():
    class NeverDone:
        def done(self):
            return False

    state = _state()
    lines = _new_job_lines()
    for i in range(50):
        lines.append(f"line {i}")
    state["jobs"]["j1"] = {"future": NeverDone(), "tool": "t",
                           "started_at": "ts", "status": "running",
                           "result": None, "log_lines": lines}

    (content,) = _handle_job_status(state, {"job_id": "j1"})
    payload = json.loads(content.text)
    tail = payload["log_tail"].splitlines()
    assert len(tail) == 25 and tail[-1] == "line 49"  # default 25, newest last

    (content,) = _handle_job_status(state, {"job_id": "j1", "tail_lines": 5})
    assert json.loads(content.text)["log_tail"].splitlines() == [
        f"line {i}" for i in range(45, 50)]

    (content,) = _handle_job_status(state, {"job_id": "j1", "tail_lines": 0})
    assert "log_tail" not in json.loads(content.text)

    (content,) = _handle_job_status(state, {"job_id": "j1",
                                            "tail_lines": 10_000})
    assert len(json.loads(content.text)["log_tail"].splitlines()) == 50


def test_job_status_backcompat_without_buffer():
    class Done:
        def done(self):
            return True

        def result(self):
            return "the result"

    state = _state()
    state["jobs"]["j2"] = {"future": Done(), "tool": "t", "started_at": "ts",
                           "status": "running", "result": None}
    (content,) = _handle_job_status(state, {"job_id": "j2"})
    payload = json.loads(content.text)
    assert payload["status"] == "completed"
    assert "log_tail" not in payload
