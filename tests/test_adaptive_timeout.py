"""Offline tests for adaptive timeout escalation (slow != broken).

A subprocess that times out is rarely broken code — usually the fit just
needs longer. `stage_and_run_adaptive` retries the SAME script with doubled
timeouts (2 escalations, 1800 s hard cap) before the correction LLM ever
sees a "timed out" error; genuine script errors pass straight through to
the correction loop. Plus: the correction prompts carry a NARROW timeout
exception (computational strategy only — model/window/data untouchable).

  UNSAFE_EXECUTION_OK=true python tests/test_adaptive_timeout.py
"""
import io
import logging
import tempfile
import time

from scilink.executors import ScriptExecutor
from scilink.agents.exp_agents._locked_exec import (
    stage_and_run_adaptive, TIMEOUT_ESCALATIONS, TIMEOUT_HARD_CAP_S,
)

import numpy as np

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


SLOW_OK = """
import time, numpy as np
time.sleep(3)
np.load("data.npy")
open("visualization.png", "wb").write(b"png")
print("done")
"""

BROKEN = """
raise RuntimeError("genuinely broken script")
"""

NEVER_ENDS = """
import time
time.sleep(60)
"""


class _CountingExecutor(ScriptExecutor):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def execute_script(self, script_content, working_dir=None, timeout=None):
        self.calls.append(timeout if timeout is not None else self.timeout)
        return super().execute_script(script_content, working_dir, timeout)


class _FakeTimeoutExecutor:
    """Always times out; records requested timeouts (no real sleeping)."""

    def __init__(self, timeout):
        self.timeout = timeout
        self.calls = []

    def execute_script(self, script_content, working_dir=None, timeout=None):
        t = timeout if timeout is not None else self.timeout
        self.calls.append(t)
        return {"status": "error",
                "message": f"Script execution timed out after {t} seconds."}


def _capture_log():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.WARNING)
    logging.getLogger().addHandler(h)
    return buf, h


def main():
    data = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)])

    # 1) Slow-but-correct script: escalates 1s -> 2s -> 4s, succeeds, no
    #    correction ever involved.
    ex = _CountingExecutor(timeout=1)
    buf, h = _capture_log()
    t0 = time.monotonic()
    run = stage_and_run_adaptive(ex, SLOW_OK, data, tempfile.mkdtemp())
    dt = time.monotonic() - t0
    logging.getLogger().removeHandler(h)
    log = buf.getvalue()
    print("1) slow-but-correct script (sleep 3, timeout 1):")
    check("succeeds after escalation", run["status"] == "success")
    check("timeout ladder 1 -> 2 -> 4", ex.calls == [1, 2, 4])
    check("escalation logged",
          "retrying same script with 2s" in log
          and "retrying same script with 4s" in log)
    check("bounded wall clock (~3+overhead, not 5x600)", dt < 30)

    # 2) Genuinely broken script: returned as-is on the FIRST call — the
    #    correction loop's job, no timeout escalation.
    ex = _CountingExecutor(timeout=5)
    run = stage_and_run_adaptive(ex, BROKEN, data, tempfile.mkdtemp())
    print("2) genuinely broken script:")
    check("error passed through", run["status"] == "error"
          and "genuinely broken" in run["exec"].get("message", ""))
    check("no escalation for non-timeout errors", ex.calls == [5])

    # 3) Never-ending script: escalation budget exhausted, final result is
    #    the standard timed-out error (what the correction LLM then sees).
    ex = _CountingExecutor(timeout=1)
    buf, h = _capture_log()
    run = stage_and_run_adaptive(ex, NEVER_ENDS, data, tempfile.mkdtemp())
    logging.getLogger().removeHandler(h)
    print("3) never-ending script:")
    check("returns timed-out error after budget",
          run["status"] == "error"
          and "timed out" in run["exec"]["message"])
    check("exactly 1 + ESCALATIONS calls",
          len(ex.calls) == 1 + TIMEOUT_ESCALATIONS)
    check("budget-exhausted handoff logged",
          "escalation budget exhausted" in buf.getvalue())

    # 4) Hard cap: base 1500 -> 1800 (cap) -> stop (next would not grow).
    fake = _FakeTimeoutExecutor(timeout=1500)
    run = stage_and_run_adaptive(fake, "x", data, tempfile.mkdtemp())
    print("4) hard cap:")
    check("caps at TIMEOUT_HARD_CAP_S then stops",
          fake.calls == [1500, TIMEOUT_HARD_CAP_S])

    # 5) executor timeout kwarg: default falls back to construction value.
    ex = ScriptExecutor(timeout=60)
    out = ex.execute_script("print('hi')")
    check("5) default timeout path still works", out["status"] == "success")
    out = ex.execute_script("import time; time.sleep(3)", timeout=1)
    check("per-call override honored in message",
          out["status"] == "error" and "after 1 seconds" in out["message"])

    # 6) Correction prompts: narrow timeout exception present, locked-model
    #    rule intact.
    from scilink.agents.exp_agents.instruct import (
        FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
    )
    print("6) correction-prompt carve-outs:")
    for name, tpl, keep in (
        ("curve", FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
         "never narrow the window or truncate the data"),
        ("image", IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
         "Do NOT change the analysis pipeline"),
    ):
        check(f"{name}: locked rule intact", keep in tpl)
        check(f"{name}: timeout exception present",
              "timeout errors ONLY" in tpl and "too slow, not wrong" in tpl)
    check("curve: exception preserves model/window/data",
          "same model, same parameters, same fit domain/window, and ALL of "
          "the data" in FITTING_SCRIPT_CORRECTION_INSTRUCTIONS)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"ADAPTIVE TIMEOUT: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
