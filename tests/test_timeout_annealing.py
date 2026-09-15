"""Offline tests for timeout-aware annealing (last-resort model escalation).

Two additions, both pure additions on paths that previously ended in
failure: (1) qc_refit tells the (possibly hot) rewrite when trailing
attempts died on execution timeouts — the stall counter that drives
annealing is cause-blind; (2) the final correction of a fresh fit whose
failures are >= 2 consecutive timeouts gets a LAST-RESORT clause allowing
model restructure (window/data still untouchable), with the annealing
level raised to hot for that one call and restored afterwards.

  conda run -n scilink python tests/test_timeout_annealing.py
"""
import json
import logging
import tempfile

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
    _is_timeout_error,
    _trailing_timeout_failures,
)
from scilink.agents.exp_agents.instruct import (
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
)

results = {}
logging.disable(logging.CRITICAL)

TIMEOUT_MSG = "Script execution timed out after 1800 seconds."


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _attempt(success, error=None):
    return {"result": {"success": success, "error": error}}


class _StubModel:
    def __init__(self, raise_error=False):
        self.prompts = []
        self.raise_error = raise_error

    def generate_content(self, prompt, **kw):
        self.prompts.append(prompt)
        if self.raise_error:
            raise RuntimeError("stub LLM failure")
        return json.dumps({"diagnosis": "stub diagnosis",
                           "script": "import numpy as np\nprint('ok')\n"})


def _controller(model):
    return UnifiedSeriesProcessingController(
        model=model, logger=logging.getLogger("t"),
        generation_config=None, safety_settings=None, parse_fn=None,
        executor=None, script_instructions="",
        correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir=tempfile.mkdtemp(),
        plot_fn=None,
    )


def main():
    # 1) Timeout-error detection + trailing-failure counting.
    print("1) helpers:")
    check("timeout message detected", _is_timeout_error(TIMEOUT_MSG))
    check("other errors not timeout",
          not _is_timeout_error("NameError: x") and not _is_timeout_error(None))
    check("trailing count stops at non-timeout",
          _trailing_timeout_failures([
              _attempt(False, "NameError"),
              _attempt(False, TIMEOUT_MSG),
              _attempt(False, TIMEOUT_MSG)]) == 2)
    check("success breaks the trail",
          _trailing_timeout_failures([
              _attempt(False, TIMEOUT_MSG),
              _attempt(True),
              _attempt(False, TIMEOUT_MSG)]) == 1)
    check("empty/None attempts -> 0",
          _trailing_timeout_failures([]) == 0
          and _trailing_timeout_failures(None) == 0)

    # 2) Escalation trigger matrix — final correction of a FRESH fit with
    #    >= 2 consecutive timeouts, and nothing else.
    should = UnifiedSeriesProcessingController._should_escalate_timeout_model
    print("2) trigger matrix:")
    check("fires: fresh fit, final attempt, 2 timeouts",
          should(None, 5, 5, 2))
    check("fires on the penultimate attempt too (debug-chance design)",
          should(None, 4, 5, 2))
    check("locked-reuse fit never escalates",
          not should("locked script", 5, 5, 5))
    check("not on earlier attempts", not should(None, 3, 5, 3))
    check("not on a single timeout", not should(None, 5, 5, 1))
    check("not on non-timeout failures", not should(None, 5, 5, 0))

    # 3) Escalated correction: clause injected, hot annealing for that one
    #    call, state fully restored afterwards.
    model = _StubModel()
    ctrl = _controller(model)
    state = {"locked_fitting_config": {"analysis_approach": "a",
                                       "physical_model": "gaussian"},
             "_annealing_level": 0}
    script, diagnosis = ctrl._correct_script_with_timeout_escalation(
        state, "print('slow')", TIMEOUT_MSG)
    prompt = model.prompts[-1]
    print("3) escalated correction:")
    check("script returned", bool(script))
    check("LAST RESORT clause in prompt", "TIMEOUT ESCALATION" in prompt
          and "RESTRUCTURE the model" in prompt)
    check("clause declares precedence", "SUPERSEDES" in prompt)
    check("clause injected before the response footer",
          prompt.index("TIMEOUT ESCALATION") < prompt.index("**Response:**"))
    check("window/data still fenced in the clause",
          "full window, ALL of" in prompt)
    check("locked-model CRITICAL rule still present",
          "never narrow the window or truncate the data" in prompt)
    check("standard carve-out still present",
          "timeout errors ONLY" in prompt)
    check("annealing level restored", state.get("_annealing_level") == 0)
    check("escalation flag cleared",
          "_timeout_model_escalation" not in state)

    # 4) State restored even when the LLM call raises.
    ctrl = _controller(_StubModel(raise_error=True))
    state = {"locked_fitting_config": {}, "_annealing_level": 1}
    try:
        ctrl._correct_script_with_timeout_escalation(state, "x", TIMEOUT_MSG)
        raised = False
    except RuntimeError:
        raised = True
    print("4) exception safety:")
    check("stub error propagated", raised)
    check("level restored on exception", state.get("_annealing_level") == 1)
    check("flag cleared on exception",
          "_timeout_model_escalation" not in state)

    # 5) No-regression: a PLAIN correction never carries the clause.
    model = _StubModel()
    ctrl = _controller(model)
    state = {"locked_fitting_config": {}}
    ctrl._correct_script(state, "print('x')", "NameError: x")
    print("5) plain correction unchanged:")
    check("no escalation clause without the flag",
          "TIMEOUT ESCALATION" not in model.prompts[-1])
    check("locked rule present as always",
          "never narrow the window" in model.prompts[-1])

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"TIMEOUT ANNEALING: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
