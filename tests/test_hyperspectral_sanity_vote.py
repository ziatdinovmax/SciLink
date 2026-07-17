"""Offline test for the physics-sanity-check VOTING logic.

A single LLM judgment is too noisy near the decision boundary (it false-rejects
a correct result ~half the time in validation) but never false-PASSES a gross
error. So _sanity_check_result votes SANITY_VOTES independent judgments and
rejects only when >= SANITY_REJECT_MAJORITY agree, short-circuiting once
decided. These tests stub the single judgment to make the logic deterministic.

  conda run -n scilink python -m pytest tests/test_hyperspectral_sanity_vote.py -q
"""
import logging
import types

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    RunDynamicAnalysisController as C,
)


def _ctrl(verdicts):
    """A controller whose single-judgment returns the given (valid, crit) seq,
    counting how many times it was called."""
    ctrl = C.__new__(C)
    ctrl.logger = logging.getLogger("t")
    seq = list(verdicts)
    calls = {"n": 0}

    def fake_one(self, *a, **k):
        calls["n"] += 1
        return seq.pop(0)
    ctrl._sanity_check_one = types.MethodType(fake_one, ctrl)
    ctrl._calls = calls
    return ctrl


ARGS = (b"dash", "code", None, {}, "obj", "feat", "summary")


def test_two_rejects_reject_and_short_circuit():
    # invalid, invalid -> reject after 2 calls (3rd not needed).
    ctrl = _ctrl([(False, "x"), (False, "y"), (True, "")])
    ok, crit = ctrl._sanity_check_result(*ARGS)
    assert ok is False and ctrl._calls["n"] == 2


def test_single_reject_accepts_majority_not_reached():
    # valid, invalid, valid -> only 1 reject < majority 2 -> accept.
    ctrl = _ctrl([(True, ""), (False, "x"), (True, "")])
    ok, _ = ctrl._sanity_check_result(*ARGS)
    assert ok is True


def test_two_accepts_short_circuit_before_third():
    # valid, valid -> can't reach 2 rejects in remaining -> accept after 2 calls.
    ctrl = _ctrl([(True, ""), (True, ""), (False, "x")])
    ok, _ = ctrl._sanity_check_result(*ARGS)
    assert ok is True and ctrl._calls["n"] == 2


def test_fail_open_on_judgment_error():
    ctrl = C.__new__(C)
    ctrl.logger = logging.getLogger("t")

    def boom(self, *a, **k):
        raise RuntimeError("model down")
    ctrl._sanity_check_one = types.MethodType(boom, ctrl)
    ok, crit = ctrl._sanity_check_result(*ARGS)
    assert ok is True and crit == ""  # fail-open, does not block


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
