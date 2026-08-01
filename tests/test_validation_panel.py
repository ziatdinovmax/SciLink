"""Multi-observable validation panel: validate each observable against its
reference and scope the prediction to what passed.

Pure composition — the per-observable judge is injected, so no LLM/engine.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.reference_validation import run_validation_panel  # noqa: E402


def _judge(observations, system_description):
    """Fake judge: each observation's verdict comes from a '_verdict' hint."""
    return {"per_observable": [
        {"observable": o["observable"], "verdict": o.get("_verdict", "good")}
        for o in observations
    ]}


def test_all_pass_warrants_prediction():
    obs = [{"observable": "density", "_verdict": "good"},
           {"observable": "viscosity", "_verdict": "good"}]
    r = run_validation_panel(obs, "T1", "Zn electrolyte", judge_fn=_judge)
    assert r["prediction_warranted"] is True
    assert set(r["passed"]) == {"density", "viscosity"}
    assert r["failed"] == []
    assert "backed by validation against" in r["confidence"] and "T1" in r["confidence"]


def test_failure_blocks_and_scopes_down():
    obs = [{"observable": "density", "_verdict": "good"},
           {"observable": "viscosity", "_verdict": "poor"}]
    r = run_validation_panel(obs, "T1", "x", judge_fn=_judge)
    assert r["prediction_warranted"] is False
    assert r["failed"] == ["viscosity"] and r["passed"] == ["density"]
    assert "NOT fully backed" in r["confidence"]


def test_consistent_false_counts_as_failure():
    def judge(o, s):
        return {"per_observable": [{"observable": "density", "consistent": False}]}
    r = run_validation_panel([{"observable": "density"}], "T1", "x", judge_fn=judge)
    assert r["failed"] == ["density"] and r["prediction_warranted"] is False


def test_scope_fn_injection_overrides_default():
    def judge(o, s):
        return {"per_observable": [{"observable": "density", "verdict": "good"}]}

    def scope(target, passed, failed, sysdesc):
        return f"CUSTOM {target} {passed} {failed}"

    r = run_validation_panel([{"observable": "density"}], "T1", "x",
                             judge_fn=judge, scope_fn=scope)
    assert r["confidence"] == "CUSTOM T1 ['density'] []"


def test_unrated_is_not_counted_as_passed():
    # A judge entry with no verdict/consistent is unrated — it must not land in
    # `passed` or be cited as validation, and it blocks warranting.
    def judge(o, s):
        return {"per_observable": [{"observable": "density"}]}
    r = run_validation_panel([{"observable": "density"}], "T1", "x", judge_fn=judge)
    assert r["passed"] == [] and r["unrated"] == ["density"]
    assert r["prediction_warranted"] is False
    assert "could not rate density" in r["confidence"]


def test_per_measurement_key_is_accepted():
    def judge(o, s):
        return {"per_measurement": [{"component": "water", "verdict": "good"}]}
    r = run_validation_panel([{"observable": "density"}], "T1", "x", judge_fn=judge)
    assert r["passed"] == ["water"] and r["prediction_warranted"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
