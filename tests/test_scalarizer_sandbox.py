"""
Tests for the ScalarizerAgent sandbox gate.

The check was previously in ``__init__``, which made the agent
unconstructable in any non-TTY context without ``UNSAFE_EXECUTION_OK=true``
— blocking the meta-agent's planning child in Streamlit, scripts, and CI.

These tests pin down the new behavior:
  * Construction succeeds unconditionally (no sandbox check at __init__).
  * ``scalarize()`` gates on sandbox approval — declined → structured
    failure dict with all expected keys, approved → falls through to the
    real workflow (here mocked to exercise just the gate, not the LLM).
  * The downstream caller contract (``status != "success"`` ⇒ failure)
    is preserved on the decline path.

Run:
    python tests/test_scalarizer_sandbox.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _no_sandbox_env() -> dict:
    """Env that guarantees ``require_sandbox_approval`` will decline in
    a non-TTY test runner (no env-var bypass, no Docker indicators)."""
    env = os.environ.copy()
    env.pop("UNSAFE_EXECUTION_OK", None)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_construction_without_sandbox() -> None:
    """ScalarizerAgent constructs cleanly in a non-TTY context without
    UNSAFE_EXECUTION_OK — this is the regression we're fixing."""
    from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent
    # Reset the global cache so this test runs in a known state regardless
    # of test ordering.
    import scilink.executors as _exec
    _exec._GLOBAL_SANDBOX_APPROVED = False

    os.environ.pop("UNSAFE_EXECUTION_OK", None)
    with tempfile.TemporaryDirectory() as td:
        agent = ScalarizerAgent(
            api_key="test-dummy-not-used",
            output_dir=td,
        )
        assert agent is not None
        assert agent.agent_type == "scalarizer"


def test_planning_orchestrator_constructs_planning_child() -> None:
    """The meta-agent's planning-child construction path — the original
    failure mode reported during PR #171 review."""
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode,
    )
    import scilink.executors as _exec
    _exec._GLOBAL_SANDBOX_APPROVED = False
    os.environ.pop("UNSAFE_EXECUTION_OK", None)

    with tempfile.TemporaryDirectory() as td:
        meta = MetaOrchestratorAgent(
            base_dir=td,
            api_key="test-dummy-not-used",
            meta_mode=MetaMode.AUTOPILOT,
        )
        # This call used to raise RuntimeError from inside
        # PlanningOrchestratorAgent.__init__ → ScalarizerAgent.__init__.
        planning_child = meta._get_planning_child()
        assert planning_child is not None


def test_scalarize_returns_failure_dict_on_declined_sandbox() -> None:
    """When the sandbox is declined at scalarize() time, the return value
    is a structured failure dict with the contract keys orchestrator_tools
    relies on."""
    from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent
    import scilink.executors as _exec
    _exec._GLOBAL_SANDBOX_APPROVED = False

    with tempfile.TemporaryDirectory() as td:
        agent = ScalarizerAgent(api_key="test-dummy-not-used", output_dir=td)

        # Pretend the sandbox check denies approval (e.g. non-TTY, no
        # env-var, no Docker).  Patching at the call site in the module.
        with patch(
            "scilink.agents.planning_agents.scalarizer_agent.require_sandbox_approval",
            return_value=False,
        ):
            result = agent.scalarize(
                data_path="/path/does/not/matter.csv",
                objective_query="Calculate yield",
            )

    # Contract: callers in orchestrator_tools.py do `res["status"] != "success"`
    # and read `res.get("error", "Scalarizer failed")`.
    assert isinstance(result, dict), f"got {type(result).__name__}, expected dict"
    assert result["status"] == "failure", (
        f"expected status='failure', got {result.get('status')!r}"
    )
    assert "error" in result and result["error"], (
        f"expected non-empty error message; got {result.get('error')!r}"
    )
    assert "Sandbox" in result["error"], (
        f"error message should explain the sandbox decline; got {result['error']!r}"
    )
    # All other keys the callers may read, even if defaulted:
    for key in ("metrics", "source_script", "column_roles"):
        assert key in result, f"missing contract key: {key!r}"


def test_execute_script_is_sandbox_gated_too() -> None:
    """Defense in depth — _execute_script gates on sandbox approval as
    well, so any future caller that bypasses ``scalarize()`` (e.g. a new
    public method, a test helper, an external integration) is still
    protected at the actual subprocess boundary.

    The duplicate check is essentially free at runtime — the global cache
    in ``require_sandbox_approval`` short-circuits the second call once
    ``scalarize()`` has approved (or once UNSAFE_EXECUTION_OK is set)."""
    from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent
    import scilink.executors as _exec
    _exec._GLOBAL_SANDBOX_APPROVED = False

    with tempfile.TemporaryDirectory() as td:
        agent = ScalarizerAgent(api_key="test-dummy-not-used", output_dir=td)

        # Direct call to _execute_script without going through scalarize().
        # No UNSAFE_EXECUTION_OK, cache cleared above → the gate must fire.
        os.environ.pop("UNSAFE_EXECUTION_OK", None)
        script = Path(td) / "trivial.py"
        script.write_text(
            "import json\n"
            "print(json.dumps({'metrics': {'value': 1.0}, 'plot_path': ''}))\n"
        )

        with patch(
            "scilink.agents.planning_agents.scalarizer_agent.require_sandbox_approval",
            return_value=False,
        ):
            exec_res = agent._execute_script(script)

        assert exec_res["status"] == "failure", (
            f"_execute_script should be sandbox-gated; got {exec_res!r}"
        )
        assert "Sandbox" in (exec_res.get("error") or ""), (
            f"_execute_script's sandbox-decline error should mention "
            f"'Sandbox'; got {exec_res.get('error')!r}"
        )

        # And confirm the happy path still works when approval is granted —
        # the gate's existence shouldn't break normal execution.
        os.environ["UNSAFE_EXECUTION_OK"] = "true"
        _exec._GLOBAL_SANDBOX_APPROVED = False
        try:
            exec_res = agent._execute_script(script)
        finally:
            os.environ.pop("UNSAFE_EXECUTION_OK", None)
            _exec._GLOBAL_SANDBOX_APPROVED = False

        assert exec_res["status"] == "success", (
            f"_execute_script should succeed when sandbox is approved; "
            f"got {exec_res!r}"
        )
        assert exec_res["metrics"]["value"] == 1.0


def test_unsafe_execution_ok_env_var_short_circuits() -> None:
    """If UNSAFE_EXECUTION_OK=true is set, scalarize() proceeds past the
    sandbox gate.  We don't need the actual scalarize workflow to run
    (mocking the LLM is heavyweight); we just need to confirm the gate
    isn't returning the "declined" failure dict."""
    from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent
    import scilink.executors as _exec
    _exec._GLOBAL_SANDBOX_APPROVED = False

    os.environ["UNSAFE_EXECUTION_OK"] = "true"
    try:
        with tempfile.TemporaryDirectory() as td:
            agent = ScalarizerAgent(api_key="test-dummy-not-used", output_dir=td)

            # scalarize() will fail later (no LLM, no real data) but the
            # failure must NOT be the sandbox-decline failure.
            result = agent.scalarize(
                data_path="/path/does/not/exist.csv",
                objective_query="trivial",
            )

        # The sandbox-declined sentinel is what we're checking *isn't* there.
        if result.get("status") == "failure":
            err = result.get("error") or ""
            assert "Sandbox approval declined" not in err, (
                "UNSAFE_EXECUTION_OK should bypass the sandbox decline, "
                f"but got: {err}"
            )
    finally:
        os.environ.pop("UNSAFE_EXECUTION_OK", None)
        _exec._GLOBAL_SANDBOX_APPROVED = False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(name: str, fn) -> bool:
    try:
        fn()
    except AssertionError as e:
        print(f"  ❌ {name}\n     {e}")
        return False
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  💥 {name}\n     {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    print(f"  ✅ {name}")
    return True


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    print("=== ScalarizerAgent sandbox-gate tests ===\n")
    passed = failed = 0
    for name, fn in (
        ("construction without sandbox",                test_construction_without_sandbox),
        ("planning child construction (meta-agent)",    test_planning_orchestrator_constructs_planning_child),
        ("scalarize returns failure dict on decline",   test_scalarize_returns_failure_dict_on_declined_sandbox),
        ("_execute_script is sandbox-gated too",        test_execute_script_is_sandbox_gated_too),
        ("UNSAFE_EXECUTION_OK env-var bypass",          test_unsafe_execution_ok_env_var_short_circuits),
    ):
        if _run(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n  passed: {passed}    failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
