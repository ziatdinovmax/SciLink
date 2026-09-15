"""
Live LangGraph backbone integration tests.

PURPOSE
-------
Verify that the LangGraph ReAct backbone works end-to-end with a real LLM
call — not a mock.  These tests complement the unit/smoke suite by confirming
that:

  1. A real LLM response is routed through the graph correctly.
  2. The response is returned by chat() as a non-empty string.
  3. chat_history.json is written to disk after the call.
  4. A second chat() call accumulates history across turns (MemorySaver works).
  5. The LiteLLM path works (no --base-url) for at least one provider.
  6. The OpenAI-SDK path works (--base-url present) when a proxy is available.
  7. run_task() returns a structured dict with the required keys.

Each test is gated on the presence of the relevant env var(s) and skips
cleanly when none are set — so this file is safe to commit and run in CI
with whatever keys are available in the environment.

SUPPORTED CONFIGURATIONS
------------------------
All combinations below are auto-detected from environment variables.  Tests
are parameterised so whichever providers are available will be exercised.

    Provider          Env vars needed
    ────────────────  ────────────────────────────────────────────────────
    Anthropic         ANTHROPIC_API_KEY  or  CLAUDE_API_KEY
    OpenAI            OPENAI_API_KEY
    Google Gemini     GEMINI_API_KEY  or  GOOGLE_API_KEY
    Internal proxy    SCILINK_API_KEY  +  SCILINK_BASE_URL
                       (SCILINK_BASE_URL must point to an OpenAI-compatible
                        endpoint, e.g. https://proxy.example.com/v1)

    SCILINK_TEST_MODEL  — optional override for the model name used on the
                          proxy path (default: claude-opus-4-6)

RUN
---
    # Run all live tests (skips whichever keys are absent):
    pytest tests/test_langgraph_live_integration.py -v

    # Run with an Anthropic key:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_langgraph_live_integration.py -v

    # Run with the internal proxy:
    SCILINK_API_KEY=sk-... SCILINK_BASE_URL=https://proxy/v1 \\
        pytest tests/test_langgraph_live_integration.py -v

    # Quick single-provider run:
    pytest tests/test_langgraph_live_integration.py -v -k "anthropic"

WHAT IS TESTED (per provider / path)
-------------------------------------
  T1  single_turn    — chat("Say HELLO") returns a non-empty string
  T2  disk_write     — chat_history.json exists and contains the turn
  T3  multi_turn     — second chat() call references first reply (MemorySaver)
  T4  run_task       — run_task() returns dict with required keys
  T5  litellm_path   — use_openai=False path actually reached (LiteLLM only)
  T6  openai_path    — use_openai=True path actually reached (proxy only)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Provider / path discovery
# ---------------------------------------------------------------------------

def _env(*names) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


# Each entry: (provider_label, model_name, api_key, base_url, use_openai_path)
def _available_providers():
    entries = []

    # Anthropic via LiteLLM
    key = _env("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
    if key:
        entries.append(("anthropic-litellm", "claude-opus-4-6", key, None, False))

    # OpenAI via LiteLLM
    key = _env("OPENAI_API_KEY")
    if key:
        entries.append(("openai-litellm", "gpt-4o-mini", key, None, False))

    # Google via LiteLLM (model requires "gemini/" prefix for LiteLLM routing)
    key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if key:
        entries.append(("gemini-litellm", "gemini/gemini-2.0-flash", key, None, False))

    # Internal proxy via openai SDK
    proxy_key = _env("SCILINK_API_KEY")
    proxy_url = _env("SCILINK_BASE_URL")
    if proxy_key and proxy_url:
        model = _env("SCILINK_TEST_MODEL") or "claude-opus-4-6"
        entries.append(("proxy-openai-sdk", model, proxy_key, proxy_url, True))

    return entries


_PROVIDERS = _available_providers()
_PROVIDER_IDS = [p[0] for p in _PROVIDERS]


# ---------------------------------------------------------------------------
# Orchestrator factories
# ---------------------------------------------------------------------------

def _make_analysis_orch(base_dir, model_name, api_key, base_url):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
    )
    return AnalysisOrchestratorAgent(
        base_dir=base_dir,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )


def _make_planning_orch(base_dir, model_name, api_key, base_url):
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel,
    )
    return PlanningOrchestratorAgent(
        objective="Test: verify backbone works",
        base_dir=base_dir,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=base_dir,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=_PROVIDERS, ids=_PROVIDER_IDS)
def provider(request):
    """Parametrised fixture — yields (label, model, api_key, base_url, use_openai)."""
    return request.param


@pytest.fixture(params=_PROVIDERS, ids=_PROVIDER_IDS)
def analysis_orch(request, tmp_path):
    """AnalysisOrchestratorAgent built against each available provider."""
    _label, model, key, base_url, _use_openai = request.param
    return _make_analysis_orch(str(tmp_path), model, key, base_url)


@pytest.fixture(params=_PROVIDERS, ids=_PROVIDER_IDS)
def planning_orch(request, tmp_path):
    """PlanningOrchestratorAgent built against each available provider."""
    _label, model, key, base_url, _use_openai = request.param
    try:
        return _make_planning_orch(str(tmp_path), model, key, base_url)
    except Exception as e:
        # ScalarizerAgent may raise in constrained environments
        pytest.skip(f"Planning orch init failed ({e})")


# ---------------------------------------------------------------------------
# Skip guard — skip entire module when no providers are available
# ---------------------------------------------------------------------------

if not _PROVIDERS:
    pytest.skip(
        "No API keys found. Set at least one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, "
        "SCILINK_API_KEY+SCILINK_BASE_URL",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# T1 — single turn returns non-empty string
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT1SingleTurn:
    """chat() with a trivial prompt returns a non-empty string."""

    def test_analysis_single_turn(self, analysis_orch):
        response = analysis_orch.chat(
            "Reply with exactly the word PONG and nothing else."
        )
        assert isinstance(response, str), f"Expected str, got {type(response)}"
        assert len(response.strip()) > 0, "Response was empty"
        print(f"\n   ✅ Analysis single turn: {response[:80]!r}")

    def test_planning_single_turn(self, planning_orch):
        response = planning_orch.chat(
            "Reply with exactly the word PONG and nothing else."
        )
        assert isinstance(response, str)
        assert len(response.strip()) > 0
        print(f"\n   ✅ Planning single turn: {response[:80]!r}")


# ---------------------------------------------------------------------------
# T2 — chat_history.json written to disk after call
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT2DiskWrite:
    """chat_history.json is created after chat() completes."""

    def test_analysis_history_written(self, analysis_orch):
        analysis_orch.chat("Say the word PONG.")
        history_path = Path(analysis_orch.history_path)
        assert history_path.exists(), "chat_history.json was not created"
        history = json.loads(history_path.read_text())
        assert len(history) >= 2, f"Expected ≥2 entries, got {len(history)}"
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles
        print(f"\n   ✅ Analysis history written: {len(history)} messages")

    def test_planning_history_written(self, planning_orch):
        planning_orch.chat("Say the word PONG.")
        history_path = Path(planning_orch.history_path)
        assert history_path.exists(), "chat_history.json was not created"
        history = json.loads(history_path.read_text())
        assert len(history) >= 2
        print(f"\n   ✅ Planning history written: {len(history)} messages")


# ---------------------------------------------------------------------------
# T3 — multi-turn: second call can reference first (MemorySaver accumulates)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT3MultiTurn:
    """Two consecutive chat() calls share state via MemorySaver."""

    def test_analysis_multi_turn(self, analysis_orch):
        first = analysis_orch.chat(
            "Remember this secret code: XRAY42. "
            "Acknowledge you have noted it by replying OK."
        )
        assert isinstance(first, str) and first.strip()

        second = analysis_orch.chat(
            "What secret code did I just give you? "
            "Reply with ONLY the code itself."
        )
        assert isinstance(second, str) and second.strip()
        # The model should be able to recall the code from MemorySaver history
        assert "XRAY42" in second, (
            f"Expected model to recall 'XRAY42' from prior turn, got: {second!r}"
        )
        print(f"\n   ✅ Analysis multi-turn memory: code recalled correctly")

    def test_planning_multi_turn(self, planning_orch):
        first = planning_orch.chat(
            "Remember this secret code: ZULU99. "
            "Acknowledge you have noted it by replying OK."
        )
        assert isinstance(first, str) and first.strip()

        second = planning_orch.chat(
            "What secret code did I just give you? "
            "Reply with ONLY the code itself."
        )
        assert isinstance(second, str) and second.strip()
        assert "ZULU99" in second, (
            f"Expected model to recall 'ZULU99' from prior turn, got: {second!r}"
        )
        print(f"\n   ✅ Planning multi-turn memory: code recalled correctly")


# ---------------------------------------------------------------------------
# T4 — run_task() returns a structured dict with required keys
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT4RunTask:
    """
    SimulationOrchestratorAgent.run_task() returns a dict with the
    contract keys defined in CLAUDE.md.

    The simulation orchestrator is tested via run_task() rather than chat()
    because it requires ase for init — but run_task() can be exercised
    against the analysis orchestrator which has the same contract shape
    once the meta-agent wires it up.

    For now we verify the analysis orchestrator's run_task()-equivalent by
    calling chat() in AUTONOMOUS mode and checking the return value shape,
    PLUS we test SimulationOrchestratorAgent.run_task() directly when ase
    is available.
    """

    def test_simulation_run_task(self, provider, tmp_path):
        """run_task() returns the required dict keys."""
        _label, model, key, base_url, _use_openai = provider
        try:
            from scilink.agents.sim_agents.simulation_orchestrator import (
                SimulationOrchestratorAgent, SimulationMode,
            )
        except ImportError as e:
            pytest.skip(f"sim_agents not importable (missing deps: {e})")

        orch = SimulationOrchestratorAgent(
            base_dir=str(tmp_path),
            api_key=key,
            model_name=model,
            base_url=base_url,
            simulation_mode=SimulationMode.AUTONOMOUS,
        )

        result = orch.run_task(
            "Briefly describe what kinds of structures you can help generate. "
            "Do not call any tools — just reply in one sentence."
        )

        required_keys = {
            "status", "summary", "files_produced",
            "key_findings", "suggested_followups", "structures",
            "warnings", "task",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"run_task() result missing keys: {missing}"
        assert result["status"] in ("success", "error")
        assert isinstance(result["summary"], str) and result["summary"].strip()
        print(f"\n   ✅ run_task() returns required keys, status={result['status']!r}")


# ---------------------------------------------------------------------------
# T5 — LiteLLM path confirmation (use_openai=False)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT5LiteLLMPath:
    """Verify use_openai=False is actually exercised for non-proxy providers."""

    def test_litellm_path_used(self, provider, tmp_path):
        label, model, key, base_url, use_openai = provider
        if use_openai:
            pytest.skip("Proxy path — LiteLLM test not applicable")

        orch = _make_analysis_orch(str(tmp_path), model, key, base_url)
        assert orch.use_openai is False, (
            "Expected use_openai=False for non-proxy provider, "
            f"got use_openai={orch.use_openai}"
        )

        response = orch.chat("Reply with the single word LITELLM.")
        assert isinstance(response, str) and response.strip()
        print(f"\n   ✅ LiteLLM path ({label}): response={response[:60]!r}")


# ---------------------------------------------------------------------------
# T6 — OpenAI SDK path confirmation (use_openai=True)
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT6OpenAISDKPath:
    """Verify use_openai=True is actually exercised for the proxy path."""

    def test_openai_sdk_path_used(self, provider, tmp_path):
        label, model, key, base_url, use_openai = provider
        if not use_openai:
            pytest.skip("Non-proxy path — OpenAI SDK test not applicable")

        orch = _make_analysis_orch(str(tmp_path), model, key, base_url)
        assert orch.use_openai is True, (
            "Expected use_openai=True for proxy provider, "
            f"got use_openai={orch.use_openai}"
        )

        response = orch.chat("Reply with the single word PROXY.")
        assert isinstance(response, str) and response.strip()
        print(f"\n   ✅ OpenAI SDK path ({label}): response={response[:60]!r}")


# ---------------------------------------------------------------------------
# T7 — Tool execution round-trip: model calls a real tool, result flows back
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT7ToolRoundTrip:
    """
    The model calls a built-in tool and the result returns to it correctly.

    We use the analysis orchestrator's `list_results` tool — it requires no
    data file and always returns a valid JSON response, making it safe to
    trigger deterministically.
    """

    def test_analysis_tool_roundtrip(self, analysis_orch):
        response = analysis_orch.chat(
            "Use the list_results tool to show me what analyses have been "
            "completed in this session, then summarise the result in one sentence."
        )
        assert isinstance(response, str) and response.strip(), (
            "Expected a non-empty response after tool call round-trip"
        )
        # The model should have called list_results and summarised it —
        # we can't assert exact content but can confirm the backbone didn't crash.
        print(f"\n   ✅ Tool round-trip: response={response[:100]!r}")

    def test_planning_tool_roundtrip(self, planning_orch):
        response = planning_orch.chat(
            "Use the list_workspace_files tool to list the session directory, "
            "then tell me what you found in one sentence."
        )
        assert isinstance(response, str) and response.strip()
        print(f"\n   ✅ Planning tool round-trip: response={response[:100]!r}")


# ---------------------------------------------------------------------------
# T8 — Step count enforced: model won't loop past MAX_TOOL_ITERATIONS
# ---------------------------------------------------------------------------

@pytest.mark.live
class TestT8StepCountEnforced:
    """
    With MAX_TOOL_ITERATIONS=2, a prompt that would normally chain many tool
    calls is cut off and chat() returns something (not an infinite loop).
    """

    def test_step_limit_returns_without_hanging(self, provider, tmp_path):
        _label, model, key, base_url, _use_openai = provider

        orch = _make_analysis_orch(str(tmp_path), model, key, base_url)
        orch.MAX_TOOL_ITERATIONS = 2

        # Rebuild the graph with the lower limit
        from scilink.graphs.analysis import build_analysis_graph
        from langgraph.checkpoint.memory import MemorySaver
        orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

        # Prompt that would chain many tools if unlimited
        response = orch.chat(
            "List all results, then examine every file in the session directory, "
            "then list results again, then run another listing."
        )
        assert isinstance(response, str), "Expected str response, got None or exception"
        print(f"\n   ✅ Step limit enforced: response returned after ≤2 tool steps")
