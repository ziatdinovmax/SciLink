"""A chat turn's timeout is retried once, inside litellm_completion, not
again by the chat loop.

The analysis, meta and simulation chat loops caught any error whose text
said "timeout" and re-sent the same call, up to three times, spending
orchestrator iterations; with the retry policy inside litellm_completion
(one retry for a timeout) a stuck call cost about six attempts.
"""

import litellm
import pytest


def _timeout():
    return litellm.Timeout(message="Request timed out", model="m", llm_provider="bedrock")


def _build(kind, tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    model = "anthropic/claude-sonnet-4-5"
    if kind == "meta":
        from scilink.agents.meta_agent import meta_orchestrator as mod
        from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
        agent = MetaOrchestratorAgent(base_dir=str(tmp_path / "s"), model_name=model,
                                      meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    elif kind == "analysis":
        from scilink.agents.exp_agents import analysis_orchestrator as mod
        from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode, AnalysisOrchestratorAgent
        agent = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "s"), model_name=model,
                                          analysis_mode=AnalysisMode.AUTONOMOUS)
    else:
        pytest.importorskip("ase")
        from scilink.agents.sim_agents import simulation_orchestrator as mod
        from scilink.agents.sim_agents.simulation_orchestrator import SimulationMode, SimulationOrchestratorAgent
        agent = SimulationOrchestratorAgent(base_dir=str(tmp_path / "s"), model_name=model,
                                            simulation_mode=SimulationMode.AUTONOMOUS)
    return mod, agent


@pytest.mark.parametrize("kind", ["meta", "analysis", "simulation"])
def test_a_timeout_surfaces_after_the_inner_retry(kind, tmp_path, monkeypatch):
    mod, agent = _build(kind, tmp_path, monkeypatch)
    calls = []

    def stuck(**kwargs):
        calls.append(kwargs)
        raise _timeout()

    from scilink.wrappers import litellm_wrapper      # the loops import it at call time
    monkeypatch.setattr(litellm_wrapper, "litellm_completion", stuck)
    with pytest.raises(litellm.Timeout):
        agent._handle_litellm_chat("hello")
    assert len(calls) == 1          # the loop no longer re-sends it
