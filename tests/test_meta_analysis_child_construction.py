"""The meta must be able to construct its analysis child for real.

Commit 30940a44 forwarded ``embedding_base_url`` to BOTH children, but only
the planning orchestrator takes it — every ``delegate_to_analysis`` then
failed with a TypeError before the child existed (seen live). The child
factories are monkeypatched in most meta tests, which is why nothing
caught it; this test builds the real child.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")


def test_meta_builds_a_real_analysis_child(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent)
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    meta = MetaOrchestratorAgent(base_dir=str(tmp_path), api_key="sk-dummy",
                                 embedding_base_url="http://embed.local/v1")
    child = meta._get_analysis_child()
    assert isinstance(child, AnalysisOrchestratorAgent)
    assert meta._get_analysis_child() is child   # persistent, reused
