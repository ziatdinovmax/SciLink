"""#638 — a plan refined from feedback / criticism must not be narrated as
results-driven. The HTML box was relabelled in 0d5dffa5 ("Results /
feedback received"); the two log banners on the same channel kept saying
"Results". All three surfaces now name both cases."""
from pathlib import Path


def test_refinement_banners_name_feedback_on_every_surface():
    agent = Path("scilink/agents/planning_agents/planning_agent.py").read_text()
    tools = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    html = Path("scilink/agents/planning_agents/html_generator.py").read_text()
    assert "Refining Plan based on New Results / Feedback" in agent
    assert "Refining Plan based on Results / Feedback" in tools
    assert "Results / feedback received" in html
    # No surviving results-only banner on the refinement channel.
    assert "based on New Results ---" not in agent
    assert "based on Results..." not in tools
