"""#639 — a deliverable must carry its workflow figure once.

Live: `write_technical_document` produced a proposal whose body embedded a
copy of the campaign workflow diagram under a proposal-specific name, and
the workflow-diagram append step then added its own rendering of the same
workflow at the end — the reader saw the figure twice under two captions.
The append step is now idempotent: it skips when the document already
carries a workflow figure (by alt/filename, by a prior appended section,
or by an image byte-identical to an existing campaign diagram).
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.utils import mermaid_render

PNG = b"\x89PNG\r\n\x1a\n" + b"campaign-diagram-bytes"


@pytest.fixture
def tools(tmp_path, monkeypatch):
    monkeypatch.setattr(mermaid_render, "mermaid_available", lambda: True)
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.orch = SimpleNamespace(
        base_dir=tmp_path,
        planner=SimpleNamespace(state={"current_plan": {"objective": "Recover Mg"}}))
    calls = []

    class StubDiagramAgent:
        def generate_workflow_diagram(self, plan, out_dir, stem, detail):
            calls.append(stem)
            png = Path(out_dir) / f"{stem}.png"
            png.write_bytes(b"\x89PNG\r\n\x1a\n" + stem.encode())
            return {"status": "success", "png_path": str(png),
                    "attempts": 1, "qc_rounds": 0}
    t._get_diagram_agent = lambda: StubDiagramAgent()
    t.calls = calls
    return t


def test_inline_workflow_figure_suppresses_the_append(tools, tmp_path):
    body = ("# Proposal\n\nIntro.\n\n"
            "![Process and screening workflow for Mg recovery](mg_recovery_workflow.png)\n\n"
            "## Budget\n\nNumbers.\n")
    out = tools._maybe_embed_workflow_diagram(body, tmp_path, stem="proposal_workflow")
    assert out == body                      # unchanged, no second figure
    assert tools.calls == []                # and no diagram was rendered


def test_a_copy_of_the_campaign_diagram_under_another_name_counts(tools, tmp_path):
    (tmp_path / "campaign_workflow.png").write_bytes(PNG)
    (tmp_path / "figure1.png").write_bytes(PNG)      # byte-identical copy
    body = "# Proposal\n\n![Overview of the process](figure1.png)\n\nBody.\n"
    out = tools._maybe_embed_workflow_diagram(body, tmp_path, stem="proposal_workflow")
    assert out == body and tools.calls == []


def test_append_happens_once_and_is_idempotent(tools, tmp_path):
    body = "# Proposal\n\n![Cost breakdown](costs.png)\n\nBody.\n"
    once = tools._maybe_embed_workflow_diagram(body, tmp_path, stem="proposal_workflow")
    assert once.count("![") == 2 and "## Campaign Workflow" in once
    assert tools.calls == ["proposal_workflow"]
    twice = tools._maybe_embed_workflow_diagram(once, tmp_path, stem="proposal_workflow")
    assert twice == once                     # a second pass adds nothing
    assert tools.calls == ["proposal_workflow"]


def test_unrelated_figures_do_not_block_the_diagram(tools, tmp_path):
    (tmp_path / "campaign_workflow.png").write_bytes(PNG)
    (tmp_path / "sem.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"micrograph")
    body = "# Proposal\n\n![SEM micrograph](sem.png)\n\nBody.\n"
    out = tools._maybe_embed_workflow_diagram(body, tmp_path, stem="proposal_workflow")
    assert "## Campaign Workflow" in out and tools.calls == ["proposal_workflow"]
