"""Offline tests for DiagramAgent: render-error retry + visual QC loop.

The LLM is stubbed with a scripted sequence; rendering is real
(mermaid-cli via mmdc/npx) and the whole file SKIPs cleanly when the
renderer is unavailable.

  conda run -n scilink python tests/test_diagram_agent.py
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.utils.mermaid_render import mermaid_available, render_mermaid
from scilink.agents.planning_agents.diagram_agent import DiagramAgent

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


GOOD = ('```mermaid\nflowchart TD\n  A["Prepare samples"] --> '
        'B{"Quality OK"}\n  B -->|yes| C["Run optimization loop"]\n'
        '  B -->|no| A\n  C --> D["Report"]\n```')
BROKEN = '```mermaid\nflowchart TD\n  A[--> ::bad\n```'
PLAN = {"proposed_experiments": [{
    "experiment_name": "Yield optimization",
    "hypothesis": "Temperature and time control yield",
    "experimental_steps": ["Prepare samples", "Measure yield",
                           "Optimize conditions"],
}]}


class SeqModel:
    """generate_content stub fed by a scripted reply sequence."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, parts, generation_config=None):
        self.calls.append(parts)
        return SimpleNamespace(text=self.replies.pop(0))


if not mermaid_available():
    print("SKIP: mermaid-cli/npx not available")
    raise SystemExit(0)

print("0) renderer sanity")
with tempfile.TemporaryDirectory() as t:
    ok, err = render_mermaid('flowchart TD\n  A["x"] --> B["y"]',
                             Path(t) / "s.png")
    check("valid source renders", ok and not err)
    ok2, err2 = render_mermaid("flowchart TD\n  A[--> ::bad",
                               Path(t) / "b.png")
    check("broken source errors with message", (not ok2) and bool(err2))

print("1) broken first draft -> render-error retry -> success")
with tempfile.TemporaryDirectory() as t:
    m = SeqModel([BROKEN, GOOD, '{"approved": true, "issues": []}'])
    agent = DiagramAgent(model=m, output_dir=t)
    res = agent.generate_workflow_diagram(PLAN, out_dir=t)
    check("retry succeeds", res["status"] == "success")
    check("two generation attempts", res["attempts"] == 2)
    check("png exists", Path(res["png_path"]).exists())
    check("mmd saved", Path(res["mmd_path"]).exists())
    check("render error fed back",
          "failed to render" in str(m.calls[1][0]))

print("2) QC rejection -> redraw with issues -> approval")
with tempfile.TemporaryDirectory() as t:
    m = SeqModel([GOOD,
                  '{"approved": false, "issues": ["overcrowded"]}',
                  GOOD,
                  '{"approved": true, "issues": []}'])
    agent = DiagramAgent(model=m, output_dir=t)
    res = agent.generate_workflow_diagram(PLAN, out_dir=t)
    check("qc round recorded", res["status"] == "success"
          and res["qc_rounds"] == 1)
    check("issues fed back", "overcrowded" in str(m.calls[2][0]))

print("3) attempt budget exhausts to clean error")
with tempfile.TemporaryDirectory() as t:
    m = SeqModel([BROKEN, BROKEN, BROKEN])
    agent = DiagramAgent(model=m, output_dir=t)
    res = agent.generate_workflow_diagram(PLAN, out_dir=t,
                                          max_render_attempts=3)
    check("exhaustion is an error, not a raise",
          res["status"] == "error" and res["attempts"] == 3)

print("4) simplicity contract in the prompt")
with tempfile.TemporaryDirectory() as t:
    m = SeqModel([GOOD, '{"approved": true, "issues": []}'])
    DiagramAgent(model=m, output_dir=t).generate_workflow_diagram(
        PLAN, out_dir=t)
    check("simple detail rule default",
          "Compact overview" in str(m.calls[0][0]))


print("5) truncated reply (no closing fence) is still usable")
with tempfile.TemporaryDirectory() as t:
    TRUNC = GOOD.replace("```", "", 1).rsplit("```", 1)[0]  # keep opener only
    m = SeqModel(["```mermaid\n" + TRUNC.split("\n", 1)[1],
                  '{"approved": true, "issues": []}'])
    agent = DiagramAgent(model=m, output_dir=t)
    res = agent.generate_workflow_diagram(PLAN, out_dir=t)
    check("recovers from missing closing fence", res["status"] == "success")

print("6) house palette applied, model styling stripped")
with tempfile.TemporaryDirectory() as t:
    styled = GOOD.replace("```mermaid\n", "```mermaid\nclassDef foo fill:#123456\n")
    m = SeqModel([styled, '{"approved": true, "issues": []}'])
    res = DiagramAgent(model=m, output_dir=t).generate_workflow_diagram(PLAN, out_dir=t)
    check("model classDef stripped", "#123456" not in res["code"])
    check("house palette present", "#BBDEFB" in res["code"])

print("=" * 50)
n = sum(results.values())
print(f"DIAGRAM AGENT: {n}/{len(results)} checks passed")
if n != len(results):
    raise SystemExit(1)
