"""run_analysis without the chat preamble (MCP clients, run_task).

Two hard stops greeted every fresh tool-only caller: 'No agent selected'
and 'No metadata available'. Now the agent is inferred from the data probe
when that is unambiguous, and the call's analysis_goal/objective/hints
serve as minimal metadata when no file or sidecar exists — both reported in
the response. Ambiguous data and a bare call still error, with a hint.
No LLM: create_agent_for_analysis is stubbed to return a fake agent.
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.analysis_orchestrator import (
    AnalysisOrchestratorAgent, AnalysisMode,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


class _FakeAgent:
    """Stands in for CurveFittingAgent: records the system_info it got."""
    seen = []

    def analyze(self, data=None, system_info=None, **kw):
        _FakeAgent.seen.append({"data": data, "system_info": system_info})
        return {"status": "success", "detailed_analysis": "ok",
                "scientific_claims": [], "output_dir": kw.get("output_dir")}


def make_orch(tmp):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = AnalysisOrchestratorAgent(
            base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
            model_name="claude-opus-4-6", analysis_mode=AnalysisMode.AUTONOMOUS)
    orch.create_agent_for_analysis = lambda agent_id, out_dir, **kw: _FakeAgent()
    return orch


def run(orch, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = json.loads(orch.tools.execute_tool("run_analysis", **kw))
    return out, buf.getvalue()


with tempfile.TemporaryDirectory() as tmp:
    x = np.linspace(0, 100, 300)
    y = np.exp(-0.5 * ((x - 40) / 4) ** 2)
    csv = Path(tmp) / "spectrum.csv"
    np.savetxt(csv, np.column_stack([x, y]), delimiter=",", header="x,y", comments="")

    orch = make_orch(tmp)
    out, log = run(orch, data_path=str(csv))
    check("bare_call_still_errors_with_hint",
          out["status"] == "error" and "analysis_goal" in out["message"], str(out)[:150])

    orch = make_orch(tmp)
    out, log = run(orch, data_path=str(csv),
                   analysis_goal="Raman spectrum, x=cm-1, y=counts; fit the band")
    check("agent_inferred_for_1d_csv",
          out.get("status") == "success" and out.get("agent_inferred") == "CurveFittingAgent",
          str(out)[:200])
    check("minimal_metadata_used_and_reported",
          "metadata_note" in out and _FakeAgent.seen
          and _FakeAgent.seen[-1]["system_info"].get("analysis_goal", "").startswith("Raman"),
          str(_FakeAgent.seen[-1]["system_info"])[:120])

    # a stem-matched sidecar still wins over the minimal fallback
    csv.with_suffix(".json").write_text(json.dumps({"technique": "Raman", "x_units": "cm-1"}))
    orch = make_orch(tmp)
    out, log = run(orch, data_path=str(csv), analysis_goal="fit the band")
    check("sidecar_preferred_over_minimal",
          out.get("status") == "success" and "metadata_note" not in out
          and _FakeAgent.seen[-1]["system_info"].get("technique") == "Raman")

    # explicit agent_id still honoured (no inference)
    orch = make_orch(tmp)
    out, log = run(orch, data_path=str(csv), agent_id=0, analysis_goal="fit")
    check("explicit_agent_id_not_overridden",
          out.get("status") == "success" and "agent_inferred" not in out)

print("=" * 50)
print(f"RUN_ANALYSIS TOOL-ONLY: {len(PASS)}/{len(PASS) + len(FAIL)} passed")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
