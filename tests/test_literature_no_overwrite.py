"""Two same-type literature searches must not truncate each other, and a
single search must keep the historical filename exactly."""
import os, tempfile, types
from pathlib import Path
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel)

ok = {}
def check(n, c, d=""): ok[n] = bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n} {d}")

with tempfile.TemporaryDirectory() as t:
    d = Path(t); (d/"data").mkdir()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        orch = PlanningOrchestratorAgent(base_dir=str(d), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(d/"data"))
    tools = orch.tools

    # Stub the literature agent: deterministic per-call content, no network.
    calls = {"n": 0}
    def _mk(tag):
        def f(objective, **kw):
            calls["n"] += 1
            return {"status": "success",
                    "content": f"RESULT-{calls['n']} for {objective}"}
        return f
    class FakeLit:
        search_for_hypothesis_context = staticmethod(_mk("hyp"))
        search_for_cross_domain = staticmethod(_mk("cross"))
        search_for_economic_data = staticmethod(_mk("econ"))
        search_for_fitting_models = staticmethod(_mk("fit"))
    orch.lit_agent = FakeLit()

    r1 = tools.execute_tool("search_literature", objective="first question",
                            search_type="hypothesis_context")
    r2 = tools.execute_tool("search_literature", objective="second question",
                            search_type="hypothesis_context")
    import json
    p1, p2 = json.loads(r1).get("file_path"), json.loads(r2).get("file_path")
    files = sorted(p.name for p in d.rglob("literature_search_*.md"))
    check("first keeps historical name",
          Path(p1).name == "literature_search_hypothesis_context.md",
          f"({Path(p1).name})")
    check("second gets a distinct name", p1 != p2, f"({Path(p2).name})")
    check("both files exist", len(files) == 2, f"({files})")
    check("first content intact", "RESULT-1" in Path(p1).read_text())
    check("second content intact", "RESULT-2" in Path(p2).read_text())
    # registry + newest-wins discovery still work
    reg = tools._lit_registry()
    paths = [Path(e["path"]).name for e in reg]
    check("registry holds BOTH distinct paths",
          len(set(paths)) == 2, f"({paths})")
    # Campaign scoping (#396): pending literature is only claimed once a
    # plan exists. Stamp a campaign and confirm newest-wins still works.
    for e in reg:
        e["campaign_id"] = 1
    orch.planner.state["current_plan"] = {"campaign_id": 1}
    latest = tools._latest_literature_file()
    check("newest file discoverable once claimed",
          latest is not None and latest.name == Path(p2).name,
          f"({latest.name if latest else None})")

print("=" * 50)
print(f"LIT COLLISION: {sum(ok.values())}/{len(ok)} passed")
raise SystemExit(0 if all(ok.values()) else 1)
