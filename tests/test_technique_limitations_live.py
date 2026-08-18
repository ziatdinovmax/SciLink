"""Live: the adversarial literature leg surfaces the technique pitfalls an
expert reviewer cited, and the critic then raises them.

Needs FUTUREHOUSE_API_KEY (Edison; ~10-15 min per search) and Bedrock creds.
Run: python tests/test_technique_limitations_live.py
"""
from __future__ import annotations
import json, os, re, shutil, sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
RUN = Path("tests/_technique_limitations_live_runs").resolve()
QUESTION = ("A proof-of-concept that estimates palladium-hydride loading (H:Pd) online "
            "from optical/LSPR, Mach-Zehnder interferometric and coulometric channels on Pd "
            "islands supported on Co3O4 in acidic aqueous electrolyte during hydrogen "
            "evolution, fires an electrochemical charge/discharge step on the estimated "
            "state, measures H2 production as the endpoint, and maps the driven state "
            "afterwards with SECCM and KPFM under electrolyte.")
EXPECT = {"kpfm_electrolyte": r"kpfm.{0,200}(electrolyte|liquid|aqueous|screening)",
          "coulometry_artifact": r"(coulometr|chronocoulometr).{0,250}(overestimat|competing|HER|hydrogen evolution|adsorb)",
          "co3o4_stability": r"co3o4.{0,200}(dissol|unstable|instab|leach|acid)",
          "h_inventory": r"(stored|trapped|absorbed) hydrogen.{0,200}(release|desorb)"}

if __name__ == "__main__":
    if not os.getenv("FUTUREHOUSE_API_KEY"):
        print("FUTUREHOUSE_API_KEY not set — skipping"); sys.exit(0)
    from scilink.agents.planning_agents.planning_orchestrator import (PlanningOrchestratorAgent, AutonomyLevel)
    if RUN.exists(): shutil.rmtree(RUN)
    (RUN / "data").mkdir(parents=True)
    orch = PlanningOrchestratorAgent(objective="technique limitations check", base_dir=str(RUN / "session"),
                                     api_key=None, model_name=MODEL, autonomy_level=AutonomyLevel.AUTONOMOUS,
                                     data_dir=str(RUN / "data"))
    res = json.loads(orch.tools.execute_tool("search_literature", objective={"technique_limitations": QUESTION}))
    print(json.dumps(res, indent=1)[:1500])
    files = list(Path(orch.base_dir).rglob("literature_search_*technique_limitations*.md"))
    text = "\n".join(f.read_text(errors="replace") for f in files).lower()
    ok = 0
    for k, pat in EXPECT.items():
        hit = re.search(pat, text, re.I | re.S) is not None
        ok += hit; print(f"  [{'PASS' if hit else 'FAIL'}] retrieved: {k}")
    print(f"{ok}/{len(EXPECT)} pitfalls surfaced by the adversarial leg")
    sys.exit(0 if ok >= 2 else 1)
