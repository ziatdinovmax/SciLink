"""Live: do the reviewer-grade critic lenses catch what an expert reviewer caught?

Fixtures are LOCAL (tests/_critic_lenses_fixtures/, gitignored): two plan JSONs
from a real session, plus a checklist of the plan-level issues an external
reviewer raised on the resulting white papers. Skips when absent.

Run: python tests/test_critic_lenses_live.py
Reports, per plan: findings by dimension, checklist coverage (judged by the
model), and the same for a Phase-0-off variant (critic sees only the clipped
conformance summary) so the input-visibility gain is measured separately.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
FIX = Path("tests/_critic_lenses_fixtures").resolve()
RUN = Path("tests/_critic_lenses_live_runs").resolve()
results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _model():
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    if RUN.exists():
        shutil.rmtree(RUN)
    (RUN / "data").mkdir(parents=True)
    orch = PlanningOrchestratorAgent(
        objective="critic lens evaluation", base_dir=str(RUN / "session"),
        api_key=None, model_name=MODEL, autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(RUN / "data"))
    return orch.planner.model, orch.planner.generation_config


def _judge(model, gen, checklist, findings):
    from scilink.knowledge import parse_json_from_response
    prompt = (
        "You are scoring whether a set of critic findings covers a reviewer's checklist.\n"
        "For each checklist item, answer covered=true only if at least one finding raises "
        "the SAME substantive concern (same mechanism / same design flaw), not merely a "
        "related topic. Return JSON: {\"items\": [{\"item\": <index>, \"covered\": bool, "
        "\"finding\": <index or null>}]}\n\n"
        "CHECKLIST:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(checklist)) +
        "\n\nFINDINGS:\n" + "\n".join(f"{i}. [{f.get('dimension')}/{f.get('severity')}] {f.get('issue')}"
                                    for i, f in enumerate(findings)) + "\n")
    resp = model.generate_content([prompt], generation_config=gen)
    out, _ = parse_json_from_response(resp)
    return [bool(x.get("covered")) for x in (out or {}).get("items", [])]


def run(tag, plan_path, checklist, model, gen, phase0=True):
    from scilink.agents.planning_agents import planning_rag as pr
    plan = json.loads(Path(plan_path).read_text())
    objective = f"{plan.get('portfolio_title', '')}. {plan.get('thesis', '')}"
    lit = ""
    ls = plan.get("literature_search")
    if isinstance(ls, dict):
        lit = json.dumps(ls)[:60000]
    elif isinstance(ls, str):
        lit = ls[:60000]
    if not phase0:
        orig = pr.summarize_plan_for_critic
        pr.summarize_plan_for_critic = lambda r: "\n".join(
            pr.summarize_experiment(e, i + 1) for i, e in enumerate(r.get("proposed_experiments", [])))
    try:
        verdict = pr.critique_plan(objective, plan, model, gen, retrieved_context=lit or None)
    finally:
        if not phase0:
            pr.summarize_plan_for_critic = orig
    findings = verdict.get("findings", [])
    print(f"\n=== {tag} (phase0={'on' if phase0 else 'off'}): {len(findings)} findings")
    for f in findings:
        print(f"   - [{f.get('dimension')}/{f.get('severity')}] {str(f.get('issue'))[:220]}")
    covered = _judge(model, gen, checklist, findings) if findings else [False] * len(checklist)
    hits = sum(covered)
    print(f"   coverage: {hits}/{len(checklist)}  " +
          " ".join(f"{'✓' if c else '✗'}{i}" for i, c in enumerate(covered)))
    dims = {f.get("dimension") for f in findings}
    return hits, len(checklist), dims, findings


if __name__ == "__main__":
    if not (FIX / "plan_oer.json").exists():
        print("fixtures absent — skipping"); sys.exit(0)
    chk = json.loads((FIX / "reviewer_checklist.json").read_text())
    model, gen = _model()
    # the session copies with literature_search intact (for the evidence lens)
    S = Path("meta_session_20260817_162104/planning/delegations")
    plans = {"oer": (S / "01_propose_a_killer_proof_of_concept_use_ca/plan.json"),
             "pd": (S / "06_assess_feasibility_first_then_only_if_it/plan.json")}
    for k, p in plans.items():
        if not p.exists():
            p = FIX / f"plan_{k}.json"
        plans[k] = p
    summary = {}
    for k in ("oer", "pd"):
        h, n, dims, _ = run(k, plans[k], chk[k], model, gen, phase0=True)
        h0, _, _, _ = run(k, plans[k], chk[k], model, gen, phase0=False)
        summary[k] = (h, n, h0, dims)
    print("\n" + "=" * 60)
    for k, (h, n, h0, dims) in summary.items():
        print(f"{k}: lenses+phase0 {h}/{n}   |   lenses only (phase0 off) {h0}/{n}   | dims={sorted(d for d in dims if d)}")
        # Stochastic (critic + judge): observed 3-4/5 (oer) and 2-3/6 (pd) across
        # runs, against ~0-1 for the pre-lens critic on the same plans.
        check(f"{k} coverage >= 50% with phase0", h / n >= 0.5)
        check(f"{k} uses at least one new lens", bool(dims & {"design", "statistics", "method", "evidence"}))
    npass = sum(results.values())
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
