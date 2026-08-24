#!/usr/bin/env python3
"""Live check: the analyze-mode orchestrator auto-injects n_candidates=3 for
image analysis and narrates the candidate table.

Runs one autonomous run_task against a real image and checks:
  - the LLM did NOT pass n_candidates in its run_analysis tool call
    (proves the tool-body default fired, not prompt-driven behavior);
  - the "Best-of-3" auto-inject log line fired;
  - the stored analysis record carries anchor_candidates;
  - the orchestrator's final answer narrates the comparison (mentions
    candidates) — also printed for eyeballing.

Needs a vendor key (e.g. AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME) and
UNSAFE_EXECUTION_OK=true.

Usage: python tests/test_best_of_n_orchestrator_live.py [model_name]
"""
import io
import json
import logging
import re
import shutil
import sys
import tempfile
from pathlib import Path

from scilink import auth

GRAINS = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)


def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment for", model_name)
        return 2

    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
    )

    session = Path(tempfile.mkdtemp(prefix="bestofn_orch_"))
    data_dir = session / "uploads"
    data_dir.mkdir()
    img = data_dir / "image.npy"
    shutil.copy2(GRAINS, img)
    meta = data_dir / "metadata.json"
    meta.write_text(json.dumps({
        "sample": "polycrystalline thin film",
        "technique": "SEM",
        "description": "Plan-view SEM image of a polycrystalline "
                       "microstructure with visible grain boundaries.",
    }))

    orch = AnalysisOrchestratorAgent(
        base_dir=str(session),
        api_key=api_key,
        model_name=model_name,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    try:
        task_result = orch.run_task(
            f"Analyze the microscopy image at {img} (metadata file: {meta}). "
            f"Segment the grains and report grain size statistics."
        )
    finally:
        logging.getLogger().removeHandler(handler)
    log = buf.getvalue()

    print("run_task status:", task_result.get("status"))
    summary = task_result.get("summary") or ""

    # 1. LLM did not pass n_candidates in the run_analysis tool call.
    llm_passed_n = None
    for msg in getattr(orch, "messages", []) or []:
        for tc in (msg.get("tool_calls") or []) if isinstance(msg, dict) else []:
            fn = tc.get("function", {})
            if fn.get("name") == "run_analysis":
                args = json.loads(fn.get("arguments") or "{}")
                llm_passed_n = "n_candidates" in args
                print("run_analysis tool-call args keys:", sorted(args))
    if llm_passed_n is None:
        # Fallback: scan persisted chat history for the tool call.
        hist_file = session / "chat_history.json"
        if hist_file.exists():
            hist = hist_file.read_text()
            m = re.search(r'run_analysis.{0,2000}?n_candidates', hist,
                          re.DOTALL)
            llm_passed_n = bool(m)
            print("(checked persisted chat history)")
    auto_injected = llm_passed_n is False

    # 2. Auto-inject log line — the default path runs in escalation mode (v2).
    inject_logged = "Best-of-3 (escalation)" in log

    # 3. Stored record carries the candidate table (1 entry on fast-accept,
    #    3 after escalation — both valid under the v2 default).
    rec = (orch.analysis_results or [{}])[-1]
    full = rec.get("full_result") or {}
    table = full.get("anchor_candidates") or []
    sel = [c for c in table if c.get("selected")] if isinstance(table, list) else []
    escalated = (full.get("anchor_judge") or {}).get("escalated")

    # 4. Final answer narrates the comparison (or the fast-accepted attempt).
    narrated = bool(re.search(r"candidate|attempt", summary, re.IGNORECASE))

    checks = [
        ("task status success", task_result.get("status") == "success"),
        ("LLM left n_candidates unset (tool default fired)", auto_injected),
        ("Best-of-3 auto-inject log line", inject_logged),
        ("candidate table consistent with escalation flag",
         isinstance(table, list) and len(sel) == 1
         and ((escalated is True and len(table) == 3)
              or (escalated is False and len(table) == 1))),
        ("final answer mentions candidates", narrated),
    ]
    print()
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name}")

    print("\n--- final answer (for eyeballing) ---")
    print(summary[:2000])
    print("--- end ---")
    print("session dir:", session)

    ok = all(c for _, c in checks)
    print("\nRESULT:", "PASS — orchestrator auto-injects and narrates"
          if ok else "CHECK — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
