#!/usr/bin/env python3
"""Exercise the analysis agent's session-file tools (#481) on the EELS
identification example — read_document, save_file, edit_file, read_file
(offset / tail / search), append_file, rename_file, and a real analysis
whose results JSON is read back with search.

Run from the repo root. Credentials come from the environment, exactly as
in the UI:

    # Bedrock (default model)
    export AWS_BEARER_TOKEN_BEDROCK=...   # or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
    export AWS_REGION_NAME=us-east-1
    python examples/file_tools_demo.py

    # Any other provider: pass the model and set its usual env var
    export ANTHROPIC_API_KEY=...
    python examples/file_tools_demo.py --model claude-opus-4-6

Each turn prints the agent's answer; the script ends with an on-disk
check of what the tools should have produced (backups, renamed file,
literature file) and a PASS / FAIL line per item. Use --skip-analysis to
stop before the (slower) run_analysis turn.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEMO = REPO / "examples" / "eels_identification_demo"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bedrock/us.anthropic.claude-opus-4-8")
ap.add_argument("--session-dir", default=None,
                help="Where to put the session (default: ./file_tools_demo_session_<time>)")
ap.add_argument("--skip-analysis", action="store_true")
args = ap.parse_args()

session = Path(args.session_dir or f"file_tools_demo_session_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
session.mkdir(parents=True, exist_ok=True)

# A long, sectioned report the agent must NOT read whole: the point of
# offset / tail / search.
long_report = session / "long_report.md"
long_report.write_text(
    "# Beamtime report\n\n"
    + "".join(f"## Section {i}\n\n" + "filler line\n" * 30 for i in range(1, 15))
    + "\n## Conclusions\n\nThe O K pre-peak splitting is 2.6 eV.\n")

from scilink.agents.exp_agents.analysis_orchestrator import (  # noqa: E402
    AnalysisMode, AnalysisOrchestratorAgent)
import scilink.executors as executors  # noqa: E402

executors._GLOBAL_SANDBOX_APPROVED = True   # the demo consents to code execution
agent = AnalysisOrchestratorAgent(base_dir=str(session), api_key=None,
                                  model_name=args.model,
                                  analysis_mode=AnalysisMode.AUTONOMOUS)

turns = [
    f"Read the example's README at `{DEMO / 'README.md'}` with read_document, then examine the "
    f"spectrum at `{DEMO / 'spectrum.npy'}` and load its metadata `{DEMO / 'spectrum.json'}`. "
    "Do NOT run the analysis yet — tell me what the README says the agent should find.",

    "Save a file notes/plan.md (save_file, subfolder 'notes') with exactly three lines describing "
    "how you would identify the material. Then use edit_file to change the word 'identification' "
    "in it to 'fingerprinting' (if it is not there, change the first word of line 1 to 'REVISED'). "
    "Then read the file back with read_file and quote it.",

    f"There is a long report at `{long_report}`. Tell me what its Conclusions section says and "
    "which line the heading is on. Do not read the whole file — use read_file's search or offset.",

    "Append two more lines to notes/plan.md with append_file, then overwrite the whole file with "
    "save_file using one new line. Tell me the exact backup filename the overwrite reported, then "
    "rename the file to final_plan.md with rename_file.",
]
if not args.skip_analysis:
    turns.append(
        "Now run the analysis in identification mode on the spectrum, grounded on the README you "
        "read (pass the literature file). When it finishes, open the analysis_results.json it "
        "produced with read_file — search for the candidate materials rather than reading the "
        "whole file — and report the top candidate.")

for i, t in enumerate(turns, 1):
    print(f"\n{'=' * 78}\nTURN {i}: {t}\n{'=' * 78}", flush=True)
    t0 = time.time()
    try:
        ans = agent.chat(t)
    except Exception as e:  # noqa: BLE001
        ans = f"<<EXCEPTION: {e!r}>>"
    print(f"\n--- ANSWER {i} ({time.time() - t0:.0f}s) ---\n{ans}\n", flush=True)

# ── what the tools should have left on disk ─────────────────────
print("\n" + "=" * 78 + "\nON-DISK CHECKS\n" + "=" * 78)
checks = {
    "literature file from read_document": any((session / "literature").glob("provided_documents_*.md")),
    "edit backup notes/plan.before_edit.md": (session / "notes" / "plan.before_edit.md").is_file(),
    "overwrite backup notes/plan.before_overwrite.md": (session / "notes" / "plan.before_overwrite.md").is_file(),
    "renamed file notes/final_plan.md": (session / "notes" / "final_plan.md").is_file(),
    "original notes/plan.md gone after rename": not (session / "notes" / "plan.md").exists(),
}
if not args.skip_analysis:
    checks["analysis_results.json produced"] = any(session.glob("results/*/analysis_results.json"))
ok = True
for label, cond in checks.items():
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + label)
print(f"\nSession: {session}")
print("ALL PASSED" if ok else "SOME CHECKS FAILED — see above")
sys.exit(0 if ok else 1)
