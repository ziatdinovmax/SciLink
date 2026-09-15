"""Live validation: the workspace listing ends blind path-probing
(Bedrock Opus 4.8).

Replays the observed struggle: a document whose TITLE ("MZI engineering
companion") matches neither its filename (cdoc_platform_engineering_
companion.md) nor its delegation folder (named by task slug). Pre-fix,
the agent probed six guessed paths and only recovered by reading
deliverables.json as a last resort. With the deliverables index folded
into list_workspace_files, one listing call must be enough.

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    python tests/test_workspace_listing_live.py
"""
from __future__ import annotations

import contextlib
import io
import re
import shutil
import sys
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_workspace_listing_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


COMPANION = """# CDOC Platform Engineering Companion

## Characterization
Mach-Zehnder interferometer quantitative phase imaging (MZI-QPI) is the
platform's primary channel; the MARKER-MZI-COST budget line is $46k.

## Cell design
Flow cell with top optical access; reset by rinse-in-place.
"""


def main():
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel)
    from scilink.agents.planning_agents.user_interface import (
        record_deliverable)

    run = BASE / "p1"
    if run.exists():
        shutil.rmtree(run)
    data_dir = run / "data"
    data_dir.mkdir(parents=True)
    orch = PlanningOrchestratorAgent(
        objective="Chemical dynamics observatory campaign design",
        base_dir=str(run / "session"),
        api_key=None, model_name=MODEL,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    base = Path(orch.base_dir)
    # Title matches neither the filename nor the task-slug folder name.
    d01 = base / "delegations" / "01_brainstorm_and_sketch_a_chemical_dynamic"
    d01.mkdir(parents=True)
    doc = d01 / "cdoc_platform_engineering_companion.md"
    doc.write_text(COMPANION)
    record_deliverable(base, doc, "MZI Engineering Companion (Track 1)",
                       deliverable=True)
    # decoys: other delegations with similar-looking files
    for slug, name in (("09_revise_in_place_the_spm_centered", "spm_notes.md"),
                       ("19_brainstorm_a_saxs_waxs_version", "saxs_notes.md")):
        d = base / "delegations" / slug
        d.mkdir(parents=True)
        (d / name).write_text(f"# {name}\nnot the MZI doc\n")

    buf = Tee()
    with contextlib.redirect_stdout(buf):
        reply = orch.chat(
            "What does the MZI engineering companion say the MZI-QPI "
            "budget line is? Quote the exact dollar figure.")
    log = buf.getvalue()

    check("listing was consulted", "Listing files in" in log)
    reads = re.findall(r"Reading file '([^']+)'", log)
    check("read the real file on the first read attempt",
          bool(reads) and reads[0].endswith(
              "cdoc_platform_engineering_companion.md"))
    check("no blind path probes",
          "No such file" not in log and "Could not find" not in log)
    check("answer extracted from the right document", "46k" in reply)

    print()
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
