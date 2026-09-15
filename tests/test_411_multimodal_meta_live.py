"""Live multimodal reproduction for issue #411 (meta agent, Bedrock).

Reproduces the cross-technique metadata failure in one meta session with
three analysis delegations over real instrument exports:

  1. A thermal measurement (heat flow vs temperature) analyzed with ONE
     combined instrument-export metadata JSON loaded into the session slot
     — the state that used to go stale.
  2. An in-situ diffraction series (per-file sidecar JSONs) requested with
     NO technique named. Pre-fix, the slot's thermal document was silently
     reused — sidecar detection was skipped entirely once any metadata
     occupied the slot — so the planner read the diffraction axes through
     the thermal schema and dropped the technique skill. Post-fix, the
     dataset must resolve its own metadata (ownership backstop or explicit
     re-resolution), select the diffraction skill, and plan diffraction.
  3. An in-situ spectroscopy series (sidecars), again technique-unnamed,
     to confirm the slot re-resolves per dataset every time.

Point the env vars at any dataset trio with this shape (dirs or globs;
data files need stem-matched .json sidecars for datasets 2 and 3):

    SCILINK_411_THERMAL_FILE=...   SCILINK_411_THERMAL_META=...
    SCILINK_411_DIFFRACTION_DATA=... SCILINK_411_SPECTROSCOPY_DATA=...
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true \
    python tests/test_411_multimodal_meta_live.py
"""
from __future__ import annotations

import contextlib
import glob as _glob
import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_411_multimodal_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _sidecar_files(spec: str, cap: int) -> list[Path]:
    """Data files (with stem-matched sidecars) from a dir or glob, evenly
    subsampled to at most *cap* for wall time."""
    p = Path(spec)
    files = sorted(p.glob("*.txt")) if p.is_dir() else sorted(
        Path(f) for f in _glob.glob(spec))
    files = [f for f in files if f.suffix.lower() != ".json"
             and f.with_suffix(".json").is_file()]
    if len(files) > cap:
        step = max(1, len(files) // cap)
        files = files[::step][:cap]
    return files


def stage_data() -> tuple[Path, Path, Path, Path, list[str]]:
    thermal_file = Path(os.environ["SCILINK_411_THERMAL_FILE"])
    thermal_meta = Path(os.environ["SCILINK_411_THERMAL_META"])
    diff_spec = os.environ["SCILINK_411_DIFFRACTION_DATA"]
    spec_spec = os.environ["SCILINK_411_SPECTROSCOPY_DATA"]

    data = BASE / "data"
    thermal = data / "thermal"
    diffraction = data / "insitu_diffraction"
    spectroscopy = data / "insitu_spectroscopy"
    for d in (thermal, diffraction, spectroscopy):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(thermal_file, thermal / "thermal_run.txt")
    meta_path = thermal / "instrument_metadata.json"
    shutil.copy(thermal_meta, meta_path)

    for f in _sidecar_files(diff_spec, cap=8):
        shutil.copy(f, diffraction / f.name)
        shutil.copy(f.with_suffix(".json"), diffraction / (f.stem + ".json"))
    for f in _sidecar_files(spec_spec, cap=4):
        shutil.copy(f, spectroscopy / f.name)
        shutil.copy(f.with_suffix(".json"), spectroscopy / (f.stem + ".json"))

    # Distinctive string values from the thermal document, used to detect
    # any leakage of that document into a later dataset's metadata.
    doc = json.loads(meta_path.read_text())
    markers = [v for v in doc.values()
               if isinstance(v, str) and len(v) > 6][:5]
    return thermal, meta_path, diffraction, spectroscopy, markers


def main() -> int:
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "SCILINK_411_THERMAL_FILE",
                "SCILINK_411_THERMAL_META", "SCILINK_411_DIFFRACTION_DATA",
                "SCILINK_411_SPECTROSCOPY_DATA"):
        if not os.environ.get(var):
            print(f"ERROR: {var} not set", file=sys.stderr)
            return 1
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    if BASE.exists():
        shutil.rmtree(BASE)
    thermal, meta_path, diffraction, spectroscopy, markers = stage_data()

    log_buf = io.StringIO()
    capture = logging.StreamHandler(log_buf)
    capture.setLevel(logging.INFO)
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        _same_dataset)

    meta = MetaOrchestratorAgent(
        base_dir=str(BASE / "meta_session"),
        api_key=None, model_name=MODEL,
        meta_mode=MetaMode.AUTONOMOUS,
    )

    turns = [
        ("thermal",
         f"Delegate to the analysis specialist: fit the thermal analysis "
         f"measurement at {thermal / 'thermal_run.txt'} (multi-column "
         f"instrument export; fit heat flow vs temperature). First load the "
         f"combined instrument metadata JSON at {meta_path}. This is a smoke "
         f"test: tell the specialist to use max_verification_iterations=1 "
         f"and accept the first reasonable fit."),
        ("diffraction",
         f"Next, delegate analysis of the in-situ series of the same sample "
         f"in {diffraction} — track how the patterns evolve across the "
         f"series. Same smoke-test settings (max_verification_iterations=1). "
         f"Proceed autonomously."),
        ("spectroscopy",
         f"Finally, delegate analysis of the in-situ series in "
         f"{spectroscopy} for the same sample — what changes across the "
         f"series? Same smoke-test settings (max_verification_iterations=1). "
         f"Proceed autonomously."),
    ]

    captured = {}
    logged = {}
    for name, prompt in turns:
        buf = io.StringIO()
        log_start = len(log_buf.getvalue())
        print(f"\n>>> turn [{name}]")
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(Tee(sys.__stdout__, buf)):
            reply = meta.chat(prompt)
        captured[name] = buf.getvalue()
        logged[name] = log_buf.getvalue()[log_start:]
        print(f"<<< [{name}] done in {time.perf_counter() - t0:.0f}s: "
              f"{reply[:180]}")

    logging.getLogger().removeHandler(capture)
    for name in captured:
        (BASE / f"captured_{name}.log").write_text(captured[name])
        (BASE / f"logging_{name}.log").write_text(logged[name])
    (BASE / "root_logging.log").write_text(log_buf.getvalue())

    child = meta._children.get("analysis")
    ledger = meta._delegation_ledger

    def ok(entry):
        # Known defect (plan 1.4, unfixed): status can read "success" on a
        # failed child - also require the summary isn't an error banner.
        return (entry.get("status") == "success"
                and not (entry.get("summary") or "").strip().startswith("❌"))

    print("\n--- checks ---")
    check("I1: three delegations recorded", len(ledger) >= 3)
    check("I2: thermal delegation succeeded", bool(ledger) and ok(ledger[0]))

    diff_txt = captured["diffraction"]
    # No-silent-reuse can be satisfied two ways: the run_analysis ownership
    # backstop fires (displacement/refusal message), OR the child explicitly
    # re-resolves metadata for the diffraction dataset before running (fresh
    # load, binds to it). Only a run with NEITHER means the thermal document
    # leaked.
    backstop = ("does not cover this dataset" in diff_txt
                or "belongs to a different dataset" in diff_txt)
    re_resolved = (("Loading metadata from" in diff_txt
                    and "insitu_diffraction"
                    in diff_txt.split("Running analysis")[0])
                   or "Synthesized global metadata" in diff_txt
                   or "using its sidecar JSONs" in diff_txt)
    check("I3: no silent reuse of the thermal metadata "
          "(backstop fired or child re-resolved)",
          backstop or re_resolved)
    check("I4: diffraction delegation succeeded",
          len(ledger) >= 2 and ok(ledger[1]))

    dl = (diff_txt + logged["diffraction"]).lower()
    check("I5: xrd_profile skill auto-selected for the diffraction run",
          "auto-selected domain skill(s): xrd_profile" in dl
          or "skill loaded: xrd_profile" in dl)
    check("I6: diffraction plan is diffraction, not thermogram",
          any(k in dl for k in ("2theta", "2-theta", "2θ",
                                "pseudo-voigt", "d-spacing", "diffraction",
                                "bragg")))
    # Scan ONLY this turn's captured stdout: the turn's logging includes the
    # child planner/validator prose which may legitimately reference the
    # prior thermal findings as context.
    ds = diff_txt.lower()
    check("I7: no thermal-analysis plan leaked into the diffraction run",
          "heat flow" not in ds and "heat-flow" not in ds
          and "j/g" not in ds and "enthalpy" not in ds)

    check("I8: spectroscopy delegation succeeded",
          len(ledger) >= 3 and ok(ledger[2]))
    check("I9: spectroscopy run re-resolved metadata again",
          child is not None
          and child.current_metadata_owner is not None
          and _same_dataset(child.current_metadata_owner, str(spectroscopy)))
    md_txt = json.dumps(child.current_metadata or {}) if child else ""
    check("I10: final child metadata carries nothing from the thermal doc",
          child is not None and not any(m in md_txt for m in markers))

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"MULTIMODAL META LIVE (#411): {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    return 0 if npass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
