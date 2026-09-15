"""Offline tests for glob-pattern `data_path` support in run_analysis.

A directory can hold several distinct datasets (e.g. two in-situ series in one
upload folder); a glob selects ONE of them. This must work WITHOUT changing how
an existing directory or single-file path behaves.

  conda run -n scilink python tests/test_run_analysis_glob.py
"""
import os
import tempfile

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import json
from pathlib import Path

import numpy as np

from scilink.agents.exp_agents.analysis_orchestrator_tools import (
    _is_glob, _resolve_glob_files, _detect_sidecar_jsons,
)

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    d = tempfile.mkdtemp()
    # One directory, TWO distinct series + a standalone file + sidecars.
    for i, t in enumerate([35, 40, 50]):
        np.savetxt(os.path.join(d, f"ftir_{t}C.txt"), np.zeros((4, 2)))
        Path(d, f"ftir_{t}C.json").write_text(json.dumps({"temperature_C": t}))
    for i, t in enumerate([28, 30]):
        np.savetxt(os.path.join(d, f"xrd_{t}C.txt"), np.ones((4, 2)))
        Path(d, f"xrd_{t}C.json").write_text(json.dumps({"temperature_C": t}))
    np.savetxt(os.path.join(d, "dsc.txt"), np.full((4, 2), 2.0))
    Path(d, "metadata.json").write_text("{}")

    print("1) _is_glob discrimination (no behavior change for real paths):")
    check("existing directory is NOT a glob", not _is_glob(d))
    check("existing file is NOT a glob", not _is_glob(os.path.join(d, "dsc.txt")))
    check("pattern IS a glob", _is_glob(os.path.join(d, "ftir_*.txt")))
    check("'?' pattern IS a glob", _is_glob(os.path.join(d, "xrd_?0C.txt")))
    check("non-existent plain path is NOT a glob (stays an error downstream)",
          not _is_glob(os.path.join(d, "nope.txt")))
    check("absurdly long inline string is NOT a glob (no OSError)",
          not _is_glob("some prose description * with a star " * 50))

    print("2) _resolve_glob_files selects ONE series out of the shared dir:")
    data_files, all_files = _resolve_glob_files(os.path.join(d, "ftir_*.txt"))
    names = sorted(f.name for f in data_files)
    check("only the FTIR series matched",
          names == ["ftir_35C.txt", "ftir_40C.txt", "ftir_50C.txt"])
    check("no XRD/DSC file leaked in",
          not any("xrd" in n or "dsc" in n for n in names))
    check("stem-matched sidecars pulled in for series metadata",
          sorted(f.name for f in all_files if f.suffix == ".json") ==
          ["ftir_35C.json", "ftir_40C.json", "ftir_50C.json"])
    smap, _ = _detect_sidecar_jsons(data_files, all_files)
    check("sidecar map resolves for the globbed subset", len(smap) == 3)

    print("3) metadata files excluded from a glob that would match them:")
    dfiles, _ = _resolve_glob_files(os.path.join(d, "*.txt"))
    check("all .txt data files matched (none are metadata)", len(dfiles) == 6)
    dfiles2, _ = _resolve_glob_files(os.path.join(d, "*"))
    check("'*' excludes .json/metadata files",
          all(f.suffix.lower() != ".json" for f in dfiles2)
          and "metadata.json" not in [f.name for f in dfiles2])

    print("4) empty match is reported, not silently treated as a file:")
    empty, _ = _resolve_glob_files(os.path.join(d, "nomatch_*.txt"))
    check("no match -> empty list (caller errors)", empty == [])

    # examine_data must ACCEPT a glob too: an agent handed a pattern examines
    # the pattern, and is told to keep it (not fall back to the parent dir).
    print("5) examine_data accepts a glob (through the real tool):")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)
    bd = tempfile.mkdtemp()
    ag = AnalysisOrchestratorAgent(base_dir=bd, api_key="sk-dummy",
                                   model_name="claude-opus-4-6",
                                   restore_checkpoint=False,
                                   analysis_mode=AnalysisMode.AUTONOMOUS)
    ex = ag.tools.functions_map["examine_data"]
    r = json.loads(ex(os.path.join(d, "ftir_*.txt")))
    check("glob examined successfully", r.get("status") == "success")
    check("reported as a pattern", r.get("is_pattern") is True)
    check("only this series' files counted", r.get("series_count") == 3)
    check("sidecars of the globbed subset detected",
          len(r.get("sidecar_json_files") or []) == 3)
    check("hint tells the agent to keep the pattern",
          "NOT the parent directory" in (r.get("pattern_hint") or ""))
    rdir = json.loads(ex(d))
    check("plain directory still examined as before (all 6 data files)",
          rdir.get("status") == "success" and rdir.get("series_count") == 6
          and not rdir.get("is_pattern"))
    rfile = json.loads(ex(os.path.join(d, "dsc.txt")))
    check("single file still examined as before",
          rfile.get("status") == "success" and not rfile.get("is_directory"))
    rbad = json.loads(ex(os.path.join(d, "nomatch_*.txt")))
    check("non-matching pattern -> clean error", rbad.get("status") == "error")

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"RUN_ANALYSIS GLOB: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
