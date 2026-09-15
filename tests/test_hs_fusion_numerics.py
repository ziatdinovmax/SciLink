"""Offline tests for the hyperspectral features-to-fusion chain.

Hyperspectral branches previously could never feed the quantitative fusion:
the agent kept its numeric results in memory only, so the shared feature-
table writer found nothing, run_task collected no feature_tables, and the
fusion numerics bundle saw the branch as table-less. These tests cover the
new link end to end: _write_results_file flattening -> analysis_results.json
-> write_feature_table -> features.csv -> _branch_numerics preview.

  conda run -n scilink python tests/test_hs_fusion_numerics.py
"""
import csv
import json
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

results = {}
LOG = logging.getLogger("hs_fusion_test")


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _Host:
    """Just enough of the agent for _write_results_file."""

    def __init__(self, out):
        self.output_dir = Path(out)
        self.logger = LOG


def _write(response, out):
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    return HyperspectralAnalysisAgent._write_results_file(
        _Host(out), response)


RESPONSE = {
    "status": "success",
    "extracted_features": [
        {"name": "Peak_Position", "units": "nm",
         "description": "per-pixel emission center",
         "stats": {"min": 601.2, "max": 638.9, "mean": 620.05}},
        {"name": "Peak_FWHM", "units": "a.u.",
         "stats": {"min": 30.0, "max": 42.0, "mean": 35.5}},
        {"not_measurable": {"feature": "second UV peak",
                            "evidence": "prominence 0.01 vs sigma 0.4"}},
        {"name": "weird", "stats": {"mean": float("nan")}},   # non-finite
        "not-a-dict",                                          # junk entry
    ],
}


def main():
    print("1) _write_results_file flattening:")
    d = tempfile.mkdtemp(prefix="hsft_")
    p = _write(RESPONSE, d)
    check("file written", p is not None and Path(p).name == "analysis_results.json")
    data = json.loads(Path(p).read_text())
    feats = data["extracted_features"]
    check("stats flattened with units in the column",
          feats.get("Peak_Position_mean_nm") == 620.05
          and feats.get("Peak_Position_min_nm") == 601.2)
    check("a.u. units omitted from the column name",
          feats.get("Peak_FWHM_mean") == 35.5)
    check("judged null recorded as data, not dropped",
          feats.get("second_UV_peak_not_measurable") == 1)
    check("non-finite and junk entries skipped",
          not any(k.startswith("weird") for k in feats))

    d2 = tempfile.mkdtemp(prefix="hsft_empty_")
    check("no numeric features -> no file (no empty tables downstream)",
          _write({"status": "success", "extracted_features": []}, d2) is None
          and not (Path(d2) / "analysis_results.json").exists())

    print("2) shared feature-table writer picks it up:")
    from scilink.agents.exp_agents.feature_table import write_feature_table
    ft = write_feature_table(d)
    check("features.csv emitted from the HS results file",
          ft is not None and Path(ft).name == "features.csv")
    with open(ft) as fh:
        rows = list(csv.DictReader(fh))
    check("one row with the flattened columns",
          len(rows) == 1 and rows[0]["Peak_Position_mean_nm"] == "620.05"
          and rows[0]["second_UV_peak_not_measurable"] == "1")

    print("3) fusion numerics bundle reads the table:")
    from scilink.agents.meta_agent.fanout import _branch_numerics
    entry = {"label": "emission map", "feature_tables": [ft],
             "files_produced": [str(Path(d) / "analysis_results.json")]}
    num = _branch_numerics(entry, join_axis=None)
    check("branch registers numerics",
          isinstance(num, dict) and num.get("feature_tables"))
    prev = json.dumps(num, default=str)
    check("preview carries the physical columns",
          "Peak_Position_mean_nm" in prev)

    check("table-less branch still degrades to None (guardrail intact)",
          _branch_numerics({"label": "x", "files_produced": []}, None) is None)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"HS FUSION NUMERICS: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
