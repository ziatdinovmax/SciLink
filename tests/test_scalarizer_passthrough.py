"""Offline tests for the scalarizer table pass-through + row-count trap
(#366): requested quantities that are already table columns are READ, not
re-derived; genuine derivations still take the codegen path; a metric equal
to the row count while a same-named column disagrees is caught.

  conda run -n scilink python tests/test_scalarizer_passthrough.py
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pandas as pd

from scilink.agents.planning_agents.scalarizer_agent import ScalarizerAgent

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _Host:
    state: dict
    def __init__(self):
        self.state = {}
    def _log_action(self, **kw):
        self.state.setdefault("log", []).append(kw["action"])
    _norm_tokens = staticmethod(ScalarizerAgent._norm_tokens)
    _DERIVATION_TERMS = ScalarizerAgent._DERIVATION_TERMS
    _load_flat_table = ScalarizerAgent._load_flat_table
    _try_table_passthrough = ScalarizerAgent._try_table_passthrough
    _rowcount_suspects = staticmethod(ScalarizerAgent._rowcount_suspects)


def _csv(d, name, text):
    p = Path(d) / name
    p.write_text(text)
    return str(p)


def main():
    h = _Host()
    d = tempfile.mkdtemp(prefix="scalpt_")

    hs = _csv(d, "features.csv",
              "unit,Peak_Position_mean_nm,Peak_FWHM_mean\n"
              "emission_map,620.05,35.5\n")
    img = _csv(d, "img_features.csv",
               "unit,particle_count,diameter_mean_nm\nimage_0000,8,11.75\n")
    tidy = _csv(d, "tidy.csv",
                "temperature_C,pH,product_area,byproduct_area\n"
                "15.18,2.38,7.12,21.10\n15.45,4.11,11.91,22.97\n")

    print("1) loader:")
    check("features table parses", h._load_flat_table(hs) is not None)
    raw = _csv(d, "spectrum.csv", "400.0,0.11\n400.5,0.12\n401.0,0.13\n")
    check("headerless raw spectrum -> None (codegen path)",
          h._load_flat_table(raw) is None)
    check("missing file -> None", h._load_flat_table("/nope.csv") is None)

    print("2) explicit-schema pass-through:")
    ctx = {"_schema_requirements": {"input_columns": ["temperature_C", "pH"],
                                    "target_columns": ["product_area"]}}
    r = h._try_table_passthrough(h._load_flat_table(tidy), "optimize", ctx, None)
    check("all schema columns present -> pass-through, values read",
          r is not None and r["passthrough"]
          and r["metrics"]["product_area"] == [7.12, 11.91]
          and r["metrics"]["temperature_C"] == [15.18, 15.45])
    check("column_roles carried from schema",
          r["column_roles"]["targets"] == ["product_area"])
    check("no 'unit' column -> no units on the result", "units" not in r)
    sib = _csv(d, "siblings.csv",
               "unit,metric_value,metric\nu0,0.5,\nu1,,0.9\n")
    r_sib = h._try_table_passthrough(
        h._load_flat_table(sib), "x",
        {"_schema_requirements": {"target_columns": ["metric"]}}, None)
    check("exact column name wins over a token-subset sibling (#534)",
          r_sib is not None and r_sib["metrics"]["metric"] == [None, 0.9])
    r_sib2 = h._try_table_passthrough(
        h._load_flat_table(sib), "x",
        {"_schema_requirements": {"target_columns": ["Metric_Value"]}}, None)
    check("case-insensitive exact match",
          r_sib2 is not None and r_sib2["metrics"]["Metric_Value"] == [0.5, None])
    r_hs = h._try_table_passthrough(
        h._load_flat_table(hs), "x",
        {"_schema_requirements": {"target_columns": ["Peak_FWHM_mean"]}}, None)
    check("feature table -> row identities carried as 'units' (#534)",
          r_hs is not None and r_hs["units"] == ["emission_map"])
    ctx2 = {"_schema_requirements": {"input_columns": ["temperature_C", "pH"],
                                     "target_columns": ["selectivity"]}}
    check("derived target (no matching column) -> codegen path",
          h._try_table_passthrough(h._load_flat_table(tidy),
                                   "maximize selectivity", ctx2, None) is None)

    print("3) goal-derived pass-through (feature tables only):")
    r = h._try_table_passthrough(
        h._load_flat_table(img),
        "Extract the particle count and mean diameter as scalar metrics.",
        None, None)
    check("one-row image table: column VALUES returned (not the row count)",
          r is not None and r["metrics"]["particle_count"] == 8.0
          and r["metrics"]["diameter_mean_nm"] == 11.75)
    r = h._try_table_passthrough(
        h._load_flat_table(hs),
        "Extract the emission peak position as a scalar metric.", None, None)
    check("one-row HS table: peak position read directly",
          r is not None and r["metrics"]["Peak_Position_mean_nm"] == 620.05)
    check("no 'unit' identity column -> goal matching stays off",
          h._try_table_passthrough(h._load_flat_table(tidy),
                                   "extract the product area", None, None)
          is None)
    check("derivation language blocks goal pass-through",
          h._try_table_passthrough(h._load_flat_table(img),
                                   "ratio of particle count to area",
                                   None, None) is None)
    # #535: an objective NAMED after a derivation term ('yield',
    # 'selectivity', 'rate', ...) that is a literal column is read, not
    # vetoed — a column that exists is read regardless of its name.
    camp = _csv(d, "campaign_features.csv",
                "unit,temperature_C,yield\nrun_07,62.5,0.81\n")
    r = h._try_table_passthrough(
        h._load_flat_table(camp),
        "Research objective: maximize yield of the coupling step.\n\n"
        "Extract the temperature and yield for this run.", None, None)
    check("goal naming a literal 'yield' column -> pass-through fires",
          r is not None and r["metrics"]["yield"] == 0.81
          and r["metrics"]["temperature_C"] == 62.5)
    check("derivation term absent from the columns still vetoes",
          h._try_table_passthrough(
              h._load_flat_table(camp),
              "Research objective: maximize yield.\n\nCompute the rate of "
              "yield change with temperature", None, None) is None)

    print("4) row-count trap:")
    df1 = h._load_flat_table(img)
    check("row-count metric flagged when the column disagrees",
          h._rowcount_suspects({"Particle_Count": 1.0}, df1)
          == ["Particle_Count"])
    check("correct value not flagged",
          h._rowcount_suspects({"Particle_Count": 8.0}, df1) == [])
    check("unrelated metric not flagged",
          h._rowcount_suspects({"Selectivity": 1.0}, df1) == [])
    check("no table -> no false alarms",
          h._rowcount_suspects({"x": 1.0}, None) == [])

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"SCALARIZER PASSTHROUGH: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
