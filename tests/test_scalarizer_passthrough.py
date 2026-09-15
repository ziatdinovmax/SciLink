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
