"""describe_feature_table: schema summary that travels with a feature-table
path (run_analysis response, run_task result, meta ledger) so callers that
cannot open server files can pick BO inputs/targets and see the holes.

#534: write_feature_table merges per-unit key-name variants of one quantity
(``metric`` vs ``metric_value``) into one column, and describe_feature_table
warns about a column empty for SOME units / a pair of complementary columns —
the two shapes that silently dropped a unit from BO."""
import json
import tempfile
from pathlib import Path

from scilink.agents.exp_agents.feature_table import (
    describe_feature_table, write_feature_table, merge_variant_columns,
)


def test_columns_rows_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "features.csv"
        p.write_text("unit,T,A400,n\nw1,300,0.5,1.1\nw2,400,,1.2\nw3,500,0.7,nan\n")
        d = describe_feature_table(p)
        assert d["columns"] == ["unit", "T", "A400", "n"]
        assert d["n_rows"] == 3
        assert d["missing"] == {"A400": 1, "n": 1}
        # Partial columns are named with the units they are empty for.
        assert len(d["warnings"]) == 2
        assert "'A400' is empty for 1 of 3 units (w2)" in d["warnings"][0]
        assert "'n' is empty for 1 of 3 units (w3)" in d["warnings"][1]


def test_short_rows_count_as_missing_and_no_missing_key_when_clean():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.csv"
        p.write_text("a,b\n1,2\n3\n")
        assert describe_feature_table(p)["missing"] == {"b": 1}
        p.write_text("a,b\n1,2\n")
        assert describe_feature_table(p) == {"columns": ["a", "b"], "n_rows": 1,
                                             "missing": {}, "warnings": []}


def test_never_raises():
    assert describe_feature_table("/nonexistent/x.csv") is None
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "empty.csv"
        p.write_text("")
        assert describe_feature_table(p) is None


def test_fully_empty_column_is_missing_but_not_partial_warning():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.csv"
        p.write_text("unit,x,y\nu1,1,\nu2,2,\n")
        d = describe_feature_table(p)
        assert d["missing"] == {"y": 2}
        assert d["warnings"] == []  # all-empty is a missing column, not a split


def test_complementary_columns_flagged_as_split_quantity():
    """The #534 shape: 8 units under one name, 1 under a variant."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.csv"
        rows = ["unit,T,metric_value,metric"]
        rows += [f"u{i},{300 + i},{0.1 * i:.2f}," for i in range(8)]
        rows.append("u8,308,,0.99")
        p.write_text("\n".join(rows) + "\n")
        d = describe_feature_table(p)
        assert d["missing"] == {"metric_value": 1, "metric": 8}
        split = [w for w in d["warnings"] if "complementary" in w]
        assert len(split) == 1
        assert "'metric_value' (8 units)" in split[0]
        assert "'metric' (1 units)" in split[0]
        # The per-column warning names the dropped unit.
        assert any("'metric_value' is empty for 1 of 9 units (u8)" in w
                   for w in d["warnings"])


def test_overlapping_partial_columns_not_called_complementary():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.csv"
        p.write_text("unit,a,b\nu1,1,\nu2,2,5\nu3,,6\n")
        d = describe_feature_table(p)
        assert not any("complementary" in w for w in d["warnings"])
        assert len(d["warnings"]) == 2


# ---------------------------------------------------------------- merge --

def _series_run(tmp, params_by_unit):
    out = Path(tmp) / "run"
    out.mkdir()
    results = [{"index": i, "name": name, "success": True, "parameters": params,
                "fit_quality": {"r_squared": 0.99}}
               for i, (name, params) in enumerate(params_by_unit.items())]
    (out / "series_fit_results.json").write_text(json.dumps({"results": results}))
    return out


def _read_csv(path):
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_variant_keys_merge_into_majority_name():
    with tempfile.TemporaryDirectory() as tmp:
        params = {f"u{i}": {"metric_value": 0.1 * i, "width": 2.0} for i in range(8)}
        params["u8"] = {"metric": 0.99, "width": 2.5}
        out = _series_run(tmp, params)
        path = write_feature_table(out)
        rows = _read_csv(path)
        header = list(rows[0].keys())
        assert "metric" not in header and "metric_value" in header
        assert header == ["unit", "metric_value", "width", "fit_r_squared"]
        by_unit = {r["unit"]: r for r in rows}
        assert by_unit["u8"]["metric_value"] == "0.99"
        assert all(r["metric_value"] != "" for r in rows)
        d = describe_feature_table(path)
        assert d["missing"] == {} and d["warnings"] == []


def test_variant_keys_tie_keeps_first_seen_name():
    with tempfile.TemporaryDirectory() as tmp:
        out = _series_run(tmp, {"u0": {"Peak_Area_val": 1.0},
                                "u1": {"peak_area": 2.0}})
        rows = _read_csv(write_feature_table(out))
        assert list(rows[0].keys()) == ["unit", "Peak_Area_val", "fit_r_squared"]
        assert [r["Peak_Area_val"] for r in rows] == ["1.0", "2.0"]


def test_both_populated_in_a_unit_is_not_merged():
    with tempfile.TemporaryDirectory() as tmp:
        out = _series_run(tmp, {"u0": {"x": 1.0, "x_value": 10.0},
                                "u1": {"x": 2.0}})
        rows = _read_csv(write_feature_table(out))
        assert set(rows[0].keys()) == {"unit", "x", "x_value", "fit_r_squared"}
        assert rows[0]["x"] == "1.0" and rows[0]["x_value"] == "10.0"


def test_statistics_suffixes_are_not_variants():
    with tempfile.TemporaryDirectory() as tmp:
        out = _series_run(tmp, {"u0": {"fwhm_mean": 1.0},
                                "u1": {"fwhm": 2.0}})
        rows = _read_csv(write_feature_table(out))
        assert set(rows[0].keys()) == {"unit", "fwhm_mean", "fwhm", "fit_r_squared"}


def test_merge_notes_report_moved_units():
    rows = [{"unit": "a", "m_value": 1.0}, {"unit": "b", "m": 2.0},
            {"unit": "c", "m_value": 3.0}]
    notes = merge_variant_columns(rows)
    assert notes == [{"column": "m_value", "merged_from": ["m"], "units": ["b"]}]
    assert rows[1] == {"unit": "b", "m_value": 2.0}
    assert merge_variant_columns([{"unit": "a", "x": 1}]) == []
