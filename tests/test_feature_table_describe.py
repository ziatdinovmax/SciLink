"""describe_feature_table: schema summary that travels with a feature-table
path (run_analysis response, run_task result, meta ledger) so callers that
cannot open server files can pick BO inputs/targets and see the holes."""
import tempfile
from pathlib import Path

from scilink.agents.exp_agents.feature_table import describe_feature_table


def test_columns_rows_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "features.csv"
        p.write_text("unit,T,A400,n\nw1,300,0.5,1.1\nw2,400,,1.2\nw3,500,0.7,nan\n")
        d = describe_feature_table(p)
        assert d == {"columns": ["unit", "T", "A400", "n"], "n_rows": 3,
                     "missing": {"A400": 1, "n": 1}}


def test_short_rows_count_as_missing_and_no_missing_key_when_clean():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.csv"
        p.write_text("a,b\n1,2\n3\n")
        assert describe_feature_table(p)["missing"] == {"b": 1}
        p.write_text("a,b\n1,2\n")
        assert describe_feature_table(p) == {"columns": ["a", "b"], "n_rows": 1,
                                             "missing": {}}


def test_never_raises():
    assert describe_feature_table("/nonexistent/x.csv") is None
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "empty.csv"
        p.write_text("")
        assert describe_feature_table(p) is None
