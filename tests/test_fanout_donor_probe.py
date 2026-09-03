"""Harmonized fan-out donor probe recognises every foundation agent's
approved-script layout (hyperspectral records, image/curve locked scripts)."""
import json
from pathlib import Path

import pytest

from scilink.agents.meta_agent.fanout import find_donor_reuse_dir


def _run_dir(root: Path, name: str, status="success", scripts=(), records=None,
             candidates=False) -> Path:
    d = root / "results" / name
    if candidates:
        d = d / "_candidates" / "cand_01"
    d.mkdir(parents=True, exist_ok=True)
    (d / "analysis_results.json").write_text(json.dumps({"status": status}))
    if scripts:
        (d / "scripts").mkdir(exist_ok=True)
        for s in scripts:
            (d / "scripts" / s).write_text("print('hi')\n")
    if records is not None:
        (d / "dynamic_analysis_records.json").write_text(json.dumps(records))
    return d


def test_missing_donor_dir(tmp_path):
    assert find_donor_reuse_dir(tmp_path / "nope") == (None, None)


def test_hyperspectral_records_still_found(tmp_path):
    d = _run_dir(tmp_path, "hs_run", records=[{"script": "x=1", "task_success": True}])
    assert find_donor_reuse_dir(tmp_path) == (d, "records")


def test_hyperspectral_rejected_records_do_not_fall_back_to_script(tmp_path):
    # A hyperspectral run whose every record was rejected must NOT be revived
    # through its scripts/ folder: the records are the authority there.
    _run_dir(tmp_path, "hs_run", scripts=("analysis_script.py",),
             records=[{"script": "x=1", "task_success": False}])
    assert find_donor_reuse_dir(tmp_path) == (None, None)


def test_image_locked_script_found(tmp_path):
    d = _run_dir(tmp_path, "img_run_ImageAnalysis_001", scripts=("analysis_script.py",))
    assert find_donor_reuse_dir(tmp_path) == (d, "script")


def test_curve_locked_script_found(tmp_path):
    d = _run_dir(tmp_path, "curve_run_CurveFit_001", scripts=("fitting_script.py",))
    assert find_donor_reuse_dir(tmp_path) == (d, "script")


def test_series_per_item_scripts_found(tmp_path):
    d = _run_dir(tmp_path, "series_run", scripts=("spectrum_a.py", "spectrum_b.py"))
    assert find_donor_reuse_dir(tmp_path) == (d, "script")


def test_failed_run_with_script_is_not_replayable(tmp_path):
    _run_dir(tmp_path, "img_run", status="error", scripts=("analysis_script.py",))
    assert find_donor_reuse_dir(tmp_path) == (None, None)


def test_success_without_script_is_not_replayable(tmp_path):
    _run_dir(tmp_path, "img_run")
    assert find_donor_reuse_dir(tmp_path) == (None, None)


def test_best_of_n_candidate_dirs_are_ignored(tmp_path):
    _run_dir(tmp_path, "img_run", scripts=("analysis_script.py",), candidates=True)
    assert find_donor_reuse_dir(tmp_path) == (None, None)


def test_records_preferred_over_script_and_newest_script_wins(tmp_path):
    import os, time
    old = _run_dir(tmp_path, "img_old", scripts=("analysis_script.py",))
    new = _run_dir(tmp_path, "img_new", scripts=("analysis_script.py",))
    t = time.time()
    os.utime(old / "analysis_results.json", (t - 100, t - 100))
    os.utime(new / "analysis_results.json", (t, t))
    assert find_donor_reuse_dir(tmp_path) == (new, "script")
    hs = _run_dir(tmp_path, "hs_run", records=[{"script": "x", "task_success": True}])
    assert find_donor_reuse_dir(tmp_path) == (hs, "records")
