"""A hyperspectral reuse replays ONE recipe, with its own gate (#751).

A reuse pointed at a folder used to collect the approved records of every
records file under it, so a series run folder (one per dataset) or the board's
folder of a series' regime copies replayed every regime's scripts as one
recipe, held to whichever map gate it found first. Now, as for curves and
images, a series run replays its first regime's locked recipe, a folder of
copies replays the first by the lock order the board recorded, a records file
and a single run are what they were, and the pick and why are on
``script_reuse.source``.

The end-to-end cases run a whole strict replay offline (no model call): a
merge shows as a second map, and a recipe held to another regime's gate fails.
"""
import json
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from test_hs_locked_replay import AXIS_OK, PEAK_SCRIPT, _peak_cube, _strict_agent

OTHER_SCRIPT = '''
def analyze_feature(data, axis):
    return {"maps": {"Other_Map": data.max(axis=2)}, "units": "a.u.", "description": "max"}
'''
# regime A's plausible range holds the frame (a peak at 660); regime B's does not
REF_A = {"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0, "coverage": 1.0}}
REF_B = {"Peak_Position": {"min": 400.0, "max": 420.0, "mean": 410.0, "coverage": 1.0}}


def _recs(script, target):
    return [{"target": target, "task_success": True, "script": script,
             "required_outputs": ["Peak_Position"] if script == PEAK_SCRIPT else ["Other_Map"],
             "quality_history": {"approved": True}}]


RECS_A, RECS_B = _recs(PEAK_SCRIPT, "peak position"), _recs(OTHER_SCRIPT, "max map")


def _series_dir(tmp_path):
    """A two-regime series run: each dataset's run folder, and the locked
    recipes the driver recorded (A anchored on dataset 0, B on dataset 3)."""
    d = tmp_path / "series"
    for idx, recs in ((0, RECS_A), (1, RECS_A), (3, RECS_B), (4, RECS_B)):
        (d / f"dataset_{idx:04d}").mkdir(parents=True)
        (d / f"dataset_{idx:04d}" / "dynamic_analysis_records.json").write_text(json.dumps(recs))

    def lock(regime, idx, recs, ref):
        return {"unit": f"cube_{idx}", "index": idx, "regime": regime, "file": "dynamic_analysis_records.json",
                "script": json.dumps(recs), "verdict": {"verified": True},
                "gate": {"kind": "map_health", "reference_maps": ref},
                "certification_reference": {"kind": "maps", "reference_maps": ref}}
    # recorded B first: the choice is the lock order (the anchors' index), not the dict's
    (d / "analysis_results.json").write_text(json.dumps({"status": "success", "locked_recipes": {
        "B": lock("B", 3, RECS_B, REF_B), "A": lock("A", 0, RECS_A, REF_A)}}))
    return d


def _board_copies(tmp_path, with_index=True):
    """The board's folder of a series' regime copies: one unit folder per
    regime, each with its sidecar. By name B's folder comes first; by the lock
    order the board recorded, A's."""
    d = tmp_path / "recipes" / "03_series" / "a1"
    for unit, regime, idx, recs, ref in (("b_unit", "B", 3, RECS_B, REF_B), ("z_unit", "A", 0, RECS_A, REF_A)):
        (d / unit).mkdir(parents=True)
        (d / unit / "dynamic_analysis_records.json").write_text(json.dumps(recs))
        side = {"regime": regime, "unit": unit, "quality_gate": {"kind": "map_health", "reference_maps": ref}}
        if with_index:
            side["index"] = idx
        (d / unit / "dynamic_analysis_records.recipe.json").write_text(json.dumps(side))
    return d


def _replay(tmp_path, paths, name="frame"):
    np.save(tmp_path / "cube.npy", _peak_cube(center=660.0, seed=1))
    return _strict_agent(tmp_path, name).analyze(
        str(tmp_path / "cube.npy"), system_info=dict(AXIS_OK),
        prior_analysis_paths=[str(p) for p in paths], reuse_locked_script=True, strict_replay=True)


def _maps(res):
    return {f["name"] for f in res.get("extracted_features") or []}


def test_a_series_run_replays_its_first_regimes_locked_recipe_with_its_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    res = _replay(tmp_path, [_series_dir(tmp_path)])
    assert res["status"] == "success", res.get("error")
    assert _maps(res) == {"Peak_Position"}                       # regime B's script did not run
    assert res["script_reuse"]["n_replayed"] == 1
    src = res["script_reuse"]["source"]
    assert "regime A" in src and "cube_0" in src and "1 of 2 in lock order" in src
    assert res["script_reuse"]["regime_choice"]["chosen"] == 1


def test_a_board_folder_of_regime_copies_replays_the_first_by_lock_order(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    res = _replay(tmp_path, [_board_copies(tmp_path)])
    # A's records held to A's gate: B's gate would reject the frame, B's script
    # would add its own map
    assert res["status"] == "success", res.get("error")
    assert _maps(res) == {"Peak_Position"}
    assert "regime A (unit z_unit)" in res["script_reuse"]["source"]
    assert res["script_reuse"]["recipe_path"].endswith("z_unit/dynamic_analysis_records.json")


def test_copies_with_no_recorded_order_replay_the_first_by_name_and_its_own_gate(tmp_path):
    ag = _strict_agent(tmp_path, "x")
    rec = ag._prior_recipe([_board_copies(tmp_path, with_index=False)])
    assert rec["records"] == RECS_B and rec["reference_maps"] == REF_B      # chosen together
    assert "1 of 2 in lock order" in rec["source"] and "never a merge" in rec["source"]
    # the other copy is an alternative, held to ITS gate if the first fails
    [alt] = rec["alternatives"]
    assert alt["records"] == RECS_A and alt["reference_maps"] == REF_A


REF_B2 = {"Peak_Position": {"min": 698.0, "max": 722.0, "mean": 710.0, "coverage": 1.0}}


def _same_method_series(tmp_path):
    """Two regimes measuring the same quantity with the same script, their
    gates holding different ranges (the band moved between the regimes)."""
    d = tmp_path / "series2"
    d.mkdir()
    lock = lambda regime, idx, ref: {
        "unit": f"cube_{idx}", "index": idx, "regime": regime, "file": "dynamic_analysis_records.json",
        "script": json.dumps(RECS_A), "verdict": {"verified": True},
        "gate": {"kind": "map_health", "reference_maps": ref},
        "certification_reference": {"kind": "maps", "reference_maps": ref}}
    (d / "analysis_results.json").write_text(json.dumps({"status": "success", "locked_recipes": {
        "A": lock("A", 0, REF_A), "B": lock("B", 3, REF_B2)}}))
    return d


@pytest.mark.parametrize("center, kept, verified", [
    (660.0, "A", True),          # regime A's frame: A, as before
    (710.0, "B", True),          # regime B's: A's gate rejects it, B's own holds it
    (800.0, "A", False),         # neither's: the first kept, not verified
])
def test_a_reuse_falls_through_the_regimes_on_their_own_gates(tmp_path, monkeypatch, center, kept, verified):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    np.save(tmp_path / "cube.npy", _peak_cube(center=center, seed=1))
    res = _strict_agent(tmp_path, "frame").analyze(
        str(tmp_path / "cube.npy"), system_info=dict(AXIS_OK),
        prior_analysis_paths=[str(_same_method_series(tmp_path))], reuse_locked_script=True, strict_replay=True)
    sr = res["script_reuse"]
    assert sr["regime_choice"]["chosen_regime"] == kept and f"regime {kept}" in sr["source"]
    assert bool((res.get("verdict") or {}).get("verified")) is verified
    assert [t["regime"] for t in sr["regime_choice"]["tried"]] == (["A"] if kept == "A" and verified else ["A", "B"])
    if verified:
        assert {f["name"] for f in res["extracted_features"]} == {"Peak_Position"}
    # the kept run is this run, at this run's folder
    assert res.get("output_directory") == str(tmp_path / "frame")
    if verified:
        assert (tmp_path / "frame" / "analysis_results.json").is_file()


def test_of_several_paths_the_first_is_replayed_and_the_rest_named(tmp_path):
    copies = _board_copies(tmp_path)
    b_file = copies / "b_unit" / "dynamic_analysis_records.json"
    a_file = copies / "z_unit" / "dynamic_analysis_records.json"
    rec = _strict_agent(tmp_path, "x")._prior_recipe([b_file, a_file])
    assert rec["records"] == RECS_B and rec["reference_maps"] == REF_B
    assert f"not replayed: {a_file}" in rec["source"]


def test_a_records_file_and_a_single_run_are_what_they_were(tmp_path):
    ag = _strict_agent(tmp_path, "x")
    run = tmp_path / "run"
    run.mkdir()
    (run / "dynamic_analysis_records.json").write_text(json.dumps(RECS_A + [dict(RECS_B[0], task_success=False)]))
    for p in (run, run / "dynamic_analysis_records.json"):
        rec = ag._prior_recipe([p])
        assert rec["records"] == ag._load_prior_dynamic_records([p]) == RECS_A
    assert ag._prior_recipe([run])["source"] is None              # nothing to say beyond the run
    assert "records file named" in ag._prior_recipe([run / "dynamic_analysis_records.json"])["source"]
    # a session folder holding one run under results/ is still found
    sess = tmp_path / "sess" / "results" / "analysis_1"
    sess.mkdir(parents=True)
    (sess / "dynamic_analysis_records.json").write_text(json.dumps(RECS_A))
    assert ag._prior_recipe([tmp_path / "sess"])["records"] == RECS_A


def test_nothing_approved_is_still_refused(tmp_path):
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "dynamic_analysis_records.json").write_text(json.dumps([dict(RECS_A[0], task_success=False)]))
    np.save(tmp_path / "cube.npy", _peak_cube())
    res = _strict_agent(tmp_path, "f").analyze(str(tmp_path / "cube.npy"), system_info=dict(AXIS_OK),
                                               prior_analysis_paths=[str(prior)], reuse_locked_script=True)
    assert res["status"] == "error" and "No approved prior script" in res["error"]["error"]


def test_the_board_records_each_regime_copys_lock_order(tmp_path):
    from scilink.agents.meta_agent import board as board_mod
    b = board_mod.Board(tmp_path)
    rec = {"agent_name": "HyperspectralAnalysisAgent", "recipes": [
        {"unit": "cube_0", "regime": "A", "index": 0, "script": json.dumps(RECS_A), "verified": True,
         "file": "dynamic_analysis_records.json", "gate": {"kind": "map_health", "reference_maps": REF_A}}]}
    [spec] = board_mod._recipe_specs("a1", rec)
    spec = board_mod._materialize_recipe(b, {"index": 3, "label": "series"}, spec)
    side = json.loads((tmp_path / "swarm" / "recipes" / "03_series" / "a1" / "cube_0"
                       / "dynamic_analysis_records.recipe.json").read_text())
    assert side["index"] == 0 and side["regime"] == "A"
