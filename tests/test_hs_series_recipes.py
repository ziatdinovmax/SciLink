"""A hyperspectral series records one replayable recipe per regime (#734).

A series' recipes are read from ``locked_recipes``, which only the curve and
image drivers wrote, so a verified hyperspectral series carried no recipe:
nothing a later run could replay. The hyperspectral driver now records each
regime's recipe where it locks — the anchor unit, its verdict then, and its
``dynamic_analysis_records.json`` (a cube's recipe is a records file, not a
script) with the map gate's reference — so a copy of the records file under
its unit's own folder, by the name the agent reads it by, is a path
``prior_analysis_paths`` replays, held to the gate its sidecar records.

  conda run -n scilink python -m pytest tests/test_hs_series_recipes.py -q
"""
import json

from scilink.agents.exp_agents._verification_record import series_recipes

import test_hs_series as ths

PLAN = {"rationale": "edge shifts", "regimes": [
    {"name": "low_T", "dataset_indices": [0, 1, 2], "description": "L3 near 458 eV"},
    {"name": "high_T", "dataset_indices": [3, 4, 5], "description": "L3 shifted +3 eV"}],
    "transition_points": [{"between_indices": [2, 3], "description": "jump"}]}
META = {"variable": "temperature", "values": [300, 350, 400, 450, 500, 550], "unit": "K"}


def _two_regime_series(tmp_path, monkeypatch):
    calls = ths._Calls()
    ths._install_fake_pipeline(monkeypatch, calls)
    agent, out = ths._agent(tmp_path, calls, monkeypatch, plan=PLAN)
    res = agent.analyze(ths._cubes(tmp_path), system_info=dict(ths.AXIS), series_metadata=META)
    assert res["status"] == "success"
    return agent, res


def _copy_of(recipe, folder):
    """A copy of one regime's recipe as a later run names it: the records file
    under the unit's own folder, its gate and lock order in the sidecar."""
    unit_dir = folder / str(recipe["unit"])
    unit_dir.mkdir(parents=True)
    (unit_dir / recipe["file"]).write_text(recipe["script"])
    (unit_dir / "dynamic_analysis_records.recipe.json").write_text(json.dumps(
        {"regime": recipe["regime"], "unit": recipe["unit"], "index": recipe.get("index"),
         "quality_gate": recipe.get("gate")}))
    return str(unit_dir / recipe["file"])


def test_each_regime_records_its_recipe_where_it_locks(tmp_path, monkeypatch):
    _, res = _two_regime_series(tmp_path, monkeypatch)
    recs = res["locked_recipes"]
    assert set(recs) == {"low_T", "high_T"}
    assert recs["low_T"]["index"] == 0 and recs["high_T"]["index"] == 3
    for r in recs.values():
        assert r["file"] == "dynamic_analysis_records.json"
        assert isinstance(json.loads(r["script"]), list)                # the records file's own text
        assert r["verdict"]["verified"] is True
        assert r["gate"]["kind"] == "map_health" and r["gate"]["reference_maps"]
    assert [r["regime"] for r in series_recipes(res)] == ["low_T", "high_T"]


def test_each_regimes_recipe_is_one_replayable_copy(tmp_path, monkeypatch):
    agent, res = _two_regime_series(tmp_path, monkeypatch)
    recipes = series_recipes(res)
    assert len(recipes) == 2 and all(r["verified"] for r in recipes)
    paths = sorted(_copy_of(r, tmp_path / "copies") for r in recipes)
    assert all(p.endswith("/dynamic_analysis_records.json") for p in paths)
    assert len(set(paths)) == 2                                          # each regime in its unit's own folder
    # the copy is a recipe file the agent replays from (the records loader the reuse path calls)
    for p in paths:
        assert agent._load_prior_dynamic_records([p]), p
    # ...and the pick among the copies is by the lock order the sidecars carry, each held to its own gate
    rec = agent._prior_recipe([str(tmp_path / "copies")])
    assert rec["regime"] == "low_T" and rec["reference_maps"] == recipes[0]["gate"]["reference_maps"]
    [alt] = rec["alternatives"]
    assert alt["regime"] == "high_T" and alt["reference_maps"] == recipes[1]["gate"]["reference_maps"]


def test_a_replay_of_a_recipe_copy_is_held_to_the_recorded_gate(tmp_path, monkeypatch):
    """The issue's second consequence, "the gate is lost": a later run replays
    a copy of the recipe (``prior_analysis_paths=[copy]``) with no reference of
    its own. The copy's sidecar carries the map gate it was locked under, so a
    cube far outside the anchor's range is rejected, and one inside it certifies
    — without the sidecar, both replays ran on coverage alone."""
    import numpy as np
    import test_hs_locked_replay as tlr
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    records = [{"target": "peak position", "task_success": True, "required_outputs": ["Peak_Position"],
                "script": tlr.PEAK_SCRIPT, "quality_history": {"approved": True}}]
    recipe = {"regime": "A", "unit": "cube_0", "index": 0, "verified": True, "reason": "approved",
              "script": json.dumps(records), "file": "dynamic_analysis_records.json",
              "gate": {"kind": "map_health",
                       "reference_maps": {"Peak_Position": {"min": 640.0, "max": 662.0, "mean": 650.0,
                                                            "coverage": 1.0}}}}
    copy = _copy_of(recipe, tmp_path / "copies")

    def replay(center, name):
        np.save(tmp_path / f"{name}.npy", tlr._peak_cube(center=center, seed=3))
        return tlr._strict_agent(tmp_path, name).analyze(
            str(tmp_path / f"{name}.npy"), system_info=dict(tlr.AXIS_OK),
            prior_analysis_paths=[copy], reuse_locked_script=True, strict_replay=True)
    far = replay(820.0, "far")                                  # another material: outside the anchor's range
    assert far["status"] != "success" and not (far.get("verdict") or {}).get("verified")
    near = replay(652.0, "near")                                # the regime's own state
    assert near["status"] == "success" and near["verdict"]["verified"] is True
    assert near["verdict"]["interpretation_checked"] is True              # certified, not only verified
