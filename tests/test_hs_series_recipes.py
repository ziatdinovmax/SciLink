"""A hyperspectral series posts one recipe per regime to the board (#734).

The board reads a series' recipes from ``locked_recipes``, which only the curve
and image drivers wrote, so a verified hyperspectral series posted claims but
no recipe: nothing a later swarm item could replay. The hyperspectral driver
now records each regime's recipe where it locks — the anchor unit, its verdict
then, and its ``dynamic_analysis_records.json`` (a cube's recipe is a records
file, not a script) with the map gate's reference — and the board copies each
under its unit's own folder by the name the agent reads it by, so the copy is
a path ``prior_analysis_paths`` replays.

  conda run -n scilink python -m pytest tests/test_hs_series_recipes.py -q
"""
import json

from scilink.agents.exp_agents._verification_record import series_recipes
from scilink.agents.meta_agent import board as board_mod

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


def test_the_board_posts_one_replayable_copy_per_regime(tmp_path, monkeypatch):
    agent, res = _two_regime_series(tmp_path, monkeypatch)
    row = {"analysis_id": "hs1", "agent_name": "HyperspectralAnalysisAgent", "status": "success",
           "verified": True, "reason": "every unit verified", "decided_by": "qc_gate", "series": True,
           "recipes": series_recipes(res), "ungated_outputs": []}
    result = {"status": "success", "analyses": [row], "key_findings": []}
    entry = {"index": 1, "label": "cubes", "mode": "analysis", "status": "success"}
    b = board_mod.Board(tmp_path / "swarm")
    board_mod.post_delegation(b, entry, result)
    recipes = [r for r in b.records() if r.get("kind") == "recipe"]
    assert len(recipes) == 2 and all(r["status"] == "verified" for r in recipes)
    paths = sorted(r["payload"]["path"] for r in recipes)
    assert all(p.endswith("/dynamic_analysis_records.json") for p in paths)
    assert len(set(paths)) == 2                                          # each regime in its unit's own folder
    # the copy is a recipe file the agent replays from (the records loader the reuse path calls)
    for p in paths:
        assert agent._load_prior_dynamic_records([p]), p
