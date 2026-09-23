"""What an instrument has learned is kept across runs, keyed on the instrument.

A loop opened with ``remember=`` tries the instrument's known recipes on the
reference by strict replay before it analyses anything: the second run on an
instrument arms in seconds with no model call. The REAL image agent runs here,
with a model that raises if it is ever called.
"""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.live import MeasurementLoop, run_experiment
from scilink.live.instrument_home import InstrumentHome
from scilink.live.simulators import get_simulator
from tests.test_image_measurement_loop import SCRIPT, _anchor, _factory


def _loop(tmp_path, name, sim, **kw):
    return MeasurementLoop(str(tmp_path / name), system_info=sim.system_info, instrument=sim,
                           outputs=sim.outputs, agent_factory=_factory, breach_patience=2,
                           check_portability=False, remember=str(tmp_path / "instruments"), **kw)


def test_the_second_run_on_an_instrument_starts_from_what_the_first_learned(tmp_path):
    sim = get_simulator("particle_coarsening_images", seed=2)
    first_frame = sim.acquire({})
    ref = first_frame.save(str(tmp_path / "ref1"), 0, stem="reference")
    with _loop(tmp_path, "run1", sim) as run1:
        s1 = run1.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})),
                        reference_data=ref)
        assert "recalled_from_instrument" not in s1
        run_experiment(sim, run1, 4, apply="never")
    home = InstrumentHome(sim, root=str(tmp_path / "instruments"))
    [kept] = home.recipes(modality="image")
    assert kept["recipe_id"] == s1["recipe"]["id"] and kept["technique"].startswith("TEM")
    assert {"particle_count", "mean_diameter_nm"} <= set(kept["reports"])
    [run] = home.runs()
    assert run["frames"] == 4 and run["recipes"] == [s1["recipe"]["id"]]

    # another day, another sample on the same instrument: a reference, and NO anchor
    sim2 = get_simulator("particle_coarsening_images", seed=9)
    ref2 = sim2.acquire({}).save(str(tmp_path / "ref2"), 0, stem="reference")
    with _loop(tmp_path, "run2", sim2) as run2:
        s2 = run2.setup(reference=ref2)                       # the agent's model raises if called
        assert s2["recalled_from_instrument"] == s1["recipe"]["id"] and s2["llm_calls"] == 0
        assert s2["source"].startswith("instrument:")
        assert abs(s2["reference_features"]["particle_count"] - 70) < 8      # replayed on THIS reference
        recs = run_experiment(sim2, run2, 3, apply="never")
        assert all(r["llm_calls"] == 0 and "fit_failed" not in r["flags"] for r in recs)
    [again] = home.recipes(modality="image")
    assert again["uses"] == 1 and len(home.runs()) == 2


def test_a_recipe_is_offered_only_for_the_same_kind_of_measurement(tmp_path):
    sim = get_simulator("particle_coarsening_images", seed=2)
    ref = sim.acquire({}).save(str(tmp_path / "ref"), 0, stem="reference")
    with _loop(tmp_path, "run1", sim) as loop:
        loop.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})), reference_data=ref)
    home = InstrumentHome(sim, root=str(tmp_path / "instruments"))
    assert home.recipes(modality="curve") == []
    assert home.recipes(modality="image", technique="scanning tunnelling spectroscopy") == []
    assert len(home.recipes(modality="image", technique="tem bright-field  imaging, in-situ heating")) == 1
    other = InstrumentHome({"id": "another-microscope"}, root=str(tmp_path / "instruments"))
    assert other.recipes() == [] and other.dir != home.dir


def test_a_known_recipe_that_does_not_report_what_is_tracked_is_not_used(tmp_path):
    sim = get_simulator("particle_coarsening_images", seed=2)
    ref = sim.acquire({}).save(str(tmp_path / "ref"), 0, stem="reference")
    with _loop(tmp_path, "run1", sim) as loop:
        loop.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})), reference_data=ref)

    wants_more = MeasurementLoop(str(tmp_path / "run2"), system_info=sim.system_info, instrument=sim,
                                 outputs={"area_fraction": "fraction of the field covered"},
                                 agent_factory=_factory, remember=str(tmp_path / "instruments"))
    assert wants_more._recall_from_home(ref) is None       # it fits, and does not report area_fraction


def test_remember_needs_an_instrument_and_keeps_heavy_artifacts_out(tmp_path):
    with pytest.raises(ValueError, match="instrument"):
        MeasurementLoop(str(tmp_path / "x"), remember=True)
    sim = get_simulator("particle_coarsening_images", seed=2)
    ref = sim.acquire({}).save(str(tmp_path / "ref"), 0, stem="reference")
    anchor = _anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})
    (anchor / "dcnn_trained").mkdir()
    (anchor / "dcnn_trained" / "weights.tar").write_bytes(b"x" * 1000)
    with _loop(tmp_path, "run1", sim) as loop:
        loop.setup(anchor=str(anchor), reference_data=ref)
    [kept] = InstrumentHome(sim, root=str(tmp_path / "instruments")).recipes()
    assert not list((tmp_path / "instruments").rglob("weights.tar"))
    assert (os.path.isfile(os.path.join(kept["anchor_dir"], "scripts", "analysis_script.py")))


def _remembered_run(tmp_path, seed=2):
    sim = get_simulator("particle_coarsening_images", seed=seed)
    ref = sim.acquire({}).save(str(tmp_path / f"ref{seed}"), 0, stem="reference")
    with _loop(tmp_path, f"run{seed}", sim) as loop:
        setup = loop.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})),
                           reference_data=ref)
        run_experiment(sim, loop, 3, apply="never")
    return sim, setup


def test_what_an_instrument_remembers_can_be_read_without_touching_it(tmp_path):
    from scilink.live.instrument_home import known_instruments, remembered
    root = str(tmp_path / "instruments")
    assert known_instruments(root) == [] and remembered("nobody", root) is None
    assert not (tmp_path / "instruments").exists()            # looking creates nothing
    sim, setup = _remembered_run(tmp_path)
    [info] = known_instruments(root)
    assert info["id"] == sim.id and info["modality"] == "image" and (info["recipes"], info["runs"]) == (1, 1)
    seen_before = info["last_seen"]
    record = remembered(sim.id, root)
    assert record == remembered(info["key"], root)            # by id or by folder name
    [recipe] = record["recipes"]
    assert recipe["recipe_id"] == setup["recipe"]["id"] and "anchor_dir" not in recipe
    assert set(recipe["outputs"]) == {"particle_count", "mean_diameter_nm"} and recipe["size_mb"] >= 0
    [run] = record["runs"]
    assert run["frames"] == 3
    assert known_instruments(root)[0]["last_seen"] == seen_before


def test_a_recipe_can_be_forgotten_and_the_runs_stay_on_the_record(tmp_path):
    from scilink.live.instrument_home import forget_instrument, forget_recipe, known_instruments, remembered
    root = str(tmp_path / "instruments")
    sim, setup = _remembered_run(tmp_path)
    rid = setup["recipe"]["id"]
    assert not forget_recipe(sim.id, "../" + rid, root)       # only a plain id names a recipe
    assert not forget_recipe(sim.id, "0" * 12, root) and not forget_recipe("nobody", rid, root)
    assert forget_recipe(sim.id, rid, root)
    record = remembered(sim.id, root)
    assert record["recipes"] == [] and len(record["runs"]) == 1
    assert forget_instrument(sim.id, root) and known_instruments(root) == []


def test_a_recalled_recipe_is_replayed_from_the_runs_own_copy(tmp_path):
    """The store is trimmed and a person can forget a recipe: neither may break
    a run that is using it."""
    from scilink.live.instrument_home import forget_recipe
    root = tmp_path / "instruments"
    sim, setup = _remembered_run(tmp_path)
    sim2 = get_simulator("particle_coarsening_images", seed=9)
    ref2 = sim2.acquire({}).save(str(tmp_path / "ref9"), 0, stem="reference")
    with _loop(tmp_path, "second", sim2) as loop:
        s2 = loop.setup(reference=ref2)
        assert s2["recalled_from_instrument"] == setup["recipe"]["id"]
        assert root.resolve() not in loop.anchor_dir.resolve().parents
        assert (tmp_path / "second").resolve() in loop.anchor_dir.resolve().parents
        assert forget_recipe(sim.id, setup["recipe"]["id"], str(root))      # mid-run
        recs = run_experiment(sim2, loop, 3, apply="never")
        assert all("fit_failed" not in r["flags"] for r in recs)


def test_the_cli_lists_shows_and_forgets(tmp_path, monkeypatch, capsys):
    import sys
    from scilink.cli import instrument as cli
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    home = InstrumentHome({"id": "raman-1", "technique": "Raman spectroscopy", "modality": "curve"})
    recipe = home.dir / "recipes" / "abc123"
    (recipe / "anchor").mkdir(parents=True)
    (recipe / "recipe.json").write_text(json.dumps({"recipe_id": "abc123", "modality": "curve", "uses": 3,
                                                    "outputs": {"g_position": "position of the G band"}}))

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["scilink instrument", *argv])
        code = cli.main()
        return code, capsys.readouterr().out

    code, out = run("list")
    assert code == 0 and "raman-1" in out and "recipes 1, runs 0" in out
    code, out = run("show", "raman-1")
    assert code == 0 and "abc123" in out and "recalled 3x" in out and "g_position" in out
    assert run("show", "nobody")[0] == 1
    assert run("forget", "raman-1")[0] == 1                       # neither a recipe nor --all
    code, out = run("forget", "raman-1", "abc123", "--yes")
    assert code == 0 and not recipe.exists() and home.dir.is_dir()
    assert run("forget", "raman-1", "--all", "-y")[0] == 0 and not home.dir.exists()


def test_a_contested_recipe_is_remembered_as_contested_and_tried_last(tmp_path):
    """Seen live: two audits rejected a recipe without agreeing with each other,
    the deeper one's recipe was adopted as contested, and the store kept it as
    the newest, so the next run would have tried the unverified recipe first."""
    sim, setup = _remembered_run(tmp_path)
    home = InstrumentHome(sim, root=str(tmp_path / "instruments"))
    verified = home.recipes()[0]
    newer = home.dir / "recipes" / "zzz999"
    (newer / "anchor").mkdir(parents=True)
    (newer / "recipe.json").write_text(json.dumps({**{k: v for k, v in verified.items() if k != "anchor_dir"},
                                                   "recipe_id": "zzz999", "contested": True,
                                                   "last_used": "2099-01-01T00:00:00"}))
    assert [r["recipe_id"] for r in home.recipes()] == [verified["recipe_id"], "zzz999"]

    class Loop:                                   # what save_recipe reads off a loop
        recipe = {"id": "abc", "source": "reanchor:thorough", "contested": True}
        anchor_dir, modality = verified["anchor_dir"], type("M", (), {"name": "image"})
        system_info, instrument, outputs, targets, _edits, _reference_features = {}, {}, {}, [], [], {}
    assert home.save_recipe(Loop)["contested"] is True
