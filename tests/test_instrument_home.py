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
