"""Will the locked recipe travel? A deterministic check at setup().

Observed on a real STEM-EELS line scan: a reference taken on a weak pixel
produced ``max=600`` on the amplitude; every frame over a bright crystal came
back pinned at 600. Replaying the recipe on the reference with its signal scaled
asks the question before the stream does. No LLM calls.
"""

from pathlib import Path

import numpy as np

from scilink.live import MeasurementLoop
from scilink.live.portability import check_portability, describe


def _reference(tmp_path):
    x = np.linspace(0, 10, 200)
    p = tmp_path / "reference.csv"
    np.savetxt(p, np.column_stack([x, 400 * np.exp(-0.5 * ((x - 5) / 0.6) ** 2) + 10]),
               delimiter=",", header="energy,counts", comments="")
    return str(p)


def test_an_invariant_recipe_travels(tmp_path):
    seen = []

    def replay(path, tag):
        seen.append(float(np.loadtxt(path, delimiter=",", skiprows=1)[:, 1].max()))
        return 0.97
    rep = check_portability(replay, _reference(tmp_path), 0.97, str(tmp_path / "w"))
    assert rep["portable"] and [t["signal_scale"] for t in rep["trials"]] == [3.0, 0.35]
    assert seen[0] > 1200 and seen[1] < 150                       # the data really was scaled
    assert "travels" in describe(rep) and "—" not in describe(rep)


def test_a_baked_in_ceiling_is_named(tmp_path):
    def replay(path, tag):                                        # amplitude capped at 600
        return 0.97 if np.loadtxt(path, delimiter=",", skiprows=1)[:, 1].max() <= 600 else 0.80
    rep = check_portability(replay, _reference(tmp_path), 0.97, str(tmp_path / "w"))
    assert rep["portable"] is False
    assert [t["ok"] for t in rep["trials"]] == [False, True]
    assert "at ×3 signal R² 0.8" in describe(rep) and "baked in" in describe(rep)


def test_a_recipe_that_cannot_run_on_the_scaled_copy_fails_the_check(tmp_path):
    rep = check_portability(lambda p, t: None, _reference(tmp_path), 0.97, str(tmp_path / "w"))
    assert rep["portable"] is False and "it did not run" in describe(rep)


def test_nothing_to_compare_is_no_verdict(tmp_path):
    assert check_portability(lambda p, t: 0.9, _reference(tmp_path), None, str(tmp_path / "w")) == {}
    assert check_portability(lambda p, t: 0.9, str(tmp_path / "missing.csv"), 0.9, str(tmp_path / "w")) == {}


def test_setup_records_the_verdict_and_never_calls_a_model(tmp_path):
    from tests.test_measurement_loop import make_anchor
    anchor = make_anchor(tmp_path)
    calls = []

    class Agent:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def analyze(self, data, **kw):
            calls.append((Path(str(data)).name, kw.get("strict_replay")))
            big = np.loadtxt(data, delimiter=",", skiprows=1)[:, 1].max() > 600
            return {"status": "success", "output_directory": self.output_dir,
                    "fitting_parameters": {"peak_1": {"center": 5.0}},
                    "fit_quality": {"r_squared": 0.80 if big else 0.99},
                    "stage_timings": {"llm_calls": 0}}
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=Agent)
    rec = loop.setup(anchor=str(anchor), reference_data=_reference(tmp_path))
    assert rec["portability"]["portable"] is False and "may not travel" in rec["portability"]["summary"]
    assert calls == [("reference_x3.csv", True), ("reference_x0.35.csv", True)]
    quiet = MeasurementLoop(str(tmp_path / "loop2"), agent_factory=Agent, check_portability=False)
    assert "portability" not in quiet.setup(anchor=str(anchor), reference_data=_reference(tmp_path))
