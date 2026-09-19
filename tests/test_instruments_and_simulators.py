"""The instrument interface, the four simulated experiments, and the driver.

A simulator and a real instrument adapter implement the same two things — an
acquisition-parameter schema and ``acquire(params) -> Frame`` — so that a
scientist can learn the loop on a simulated experiment and then swap in their
own controller. No LLM calls anywhere.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scilink.live import Frame, Instrument, InstrumentSchema, MeasurementLoop, run_experiment
from scilink.live.simulators import (SIMULATORS, AFMForceCurve, BeamlineXRD,
                                     InSituRaman, STMdIdV, get_simulator)


@pytest.mark.parametrize("name", sorted(SIMULATORS))
class TestEverySimulator:
    def test_implements_the_instrument_interface(self, name):
        sim = get_simulator(name)
        assert isinstance(sim, Instrument) and sim.name == name
        assert sim.schema.validate(sim.defaults) == []
        assert sim.outputs and sim.targets and sim.system_info.get("technique")
        assert sim.events and all({"frame", "what"} <= set(e) for e in sim.events)

    def test_frames_are_finite_reproducible_and_carry_truth(self, name):
        a, b = get_simulator(name, seed=3), get_simulator(name, seed=3)
        for _ in range(3):
            fa, fb = a.acquire({}), b.acquire({})
        assert np.array_equal(fa.y, fb.y) and np.all(np.isfinite(fa.y))
        assert len(fa.x) == len(fa.y) > 100 and fa.truth
        assert not np.array_equal(get_simulator(name, seed=4).acquire({}).y,
                                  get_simulator(name, seed=3).acquire({}).y)

    def test_out_of_bounds_parameters_are_refused_by_the_instrument(self, name):
        sim = get_simulator(name)
        first = sim.schema.numeric[0]
        with pytest.raises(ValueError, match="refusing to acquire"):
            sim.acquire({first.name: first.high * 10})
        with pytest.raises(ValueError, match="not a parameter"):
            sim.acquire({"warp_factor": 9})

    def test_a_frame_saves_as_csv_plus_sidecar(self, name, tmp_path):
        f = get_simulator(name).acquire({})
        path = Path(f.save(str(tmp_path), 7))
        assert path.name == "frame_000007.csv"
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        assert data.shape == (len(f.x), 2)
        side = json.loads(path.with_suffix(".json").read_text())
        assert side["params"] == f.params and side["truth"] == f.truth


class TestThePhysicsIsWorthOptimizing:
    def test_xrd_transforms_and_expands(self):
        sim = BeamlineXRD()
        frames = [sim.acquire({}) for _ in range(60)]
        assert frames[0].truth["majority_phase"] == "A" and frames[-1].truth["majority_phase"] == "B"
        assert frames[20].truth["main_peak_2theta"] < frames[0].truth["main_peak_2theta"]   # expansion
        noisy = BeamlineXRD(seed=1).acquire({"exposure_s": 0.1}).y
        quiet = BeamlineXRD(seed=1).acquire({"exposure_s": 20.0}).y
        assert np.std(noisy[-200:]) > 3 * np.std(quiet[-200:])

    def test_raman_anneals_spikes_and_punishes_too_much_power(self):
        sim = InSituRaman()
        frames = [sim.acquire({}) for _ in range(60)]
        assert frames[0].truth["d_over_g"] > 1.1 and frames[-1].truth["d_over_g"] < 0.5
        assert frames[22].truth["cosmic_ray"] and frames[22].y.max() > 3 * frames[21].y.max()
        cool = InSituRaman().acquire({"laser_power_mw": 4.0}).truth["g_position"]
        hot = InSituRaman().acquire({"laser_power_mw": 18.0}).truth["g_position"]
        assert hot < cool - 5                                       # laser heating softens G

    def test_afm_steps_onto_a_stiffer_higher_inclusion(self):
        sim = AFMForceCurve()
        frames = [sim.acquire({}) for _ in range(30)]
        assert frames[0].truth["material"] == "matrix" and frames[-1].truth["material"] == "inclusion"
        assert frames[-1].truth["stiffness_N_per_m"] > 3 * frames[0].truth["stiffness_N_per_m"]
        assert frames[-1].truth["contact_point_nm"] < frames[0].truth["contact_point_nm"] - 10

    def test_sts_modulation_buys_quiet_at_the_price_of_smearing(self):
        def peak_height(mod):
            f = STMdIdV(seed=2).acquire({"modulation_mv": mod, "averages": 64})
            return float(np.max(f.y[f.x > 0]))
        assert peak_height(0.05) > peak_height(0.8) + 0.3           # smeared coherence peak
        lo = STMdIdV(seed=2).acquire({"modulation_mv": 0.03, "averages": 1}).y
        hi = STMdIdV(seed=2).acquire({"modulation_mv": 0.6, "averages": 1}).y
        flat = slice(230, 270)                                       # inside the gap
        assert np.std(lo[flat]) > 5 * np.std(hi[flat])

    def test_unknown_simulator(self):
        with pytest.raises(ValueError, match="available"):
            get_simulator("tem")


# ──────────────────────────────────────────────────────────────
# the driver — and swapping the simulator for "your" instrument
# ──────────────────────────────────────────────────────────────

class MyInstrument(Instrument):
    """What a scientist writes to replace a simulator."""
    name = "my_instrument"
    system_info = {"technique": "generic 1D spectroscopy"}
    schema = InstrumentSchema.from_dict({"dwell": {"low": 1.0, "high": 100.0}})
    defaults = {"dwell": 10.0}

    def __init__(self):
        self.seen = []

    def acquire(self, params):
        params = self.check(params)
        self.seen.append(dict(params))
        x = np.linspace(0, 1, 50)
        return Frame(x=x, y=np.sin(x) * params["dwell"], params=params)


class Doubler:
    name, clock = "doubler", "fast"

    def observe(self, params, features, step=None, flags=None): self.last = params

    def suggest(self): return {"params": {"dwell": self.last["dwell"] * 2}}


def _armed_loop(tmp_path, **kw):
    from tests.test_measurement_loop import FakeAgent, make_anchor
    FakeAgent.calls, FakeAgent.replies = [], {}
    loop = MeasurementLoop(str(tmp_path / "loop"), agent_factory=FakeAgent, **kw)
    loop.setup(anchor=str(make_anchor(tmp_path)))
    return loop


class TestRunExperiment:
    def test_a_user_instrument_drops_in(self, tmp_path):
        inst = MyInstrument()
        records = run_experiment(inst, _armed_loop(tmp_path), 3)
        assert [r["step"] for r in records] == [1, 2, 3]
        assert [r["params"] for r in records] == [{"dwell": 10.0}] * 3
        assert sorted(p.name for p in (tmp_path / "loop" / "incoming").glob("*.csv")) == [
            "frame_000001.csv", "frame_000002.csv", "frame_000003.csv"]

    def test_valid_recommendations_are_applied_when_the_policy_says_so(self, tmp_path):
        inst = MyInstrument()
        loop = _armed_loop(tmp_path, recommender=Doubler(), schema=inst.schema)
        run_experiment(inst, loop, 4, apply="valid")
        assert [p["dwell"] for p in inst.seen] == [10.0, 20.0, 40.0, 80.0]

    def test_advisory_recommendations_wait_for_approval(self, tmp_path):
        inst = MyInstrument()
        loop = _armed_loop(tmp_path, recommender=Doubler(), schema=inst.schema, closed_loop=False)
        records = run_experiment(inst, loop, 3, apply="approved")
        assert [p["dwell"] for p in inst.seen] == [10.0] * 3
        assert records[-1]["recommendation"]["requires_approval"] is True
        closed = _armed_loop(tmp_path / "c", recommender=Doubler(), schema=inst.schema,
                             closed_loop=True)
        inst2 = MyInstrument()
        run_experiment(inst2, closed, 3, apply="approved")
        assert [p["dwell"] for p in inst2.seen] == [10.0, 20.0, 40.0]

    def test_a_recommendation_outside_the_limits_never_reaches_the_instrument(self, tmp_path):
        inst = MyInstrument()
        loop = _armed_loop(tmp_path, recommender=Doubler(), schema=inst.schema, closed_loop=True)
        records = run_experiment(inst, loop, 6, apply="valid")
        assert max(p["dwell"] for p in inst.seen) == 80.0           # 160 was refused by the loop
        assert records[-1]["recommendation"]["valid"] is False

    def test_never_and_stop(self, tmp_path):
        inst = MyInstrument()
        loop = _armed_loop(tmp_path, recommender=Doubler(), schema=inst.schema, closed_loop=True)
        n = []
        run_experiment(inst, loop, 10, apply="never", on_frame=lambda f, r: n.append(r["step"]),
                       stop=lambda: len(n) >= 4)
        assert n == [1, 2, 3, 4] and {p["dwell"] for p in inst.seen} == {10.0}
        with pytest.raises(ValueError, match="apply must be"):
            run_experiment(inst, loop, 1, apply="always")

    def test_simulator_truth_rides_along_for_checking(self, tmp_path):
        records = run_experiment(get_simulator("afm_force_curve"), _armed_loop(tmp_path), 2)
        assert records[0]["truth"]["material"] == "matrix"
