"""The measurement loop following a stream of DATACUBES.

The loop is modality-neutral; what differs for a datacube lives in
``scilink/live/modality.py``: the hyperspectral agent locks and replays the
recipe (a strict replay: the whole analysis with no model call), the features
are per-map means, the fit verdict is the agent's deterministic replay gate, and
the change signal reads the cube's mean spectrum.

These run the REAL hyperspectral agent on the fast path, with a model that
raises if it is ever called. No LLM calls anywhere.
"""

import json
import logging
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.live import MeasurementLoop, ReplayInstrument, run_experiment
from scilink.live.modality import HyperspectralModality, load_cube, resolve_modality
from scilink.live.simulators import SpectrumImageSeries, get_simulator

SCRIPT = '''
def analyze_feature(data, axis):
    import numpy as np
    win = (axis > 0.45) & (axis < 0.80)
    sub = data[:, :, win] - data[:, :, win].min(axis=2, keepdims=True)
    e = axis[win]
    top = sub >= 0.6 * sub.max(axis=2, keepdims=True)
    pos = (sub * top * e).sum(axis=2) / np.maximum((sub * top).sum(axis=2), 1e-12)
    return {"maps": {"Plasmon_Energy": pos}, "units": "eV", "description": "plasmon maximum"}
'''


class _ExplodingModel:
    def generate_content(self, *a, **k):
        raise AssertionError("a model was called on the fast path")


def _factory(out_dir):
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    ag = HyperspectralAnalysisAgent(api_key="sk-dummy", model_name="claude-opus-4-6",
                                    output_dir=out_dir, enable_human_feedback=False)
    ag.model = _ExplodingModel()
    for stage in list(getattr(ag, "pipeline", [])) + list(getattr(ag, "synthesis_pipeline", [])):
        if hasattr(stage, "model"):
            stage.model = _ExplodingModel()
    return ag


def _anchor(tmp_path, reference_cube):
    """A finished hyperspectral run, as the agent leaves it on disk."""
    d = tmp_path / "anchor"
    d.mkdir()
    (d / "dynamic_analysis_records.json").write_text(json.dumps([{
        "target": "plasmon resonance energy map", "task_success": True,
        "required_outputs": ["Plasmon_Energy"], "script": SCRIPT,
        "quality_history": {"approved": True}}]))
    scope = {}
    exec(SCRIPT, scope)
    pos = scope["analyze_feature"](reference_cube, np.linspace(0.30, 1.20, reference_cube.shape[2]))[
        "maps"]["Plasmon_Energy"]
    stats = {"min": float(pos.min()), "max": float(pos.max()), "mean": float(pos.mean())}
    (d / "analysis_results.json").write_text(json.dumps({
        "agent_type": "hyperspectral", "status": "success",
        "extracted_features": {f"Plasmon_Energy_{k}_eV": v for k, v in stats.items()},
        "feature_records": [{"name": "Plasmon_Energy", "units": "eV", "stats": stats, "coverage": 1.0}]}))
    return d


@pytest.fixture
def armed(tmp_path):
    sim = get_simulator("spectrum_image_series", seed=3)
    first = sim.acquire({})
    ref = first.save(str(tmp_path / "reference"), 0, stem="reference")
    loop = MeasurementLoop(str(tmp_path / "loop"), system_info=sim.system_info, instrument=sim,
                           outputs={"plasmon_energy": "energy of the plasmon maximum"},
                           agent_factory=_factory, breach_patience=2)
    setup = loop.setup(anchor=str(_anchor(tmp_path, first.cube)), reference_data=ref)
    return sim, loop, setup


def test_the_instrument_says_what_a_frame_is_and_the_loop_follows(armed):
    sim, loop, setup = armed
    assert sim.describe()["modality"] == "hyperspectral" and loop.modality.name == "hyperspectral"
    assert setup["instrument"]["modality"] == "hyperspectral"
    # the named output is matched to what the recipe reports; nothing is pinned
    assert list(loop.outputs) == ["Plasmon_Energy_mean_eV"] and "pinned_outputs" not in setup
    assert abs(setup["reference_features"]["Plasmon_Energy_mean_eV"] - 0.62) < 0.02
    assert not any("min" in k or "max" in k for k in setup["reference_features"])
    assert loop._modality_state["replay_reference"]["Plasmon_Energy"]["coverage"] == 1.0
    assert loop._modality_state["replay_reference"]["Plasmon_Energy"]["values_may_move"] is True
    assert loop._modality_state["locked_targets"][0]["required_outputs"] == ["Plasmon_Energy"]
    assert "portability" not in setup                     # a curve mechanism


def test_frames_are_answered_by_a_strict_replay_and_the_red_shift_is_tracked(armed, tmp_path):
    sim, loop, _ = armed
    records = run_experiment(sim, loop, 8, apply="never")
    assert [r["flags"] for r in records] == [[]] * 8      # and the exploding model was never reached
    assert all(r["llm_calls"] == 0 and r["gate"]["verdict"] == "good" for r in records)
    assert all(r["data"].endswith(".npy") for r in records)
    e = [r["features"]["Plasmon_Energy_mean_eV"] for r in records]
    truth = [sim.energy(f) for f in range(2, 10)]
    assert np.allclose(e, truth, atol=0.012) and e[0] - e[-1] > 0.006       # 1.5 meV per frame
    assert all("drift_fraction" in r["gate"] for r in records)


def test_a_new_mode_is_announced_and_located_on_the_energy_axis(armed):
    sim, loop, _ = armed
    records = run_experiment(sim, loop, 20, apply="never")
    [novelty] = [e for e in loop.read_log() if e["event"] == "novelty"]
    assert 13 <= novelty["since_step"] + 1 <= 16          # frame 13 of the run (the reference was frame 1)
    where = novelty["where"][0]
    assert where["kind"] == "new" and abs(where["x_peak"] - 0.95) < 0.04    # in eV, not in channels
    # The mode covers one corner: 3 % of the whole field's mean spectrum, which a
    # monitor of the mean alone did not see. The field is watched by region too.
    assert novelty["region"] == "lower right" and where["region"] == "lower right"
    assert novelty["recipe_fits"] is True                 # the plasmon map is still right
    assert any(e["event"] == "state_accepted" for e in loop.read_log())
    assert records[-1]["flags"] == []


def test_a_cube_the_recipe_cannot_run_on_fails_the_frame_without_a_model(armed, tmp_path):
    sim, loop, _ = armed
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((4, 4, 12), dtype=np.float32))  # too few channels for the window
    rec = loop.step(str(bad))
    assert "fit_failed" in rec["flags"] or "gate_poor" in rec["flags"]


def test_state_survives_a_restart(armed, tmp_path):
    sim, loop, _ = armed
    run_experiment(sim, loop, 3, apply="never")
    again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=_factory)
    assert again.modality.name == "hyperspectral"
    assert again._modality_state == loop._modality_state
    assert again.step(sim.acquire({}).save(str(tmp_path / "more"), 0))["flags"] == []


def test_a_folder_of_recorded_cubes_replays_as_cubes(tmp_path):
    sim = SpectrumImageSeries(seed=1)
    for i in range(3):
        sim.acquire({}).save(str(tmp_path / "rec"), i, stem="si")
    inst = ReplayInstrument(str(tmp_path / "rec"), system_info=sim.system_info)
    assert inst.modality == "hyperspectral" and len(inst) == 3
    frame = inst.acquire({})
    assert frame.cube.shape == (14, 14, 160) and abs(frame.x[0] - 0.30) < 1e-6   # the physical axis
    assert frame.save(str(tmp_path / "out"), 0).endswith(".npy")
    curves = tmp_path / "curves"
    curves.mkdir()
    np.save(curves / "a.npy", np.c_[np.arange(30.0), np.ones(30)])
    assert ReplayInstrument(str(curves)).modality == "curve"


def test_the_modality_helpers(tmp_path):
    assert resolve_modality(None).name == "curve" and resolve_modality("hyperspectral").pinning is False
    with pytest.raises(ValueError):
        resolve_modality("movies")
    np.save(tmp_path / "c.npy", np.ones((3, 4, 20)))
    assert load_cube(tmp_path / "c.npy").shape == (3, 4, 20)
    m = HyperspectralModality()
    assert m.validity({"status": "success", "script_reuse": {"verbatim": True}})["verdict"] == "good"
    poor = m.validity({"status": "partial", "degraded_outputs": [{"missing_required": ["Width"]}]})
    assert poor["verdict"] == "poor" and poor["withheld"] == ["Width"]
    feats = m.features({"extracted_features": [
        {"name": "Peak", "units": "eV", "stats": {"min": 1.0, "max": 3.0, "mean": 2.0}},
        {"name": "Total", "scalar": 7.0}]})
    assert feats == {"Peak_mean_eV": 2.0, "Total": 7.0}


def test_a_reference_whose_required_map_never_passed_does_not_arm_the_loop(tmp_path):
    # Live, real EELS tile under `quick`: the per-pixel map was rejected three times,
    # the run ended `partial` with no approved script, and the loop said "holds no
    # reusable run", which was not what had happened.
    class Agent:
        def __init__(self, out):
            self.out = out

        def analyze(self, data, **kw):
            os.makedirs(self.out, exist_ok=True)
            with open(os.path.join(self.out, "dynamic_analysis_records.json"), "w") as fh:
                json.dump([{"target": "t", "task_success": False, "script": "x",
                            "required_outputs": ["plasmon_energy"]}], fh)
            return {"status": "partial", "output_directory": self.out,
                    "degraded_outputs": [{"missing_required": ["plasmon_energy"]}]}

    np.save(tmp_path / "reference_000000.npy", np.ones((4, 4, 20), dtype=np.float32))
    loop = MeasurementLoop(str(tmp_path / "loop"), modality="hyperspectral", agent_factory=Agent)
    with pytest.raises(RuntimeError, match=r"without an approved script.*plasmon_energy.*never passed"):
        loop.setup(reference=str(tmp_path / "reference_000000.npy"))
