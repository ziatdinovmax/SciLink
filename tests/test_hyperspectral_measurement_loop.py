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
    # the recipe was replayed on the reference at 3x and 0.35x the counts: this one
    # weights by the data's own maximum, so the plasmon energy does not move
    port = setup["portability"]
    assert port["kind"] == "datacube" and port["portable"] is True
    assert port["tracked"] == ["Plasmon_Energy_mean_eV"] and "does not depend" in port["summary"]


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
    [novelty] = [e for e in loop.read_log() if e["event"] == "novelty" and not e.get("onset")]
    assert 13 <= novelty["since_step"] + 1 <= 16          # frame 13 of the run (the reference was frame 1)
    where = novelty["where"][0]
    assert where["kind"] == "new" and abs(where["x_peak"] - 0.95) < 0.04    # in eV, not in channels
    # The mode covers one corner: 3 % of the whole field's mean spectrum, which a
    # monitor of the mean alone did not see. The field is watched by region too.
    # named by the smallest region that holds it, on a pyramid of grids
    lower_right = {"lower right quarter", "lower right ninth", "row 4, column 4 of a 4 x 4 grid"}
    assert novelty["region"] in lower_right and where["region"] == novelty["region"]
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


def test_a_slow_red_shift_is_announced_as_the_whole_field_moving(tmp_path):
    """No frame of the ramp ever looks new. The slow alarm says the stream has
    moved from its reference, what moved, and that no one region did it."""
    sim = get_simulator("spectrum_image_series", seed=3)
    first = sim.acquire({})
    ref = first.save(str(tmp_path / "reference"), 0, stem="reference")
    loop = MeasurementLoop(str(tmp_path / "loop"), system_info=sim.system_info, instrument=sim,
                           agent_factory=_factory, breach_patience=2, gradual_bar=0.15)
    loop.setup(anchor=str(_anchor(tmp_path, first.cube)), reference_data=ref)
    records = run_experiment(sim, loop, 11, apply="never")           # ends before the second mode
    assert all(r["flags"] == [] for r in records)
    [slow] = [e for e in loop.read_log() if e["event"] == "novelty"]
    assert slow["onset"] == "gradual" and slow["fraction"] > 0.15 and "region" not in slow
    where = slow["where"][0]
    assert where["kind"] == "shifted" and abs(where["x_peak"] - 0.61) < 0.03
    assert where["region"] == "whole field"


# ── a rebuild first tries the recipes this run has already used ─────────────

def _window_script(lo, hi):
    return f'''
def analyze_feature(data, axis):
    import numpy as np
    mean = data.reshape(-1, data.shape[-1]).mean(axis=0)
    if not ({lo} < axis[int(np.argmax(mean))] < {hi}):
        raise ValueError("the resonance is outside this recipe's window")
    win = (axis > {lo}) & (axis < {hi})
    sub = data[:, :, win] - data[:, :, win].min(axis=2, keepdims=True)
    pos = (sub * axis[win]).sum(axis=2) / np.maximum(sub.sum(axis=2), 1e-12)
    return {{"maps": {{"Resonance": pos}}, "units": "eV", "description": "resonance"}}
'''


def _state_cube(path, centre, seed):
    rng = np.random.default_rng(seed)
    e = np.linspace(0.30, 1.20, 120)
    u = np.linspace(-1, 1, 8)[:, None] * np.ones((1, 8))
    cube = 100.0 * np.exp(-0.5 * ((e - (centre + 0.01 * u)[..., None]) / 0.05) ** 2) + rng.normal(0, 1.0, (8, 8, 120))
    np.save(path, cube.astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps({"meta": {"energy_range": {"start": 0.30, "end": 1.20, "units": "eV"}}}))
    return str(path)


def _recipe_dir(root, name, lo, hi, mean):
    d = root / name
    d.mkdir()
    (d / "dynamic_analysis_records.json").write_text(json.dumps([{
        "target": "resonance energy map", "task_success": True, "required_outputs": ["Resonance"],
        "script": _window_script(lo, hi), "quality_history": {"approved": True}}]))
    (d / "analysis_results.json").write_text(json.dumps({"agent_type": "hyperspectral", "status": "success",
        "extracted_features": {"Resonance_mean_eV": mean},
        "feature_records": [{"name": "Resonance", "units": "eV", "coverage": 1.0,
                             "stats": {"min": mean - 0.01, "max": mean + 0.01, "mean": mean}}]}))
    return d


def test_a_state_seen_before_is_served_by_recall_with_no_model_call(tmp_path):
    from scilink.live.measurement_loop import _InlineEscalation
    low, high = _recipe_dir(tmp_path, "low", 0.45, 0.80, 0.60), _recipe_dir(tmp_path, "high", 0.85, 1.15, 1.00)
    loop = MeasurementLoop(str(tmp_path / "loop"), modality="hyperspectral", agent_factory=_factory,
                           system_info={"technique": "EELS"}, breach_patience=2, auto_escalate=True,
                           escalation_runner=_InlineEscalation, gradual_bar=None)
    loop.setup(anchor=str(low), reference_data=_state_cube(tmp_path / "ref.npy", 0.60, 0))
    loop._known_recipes = [{"anchor_dir": str(high), "edits": [], "recipe_id": "the-high-state"}]
    assert loop.step(_state_cube(tmp_path / "a.npy", 0.60, 1))["flags"] == []
    failing = [loop.step(_state_cube(tmp_path / f"b{i}.npy", 1.00, 10 + i)) for i in range(2)]
    assert all("fit_failed" in r["flags"] for r in failing)          # the low-state recipe cannot run here
    served = loop.step(_state_cube(tmp_path / "b2.npy", 1.00, 12))   # the rebuild landed before this frame
    rebuilt = next(e for e in loop.read_log() if e["event"] == "reanchor")
    assert rebuilt["source"] == "recalled" and rebuilt["llm_calls"] == 0
    assert "fit_failed" not in served["flags"] and abs(served["features"]["Resonance_mean_eV"] - 1.00) < 0.02
    # and the recipe it left is now known: going back is a recall too
    assert [r["anchor_dir"] for r in loop._known_recipes] == [str(low)]
    for i in range(2):
        loop.step(_state_cube(tmp_path / f"c{i}.npy", 0.60, 20 + i))
    back = loop.step(_state_cube(tmp_path / "c2.npy", 0.60, 22))
    assert [e["source"] for e in loop.read_log() if e["event"] == "reanchor"] == ["recalled", "recalled"]
    assert abs(back["features"]["Resonance_mean_eV"] - 0.60) < 0.02


def test_several_first_cubes_are_planned_as_a_series_and_the_last_ones_regime_is_locked(tmp_path):
    """One noisy or unrepresentative cube should not decide the method (live, a
    real EELS tile under a reduced profile never got its map approved). A list of
    reference cubes goes to the hyperspectral series driver; the loop locks the
    recipe of the regime the LAST cube belongs to and seeds its monitor from all."""
    early, late = _recipe_dir(tmp_path, "early", 0.45, 0.80, 0.60), _recipe_dir(tmp_path, "late", 0.85, 1.15, 1.00)
    seen = []

    class SeriesAgent:
        def __init__(self, out):
            self.out = out

        def analyze(self, data, **kw):
            seen.append((data, kw))
            os.makedirs(self.out, exist_ok=True)
            with open(os.path.join(self.out, "series_analysis_results.json"), "w") as fh:
                json.dump({"results": [{"index": 0, "success": True, "regime": "a"},
                                       {"index": 1, "success": True, "regime": "a"},
                                       {"index": 2, "success": True, "regime": "b"},
                                       {"index": 3, "success": False, "regime": "b"}],
                           "locked_config": {"regimes": {
                               "a": {"anchor_index": 0, "anchor_output_dir": str(early), "targets": [{"target": "low"}]},
                               "b": {"anchor_index": 2, "anchor_output_dir": str(late), "targets": [{"target": "high"}]}}}}, fh)
            return {"status": "success", "output_directory": self.out, "stage_timings": {"llm_calls": 11}}

    cubes = [_state_cube(tmp_path / f"r{i}.npy", c, i) for i, c in enumerate([0.60, 0.61, 1.00, 1.00])]
    loop = MeasurementLoop(str(tmp_path / "loop"), modality="hyperspectral", agent_factory=SeriesAgent,
                           system_info={"technique": "EELS"}, targets=["resonance energy map"],
                           check_portability=False)             # its replays are not this test's subject
    setup = loop.setup(reference=cubes, profile="quick")
    [(data, kw)] = seen
    assert data == cubes and kw["series_metadata"]["values"] == [0, 1, 2, 3]
    assert kw["profile"] == {"base": "quick", "trend": False, "synthesis": "none", "adaptive_refit": False}
    assert "resonance energy map" in kw["objective"]
    assert str(loop.anchor_dir) == str(late)                     # the regime the stream continues from
    assert setup["reference_frames"] == {"n": 4, "fitted": 3, "anchored_on": 2, "regimes": 2,
                                         "model": "high", "llm_calls": 11}
    assert setup["source"] == "reference:quick:4 frames"
    assert setup["reference_features"] == {"Resonance_mean_eV": 1.0}
    assert loop._modality_state["locked_targets"][0]["required_outputs"] == ["Resonance"]
    assert loop._drift.n_learned == 4                           # every reference cube seeds the change signal


def test_the_field_is_watched_on_a_pyramid_of_grids(tmp_path):
    m = HyperspectralModality()
    np.save(tmp_path / "c.npy", np.random.default_rng(0).random((14, 14, 40)))
    names = list(m.read_signals(str(tmp_path / "c.npy")))
    assert names[0] == "whole field" and len(names) == 1 + 4 + 9 + 16
    assert {"lower right quarter", "lower right ninth", "upper centre ninth", "centre ninth",
            "row 4, column 1 of a 4 x 4 grid"} <= set(names)
    np.save(tmp_path / "small.npy", np.random.default_rng(0).random((6, 6, 40)))
    assert len(m.read_signals(str(tmp_path / "small.npy"))) == 1 + 4      # 3 x 3 would leave 4 pixels a region
    np.save(tmp_path / "line.npy", np.random.default_rng(0).random((1, 30, 40)))
    assert list(m.read_signals(str(tmp_path / "line.npy"))) == ["whole field"]


def test_a_small_feature_is_found_by_the_grid_it_fits_in(tmp_path):
    """Measured: a 3 x 3-pixel feature (4.6 % of a 14 x 14 field) at random positions
    was never seen by quadrants alone and 9 times of 12 by the pyramid, with no
    false alarm in 132 quiet frames."""
    from scilink.live.drift import DriftBank
    sim, m, bank = SpectrumImageSeries(seed=2), HyperspectralModality(), DriftBank()
    found = None
    for k in range(1, 12):
        f = sim.acquire({})
        cube = f.cube.copy()
        if k >= 9:
            cube[5:9, 0:4] += 40.0 * np.exp(-0.5 * ((f.x - 0.95) / 0.04) ** 2)
        np.save(tmp_path / "f.npy", cube)
        sig = m.read_signals(str(tmp_path / "f.npy"), sim.system_info)
        if k == 1:
            bank.seed([sig])
            continue
        v = bank.judge(sig)
        assert v["suspected"] == (k >= 9), (k, v.get("region"), v["fraction"])
        if v["suspected"]:
            found = found or v["region"]
        else:
            bank.learn(sig, v)
    assert found == "middle left ninth"
    [where] = bank.locate()
    assert where["region"] == "middle left ninth" and abs(where["x_peak"] - 0.95) < 0.03


def test_a_cube_recipe_can_be_amended_without_a_rebuild(armed, tmp_path):
    """One knob of the locked script, changed by hand: the amendment is a copy of
    the anchor run with the edited script, so the replay stays a plain replay."""
    sim, loop, _ = armed
    before, first_anchor = loop.recipe["id"], loop.anchor_dir
    assert loop.step(sim.acquire({}).save(str(tmp_path / "in"), 1))["flags"] == []
    with pytest.raises(ValueError, match="do not apply"):
        loop.amend([{"old_text": "axis > 0.99", "new_text": "axis > 0.5"}])
    assert loop.anchor_dir == first_anchor and loop.recipe["id"] == before      # nothing changed
    rec = loop.amend([{"old_text": "(axis > 0.45)", "new_text": "(axis > 0.50)"}], note="tighter window")
    assert rec["event"] == "amend" and loop.recipe["id"] != before
    assert loop.anchor_dir != first_anchor and (loop.anchor_dir / "amended_from.json").is_file()
    script = json.loads((loop.anchor_dir / "dynamic_analysis_records.json").read_text())[0]["script"]
    assert "(axis > 0.50)" in script and "(axis > 0.45)" not in script
    after = loop.step(sim.acquire({}).save(str(tmp_path / "in"), 2))
    assert after["flags"] == [] and after["recipe_id"] == loop.recipe["id"]
    assert abs(after["features"]["Plasmon_Energy_mean_eV"] - sim.energy(sim.frame)) < 0.012
    # the gate's reference and the targets a rebuild must keep travel with the copy
    assert loop._modality_state["locked_targets"][0]["required_outputs"] == ["Plasmon_Energy"]
    again = MeasurementLoop.resume(str(tmp_path / "loop"), agent_factory=_factory)
    assert again.anchor_dir == loop.anchor_dir


def test_a_cube_recipe_with_a_constant_read_off_the_reference_is_caught_at_setup(tmp_path):
    """The risk the curve check was built for (a bound of 600 counts read off a weak
    reference pixel) is the same for a per-pixel cube script. Asked of the outputs:
    under a change of signal level each must stay put or scale with the counts."""
    sim = get_simulator("spectrum_image_series", seed=3)
    first = sim.acquire({})
    ref = first.save(str(tmp_path / "reference"), 0, stem="reference")
    threshold = SCRIPT.replace("top = sub >= 0.6 * sub.max(axis=2, keepdims=True)",
                               "top = sub >= 70.0")           # counts, read off this cube
    anchor = _anchor(tmp_path, first.cube)
    records = json.loads((anchor / "dynamic_analysis_records.json").read_text())
    records[0]["script"] = threshold
    (anchor / "dynamic_analysis_records.json").write_text(json.dumps(records))
    loop = MeasurementLoop(str(tmp_path / "loop"), system_info=sim.system_info, instrument=sim,
                           outputs={"plasmon_energy": "energy of the plasmon maximum"},
                           agent_factory=_factory)
    port = loop.setup(anchor=str(anchor), reference_data=ref)["portability"]
    assert port["portable"] is False and "depends on the signal level" in port["summary"]
    low = next(t for t in port["trials"] if t["scale"] == 0.35)
    assert not low["ok"]                                  # at a third of the counts nothing clears 70
    assert loop.recipe is not None                        # advisory: the loop is armed, and told


def test_the_cube_check_tells_a_level_from_an_amplitude():
    from scilink.live.portability import check_cube_portability, describe_cube
    import tempfile
    d = tempfile.mkdtemp()
    np.save(os.path.join(d, "ref.npy"), np.ones((4, 4, 20)))
    scale_of = lambda path: float(np.load(path).mean())   # noqa: E731
    honest = lambda path, tag: {"energy": 0.6, "amplitude": 100.0 * scale_of(path), "n_pixels": 16.0}   # noqa: E731
    good = check_cube_portability(honest, os.path.join(d, "ref.npy"), os.path.join(d, "w1"))
    assert good["portable"] and all(not t["depends_on_level"] for t in good["trials"])
    bound = lambda path, tag: {"energy": 0.6, "amplitude": min(100.0 * scale_of(path), 150.0)}          # noqa: E731
    bad = check_cube_portability(bound, os.path.join(d, "ref.npy"), os.path.join(d, "w2"))
    assert not bad["portable"] and "amplitude" in bad["trials"][0]["depends_on_level"]
    only_energy = check_cube_portability(bound, os.path.join(d, "ref.npy"), os.path.join(d, "w3"),
                                         tracked=["energy"])
    assert only_energy["portable"] and "untracked amplitude does" in describe_cube(only_energy)
    assert check_cube_portability(lambda p, t: None, os.path.join(d, "ref.npy"), os.path.join(d, "w4")) == {}
