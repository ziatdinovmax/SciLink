"""The measurement loop following a stream of IMAGES.

Third modality of the same loop (``scilink/live/modality.py``): the image agent
locks and replays the recipe (a strict replay: a whole analysis with no model
call, judged by ``_replay_feature_gate``), the features are what the approved
script reports, and the change signal reads the image's radial power spectrum,
whole and by quarter.

The REAL image agent runs on the fast path with a model that raises if called.
"""

import json
import logging
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.live import MeasurementLoop, ReplayInstrument, run_experiment
from scilink.live.modality import ImageModality, resolve_modality
from scilink.live.simulators import ParticleCoarseningImages, get_simulator

SCRIPT = '''
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

img = np.asarray(np.load("data.npy"), dtype=float)
try:
    fov = json.load(open("metadata.json"))["experimental_details"]["spatial_info"]["field_of_view_x"]
    nm_per_px = float(fov) / img.shape[1]
except Exception:
    nm_per_px = 1.0
smooth = ndimage.gaussian_filter(img, 1.0)
med = np.median(smooth)
mad = 1.4826 * np.median(np.abs(smooth - med))
labels, n = ndimage.label(smooth > med + 4.0 * mad)          # relative to this image's own statistics
areas = ndimage.sum(np.ones_like(img), labels, index=np.arange(1, n + 1)) if n else np.array([])
areas = areas[areas >= 6]
diam = np.sqrt(4.0 * areas / np.pi) * nm_per_px
plt.figure(figsize=(4, 4)); plt.imshow(img, cmap="gray"); plt.contour(labels > 0, [0.5], colors="r", linewidths=0.4)
plt.axis("off"); plt.savefig("visualization.png", dpi=70); plt.close()
print("IMAGE_ANALYSIS_RESULTS_JSON:" + json.dumps({
    "analysis_type": "particle segmentation",
    "extracted_features": {"particle_count": int(areas.size),
                           "mean_diameter_nm": float(diam.mean()) if areas.size else 0.0},
    "quality_metrics": {}, "summary": "particles", "saved_arrays": {}}))
'''


class _ExplodingModel:
    def generate_content(self, *a, **k):
        raise AssertionError("a model was called on the fast path")


def _factory(out_dir):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    ag = ImageAnalysisAgent(api_key="sk-dummy", output_dir=out_dir, enable_human_feedback=False,
                            use_literature=False)
    ag.model = _ExplodingModel()
    return ag


def _anchor(tmp_path, features):
    d = tmp_path / "anchor"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "analysis_script.py").write_text(SCRIPT)
    (d / "analysis_results.json").write_text(json.dumps({"status": "success", "extracted_features": features}))
    return d


@pytest.fixture
def armed(tmp_path):
    sim = get_simulator("particle_coarsening_images", seed=2)
    first = sim.acquire({})
    ref = first.save(str(tmp_path / "reference"), 0, stem="reference")
    loop = MeasurementLoop(str(tmp_path / "loop"), system_info=sim.system_info, instrument=sim,
                           outputs=sim.outputs, agent_factory=_factory, breach_patience=2,
                           check_portability=False)
    setup = loop.setup(anchor=str(_anchor(tmp_path, {"particle_count": 70, "mean_diameter_nm": 5.0})),
                       reference_data=ref)
    return sim, loop, setup


def test_the_instrument_says_its_frames_are_images(armed):
    sim, loop, setup = armed
    assert sim.describe()["modality"] == "image" and loop.modality.name == "image"
    assert list(loop.outputs) == ["particle_count", "mean_diameter_nm"] and "pinned_outputs" not in setup
    assert loop._modality_state["replay_reference"] == {"particle_count": 70.0, "mean_diameter_nm": 5.0}


def test_frames_are_answered_by_a_strict_replay_and_coarsening_is_tracked(armed):
    sim, loop, _ = armed
    records = run_experiment(sim, loop, 12, apply="never")
    assert all(r["llm_calls"] == 0 and r["gate"]["verdict"] == "good" for r in records)
    assert all(r["data"].endswith(".npy") for r in records)
    assert not any("fit_failed" in r["flags"] or "gate_poor" in r["flags"] for r in records)
    d = [r["features"]["mean_diameter_nm"] for r in records]
    n = [r["features"]["particle_count"] for r in records]
    assert d[-1] > d[0] + 0.5 and n[-1] < n[0]                      # they grow, and there are fewer
    assert abs(n[0] - 69) <= 6 and abs(d[0] - 5.1) < 0.8            # against the simulator's truth
    assert all("drift_fraction" in r["gate"] for r in records)


def test_an_empty_field_is_flagged_on_evidence_and_a_broken_recipe_fails_the_frame(armed, tmp_path):
    sim, loop, _ = armed
    empty = tmp_path / "empty.npy"
    np.save(empty, np.random.default_rng(0).normal(100.0, 3.0, (256, 256)).astype(np.float32))
    rec = loop.step(str(empty))
    assert "gate_poor" in rec["flags"] and rec["features"]["particle_count"] == 0
    tiny = tmp_path / "tiny.npy"
    np.save(tiny, np.zeros((3, 40), dtype=np.float32))
    assert "fit_failed" in loop.step(str(tiny))["flags"] or "gate_poor" in loop.step(str(tiny))["flags"]


def test_a_recipe_edit_travels_to_every_frame(armed):
    sim, loop, _ = armed
    loop.amend([{"old_text": "med + 4.0 * mad", "new_text": "med + 6.0 * mad"}])
    rec = loop.step(sim.acquire({}).save(str(loop.output_dir / "in"), 1))
    assert rec["flags"] == [] or rec["flags"] == ["drift_suspected"]
    assert "6.0 * mad" in (next((loop.output_dir / "frames").rglob("*.py"))).read_text()


def test_a_folder_of_images_replays_as_images(tmp_path):
    sim = ParticleCoarseningImages(seed=1)
    for i in range(3):
        sim.acquire({}).save(str(tmp_path / "rec"), i, stem="img")
    inst = ReplayInstrument(str(tmp_path / "rec"), system_info=sim.system_info)
    assert inst.modality == "image" and len(inst) == 3
    frame = inst.acquire({})
    assert frame.image.shape == (256, 256) and len(frame.x) == 96
    assert frame.save(str(tmp_path / "out"), 0).endswith(".npy")


def test_the_change_signal_reads_the_power_spectrum_whole_and_by_quarter(tmp_path):
    m = resolve_modality("image")
    np.save(tmp_path / "a.npy", np.random.default_rng(0).random((128, 128)))
    names = list(m.read_signals(str(tmp_path / "a.npy")))
    assert names == ["whole field", "upper left quarter", "upper right quarter",
                     "lower left quarter", "lower right quarter"]
    np.save(tmp_path / "small.npy", np.random.default_rng(0).random((40, 40)))
    assert list(m.read_signals(str(tmp_path / "small.npy"))) == ["whole field"]
    # gain and offset are not a change
    a = np.random.default_rng(1).random((128, 128))
    np.save(tmp_path / "b.npy", a)
    np.save(tmp_path / "c.npy", 7.0 * a + 300.0)
    x1, y1 = m.read_signal(str(tmp_path / "b.npy"))
    x2, y2 = m.read_signal(str(tmp_path / "c.npy"))
    assert np.allclose(y1, y2, atol=1e-6)
    assert ImageModality().features({"extracted_features": {"n": 3, "ok": True, "label": "x", "d": 2.5}}) == {"n": 3.0, "d": 2.5}


def test_a_located_change_is_given_as_a_length_scale(tmp_path):
    m = ImageModality()
    np.save(tmp_path / "f.npy", np.zeros((256, 256), dtype=np.float32))
    info = {"experimental_details": {"spatial_info": {"field_of_view_x": 102.4, "field_of_view_y": 102.4,
                                                        "field_of_view_units": "nm"}}}
    where = [{"kind": "new", "x_from": 0.08, "x_to": 0.125, "x_peak": 0.10, "share": 1.0}]
    [w] = m.annotate_where(where, str(tmp_path / "f.npy"), info)          # 0.4 nm per pixel
    assert w["length_units"] == "nm" and w["length_peak"] == 4.0
    assert (w["length_from"], w["length_to"]) == (3.2, 5.0) and w["x_peak"] == 0.10
    [px] = m.annotate_where(where, str(tmp_path / "f.npy"), {})
    assert px["length_units"] == "px" and px["length_peak"] == 10.0
    assert resolve_modality("curve").annotate_where(where, "x.csv") == where


def test_a_rebuild_that_has_no_value_for_a_tracked_output_is_refused(armed):
    """Names are only asked for, so they are checked. Live on a real HAADF tile: a
    quick audit found no atomic columns and reported the tracked distance as null."""
    from types import SimpleNamespace
    sim, loop, _ = armed
    before = loop.recipe["id"]

    class Finished:
        def __init__(self, spec):
            pass

        def poll(self):
            return {"status": "success", "output_directory": str(loop.anchor_dir), "llm_calls": 9, "seconds": 280,
                    "pin_features": {"particle_count": 0.0},
                    "reported": ["mean_diameter_nm", "particle_count"]}
    loop._escalation_runner = Finished
    out = loop.escalate(sim.acquire({}).save(str(loop.output_dir / "in"), 5), background=False)
    assert out["event"] == "escalation_failed" and loop.recipe["id"] == before
    assert "no value for the tracked output(s) ['mean_diameter_nm']" in out["error"]
