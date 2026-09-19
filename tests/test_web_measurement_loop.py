"""The Live tab's backend (scilink/server/live_api.py) and its routes.

A live run is a background job the page only observes. These tests drive one
end to end with a fake curve agent behind the real MeasurementLoop and a real
simulator: arm → stream → snapshot → operator parameters → stop, plus the
guards on what may be loaded as an instrument. No LLM calls anywhere.
"""

import functools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import scilink.live as live_pkg
from scilink.live import MeasurementLoop
from scilink.live.instruments import Frame, Instrument
from scilink.server import live_api
from scilink.server.live_api import LiveError

SCRIPT = "print('FIT_RESULTS_JSON: {}')\n"


def _anchor(root: Path) -> Path:
    d = root / "reference_run"
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "scripts" / "fitting_script.py").write_text(SCRIPT)
    (d / "series_fit_results.json").write_text(json.dumps(
        {"results": [{"model_type": "one peak"}], "locked_config": {"physical_model": "one peak"}}))
    (d / "analysis_results.json").write_text(json.dumps({
        "status": "success", "fitting_parameters": {"peak_1": {"center": 6.0}},
        "fit_quality": {"r_squared": 0.99}}))
    return d


def _fake_agent(anchor: Path):
    class Agent:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def analyze(self, data, **kw):
            out = str(anchor) if "reference" in Path(str(data)).name else self.output_dir
            return {"status": "success", "output_directory": out,
                    "fitting_parameters": {"peak_1": {"center": 6.0}},
                    "fit_quality": {"r_squared": 0.99},
                    "reuse_validity": {"verdict": "good", "r_squared": 0.99, "threshold": 0.95,
                                       "drift": "none", "fingerprint_similarity": 0.99},
                    "stage_timings": {"llm_calls": 0}}
    return Agent


@pytest.fixture
def session(tmp_path, monkeypatch):
    live_api._RUNS.clear()
    monkeypatch.setattr(live_pkg, "MeasurementLoop", functools.partial(
        MeasurementLoop, agent_factory=_fake_agent(_anchor(tmp_path))))
    sdir = tmp_path / "session"
    sdir.mkdir()
    yield SimpleNamespace(id="s1", session_dir=str(sdir),
                          agent=SimpleNamespace(model_name="m", api_key=None, base_url=None))
    for run in live_api._RUNS.values():
        run.stop()
    live_api._RUNS.clear()


def _wait(session, until, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = live_api.snapshot(session)
        if until(snap):
            return snap
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state {snap['state']} error={snap.get('error')}")


CONFIG = {"instrument": "afm_force_curve", "n_frames": 4, "interval_s": 0.0,
          "pin_outputs": False, "recommender": "none", "auto_escalate": False}


def test_the_four_simulated_experiments_are_offered():
    names = {s["name"] for s in live_api.list_simulators()}
    assert names == {"beamline_xrd", "insitu_raman", "afm_force_curve", "stm_didv"}
    for s in live_api.list_simulators():
        assert s["schema"] and s["outputs"] and s["events"] and s["about"]


def test_idle_session_lists_simulators(session):
    snap = live_api.snapshot(session)
    assert snap["state"] == "idle" and len(snap["simulators"]) == 4


def test_a_run_arms_streams_and_finishes(session):
    assert live_api.start(session, dict(CONFIG))["state"] in ("arming", "running", "done")
    snap = _wait(session, lambda s: s["state"] == "done")
    assert [f["step"] for f in snap["frames"]] == [1, 2, 3, 4]
    assert snap["latest"]["step"] == 4
    assert snap["status"]["llm_calls_in_frames"] == 0
    assert snap["frames"][-1]["features"]["peak_1_center"] == 6.0
    # the simulator's ground truth rides along, for the page only
    assert "stiffness_N_per_m" in snap["frames"][-1]["truth"]
    assert len(snap["latest"]["x"]) == len(snap["latest"]["y"]) > 10
    assert Path(snap["run_dir"]).parent == Path(session.session_dir) / "live"
    assert any(e["event"] == "setup" for e in snap["events"])


def test_one_run_per_session_and_stop(session):
    live_api.start(session, {**CONFIG, "n_frames": 500, "interval_s": 0.05})
    with pytest.raises(LiveError) as e:
        live_api.start(session, dict(CONFIG))
    assert e.value.status == 409
    with pytest.raises(LiveError):
        live_api.clear(session)
    _wait(session, lambda s: s["state"] == "running" and s["frames"])
    live_api.stop(session)
    snap = _wait(session, lambda s: s["state"] == "stopped")
    assert 0 < len(snap["frames"]) < 500
    assert live_api.clear(session)["state"] == "idle"


def test_operator_parameters_take_effect_and_bad_ones_are_refused(session):
    live_api.start(session, {**CONFIG, "n_frames": 400, "interval_s": 0.02})
    _wait(session, lambda s: s["state"] == "running" and s["frames"])
    with pytest.raises(LiveError) as e:
        live_api.set_params(session, {"trigger_force_nN": 1e9})
    assert e.value.status == 400
    with pytest.raises(LiveError):
        live_api.set_params(session, {"no_such_knob": 1})
    live_api.set_params(session, {"trigger_force_nN": 12.5})
    snap = _wait(session, lambda s: s["current_params"].get("trigger_force_nN") == 12.5)
    assert snap["frames"][-1]["params"]["trigger_force_nN"] == 12.5


def test_no_running_loop_no_parameters(session):
    with pytest.raises(LiveError) as e:
        live_api.set_params(session, {"trigger_force_nN": 5})
    assert e.value.status == 409


class MyInstrument(Instrument):
    name = "mine"

    def acquire(self, params):
        return Frame(x=[0, 1, 2], y=[1, 2, 1], params=params)


def test_custom_instrument_by_import_path():
    inst = live_api._make_instrument(f"{__name__}:MyInstrument", 0)
    assert isinstance(inst, MyInstrument)


def test_a_named_attribute_that_is_not_an_instrument_is_never_called():
    with pytest.raises(LiveError) as e:
        live_api._make_instrument("os:abort", 0)      # would kill the process if called
    assert "not a scilink.live.Instrument" in e.value.message
    with pytest.raises(LiveError):
        live_api._make_instrument("no.such.module:X", 0)
    with pytest.raises(LiveError):
        live_api._make_instrument("not_a_simulator", 0)


def test_custom_instruments_are_local_only():
    with pytest.raises(LiveError) as e:
        live_api._make_instrument(f"{__name__}:MyInstrument", 0, allow_custom=False)
    assert e.value.status == 403
    assert live_api._make_instrument("stm_didv", 0, allow_custom=False).name == "stm_didv"


def test_a_failed_run_reports_the_error(session):
    live_api.start(session, {**CONFIG, "recommender": "gp"})     # GP without an objective
    snap = _wait(session, lambda s: s["state"] == "error")
    assert "objective" in snap["error"]


# ── replaying a folder of recorded measurements ──────────────────

def _recording(tmp_path, n=5):
    d = tmp_path / "recorded"
    d.mkdir()
    for i in range(1, n + 1):
        (d / f"scan_{i}.csv").write_text("shift,counts\n" + "\n".join(
            f"{x},{100 + 50 * (x == 5) + i}" for x in range(10)))
    return d


def test_a_recorded_folder_streams_through_the_loop(session, tmp_path):
    d = _recording(tmp_path)
    live_api.start(session, {"instrument": "replay", "replay_dir": str(d), "n_frames": 60,
                             "interval_s": 0.0, "pin_outputs": False, "auto_escalate": False,
                             "system_info": {"technique": "Raman spectroscopy", "sample": " "},
                             "recommender": "llm", "objective": "anything"})
    snap = _wait(session, lambda s: s["state"] in ("done", "error"))
    assert snap["state"] == "done", snap.get("error")
    # the first file is the reference; the other four are the stream
    assert [f["step"] for f in snap["frames"]] == [1, 2, 3, 4] and snap["n_frames_total"] == 4
    assert snap["instrument"]["technique"] == "Raman spectroscopy"
    assert snap["instrument"]["schema"] == {} and snap["recommendation"] is None   # nothing to steer
    assert snap["frames"][0]["truth"] == {}


def test_replay_needs_a_folder_a_technique_and_a_local_server(session, tmp_path):
    d = _recording(tmp_path)
    for cfg, status in (({"replay_dir": str(d)}, 400),                                   # no technique
                        ({"system_info": {"technique": "XRD"}}, 400),                    # no folder
                        ({"replay_dir": str(tmp_path / "nope"), "system_info": {"technique": "XRD"}}, 400)):
        with pytest.raises(LiveError) as e:
            live_api.start(session, {"instrument": "replay", **cfg})
        assert e.value.status == status
    with pytest.raises(LiveError) as e:
        live_api.start(session, {"instrument": "replay", "replay_dir": str(d),
                                 "system_info": {"technique": "XRD"}}, allow_custom=False)
    assert e.value.status == 403


# ── routes ───────────────────────────────────────────────────────

def test_routes(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    client = TestClient(create_app(session_root=tmp_path, serve_frontend=False))
    sims = client.get("/api/v1/live/simulators").json()["simulators"]
    assert len(sims) == 4
    assert client.get("/api/v1/sessions/nope/live").status_code == 404
    assert client.post("/api/v1/sessions/nope/live/start", json={}).status_code == 404
