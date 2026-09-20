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
SEEN = []          # what the fake agent was called with


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
            SEEN.append(kw)
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


def _wait(session, until, timeout=90.0):
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


# ── the reference: first frame, or a past analysis ───────────────

def test_a_past_analysis_in_the_session_can_be_the_reference(session, tmp_path):
    past = Path(session.session_dir) / "analysis" / "results" / "analysis_ref_CurveFit_001"
    (past / "scripts").mkdir(parents=True)
    (past / "scripts" / "fitting_script.py").write_text(SCRIPT)
    (past / "series_fit_results.json").write_text(json.dumps(
        {"results": [{"model_type": "one peak"}], "locked_config": {"physical_model": "Lorentzian + linear"}}))
    (past / "analysis_results.json").write_text(json.dumps({
        "status": "success", "fitting_parameters": {"peak_1": {"center": 6.0}},
        "fit_quality": {"r_squared": 0.99}}))
    [cand] = live_api.snapshot(session)["analyses"]
    assert cand["path"] == "analysis/results/analysis_ref_CurveFit_001"
    assert cand["model"] == "Lorentzian + linear" and cand["has_data"] is False

    live_api.start(session, {**CONFIG, "reference_source": "analysis", "reference_analysis": cand["path"]})
    snap = _wait(session, lambda s: s["state"] in ("done", "error"))
    assert snap["state"] == "done", snap.get("error")
    setup = next(e for e in snap["events"] if e["event"] == "setup")
    assert setup["source"] == "anchor"                       # adopted: no reference analysis was run
    assert not (Path(snap["run_dir"]) / "reference").exists()
    assert len(snap["frames"]) == 4


def test_the_reference_must_be_inside_the_session(session, tmp_path):
    live_api.start(session, {**CONFIG, "reference_source": "analysis", "reference_analysis": "../.."})
    snap = _wait(session, lambda s: s["state"] == "error")
    assert "not a curve-fit run in this session" in snap["error"]


def test_open_ended_until_stopped_and_the_log_is_read_incrementally(session):
    live_api.start(session, {**CONFIG, "n_frames": None, "interval_s": 0.01})
    snap = _wait(session, lambda s: s["state"] == "running" and len(s["frames"]) >= 3)
    assert snap["n_frames_total"] is None
    run = live_api._RUNS[session.id]
    offset = run._tail.offset
    assert offset > 0
    _wait(session, lambda s: s["status"]["frames"] >= 6)
    assert run._tail.offset > offset                         # continued from where it stopped
    live_api.stop(session)
    done = _wait(session, lambda s: s["state"] == "stopped")
    assert done["status"]["frames"] == len(done["frames"]) == done["status"]["clean_frames"]
    assert done["output_keys"] == ["peak_1_center"]          # nothing pinned: the recipe's own names


def test_the_frame_deadline_comes_from_the_page(session):
    live_api.start(session, {**CONFIG, "frame_deadline_s": 0.000001})
    snap = _wait(session, lambda s: s["state"] == "done")
    assert all(f["flags"] == ["deadline_missed"] for f in snap["frames"])
    assert snap["status"]["clean_frames"] == 0 and not any(
        e["event"] == "escalation_started" for e in snap["events"])      # slow is not a breach
    live_api.clear(session)
    live_api.start(session, {**CONFIG, "frame_deadline_s": None})          # no deadline at all
    snap = _wait(session, lambda s: s["state"] == "done")
    assert all(f["flags"] == [] for f in snap["frames"])


# ── an instrument behind an MCP server connected in the MCP tab ──

def test_an_mcp_server_in_the_session_can_be_the_instrument(session):
    from tests.test_mcp_instrument import TOOL, FakeConnection
    x = [float(i) for i in range(40)]
    conn = FakeConnection(reply={"x": x, "y": [1.0 + (i == 20) * 5 for i in range(40)],
                                 "meta": {"T_K": 300}})
    session.agent._mcp_connections = {"raman-lab": conn}
    assert live_api.snapshot(session)["mcp_servers"] == [
        {"name": "raman-lab", "tools": ["acquire_spectrum"]}]
    base = {**CONFIG, "instrument": "mcp", "mcp_server": "raman-lab", "mcp_tool": "acquire_spectrum"}
    with pytest.raises(LiveError, match="technique"):                # the server did not describe itself
        live_api.start(session, dict(base))
    with pytest.raises(LiveError, match="Connect it in the MCP tab"):
        live_api.start(session, {**base, "mcp_server": "nope"})
    live_api.start(session, {**base, "system_info": {"technique": "Raman spectroscopy"}})
    snap = _wait(session, lambda s: s["state"] in ("done", "error"))
    assert snap["state"] == "done", snap.get("error")
    assert len(snap["frames"]) == 4 and len(conn.calls) == 5         # reference + four frames
    info = snap["instrument"]
    assert "integration_s" in info["schema"] and any("stage_z_um" in h for h in info["held"])
    # the measurement's own metadata reached the analysis through the sidecar
    assert snap["frames"][0]["params"]["integration_s"] == 1.0


def test_the_page_gets_the_change_signal_and_the_audit_setting(session):
    live_api.start(session, {**CONFIG, "n_frames": 6, "audit_every": 4})
    snap = _wait(session, lambda s: s["state"] == "done")
    assert all("drift_fraction" in f["gate"] for f in snap["frames"][1:])
    assert snap["status"]["drift_fraction_bar"] == 0.1 and snap["status"]["audits"] == 0
    assert snap["last_audit"] is None
    assert live_api._RUNS[session.id].loop.audit_every == 4


def test_a_lasting_change_reaches_the_page_as_a_novelty_with_the_frame_to_hand_to_chat(session):
    SEEN.clear()
    live_api.start(session, {**CONFIG, "n_frames": 31, "notes": "the tip may cross a stiff inclusion",
                             "on_change": "report"})
    snap = _wait(session, lambda s: s["state"] in ("done", "error"))
    assert snap["state"] == "done", snap.get("error")
    [novelty] = snap["novelties"]                              # the AFM simulator's inclusion, frame 26
    assert 24 <= novelty["since_step"] <= 27 and novelty["recipe_fits"] is True
    assert novelty["fraction"] > 0.5 and novelty["window_share"] < 0.9     # the curve now ends early
    kinds = [w["kind"] for w in novelty["where"]]                          # located where it overlaps, and
    assert kinds[-1] == "window" and "new" in kinds                        # the lost part of the axis is named
    assert novelty["frame_path"].startswith("live/run_001/loop/incoming/frame_")
    assert Path(novelty["frame_abs_path"]).is_file()           # what the Chat hand-off sends
    # a live run's per-frame files are not artifacts of whatever chat turn finishes meanwhile
    from scilink.server.artifacts import ArtifactTracker
    tracker = ArtifactTracker(session.session_dir)
    (Path(session.session_dir) / "live" / "run_001" / "frame_report.html").write_text("<html></html>")
    (Path(session.session_dir) / "analysis_report.html").write_text("<html></html>")
    assert [Path(p).name for p in tracker.find_new_html_reports()] == ["analysis_report.html"]
    assert any(e["event"] == "state_accepted" for e in snap["events"])
    assert snap["status"]["reanchors"] == 0                    # reported and tracked: nothing rebuilt
    # the user's notes are context for every analysis
    assert SEEN[0]["system_info"]["notes_from_the_user"] == "the tip may cross a stiff inclusion"
    assert live_api._RUNS[session.id].loop.on_change == "report"


def test_a_run_can_wait_at_a_novelty_for_the_person_to_decide(session):
    live_api.start(session, {**CONFIG, "n_frames": 31, "pause_on": ["novelty"]})
    snap = _wait(session, lambda s: s["state"] in ("paused", "done", "error"))
    assert snap["state"] == "paused", snap.get("error")
    assert snap["paused"]["why"] == "novelty" and 24 <= snap["paused"]["since_step"] <= 27
    assert snap["paused"]["experiment_held"] is False          # a simulator only stops acquiring
    n = len(snap["frames"])
    time.sleep(0.4)
    assert len(live_api.snapshot(session)["frames"]) == n      # nothing is acquired while it waits
    with pytest.raises(live_api.LiveError):                    # a bad answer is refused, the wait goes on
        live_api.decide(session, {"action": "resume", "params": {"trigger_force_nN": 1e9}})
    with pytest.raises(live_api.LiveError):                    # and no second run starts meanwhile
        live_api.start(session, CONFIG)
    live_api.decide(session, {"action": "resume", "params": {"trigger_force_nN": 12.5}})
    snap = _wait(session, lambda s: s["state"] == "done")
    assert snap["frames"][-1]["params"]["trigger_force_nN"] == 12.5 and snap["paused"] is None
    kinds = [e["event"] for e in snap["events"]]
    assert kinds.index("novelty") < kinds.index("paused") < kinds.index("resumed") < kinds.index("state_accepted")
    resumed = next(e for e in snap["events"] if e["event"] == "resumed")
    assert resumed["decision"] == "change" and resumed["params"] == {"trigger_force_nN": 12.5}
    assert snap["instrument"]["id"] == "afm_force_curve"
    with pytest.raises(live_api.LiveError):
        live_api.decide(session, {"action": "resume"})         # nothing to answer any more


def test_stop_ends_a_pause_and_a_time_limit_resumes_unchanged(session):
    live_api.start(session, {**CONFIG, "n_frames": 31, "pause_on": ["novelty"]})
    _wait(session, lambda s: s["state"] == "paused")
    live_api.decide(session, {"action": "stop"})
    snap = _wait(session, lambda s: s["state"] in ("stopped", "done"))
    assert snap["state"] == "stopped" and len(snap["frames"]) < 31
    live_api.clear(session)
    live_api.start(session, {**CONFIG, "n_frames": 31, "pause_on": ["novelty"], "pause_timeout_s": 0.3})
    snap = _wait(session, lambda s: s["state"] == "done")
    assert len(snap["frames"]) == 31 and "went on unchanged" in snap["note"]


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
