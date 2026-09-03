"""Tests for the web backend (scilink/server/) — the streamlit-free ports.

Covers the pure/portable pieces offline: presenter classification, artifact
sweeps, upload conventions, path-traversal guard, session discovery, SSE
ring buffer. Agent-constructing paths (create/resume with real credentials)
are exercised by the live smoke flow, not here.
"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scilink.hitl import FeedbackRequest  # noqa: E402
from scilink.server import files as files_mod  # noqa: E402
from scilink.server.app import create_app  # noqa: E402
from scilink.server.artifacts import ArtifactTracker  # noqa: E402
from scilink.server.events import EventBuffer  # noqa: E402
from scilink.server.presenter import (  # noqa: E402
    parse_bestofn_review,
    parse_fanout_confirm,
    parse_plan_candidate_review,
    present_question,
)
from scilink.server.session_manager import SessionManager  # noqa: E402


# ── presenter ────────────────────────────────────────────────────

BESTOFN_CTX = """
BEST-OF-N CANDIDATES
Candidate 1: R²=0.912, approved=True, iterations=2
Candidate 2: R²=0.947, approved=True, iterations=3  <- judge pick
Candidate 3: R²=0.801, approved=False, iterations=5
"""

PLAN_CTX = """
PLAN CANDIDATES
── Candidate 1: Doped perovskite anneal sweep ──
── Candidate 2: Solvent-ratio DoE ──  judge pick
"""

FANOUT_CTX = """
parallel multi-dataset analysis
Complementarity verdict : complementary modalities
Join axis : temperature
• XRD series (5 files)
• Raman series (5 files)
Rationale : The two series probe the same transition.
"""


def test_parse_bestofn_review():
    cands, pick = parse_bestofn_review(BESTOFN_CTX, "accept candidate 2")
    assert [c["idx"] for c in cands] == [1, 2, 3]
    assert pick == 2
    assert "✓ approved" in cands[1]["label"]
    assert "✗ below gate" in cands[2]["label"]
    # gated on the prompt — a stale review block must not hijack
    assert parse_bestofn_review(BESTOFN_CTX, "Review the plan") is None
    assert parse_bestofn_review("", "accept candidate 1") is None


def test_parse_plan_candidate_review():
    cands, pick = parse_plan_candidate_review(PLAN_CTX, "accept plan candidate 2")
    assert len(cands) == 2 and pick == 2
    assert "Solvent-ratio DoE" in cands[1]["label"]
    assert parse_plan_candidate_review(PLAN_CTX, "accept candidate 2") is None


def test_parse_fanout_confirm():
    d = parse_fanout_confirm(FANOUT_CTX)
    assert d["verdict"] == "complementary modalities"
    assert d["join_axis"] == "temperature"
    assert len(d["branches"]) == 2


def _present(tmp_path, prompt="", kind="free_text", context="", origin=None,
             options=None):
    hreq = FeedbackRequest(prompt=prompt, kind=kind, origin=origin or {},
                           options=options)
    return present_question(hreq, context, str(tmp_path))


def test_present_widget_classification(tmp_path):
    assert _present(tmp_path)["widget"] == "generic"
    assert _present(tmp_path, kind="dataset_description")["widget"] == \
        "dataset_description"
    q = _present(tmp_path, context="x" * 10 + "\nCODE REVIEW\nfiles ready")
    assert q["widget"] == "code_review"
    q = _present(tmp_path, kind="keep_or_revert", options=["keep", "revert"])
    assert q["widget"] == "keep_revert"
    q = _present(tmp_path, origin={"stage": "fanout_confirm"},
                 context=FANOUT_CTX)
    assert q["widget"] == "fanout_confirm"
    assert q["fanout"]["join_axis"] == "temperature"
    q = _present(tmp_path, prompt="accept candidate 2", context=BESTOFN_CTX)
    assert q["widget"] == "bestofn" and q["judge_pick"] == 2
    q = _present(tmp_path, prompt="accept plan candidate 2", context=PLAN_CTX)
    assert q["widget"] == "plan_candidates" and q["judge_pick"] == 2
    # label sets ride the generic surface
    q = _present(tmp_path, context="REQUESTING FEEDBACK on the plan")
    assert q["labels"]["accept"] == "Approve plan"
    q = _present(tmp_path, kind="review_metrics")
    assert q["labels"]["accept"] == "Approve extraction"


def test_present_code_files_and_previews(tmp_path):
    review = tmp_path / "temp_code_review"
    review.mkdir()
    (review / "fitting_script.py").write_text("print('hi')")
    (tmp_path / "spectrum_fit_review.png").write_bytes(b"png")
    (tmp_path / "bestofn_candidate_2_review.png").write_bytes(b"png")
    q = _present(tmp_path, context="CODE REVIEW\nReview files in temp_code_review")
    assert q["code_files"][0]["name"] == "fitting_script.py"
    assert "spectrum_fit_review.png" in q["preview_images"]
    assert q["candidate_captions"]["bestofn_candidate_2_review.png"] == \
        "Candidate 2"


# ── artifacts ────────────────────────────────────────────────────

def test_artifact_tracker_sweep(tmp_path):
    tracker = ArtifactTracker(str(tmp_path))
    (tmp_path / "plot.png").write_bytes(b"x")
    (tmp_path / "fit_review.png").write_bytes(b"x")          # excluded
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "user.png").write_bytes(b"x")    # excluded
    out = tracker.sweep_turn(autonomy="autonomous")
    assert out["images"] == ["plot.png"]
    # second sweep: nothing new
    assert tracker.sweep_turn("autonomous")["images"] == []
    # html report suppresses images in the same turn
    (tmp_path / "new.png").write_bytes(b"x")
    (tmp_path / "report.html").write_text("<html></html>")
    out = tracker.sweep_turn("autonomous")
    assert out["images"] == [] and out["html_reports"][0]["name"] == "report.html"
    # rewritten report (new mtime) is re-surfaced — path+mtime identity
    time.sleep(0.01)
    (tmp_path / "report.html").write_text("<html>v2</html>")
    assert tracker.sweep_turn("autonomous")["html_reports"]


def test_artifact_tracker_md_rules(tmp_path):
    tracker = ArtifactTracker(str(tmp_path))
    (tmp_path / "brief.md").write_text("# ok")
    (tmp_path / "literature_search_1.md").write_text("bulk")     # bulk stem
    (tmp_path / "huge.md").write_text("x" * 70_000)              # too big
    out = tracker.sweep_turn("autonomous")
    assert [d["name"] for d in out["md_reports"]] == ["brief.md"]


def test_artifact_tracker_debug_subsampling(tmp_path):
    tracker = ArtifactTracker(str(tmp_path))
    for i in range(1, 6):
        (tmp_path / f"debug_{i}.png").write_bytes(b"x")
    assert tracker.sweep_turn("autonomous")["images"] == []      # not co-pilot
    tracker2 = ArtifactTracker(str(tmp_path))
    imgs = tracker2.sweep_turn("co-pilot")["images"]
    assert imgs == ["debug_1.png", "debug_3.png", "debug_5.png"]


def test_mark_all_existing_suppresses_resume_dump(tmp_path):
    (tmp_path / "old.png").write_bytes(b"x")
    (tmp_path / "old.html").write_text("<html></html>")
    tracker = ArtifactTracker(str(tmp_path))
    tracker.mark_all_existing()
    out = tracker.sweep_turn("autonomous")
    assert out["images"] == [] and out["html_reports"] == []


# ── files ────────────────────────────────────────────────────────

def test_save_uploads_conventions(tmp_path):
    one = files_mod.save_uploads(str(tmp_path), "data", [("a.csv", b"1,2")])
    assert one["paths"] == [str(tmp_path / "uploads" / "a.csv")]
    many = files_mod.save_uploads(str(tmp_path), "data",
                                  [("a.csv", b"1"), ("b.csv", b"2")])
    assert many["series_dir"] == str(tmp_path / "uploads" / "series")
    meta = files_mod.save_uploads(str(tmp_path), "metadata",
                                  [("metadata.json", b"{}"), ("s1.json", b"{}")])
    assert meta["global_metadata"].endswith("series/metadata.json")
    plan = files_mod.save_uploads(str(tmp_path), "knowledge", [("p.pdf", b"%")])
    assert plan["paths"] == [str(tmp_path / "knowledge" / "p.pdf")]
    pdata = files_mod.save_uploads(str(tmp_path), "planning_data",
                                   [("d.csv", b"1")])
    assert pdata["paths"] == [str(tmp_path / "data" / "d.csv")]
    with pytest.raises(files_mod.UploadError):
        files_mod.save_uploads(str(tmp_path), "data", [("evil.exe", b"")])
    with pytest.raises(files_mod.UploadError):
        files_mod.save_uploads(str(tmp_path), "nope", [("a.csv", b"")])


def test_resolve_safe_traversal(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    assert files_mod.resolve_safe(str(tmp_path), "ok.txt").name == "ok.txt"
    with pytest.raises(PermissionError):
        files_mod.resolve_safe(str(tmp_path), "../../etc/passwd")


# ── session discovery ────────────────────────────────────────────

def test_discover_resumable(tmp_path):
    good = tmp_path / "analysis_session_20260101_010101"
    good.mkdir()
    (good / "checkpoint.json").write_text(json.dumps(
        {"analysis_results": [1, 2], "current_data_path": "/x/spec.csv"}))
    empty = tmp_path / "analysis_session_20260101_020202"
    empty.mkdir()  # no checkpoint, no history -> skipped
    chat_only = tmp_path / "analysis_session_20260101_030303"
    chat_only.mkdir()
    (chat_only / "chat_history.json").write_text(json.dumps(
        [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "yo"}]))
    mgr = SessionManager(tmp_path)
    found = mgr.discover_resumable("analyze")
    ids = [f["id"] for f in found]
    assert good.name in ids and chat_only.name in ids and empty.name not in ids
    by_id = {f["id"]: f for f in found}
    assert by_id[good.name]["summary"] == {"analysis_count": 2,
                                           "data_file": "spec.csv"}
    assert by_id[chat_only.name]["summary"] == {"message_count": 1}
    assert by_id[good.name]["label"].endswith("2026-01-01 01:01:01")


# ── events ───────────────────────────────────────────────────────

def test_event_buffer_replay_and_stream():
    buf = EventBuffer(ring_size=10)
    for i in range(3):
        buf.emit("log", {"chunk": f"line{i}"})
    frames = []
    done = threading.Event()

    def consume():
        for frame in buf.sse_stream(last_event_id=1):
            frames.append(frame)
            if len([f for f in frames if "event:" in f]) >= 3:
                done.set()
                return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.05)
    buf.emit("status", {"status": "idle"})
    assert done.wait(2)
    # replayed events 2,3 then the live one
    assert "line1" in frames[0] and "line2" in frames[1]
    assert "status" in frames[2]
    buf.close()


# ── HTTP surface (no agent construction) ─────────────────────────

@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path, serve_frontend=False))


def test_config_endpoint(client):
    d = client.get("/api/v1/config").json()
    assert [m["key"] for m in d["modes"]] == ["meta", "analyze", "plan"]
    assert d["autonomy_options"]["meta"] == ["autopilot", "autonomous"]
    assert "api_key" in d["credentials"]
    assert all("value" not in v for v in d["credentials"].values())
    bedrock = client.get("/api/v1/config",
                         params={"model": "bedrock/us.anthropic.claude-opus-4-8"}).json()
    assert bedrock["provider"]["name"] == "bedrock"
    assert bedrock["provider"]["fields"][0]["name"] == "region"


def test_consent_required(client):
    r = client.post("/api/v1/sessions", json={
        "mode": "analyze", "model": "gpt-5.4", "autonomy": "co-pilot"})
    assert r.status_code == 400 and "Consent" in r.json()["detail"]


def test_missing_session_404(client):
    assert client.get("/api/v1/sessions/nope").status_code == 404
    assert client.post("/api/v1/sessions/nope/stop").status_code == 404


def test_file_endpoint_guard(client, tmp_path):
    # register a fake live session without an agent
    from scilink.server.session_manager import WebSession
    sdir = tmp_path / "analysis_session_20260101_050505"
    sdir.mkdir()
    (sdir / "plot.png").write_bytes(b"\x89PNG")
    mgr = client.app.state.manager
    mgr._sessions[sdir.name] = WebSession(
        id=sdir.name, session_dir=str(sdir), mode="analyze",
        model="gpt-5.4", autonomy="autonomous", agent=SimpleNamespace())
    ok = client.get(f"/api/v1/sessions/{sdir.name}/files",
                    params={"path": "plot.png"})
    assert ok.status_code == 200 and ok.content.startswith(b"\x89PNG")
    bad = client.get(f"/api/v1/sessions/{sdir.name}/files",
                     params={"path": "../../etc/passwd"})
    assert bad.status_code == 403
    missing = client.get(f"/api/v1/sessions/{sdir.name}/files",
                         params={"path": "gone.png"})
    assert missing.status_code == 404


def _fake_session(client, tmp_path, name="analysis_session_20260101_090909"):
    from scilink.server.session_manager import WebSession
    sdir = tmp_path / name
    sdir.mkdir(exist_ok=True)
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="autonomous",
                         agent=SimpleNamespace())
    client.app.state.manager._sessions[sdir.name] = session
    return session, sdir


def test_tree_endpoint(client, tmp_path):
    session, sdir = _fake_session(client, tmp_path)
    (sdir / "results").mkdir()
    (sdir / "results" / "plot.png").write_bytes(b"x")
    (sdir / "notes.md").write_text("# hi")
    (sdir / ".hidden").write_text("no")
    session.turn_started_at = time.time() - 10  # both files are "new"
    d = client.get(f"/api/v1/sessions/{sdir.name}/tree").json()
    names = [e["name"] for e in d["entries"]]
    assert names == ["results", "notes.md"]  # dirs first, hidden skipped
    assert d["entries"][0]["children"][0]["name"] == "plot.png"
    assert d["entries"][1]["new"] is True
    session.turn_started_at = time.time() + 10
    d = client.get(f"/api/v1/sessions/{sdir.name}/tree").json()
    assert d["entries"][1]["new"] is False


def test_table_endpoint(client, tmp_path):
    _, sdir = _fake_session(client, tmp_path)
    (sdir / "data.csv").write_text("a,b\n1,2\n3,4\n")
    d = client.get(f"/api/v1/sessions/{sdir.name}/table",
                   params={"path": "data.csv"}).json()
    assert d["columns"] == ["a", "b"] and d["rows"] == [[1, 2], [3, 4]]
    assert d["total_rows"] == 2 and d["truncated"] is False
    r = client.get(f"/api/v1/sessions/{sdir.name}/table",
                   params={"path": "../x.csv"})
    assert r.status_code == 403


def test_thumb_endpoint(client, tmp_path):
    np = pytest.importorskip("numpy")
    _, sdir = _fake_session(client, tmp_path)
    np.save(sdir / "img.npy", np.random.rand(64, 64))
    r = client.get(f"/api/v1/sessions/{sdir.name}/thumb",
                   params={"path": "img.npy", "size": 64, "cmap": "gray"})
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert r.headers["X-Preview-Kind"] == "heatmap"
    r = client.get(f"/api/v1/sessions/{sdir.name}/thumb",
                   params={"path": "img.npy", "cmap": "not-a-cmap"})
    assert r.status_code == 200  # invalid cmap falls back, not 500
    (sdir / "notes.txt").write_text("x")
    r = client.get(f"/api/v1/sessions/{sdir.name}/thumb",
                   params={"path": "notes.txt"})
    assert r.status_code == 422


def test_thumb_array_kinds(client, tmp_path):
    """1D data plots as a line; unrenderable arrays fail with an explanation."""
    np = pytest.importorskip("numpy")
    _, sdir = _fake_session(client, tmp_path,
                            name="analysis_session_20260101_101010")
    base = f"/api/v1/sessions/{sdir.name}/thumb"

    np.save(sdir / "spectrum.npy", np.sin(np.linspace(0, 6, 300)))
    r = client.get(base, params={"path": "spectrum.npy"})
    assert r.status_code == 200 and r.headers["X-Preview-Kind"] == "line"

    np.save(sdir / "xy.npy", np.column_stack(
        [np.arange(50.0), np.arange(50.0) ** 2]))
    r = client.get(base, params={"path": "xy.npy"})
    assert r.status_code == 200 and r.headers["X-Preview-Kind"] == "line"

    np.save(sdir / "scalar.npy", np.array(3.14))
    r = client.get(base, params={"path": "scalar.npy"})
    assert r.status_code == 422 and "scalar" in r.json()["detail"]

    np.save(sdir / "obj.npy", np.array([{"a": 1}], dtype=object),
            allow_pickle=True)
    r = client.get(base, params={"path": "obj.npy"})
    assert r.status_code == 422
    assert "download to inspect" in r.json()["detail"]
    assert "object" in r.json()["detail"]  # dtype named from the header


def test_zip_endpoint(client, tmp_path):
    import io
    import zipfile
    _, sdir = _fake_session(client, tmp_path)
    (sdir / "results").mkdir()
    (sdir / "results" / "a.txt").write_text("A")
    (sdir / "results" / "b.txt").write_text("B")
    r = client.get(f"/api/v1/sessions/{sdir.name}/zip",
                   params={"path": "results"})
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == ["a.txt", "b.txt"]
    assert client.get(f"/api/v1/sessions/{sdir.name}/zip",
                      params={"path": "../.."}).status_code == 403
    assert client.get(f"/api/v1/sessions/{sdir.name}/zip",
                      params={"path": "gone"}).status_code == 404


def test_provenance_endpoint(client, tmp_path):
    _, sdir = _fake_session(client, tmp_path)
    abs_file = sdir / "results" / "fit.png"
    (sdir / "results").mkdir()
    abs_file.write_bytes(b"x")
    clipped = "…lipped/prefix/" + sdir.name + "/results/other.png"
    rec = {"n": 1, "ts": "2026-08-30T18:19:00", "tool": "run_analysis",
           "args": "{}", "status": "ok", "summary": "fit done",
           "files": [str(abs_file), "/elsewhere/out.txt", clipped]}
    (sdir / "events.jsonl").write_text(json.dumps(rec) + "\n")
    d = client.get(f"/api/v1/sessions/{sdir.name}/provenance").json()
    ev = d["events"][0]
    assert ev["tool"] == "run_analysis" and ev["log"] == ""
    assert "results/fit.png" in ev["files"]
    assert "/elsewhere/out.txt" in ev["files"]  # outside paths kept verbatim
    assert "results/other.png" in ev["files"]   # clipped path recovered


def test_scan_new_images(tmp_path):
    from scilink.server.runner import _scan_new_images

    # Session root deliberately containing a skip token: must NOT filter
    # everything (the skip list matches relative paths only).
    root = tmp_path / "testloop_session"
    root.mkdir()
    (root / "old.png").write_bytes(b"x")
    seen: dict = {}
    _scan_new_images(str(root), seen)  # seed pass sees old.png

    (root / "fit_overlay.png").write_bytes(b"x")
    (root / "notes.txt").write_text("x")               # not an image
    (root / "elbow_plot_k.png").write_bytes(b"x")      # skip token in name
    (root / "previews").mkdir()
    (root / "previews" / "p.png").write_bytes(b"x")    # skip dir, top-level
    out = _scan_new_images(str(root), seen)
    assert [rel for rel, _label, _branch in out] == ["fit_overlay.png"]
    # already-seen files are not re-emitted
    assert _scan_new_images(str(root), seen) == []


def test_feedback_flow_and_409(client, tmp_path):
    """A fake blocking agent exercises the full turn/feedback/stop plumbing."""
    from scilink.server import runner as runner_mod
    from scilink.server.session_manager import WebSession

    class FakeAgent:
        def chat(self, text):
            from scilink import hitl
            answer = hitl.request_human_feedback(
                "Context please", kind="dataset_description",
                origin={"filename": "a.csv"})
            return f"got: {answer}"

    sdir = tmp_path / "analysis_session_20260101_060606"
    sdir.mkdir()
    mgr = client.app.state.manager
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="co-pilot",
                         agent=FakeAgent())
    mgr._sessions[sdir.name] = session

    r = client.post(f"/api/v1/sessions/{sdir.name}/messages",
                    json={"content": "analyze it"})
    assert r.status_code == 202
    # wait for the question to park
    for _ in range(100):
        if session.turn and session.turn.pending_question:
            break
        time.sleep(0.02)
    else:
        pytest.fail("question never parked")
    snap = client.get(f"/api/v1/sessions/{sdir.name}").json()
    assert snap["status"] == "awaiting_input"
    q = snap["pending_question"]
    assert q["widget"] == "dataset_description"

    # concurrent turn -> 409
    r = client.post(f"/api/v1/sessions/{sdir.name}/messages",
                    json={"content": "again"})
    assert r.status_code == 409

    # wrong request id -> 404
    r = client.post(f"/api/v1/sessions/{sdir.name}/feedback",
                    json={"request_id": "bogus", "response": "x"})
    assert r.status_code == 404

    r = client.post(f"/api/v1/sessions/{sdir.name}/feedback",
                    json={"request_id": q["request_id"],
                          "response": "a Raman spectrum"})
    assert r.status_code == 200
    session.turn.done.wait(5)
    for _ in range(100):
        if session.status == "idle" and session.chat_messages[-1]["role"] == "assistant":
            break
        time.sleep(0.02)
    assert session.chat_messages[-1]["content"] == "got: a Raman spectrum"
    # feedback_log written by the hitl chokepoint? (only when bound by the
    # orchestrator — FakeAgent doesn't bind one, so no assertion here)


def test_reset_session(client, tmp_path):
    from scilink.server.session_manager import WebSession

    class HangingAgent:
        def chat(self, text):
            from scilink import hitl
            hitl.request_human_feedback("waiting")
            print("post-stop print")  # raises AgentStoppedError after reset
            return "done"

    sdir = tmp_path / "analysis_session_20260101_080808"
    sdir.mkdir()
    mgr = client.app.state.manager
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="co-pilot",
                         agent=HangingAgent())
    mgr._sessions[sdir.name] = session
    client.post(f"/api/v1/sessions/{sdir.name}/messages", json={"content": "go"})
    for _ in range(100):
        if session.turn and session.turn.pending_question:
            break
        time.sleep(0.02)
    # reset while a question is parked: stops the turn and drops the session
    assert client.delete(f"/api/v1/sessions/{sdir.name}").json()["ok"] is True
    assert client.get(f"/api/v1/sessions/{sdir.name}").status_code == 404
    assert client.delete(f"/api/v1/sessions/{sdir.name}").status_code == 404
    session.turn.done.wait(5)
    assert not session.turn.thread.is_alive()


def test_analysis_image_events_during_turn(client, tmp_path):
    """A figure written mid-turn must reach the SSE buffer as an
    analysis_image event (with a session-relative path), plus a final sweep
    at turn end for late figures."""
    from scilink.server.session_manager import WebSession

    sdir = tmp_path / "analysis_session_20260101_121212"
    sdir.mkdir()
    results = sdir / "results"
    results.mkdir()

    class PlottingAgent:
        def chat(self, text):
            time.sleep(0.3)  # realistic: some work precedes the first plot
            (results / "fit_overlay.png").write_bytes(b"\x89PNG mid")
            print("plot written")
            time.sleep(2.6)  # let the watcher's 2s fs tick observe it
            (results / "late_dashboard.png").write_bytes(b"\x89PNG late")
            return "done"

    mgr = client.app.state.manager
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="autonomous",
                         agent=PlottingAgent())
    mgr._sessions[sdir.name] = session
    client.post(f"/api/v1/sessions/{sdir.name}/messages", json={"content": "go"})
    session.turn.done.wait(15)

    def emitted():
        return [ev.data["path"] for ev in session.events._ring
                if ev.type == "analysis_image"]
    # The watcher's final sweep runs shortly after the turn thread finishes.
    for _ in range(100):
        if len(emitted()) >= 2:
            break
        time.sleep(0.05)
    imgs = emitted()
    assert "results/fit_overlay.png" in imgs
    assert "results/late_dashboard.png" in imgs  # final turn-end sweep
    # ...and the filmstrip rides the snapshot for mid-run reattach.
    assert [i["path"] for i in session.turn.live_images] == imgs


def test_stop_unblocks_pending_question(client, tmp_path):
    from scilink.server.session_manager import WebSession

    class HangingAgent:
        def chat(self, text):
            from scilink import hitl
            hitl.request_human_feedback("waiting forever")
            print("never reached after stop")  # raises AgentStoppedError
            return "done"

    sdir = tmp_path / "analysis_session_20260101_070707"
    sdir.mkdir()
    mgr = client.app.state.manager
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze",
                         model="gpt-5.4", autonomy="co-pilot",
                         agent=HangingAgent())
    mgr._sessions[sdir.name] = session
    client.post(f"/api/v1/sessions/{sdir.name}/messages",
                json={"content": "go"})
    for _ in range(100):
        if session.turn and session.turn.pending_question:
            break
        time.sleep(0.02)
    r = client.post(f"/api/v1/sessions/{sdir.name}/stop")
    assert r.json()["stopped"] is True
    assert session.status == "idle"
    assert session.chat_messages[-1]["content"] == "Analysis stopped by user."
    session.turn.done.wait(5)
