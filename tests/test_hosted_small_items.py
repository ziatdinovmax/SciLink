"""Three small hosted-server items: one workers knob, Bedrock through the
task role, and an interrupted turn announced after a restart."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.utils.workers import max_workers, resolve_workers


def test_one_ceiling_over_three_pools(monkeypatch):
    for v in ("SCILINK_MAX_WORKERS", "SCILINK_FANOUT_MAX_WORKERS", "SCILINK_HS_SERIES_WORKERS",
              "SCILINK_CURVE_FIT_WORKERS"):
        monkeypatch.delenv(v, raising=False)
    assert max_workers() is None
    assert resolve_workers(None, "SCILINK_FANOUT_MAX_WORKERS", 4) == 4      # old defaults survive
    assert resolve_workers(None, "SCILINK_CURVE_FIT_WORKERS", 1) == 1
    monkeypatch.setenv("SCILINK_MAX_WORKERS", "2")
    assert resolve_workers(None, "SCILINK_FANOUT_MAX_WORKERS", 4) == 2      # the ceiling caps a default
    assert resolve_workers(None, "SCILINK_CURVE_FIT_WORKERS", 1) == 2       # and fills an unset pool
    monkeypatch.setenv("SCILINK_CURVE_FIT_WORKERS", "8")
    assert resolve_workers(None, "SCILINK_CURVE_FIT_WORKERS", 1) == 2       # a pool's own var is capped
    assert resolve_workers(1, "SCILINK_CURVE_FIT_WORKERS", 1) == 1          # explicit wins, still capped
    assert resolve_workers(9, "SCILINK_CURVE_FIT_WORKERS", 1) == 2
    monkeypatch.setenv("SCILINK_MAX_WORKERS", "auto")
    assert max_workers() >= 1
    monkeypatch.setenv("SCILINK_MAX_WORKERS", "nonsense")
    assert max_workers() is None
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _resolve_parallel_workers
    from scilink.agents.exp_agents.controllers.hyperspectral_series import resolve_series_workers
    monkeypatch.setenv("SCILINK_MAX_WORKERS", "3")
    monkeypatch.delenv("SCILINK_CURVE_FIT_WORKERS")
    assert _resolve_parallel_workers(None) == 3 and resolve_series_workers(None) == 3


def test_bedrock_accepts_the_task_role_as_ambient_credentials(monkeypatch):
    from scilink.providers import provider_for
    from scilink.server import session_manager as sm
    spec = provider_for("bedrock/us.anthropic.claude-opus-4-8")
    assert "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" in spec.cred_env
    for v in spec.cred_env:
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(sm.SessionError):
        sm._resolve_credentials("bedrock/us.anthropic.claude-opus-4-8", "", "", {"region": "us-east-1"})
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/abc")   # what ECS sets
    assert sm._resolve_credentials("bedrock/us.anthropic.claude-opus-4-8", "", "", {"region": "us-east-1"}) is None
    assert "AWS_BEARER_TOKEN_BEDROCK" not in __import__("os").environ                     # no empty token exported


def test_interrupted_turn_is_marked_at_shutdown_and_announced_on_resume(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    from scilink.server.runner import TurnState
    from scilink.server.session_manager import WebSession
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    app = create_app(tmp_path, serve_frontend=False)
    mgr = app.state.manager_for_user("default")
    sdir = tmp_path / "analysis_session_20260101_000000"; sdir.mkdir()
    (sdir / "checkpoint.json").write_text("{}")
    s = WebSession(id=sdir.name, session_dir=str(sdir), mode="analyze", model="m",
                   autonomy="autonomous", agent=SimpleNamespace())
    s.chat_messages.append({"role": "user", "content": "fit the spectrum"})
    s.turn = TurnState(is_running=True)
    mgr._sessions[s.id] = s
    with TestClient(app):
        pass                                             # startup + shutdown events
    marker = json.loads((sdir / "interrupted.json").read_text())
    assert marker["user_input"] == "fit the spectrum"
    # a quiet session gets no marker
    assert not (tmp_path / "other").exists()
    # resume: the note is shown once and the marker consumed
    mgr._sessions.clear()
    monkeypatch.setattr("scilink.server.session_manager._init_analysis_agent",
                        lambda *a, **k: SimpleNamespace())
    sess = mgr.resume(resume_dir=sdir.name, mode="analyze", model="m", autonomy="autonomous",
                      api_key="k", base_url="", provider_fields={}, fh_api_key="", mp_api_key="")
    notes = [m for m in sess.chat_messages if m.get("interrupted")]
    assert len(notes) == 1 and "interrupted by a server restart" in notes[0]["content"]
    assert "fit the spectrum" in notes[0]["content"]
    assert not (sdir / "interrupted.json").exists()
