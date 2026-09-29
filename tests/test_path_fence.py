"""On a hosted server every path an agent is handed stays inside the
workspace; on a laptop nothing changes. The fence is checked by parameter
name at all four tool dispatchers and inside the resolvers that used to fall
back to the process cwd."""
import json
import os
from pathlib import Path

import pytest

from scilink.utils import path_fence as pf
from scilink.utils.file_io import resolve_user_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.delenv("SCILINK_FILE_ROOTS", raising=False)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_MODELS", raising=False)
    # the temp dir is always an allowed root, and pytest's tmp_path lives in
    # it: point the fence's idea of "temp" elsewhere so "outside" means outside
    (tmp_path / "tmpdir").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmpdir"))


def _ws(tmp_path):
    ws = tmp_path / "ws"; (ws / "sessions" / "s1" / "uploads").mkdir(parents=True)
    (ws / "sessions" / "s1" / "uploads" / "a.csv").write_text("x,y\n1,2\n")
    out = tmp_path / "elsewhere"; out.mkdir()
    (out / "secret.txt").write_text("nope")
    return ws, out


def test_fence_is_off_unless_roots_or_env(tmp_path, monkeypatch):
    assert pf.PathFence.build(tmp_path / "s") is None
    monkeypatch.setenv("SCILINK_FILE_ROOTS", str(tmp_path / "r1") + os.pathsep + str(tmp_path / "r2"))
    f = pf.PathFence.build(tmp_path / "s")
    assert f is not None and (tmp_path / "r2").resolve() in f.roots and (tmp_path / "s").resolve() in f.roots
    assert (tmp_path / "home").resolve() in f.roots          # the persistent store is always allowed


def test_allows_check_and_patterns(tmp_path):
    ws, out = _ws(tmp_path)
    f = pf.PathFence([ws])
    assert f.allows(ws / "sessions" / "s1") and f.allows(str(ws))
    assert not f.allows(out / "secret.txt") and not f.allows(ws / ".." / "elsewhere")
    with pytest.raises(pf.PathFenceError, match="outside this workspace"):
        f.check(out / "secret.txt")
    f.check_pattern(str(ws / "sessions" / "*" / "*.csv"))
    with pytest.raises(pf.PathFenceError):
        f.check_pattern(str(out / "*.txt"))
    with pytest.raises(pf.PathFenceError):
        f.check_pattern("/etc/*")


def test_refuse_tool_args_by_name(tmp_path):
    ws, out = _ws(tmp_path)
    f = pf.PathFence([ws]); base = ws / "sessions" / "s1"
    ok = lambda **kw: f.refuse_tool_args(kw, base) is None
    bad = lambda **kw: f.refuse_tool_args(kw, base) is not None
    assert ok(file_path=str(base / "uploads" / "a.csv")) and ok(file_path="uploads/a.csv")
    assert bad(file_path=str(out / "secret.txt")) and bad(file_path="../../../elsewhere/secret.txt")
    assert bad(paths=[str(base / "uploads" / "a.csv"), "/etc/hosts"])
    assert bad(data_path="/etc/*") and ok(data_path="uploads/*.csv")
    assert bad(knowledge_paths=f"{base}/uploads, {out}") and ok(knowledge_paths="uploads, knowledge")
    assert ok(literature_context="the perovskite literature says nothing here") \
        and bad(literature_context=f"{out}/secret.txt")
    assert ok(skill="xrd_profile") and bad(skill=str(out / "evil.md"))
    assert bad(branches=[{"data_path": str(base), "pattern": f"{out}/*.npy"}]) \
        and ok(branches=[{"data_path": str(base), "pattern": "*.npy"}])
    assert bad(conditions={str(out / "secret.txt"): {"T": 1}}) and ok(conditions={"a.csv": {"T": 1}})
    assert ok(task="read /etc/passwd please", context={"note": "/etc/hosts"})   # prose is not a path arg
    assert bad(output_dir=str(out)) and bad(local_dir="/tmp/../etc")


def test_resolve_user_path_skips_the_cwd_on_a_fenced_server(tmp_path, monkeypatch):
    ws, out = _ws(tmp_path)
    (out / "data.csv").write_text("x\n")
    monkeypatch.chdir(out)
    base = ws / "sessions" / "s1"
    assert resolve_user_path("data.csv", base) == out / "data.csv"          # laptop: cwd fallback
    with pf.bound(pf.PathFence([ws])):
        assert resolve_user_path("data.csv", base) == base / "data.csv"     # fenced: session form
        assert resolve_user_path("uploads/a.csv", base) == base / "uploads" / "a.csv"
        with pytest.raises(pf.PathFenceError):
            resolve_user_path(str(out / "data.csv"), base)


def _error(res):
    d = json.loads(res)
    return d.get("status") == "error" and "outside this workspace" in d.get("message", "")


def test_analysis_dispatcher_fences_reads_and_globs(tmp_path, monkeypatch):
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
    ws, out = _ws(tmp_path)
    ag = AnalysisOrchestratorAgent(api_key="sk-dummy", base_dir=str(ws / "sessions" / "s1"),
                                   file_roots=[str(ws)])
    assert _error(ag.tools.execute_tool("read_file", file_path=str(out / "secret.txt")))
    assert _error(ag.tools.execute_tool("examine_data", data_path=str(out / "*.txt")))
    good = json.loads(ag.tools.execute_tool("read_file", file_path="uploads/a.csv"))
    assert good.get("status") != "error" or "outside" not in good.get("message", "")
    open_ag = AnalysisOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path / "open"))
    assert open_ag.path_fence is None
    assert not _error(open_ag.tools.execute_tool("read_file", file_path=str(out / "secret.txt")))


def test_planning_dispatcher_fences_its_data_path_search(tmp_path, monkeypatch):
    from scilink.agents.planning_agents.planning_orchestrator import (
        AutonomyLevel, PlanningOrchestratorAgent)
    ws, out = _ws(tmp_path)
    (out / "data.csv").write_text("x,y\n1,2\n")
    monkeypatch.chdir(out)                                      # the search list reaches ./
    ag = PlanningOrchestratorAgent(api_key="sk-dummy", base_dir=str(ws / "sessions" / "p1"),
                                   autonomy_level=AutonomyLevel.CO_PILOT, file_roots=[str(ws)])
    assert _error(ag.tools.execute_tool("read_file", file_path=str(out / "data.csv")))
    res = json.loads(ag.tools.execute_tool("read_file", file_path="data.csv"))
    assert res.get("status") == "error" and "outside" in res.get("message", "")


def test_simulation_and_meta_dispatchers_and_child_inheritance(tmp_path):
    from scilink.agents.sim_agents.simulation_orchestrator import SimulationOrchestratorAgent
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    ws, out = _ws(tmp_path)
    sim = SimulationOrchestratorAgent(api_key="sk-dummy", base_dir=str(ws / "sessions" / "sim"),
                                      file_roots=[str(ws)])
    assert _error(sim.tools.execute_tool("read_file", file_path=str(out / "secret.txt")))
    assert _error(sim.tools.execute_tool("analyze_output", output_dir=str(out), research_goal="x"))
    meta = MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(ws / "sessions" / "m"),
                                 file_roots=[str(ws)])
    assert _error(meta.tools.execute_tool("read_file", file_path=str(out / "secret.txt")))
    assert _error(meta.tools.execute_tool("view_image", paths=[str(out / "secret.txt")]))
    child = meta._get_analysis_child()
    assert child.path_fence is not None and ws.resolve() in child.path_fence.roots
    assert _error(child.tools.execute_tool("read_file", file_path=str(out / "secret.txt")))


def test_web_server_fences_remote_deployments_only(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from scilink.server.app import create_app
    from scilink.server.session_manager import SessionManager
    assert SessionManager(tmp_path, fenced=True)._file_roots == [str(tmp_path.resolve())]
    assert SessionManager(tmp_path)._file_roots is None
    remote = create_app(tmp_path, serve_frontend=False, local_files=False)
    assert remote.state.manager.fenced
    local = create_app(tmp_path, serve_frontend=False)
    assert not local.state.manager.fenced
    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr("scilink.agents.exp_agents.analysis_orchestrator.AnalysisOrchestratorAgent", Fake)
    from scilink.server import session_manager as sm
    sm._init_analysis_agent(tmp_path / "s", "k", "m", "", "autonomous", "", file_roots=["/w"])
    assert seen["file_roots"] == ["/w"]
    # a skill reference is two names, never a path
    c = TestClient(local)
    assert c.get("/api/v1/memory/skills/..%2F..%2Fetc/passwd").status_code in (400, 404)
    assert c.get("/api/v1/memory/skills/curve_fitting/x%2F..%2Fy").status_code in (400, 404)
