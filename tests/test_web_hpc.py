"""The web UI's HPC panel: connect / status / disconnect for a Mission Control
session, and the hand-off to the simulation specialist. No real SSH: the
connection is a fake that records what it was asked to do."""
import sys
from pathlib import Path
from types import SimpleNamespace

import paramiko
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.server.app import create_app  # noqa: E402
from scilink.server.session_manager import WebSession  # noqa: E402

PASSWORD, CODE = "s3cret", "123456"


class FakeConn:
    instances = []
    fail_with = None

    def __init__(self, profile):
        self.profile = profile
        self.connect_kwargs = None
        self.connected = False
        self.disconnected = False
        FakeConn.instances.append(self)

    def connect(self, **kw):
        self.connect_kwargs = kw
        if FakeConn.fail_with is not None:
            raise FakeConn.fail_with
        self.connected = True

    @property
    def is_connected(self):
        return self.connected

    def home_dir(self):
        return "/global/homes/a/alice"

    def disconnect(self):
        self.connected = False
        self.disconnected = True


@pytest.fixture(autouse=True)
def fake_ssh(monkeypatch):
    import scilink.hpc as hpc
    import scilink.hpc.connection as conn_mod
    FakeConn.instances, FakeConn.fail_with = [], None
    monkeypatch.setattr(conn_mod, "HPCConnection", FakeConn)
    monkeypatch.setattr(hpc, "detect_scheduler", lambda c: SimpleNamespace(name="SLURM"))


class MetaLike:
    """Stands in for MetaOrchestratorAgent's hook."""

    def __init__(self):
        self.calls = []

    def set_hpc_connection(self, connection, scheduler=None):
        self.calls.append((connection, scheduler))


def _client(tmp_path, local_files=True):
    return TestClient(create_app(tmp_path, serve_frontend=False, local_files=local_files))


def _session(client, tmp_path, mode="meta", agent=None):
    sdir = tmp_path / f"{mode}_session_20261107_090909"
    sdir.mkdir(exist_ok=True)
    s = WebSession(id=sdir.name, session_dir=str(sdir), mode=mode, model="m",
                   autonomy="autopilot", agent=agent if agent is not None else MetaLike())
    client.app.state.manager._sessions[sdir.name] = s
    return s, f"/api/v1/sessions/{sdir.name}/hpc"


_NERSC = {"hostname": "perlmutter.nersc.gov", "username": "alice",
          "auth_method": "password", "password": PASSWORD, "otp": CODE}


def test_connect_with_a_one_time_code(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    r = c.post(f"{base}/connect", json=_NERSC)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["connected"] and d["hostname"] == "perlmutter.nersc.gov"
    assert d["mfa"] is True and d["scheduler"] == "SLURM"
    conn = FakeConn.instances[-1]
    assert conn.connect_kwargs == {"password": PASSWORD, "key_passphrase": "", "otp": CODE}
    assert s.agent.calls[-1][0] is conn and s.agent.calls[-1][1].name == "SLURM"
    # no secret is kept or returned
    assert PASSWORD not in r.text and CODE not in r.text
    assert PASSWORD not in str(s.hpc_info) and CODE not in str(s.hpc_info)
    assert c.get(base).json()["connected"] is True


def test_only_mission_control_sessions_take_a_connection(tmp_path):
    c = _client(tmp_path)
    _, base = _session(c, tmp_path, mode="analyze", agent=SimpleNamespace())
    r = c.post(f"{base}/connect", json=_NERSC)
    assert r.status_code == 400 and "Mission Control" in r.json()["detail"]


def test_a_shared_server_refuses_key_login_and_jump_hosts(tmp_path):
    c = _client(tmp_path, local_files=False)
    _, base = _session(c, tmp_path)
    key = {**_NERSC, "auth_method": "key", "password": ""}
    assert c.post(f"{base}/connect", json=key).status_code == 403
    jump = {**_NERSC, "proxy_jump": "alice@bastion"}
    assert c.post(f"{base}/connect", json=jump).status_code == 403
    assert not FakeConn.instances                      # nothing was attempted
    assert c.post(f"{base}/connect", json=_NERSC).status_code == 200


def test_a_signed_in_local_server_counts_as_shared(tmp_path):
    # A loopback bind behind sign-in (tokens, --users, an auth proxy) serves
    # other people: the server account's keys must not be theirs to use.
    from scilink.server.auth import AuthConfig
    app = create_app(tmp_path, serve_frontend=False, auth=AuthConfig.single("tok-123456789012"))
    c = TestClient(app, headers={"Authorization": "Bearer tok-123456789012"})
    _, base = _session(c, tmp_path)
    key = {**_NERSC, "auth_method": "key", "password": ""}
    assert c.post(f"{base}/connect", json=key).status_code == 403
    assert c.post(f"{base}/connect", json={**_NERSC, "proxy_jump": "a@b"}).status_code == 403
    assert c.post(f"{base}/connect", json=_NERSC).status_code == 200


def test_a_jump_host_that_is_not_one_is_refused(tmp_path):
    c = _client(tmp_path)
    _, base = _session(c, tmp_path)
    r = c.post(f"{base}/connect", json={**_NERSC, "proxy_jump": "-oProxyCommand=touch /tmp/x x"})
    assert r.status_code == 400 and not FakeConn.instances


def test_the_connection_does_not_change_during_a_turn(tmp_path):
    from scilink.server.runner import TurnState
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    first = FakeConn.instances[-1]
    s.turn = TurnState(is_running=True)
    assert c.post(f"{base}/connect", json=_NERSC).status_code == 409
    assert c.post(f"{base}/disconnect").status_code == 409
    assert s.hpc_connection is first and not first.disconnected
    s.turn = None
    assert c.post(f"{base}/disconnect").status_code == 200


def test_a_local_server_allows_key_login(tmp_path):
    c = _client(tmp_path)
    _, base = _session(c, tmp_path)
    key = {"hostname": "cluster-a", "username": "alice", "auth_method": "key",
           "key_path": "~/.ssh/id_ed25519"}
    assert c.post(f"{base}/connect", json=key).status_code == 200
    assert FakeConn.instances[-1].profile.auth_method == "key"


def test_a_refused_login_explains_and_does_not_sign_the_user_out(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    FakeConn.fail_with = paramiko.AuthenticationException(
        "The server asks for a one-time code (the server asked: 'Password + OTP:').")
    r = c.post(f"{base}/connect", json={**_NERSC, "otp": ""})
    assert r.status_code == 422          # not 401: the client treats 401 as sign-out
    assert "one-time code" in r.json()["detail"]
    assert c.get(base).json() == {"connected": False}
    assert not s.agent.calls


def test_disconnect_detaches_the_specialist(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    conn = FakeConn.instances[-1]
    d = c.post(f"{base}/disconnect").json()
    assert d == {"connected": False}
    assert conn.disconnected and s.agent.calls[-1] == (None, None)
    assert s.hpc_connection is None


def test_reconnecting_replaces_and_closes_the_old_connection(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    first = FakeConn.instances[-1]
    c.post(f"{base}/connect", json=_NERSC)
    assert first.disconnected and s.hpc_connection is FakeConn.instances[-1]


def test_a_dropped_connection_is_reported(tmp_path):
    c = _client(tmp_path)
    _, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    FakeConn.instances[-1].connected = False      # the link went away
    d = c.get(base).json()
    assert d["connected"] is False and d["dropped"] is True and d["hostname"]


def test_closing_the_session_closes_its_connection(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    c.app.state.manager.remove(s.id)
    assert FakeConn.instances[-1].disconnected


# ── Mission Control hands the connection to its simulation specialist ────

def test_meta_attaches_to_an_existing_specialist_and_detaches():
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    meta = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    child = SimpleNamespace(hpc_connection=None, hpc_scheduler=None)
    meta._children = {"simulation": child}
    conn, sched = object(), SimpleNamespace(name="SLURM")
    meta.set_hpc_connection(conn, sched)
    assert child.hpc_connection is conn and child.hpc_scheduler is sched
    meta.set_hpc_connection(None)
    assert child.hpc_connection is None and child.hpc_scheduler is None


def test_a_specialist_built_later_gets_the_connection(tmp_path):
    pytest.importorskip("ase")
    from scilink.agents.meta_agent.workers import build_child
    conn, sched = object(), SimpleNamespace(name="PBS")
    orch = SimpleNamespace(api_key="dummy", model_name="claude-opus-4-6", base_url=None,
                           futurehouse_api_key=None, path_fence=None,
                           hpc_connection=conn, hpc_scheduler=sched,
                           _propagate_extensions_to_child=lambda child: None)
    child = build_child(orch, "simulation", tmp_path / "sim", persistent=False)
    assert child.hpc_connection is conn and child.hpc_scheduler is sched


def test_no_turn_starts_while_a_connection_is_being_swapped(tmp_path):
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    with s.hpc_lock:                    # a connect is in progress
        r = c.post(f"/api/v1/sessions/{s.id}/messages", json={"content": "hi"})
    assert r.status_code == 409 and "HPC connection" in r.json()["detail"]


def test_a_stopped_turn_counts_until_its_thread_ends(tmp_path):
    import threading
    from scilink.server.runner import TurnState
    c = _client(tmp_path)
    s, base = _session(c, tmp_path)
    c.post(f"{base}/connect", json=_NERSC)
    done = threading.Event()
    t = threading.Thread(target=done.wait, daemon=True)
    t.start()
    s.turn = TurnState(thread=t, is_running=False, stopped=True)   # Stop pressed
    assert c.post(f"{base}/disconnect").status_code == 409
    done.set(); t.join()
    assert c.post(f"{base}/disconnect").status_code == 200


# ── the meta's model is told; the specialist can list the user's queue ──

def test_the_meta_prompt_says_a_cluster_is_connected():
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    meta = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    meta._children = {}
    meta.meta_mode = MetaMode.AUTOPILOT
    meta._kb_note = lambda: ""
    meta.messages = [{"role": "system", "content": "base"}]
    conn = SimpleNamespace(profile=SimpleNamespace(username="bob", hostname="cluster-a.example.org"))
    meta.set_hpc_connection(conn, SimpleNamespace(name="SLURM"))
    prompt = meta.messages[0]["content"]
    assert "HPC CLUSTER (connected)" in prompt and "bob@cluster-a.example.org (SLURM)" in prompt
    assert "list_cluster_jobs" in prompt and "delegate_to_simulation" in prompt
    meta.set_hpc_connection(None)
    assert "HPC CLUSTER" not in meta.messages[0]["content"]


def test_the_specialist_lists_the_users_queue(tmp_path):
    pytest.importorskip("ase")
    import json
    from scilink.agents.sim_agents import SimulationOrchestratorAgent, SimulationMode
    from scilink.hpc.scheduler import HPCJob, JobStatus
    asked = []

    class Sched:
        name = "SLURM"

        def queue(self, user=None):
            asked.append(user)
            return [HPCJob(job_id="1477923", name="smoke_cr", status=JobStatus.PENDING,
                           raw_status="PD", partition="slurm")]

    orch = SimulationOrchestratorAgent(base_dir=str(tmp_path / "sim"), api_key="dummy",
                                       model_name="claude-opus-4-6",
                                       simulation_mode=SimulationMode.AUTONOMOUS)
    r = json.loads(orch.tools.execute_tool("list_cluster_jobs"))
    assert r["status"] == "error"                           # not connected yet
    orch.hpc_connection = SimpleNamespace(is_connected=True,
                                          profile=SimpleNamespace(username="bob"))
    orch.hpc_scheduler = Sched()
    r = json.loads(orch.tools.execute_tool("list_cluster_jobs"))
    assert r["status"] == "success" and r["user"] == "bob" and asked == ["bob"]
    assert "cluster" in r
    assert r["jobs"][0]["job_id"] == "1477923" and r["jobs"][0]["job_status"] == "Pending"



def test_the_specialist_is_told_when_the_cluster_changes(tmp_path):
    # Live: after switching cluster A -> Perlmutter the specialist answered
    # "you are connected to cluster-a" from its own earlier messages.
    pytest.importorskip("ase")
    import json
    from scilink.agents.sim_agents import SimulationOrchestratorAgent, SimulationMode
    orch = SimulationOrchestratorAgent(base_dir=str(tmp_path / "sim"), api_key="dummy",
                                       model_name="claude-opus-4-6",
                                       simulation_mode=SimulationMode.AUTONOMOUS)
    dec = SimpleNamespace(is_connected=True,
                          profile=SimpleNamespace(username="bob", hostname="cluster-a.example.org"))
    nersc = SimpleNamespace(is_connected=True,
                            profile=SimpleNamespace(username="carol", hostname="perlmutter.nersc.gov"))
    slurm = SimpleNamespace(name="SLURM", queue=lambda user=None: [])
    orch.set_hpc_connection(dec, slurm)
    assert "bob@cluster-a.example.org (SLURM)" in orch.messages[0]["content"]
    orch.set_hpc_connection(nersc, slurm)
    prompt = orch.messages[0]["content"]
    assert "carol@perlmutter.nersc.gov" in prompt and "cluster-a" not in prompt
    assert "connection has changed" in prompt
    r = json.loads(orch.tools.execute_tool("list_cluster_jobs"))
    assert r["cluster"] == "perlmutter.nersc.gov" and r["user"] == "carol"
    orch.set_hpc_connection(None)
    assert "No cluster is connected now" in orch.messages[0]["content"]
    orch.set_simulation_mode(SimulationMode.CO_PILOT)       # survives a prompt rebuild
    assert "No cluster is connected now" in orch.messages[0]["content"]


def test_the_meta_tells_its_specialist_through_its_hook():
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent
    meta = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    seen = []
    child = SimpleNamespace(set_hpc_connection=lambda c, s=None: seen.append((c, s)))
    meta._children = {"simulation": child}
    conn, sched = object(), SimpleNamespace(name="SLURM")
    meta.set_hpc_connection(conn, sched)
    assert seen == [(conn, sched)]


def test_the_gate_is_the_panels_rule_mission_control_only(tmp_path):
    # Review #811: the endpoint and the sidebar should state the same rule.
    # An agent with the hook in a non-meta session is still refused.
    c = _client(tmp_path)
    _, base = _session(c, tmp_path, mode="plan", agent=MetaLike())
    r = c.post(f"{base}/connect", json=_NERSC)
    assert r.status_code == 400 and "Mission Control" in r.json()["detail"]


def test_an_old_paramiko_is_named_at_connect_not_at_import(monkeypatch):
    # Review #811: paramiko < 3.2 has no auth_strategy. The hpc package must
    # still import, and connect() must say which version is needed.
    import builtins
    from scilink.hpc import connection as conn_mod
    real_import = builtins.__import__

    def no_auth(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "auth" and level == 1 or name == "scilink.hpc.auth":
            raise ImportError("No module named 'paramiko.auth_strategy'")
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", no_auth)
    with pytest.raises(ImportError, match=r"paramiko >= 3\.2"):
        conn_mod._login_strategy()


# ── a job belongs to the cluster it was submitted to (review #811) ───────

def _orch(tmp_path):
    pytest.importorskip("ase")
    from scilink.agents.sim_agents import SimulationOrchestratorAgent, SimulationMode
    return SimulationOrchestratorAgent(base_dir=str(tmp_path / "sim"), api_key="dummy",
                                       model_name="claude-opus-4-6",
                                       simulation_mode=SimulationMode.AUTONOMOUS)


class _JobConn:
    def __init__(self, user, host):
        self.is_connected = True
        self.profile = SimpleNamespace(username=user, hostname=host, port=22)
        self.uploaded, self.downloaded = [], []

    def mkdir_p(self, path):
        pass

    def upload(self, local, remote):
        self.uploaded.append(remote)

    def download(self, remote, local):
        self.downloaded.append(remote)
        Path(local).write_text("x")


class _JobSched:
    name = "SLURM"

    def __init__(self):
        self.asked = []

    def submit(self, script, work_dir=None):
        return "4242"

    def status(self, job_id):
        from scilink.hpc.scheduler import HPCJob, JobStatus
        self.asked.append(job_id)
        return HPCJob(job_id=job_id, name="sim", status=JobStatus.COMPLETED,
                      raw_status="CD", partition="slurm")


def _record(tmp_path, slug, **extra):
    d = tmp_path / slug
    d.mkdir()
    (d / "POSCAR").write_text("Si\n")
    (d / "INCAR").write_text("ENCUT = 300\n")
    return {"slug": slug, "structure_dir": str(d), "structure_path": str(d / "POSCAR"),
            "input_files": {"INCAR": str(d / "INCAR")}, **extra}


CLA = ("bob", "cluster-a.example.org")
NERSC = ("carol", "perlmutter.nersc.gov")


def test_a_submitted_job_records_its_cluster(tmp_path):
    import json
    orch = _orch(tmp_path)
    rec = _record(tmp_path, "si")
    orch.generated_structures = [rec]
    orch.set_hpc_connection(_JobConn(*CLA), _JobSched())
    r = json.loads(orch.tools.execute_tool(
        "submit_simulation_job", structure_slug="si", remote_dir="/scratch/x/si",
        run_command="mpirun vasp_std"))
    assert r["status"] == "success", r
    assert rec["hpc_cluster"] == r["cluster"] == "bob@cluster-a.example.org:22"


def test_status_and_download_refuse_a_job_from_another_cluster(tmp_path):
    import json
    orch = _orch(tmp_path)
    orch.generated_structures = [_record(
        tmp_path, "si", hpc_job_id="4242", hpc_remote_dir="/scratch/x/si",
        hpc_cluster="bob@cluster-a.example.org:22")]
    nersc, sched = _JobConn(*NERSC), _JobSched()
    orch.set_hpc_connection(nersc, sched)
    for tool in ("get_job_status", "download_job_results"):
        r = json.loads(orch.tools.execute_tool(tool, job_id="4242"))
        assert r["status"] == "error", (tool, r)
        assert r["submitted_to"] == "bob@cluster-a.example.org:22"
        assert "reconnect to bob@cluster-a.example.org:22" in r["message"]
    assert sched.asked == [] and nersc.downloaded == []    # nothing ran on NERSC
    # back on cluster A, both work
    cla = _JobConn(*CLA)
    orch.set_hpc_connection(cla, sched)
    assert json.loads(orch.tools.execute_tool("get_job_status", job_id="4242"))["status"] == "success"
    r = json.loads(orch.tools.execute_tool("download_job_results", job_id="4242"))
    assert r["status"] == "success" and cla.downloaded[0].startswith("/scratch/x/si/")


def test_the_same_job_id_on_two_clusters_picks_the_live_ones_record(tmp_path):
    import json
    orch = _orch(tmp_path)
    orch.generated_structures = [
        _record(tmp_path, "si", hpc_job_id="4242", hpc_remote_dir="/scratch/si",
                hpc_cluster="bob@cluster-a.example.org:22"),
        _record(tmp_path, "cu", hpc_job_id="4242", hpc_remote_dir="/pscratch/cu",
                hpc_cluster="carol@perlmutter.nersc.gov:22"),
    ]
    nersc = _JobConn(*NERSC)
    orch.set_hpc_connection(nersc, _JobSched())
    r = json.loads(orch.tools.execute_tool("download_job_results", job_id="4242"))
    assert r["status"] == "success" and nersc.downloaded[0].startswith("/pscratch/cu/")


def test_a_job_scilink_did_not_submit_is_still_checked(tmp_path):
    # e.g. an id from list_cluster_jobs: no record, so no cluster to compare
    import json
    orch = _orch(tmp_path)
    sched = _JobSched()
    orch.set_hpc_connection(_JobConn(*NERSC), sched)
    r = json.loads(orch.tools.execute_tool("get_job_status", job_id="999"))
    assert r["status"] == "success" and sched.asked == ["999"]


def test_an_old_paramiko_is_a_501_before_any_login(tmp_path, monkeypatch):
    # 501 is in the panel's "no login was tried" set, so the code is kept.
    import scilink.hpc.connection as conn_mod

    def old():
        raise ImportError("SciLink's HPC login needs paramiko >= 3.2 (found 3.1.0)")
    monkeypatch.setattr(conn_mod, "_login_strategy", old)
    c = _client(tmp_path)
    _, base = _session(c, tmp_path)
    r = c.post(f"{base}/connect", json=_NERSC)
    assert r.status_code == 501 and "paramiko >= 3.2" in r.json()["detail"]
    assert FakeConn.instances == []                      # no login attempted


def test_the_live_clusters_record_beats_one_from_before_clusters_were_recorded(tmp_path):
    import json
    orch = _orch(tmp_path)
    orch.generated_structures = [
        _record(tmp_path, "old", hpc_job_id="4242", hpc_remote_dir="/scratch/old"),
        _record(tmp_path, "new", hpc_job_id="4242", hpc_remote_dir="/pscratch/new",
                hpc_cluster="carol@perlmutter.nersc.gov:22"),
    ]
    nersc = _JobConn(*NERSC)
    orch.set_hpc_connection(nersc, _JobSched())
    r = json.loads(orch.tools.execute_tool("download_job_results", job_id="4242"))
    assert r["status"] == "success" and nersc.downloaded[0].startswith("/pscratch/new/")
