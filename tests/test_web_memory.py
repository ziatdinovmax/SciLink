"""The web UI's persistent-memory endpoints (scilink/server/memory_api.py)
over a seeded store: the switch, the pipeline overview, bank / inbox /
skill records and actions, the technique-match target ordering, the
LLM-backed consolidate and upgrade jobs (fake model), the apply path
that forks a built-in, and the additivity check."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from scilink.server.app import create_app  # noqa: E402


SKILL_JSON = {
    "description": "Distilled Raman D/G fitting rules",
    "overview": "Two-Voigt D/G model on a linear baseline.",
    "planning": "Seed peaks from the local maxima near 1350 and 1580.",
    "implementation": "Fit with lmfit VoigtModel x2 + LinearModel.",
    "interpretation": "Report the D/G area ratio.",
    "validation": "R² above 0.98 and residual without structure.",
}


class _FakeModel:
    """Returns the canned skill JSON for any prompt (the graduation flow
    parses a JSON object with description + canonical sections)."""
    def __init__(self):
        self.prompts = []

    def generate_content(self, contents):
        self.prompts.append(contents[0])
        return SimpleNamespace(text=json.dumps(SKILL_JSON))


@pytest.fixture
def mem_client(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("SCILINK_HOME", str(home))
    monkeypatch.delenv("SCILINK_MEMORY", raising=False)
    monkeypatch.setenv("SCILINK_CONSOLIDATE_N", "2")  # two records suffice in tests
    client = TestClient(create_app(tmp_path / "sessions", serve_frontend=False))
    return client


def _seed(tmp_path):
    """The Streamlit panel test's full-pipeline seed: one banked script
    proven across three sessions and nominated, an error lesson and a
    feedback record in the same technique group, a provisional skill."""
    from scilink.skills._shared import _script_bank as sb, _staging
    from scilink.skills.loader import graduated_skills_dir

    rid = sb.add_record("curve_fitting", {
        "working_script": "# hot win\nimport numpy as np\n",
        "data_fingerprint": {"kind": "curve"},
        "measurement_context": {"technique": "Raman"},
        "technique_signals": {"model_type": "two voigt"},
        "outcome": {"metric": {"name": "r_squared", "value": 0.99}},
        "provenance": {"session": "s1"}})["id"]
    for s in ("s2", "s3"):
        sb.record_success("curve_fitting", rid, session=s)
    out = sb.promote_to_staging("curve_fitting", rid, technique="raman_dg",
                                provenance="t2_hot_win",
                                extra={"deviation_from_plan": "switched model"})
    e = _staging.stage_solution("curve_fitting", "raman_dg", {
        "provenance": "error_fix", "model": "m",
        "error_lessons": [{"error": "SNIP no converge", "fix": "fixed iter"}],
        "session": "s1"})
    f = _staging.stage_solution("curve_fitting", "raman_dg", {
        "provenance": "user_correction", "model": "m",
        "user_feedback": "always report baseline fraction", "session": "s1"})
    # a second, unnominated bank record so the bank has a plain row too
    rid2 = sb.add_record("curve_fitting", {
        "working_script": "# plain\nprint(1)\n",
        "data_fingerprint": {"kind": "curve"},
        "technique_signals": {"model_type": "gaussian"},
        "outcome": {"metric": {"name": "r_squared", "value": 0.9}},
        "provenance": {"session": "s9"}})["id"]
    d = graduated_skills_dir() / "curve_fitting" / "myskill"
    d.mkdir(parents=True)
    (d / "myskill.md").write_text(
        "---\ndescription: My provisional skill\nprovisional: true\n---\n"
        "## overview\n\nskill body\n")
    return {"bank": rid, "bank2": rid2, "staged": out["staged_id"], "err": e, "fb": f}


def _session(client, tmp_path):
    from scilink.server.session_manager import WebSession
    sdir = tmp_path / "sessions" / "meta_session_20260101_090909"
    sdir.mkdir(parents=True, exist_ok=True)
    agent = SimpleNamespace(model=_FakeModel())
    session = WebSession(id=sdir.name, session_dir=str(sdir), mode="meta",
                         model="fake", autonomy="autopilot", agent=agent)
    client.app.state.manager._sessions[sdir.name] = session
    return session


def _wait_job(client, job_id, timeout=20):
    for _ in range(int(timeout * 20)):
        j = client.get(f"/api/v1/memory/jobs/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_memory_overview_switch_and_records(mem_client, tmp_path):
    ids = _seed(tmp_path)
    c = mem_client
    ov = c.get("/api/v1/memory").json()
    assert ov["enabled"] is False and ov["env_override"] is None
    assert ov["pipeline"] == {"bank_total": 2, "bank_proven": 1, "inbox_total": 3,
                              "inbox_ready": 1, "skills_total": 1, "skills_provisional": 1}
    bank = {d["domain"]: d for d in ov["bank"]}["curve_fitting"]
    rows = {r["id"]: r for r in bank["records"]}
    assert rows[ids["bank"]]["proven"] and rows[ids["bank"]]["n_successes"] == 3
    assert rows[ids["bank"]]["promoted_to_staging"] == ids["staged"]
    assert rows[ids["bank2"]]["proven"] is False and rows[ids["bank2"]]["promoted_to_staging"] is None
    (group,) = ov["inbox"]
    assert (group["domain"], group["technique"], group["ready"]) == ("curve_fitting", "raman_dg", True)
    provs = {r["id"]: r["provenance_label"] for r in group["records"]}
    assert provs == {ids["staged"]: "solved from scratch (banked)",
                     ids["err"]: "error fix", ids["fb"]: "your feedback"}
    (skill,) = ov["skills"]
    assert skill["name"] == "myskill" and skill["provisional"] is True
    assert skill["description"] == "My provisional skill" and skill["shadows_builtin"] is False

    # the switch persists to config.json; the env var overrides it
    assert c.post("/api/v1/memory/enabled", json={"enabled": True}).json()["enabled"] is True
    assert c.get("/api/v1/memory").json()["enabled"] is True
    cfg = json.loads((tmp_path / "home" / "config.json").read_text())
    assert cfg["memory_enabled"] is True

    # bank record + inbox record inspectors
    b = c.get(f"/api/v1/memory/bank/curve_fitting/{ids['bank']}").json()
    assert b["script"].startswith("# hot win") and "working_script" not in b["fields"]
    assert b["fields"]["stats"]["n_successes"] == 3
    r = c.get(f"/api/v1/memory/inbox/curve_fitting/{ids['staged']}").json()
    assert r["bank"] == {"bank_id": ids["bank"], "n_successes": 3}
    assert r["fields"]["deviation_from_plan"] == "switched model"
    assert c.get("/api/v1/memory/inbox/curve_fitting/nope").status_code == 404
    assert c.get("/api/v1/memory/bank/curve_fitting/nope").status_code == 404

    # nominate the plain record; a second nomination is refused while staged
    out = c.post(f"/api/v1/memory/bank/curve_fitting/{ids['bank2']}/nominate").json()
    assert out["status"] == "success" and out["staged_id"]
    assert c.post(f"/api/v1/memory/bank/curve_fitting/{ids['bank2']}/nominate").status_code == 409
    # discard that staged copy → the bank row is nominatable again
    assert c.delete(f"/api/v1/memory/inbox/curve_fitting/{out['staged_id']}").json()["removed"] == 1
    ov = c.get("/api/v1/memory").json()
    row = [r for r in ov["bank"][0]["records"] if r["id"] == ids["bank2"]][0]
    assert row["promoted_to_staging"] is None
    # delete the bank record
    assert c.delete(f"/api/v1/memory/bank/curve_fitting/{ids['bank2']}").json()["removed"] == 1
    assert c.delete(f"/api/v1/memory/bank/curve_fitting/{ids['bank2']}").status_code == 404


def test_memory_skill_lifecycle(mem_client, tmp_path):
    _seed(tmp_path)
    c = mem_client
    base = "/api/v1/memory/skills/curve_fitting/myskill"
    r = c.get(base)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert "provisional: true" in r.text
    # edit: validated, backed up
    bad = c.put(base, json={"content": "no headings at all\n"})
    assert bad.status_code == 400 and "section" in bad.json()["detail"]
    ok = c.put(base, json={"content": "---\ndescription: edited\nprovisional: true\n---\n## overview\n\nnew body\n"})
    assert ok.status_code == 200 and Path(ok.json()["backup_path"]).is_file()
    assert "new body" in c.get(base).text
    # promote → approved (auto-routable); demote → provisional again
    assert c.post(f"{base}/promote").json()["promoted"] is True
    assert c.get("/api/v1/memory").json()["skills"][0]["provisional"] is False
    assert c.post(f"{base}/demote").json()["provisional"] is True
    assert c.get("/api/v1/memory").json()["skills"][0]["provisional"] is True
    assert c.post(f"{base}/bogus").status_code == 400
    # fork a built-in: shadows it; diff starts identical; prune removes it
    f = c.post("/api/v1/memory/skills/curve_fitting/raman/fork").json()
    assert f["status"] == "success" and Path(f["path"]).is_file()
    assert c.post("/api/v1/memory/skills/curve_fitting/raman/fork").status_code == 409
    d = c.post("/api/v1/memory/skills/curve_fitting/raman/diff").json()
    assert d["identical"] is True
    skills = {s["name"]: s for s in c.get("/api/v1/memory").json()["skills"]}
    assert skills["raman"]["shadows_builtin"] is True and skills["myskill"]["shadows_builtin"] is False
    assert c.post("/api/v1/memory/skills/curve_fitting/raman/prune").json()["pruned"] is True
    assert c.post(f"{base}/prune").json()["pruned"] is True
    assert c.get(base).status_code == 404
    assert c.post("/api/v1/memory/skills/curve_fitting/nope/fork").status_code == 404


def test_memory_targets_consolidate_and_upgrade_jobs(mem_client, tmp_path):
    ids = _seed(tmp_path)
    c = mem_client
    session = _session(c, tmp_path)
    c.post("/api/v1/memory/enabled", json={"enabled": True})

    # targets: the persistent skill first (unknown match), built-ins after;
    # the Raman-context records match the raman skill, not epr
    t = c.post("/api/v1/memory/inbox/curve_fitting/targets",
               json={"ids": [ids["staged"]]}).json()["targets"]
    by = {x["name"]: x for x in t}
    assert by["myskill"]["builtin"] is False
    assert by["raman"]["builtin"] is True and by["raman"]["match"] is True
    assert by["epr"]["match"] is False
    assert t.index(by["raman"]) < t.index(by["epr"])

    # consolidate guards: memory off, too few records, swept label
    c.post("/api/v1/memory/enabled", json={"enabled": False})
    r = c.post("/api/v1/memory/inbox/curve_fitting/consolidate",
               json={"ids": [ids["staged"], ids["err"]], "label": "raman dg", "session_id": session.id})
    assert r.status_code == 400 and "off" in r.json()["detail"]
    c.post("/api/v1/memory/enabled", json={"enabled": True})
    r = c.post("/api/v1/memory/inbox/curve_fitting/consolidate",
               json={"ids": [ids["staged"]], "label": "raman dg", "session_id": session.id})
    assert r.status_code == 400 and "at least" in r.json()["detail"]
    r = c.post("/api/v1/memory/inbox/curve_fitting/consolidate",
               json={"ids": [ids["staged"], ids["err"]], "label": "raman dg", "session_id": session.id})
    assert r.status_code == 409 and ids["fb"] in r.json()["detail"]
    assert c.post("/api/v1/memory/inbox/curve_fitting/consolidate",
                  json={"ids": [ids["staged"]], "label": "x", "session_id": "nope"}).status_code == 404

    # upgrade preview against the built-in raman skill (previewed on a temp
    # copy — nothing written), then apply → forks it and consumes the record
    r = c.post("/api/v1/memory/inbox/curve_fitting/propose-upgrade",
               json={"ids": [ids["fb"]], "target_domain": "curve_fitting",
                     "target_name": "raman", "session_id": session.id})
    assert r.status_code == 200, r.text
    job = _wait_job(c, r.json()["job_id"])
    assert job["status"] == "done", job
    prop = job["result"]
    assert prop["builtin_target"] is True and prop["proposed_content"] and prop["diff"]
    assert "Distilled Raman" in prop["proposed_content"]
    assert session.agent.model.prompts  # the session's model was used
    from scilink.skills.loader import graduated_skills_dir
    assert not (graduated_skills_dir() / "curve_fitting" / "raman").exists()
    chk = c.post("/api/v1/memory/check-upgrade",
                 json={"existing": prop["existing_content"], "proposed": prop["proposed_content"]}).json()
    assert chk["diff"] == prop["diff"] and isinstance(chk["warnings"], list)
    ap = c.post("/api/v1/memory/inbox/curve_fitting/apply-upgrade",
                json={"ids": [ids["fb"]], "target_domain": "curve_fitting", "target_name": "raman",
                      "content": prop["proposed_content"], "fork_builtin": True}).json()
    assert ap["status"] == "success" and ap["n_consumed"] == 1
    fork = graduated_skills_dir() / "curve_fitting" / "raman" / "raman.md"
    assert fork.is_file() and "Distilled Raman" in fork.read_text()
    assert (fork.parent / "raman.md.bak").is_file()

    # consolidate the two remaining records into a NEW provisional skill
    r = c.post("/api/v1/memory/inbox/curve_fitting/consolidate",
               json={"ids": [ids["staged"], ids["err"]], "label": "Raman D/G", "session_id": session.id})
    assert r.status_code == 200, r.text
    job = _wait_job(c, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["skill_name"] == "auto_raman_d_g"
    ov = c.get("/api/v1/memory").json()
    assert ov["pipeline"]["inbox_total"] == 0
    names = {s["name"]: s for s in ov["skills"]}
    assert names["auto_raman_d_g"]["provisional"] is True
    assert names["auto_raman_d_g"]["provenance"] == "t2_consolidated"
    assert c.get("/api/v1/memory/jobs/nope").status_code == 404


def test_memory_overview_survives_hostile_records(mem_client, tmp_path):
    """Markdown/HTML in labels, missing tiers, plain-float metrics, a corrupt
    bank file, a corrupt and a degenerate staged file must not break the
    overview (the Streamlit panel's hostile-store test, over the API)."""
    from scilink.skills._shared import _script_bank as sb, _staging
    sb.add_record("curve_fitting", {
        "working_script": "# s", "data_fingerprint": {"kind": "curve"},
        "measurement_context": {},
        "technique_signals": {"model_type": "**bold** <script>x</script> `tick`"},
        "outcome": {"metric": 0.97}, "provenance": {"session": "s"}})
    sb.add_record("curve_fitting", {
        "working_script": "# s2", "data_fingerprint": None,
        "measurement_context": None, "technique_signals": None,
        "outcome": None, "provenance": {"session": "s"}})
    (sb._domain_dir("curve_fitting") / "corrupt.json").write_text("{broken")
    d = _staging.staging_dir() / "curve_fitting"
    d.mkdir(parents=True)
    (d / "broken.json").write_text("{not json")
    (d / "weird.json").write_text(json.dumps({"id": "weird", "domain": "curve_fitting",
                                              "technique": None, "provenance": 42}))
    ov = mem_client.get("/api/v1/memory").json()
    assert ov["pipeline"]["bank_total"] == 2
    assert [g["technique"] for g in ov["inbox"]] == ["unlabeled"]
    assert ov["inbox"][0]["records"][0]["provenance_label"] == "42"
