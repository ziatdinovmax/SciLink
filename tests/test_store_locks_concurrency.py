"""Concurrent writers to the learned-knowledge stores do not lose each other's work.

Distill staging, graduated skills and instrument homes were read-modify-write
without a lock: two consolidations of one technique both distilled it and
each consumed the records; two graduations into one skill each merged into
the same old version, so the first merge was lost; an instrument's use
counts dropped increments, and a trim or a forget elsewhere could delete a
recipe while a run was copying it out. Several workers of one swarm, or two
sessions, reach all of these at once.
"""

import json
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scilink.live import instrument_home as ih
from scilink.skills._shared import _graduation, _staging

REPO = Path(__file__).resolve().parents[1]

FRESH = "FRESH {skill_name} {domain}\n{knowledge_text}"
UPDATE = "UPDATE {skill_name}\n<<<{existing_skill}>>>\n{new_knowledge}"


def _merging_model(calls=None, delay=0.2):
    """A stand-in for the model: keeps what the skill already says and adds
    the new entry's marker, slowly enough for a second writer to overlap."""
    def llm(prompt):
        if calls is not None:
            calls.append(prompt)
        time.sleep(delay)
        markers = re.findall(r"\*\*Note:\*\* (\w+)", prompt)
        if prompt.startswith("UPDATE"):
            existing = json.loads(prompt.split("<<<", 1)[1].split(">>>", 1)[0])
            overview = f"{existing.get('overview', '')} {' '.join(markers)}".strip()
        else:
            overview = " ".join(markers) or "consolidated"
        return json.dumps({"description": "learned", "overview": overview})
    return llm


def _run_threads(*targets, timeout=30):
    errors = []

    def wrap(fn):
        def run():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        return run

    threads = [threading.Thread(target=wrap(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "a writer never finished (deadlock?)"
    assert not errors, errors


# ------------------------------------------------------------ graduated skills

def test_two_graduations_into_one_skill_keep_both_merges(tmp_path):
    root = tmp_path / "skills"

    def graduate(marker):
        return lambda: _graduation.graduate_to_skill_file(
            knowledge_entry={"note": marker}, skill_name="xps_learned",
            domain="curve_fitting", llm_call=_merging_model(),
            fresh_template=FRESH, update_template=UPDATE, skills_root=root)

    _run_threads(graduate("ALPHA"), graduate("BETA"))
    text = (root / "curve_fitting" / "xps_learned" / "xps_learned.md").read_text()
    assert "ALPHA" in text and "BETA" in text


def test_a_proposal_does_not_wait_for_a_write(tmp_path):
    """Only a write holds the skill's lock; a review-time proposal reads freely."""
    root = tmp_path / "skills"
    path = root / "curve_fitting" / "xps_learned" / "xps_learned.md"
    done = []
    with _graduation.skill_lock(path):
        t = threading.Thread(target=lambda: done.append(_graduation.graduate_to_skill_file(
            knowledge_entry={"note": "ALPHA"}, skill_name="xps_learned",
            domain="curve_fitting", llm_call=_merging_model(delay=0),
            fresh_template=FRESH, update_template=UPDATE, skills_root=root, write=False)))
        t.start()
        t.join(10)
    assert done and done[0]["status"] == "success"


def test_the_lock_file_is_not_a_skill_and_a_failed_distillation_leaves_no_bundle(tmp_path):
    root = tmp_path / "skills"

    def broken(_prompt):
        return "no json here"

    with pytest.raises(ValueError):
        _graduation.graduate_to_skill_file(
            knowledge_entry={"note": "X"}, skill_name="never", domain="curve_fitting",
            llm_call=broken, fresh_template=FRESH, update_template=UPDATE, skills_root=root)
    assert not (root / "curve_fitting" / "never").exists()
    assert (root / "curve_fitting" / ".never.lock").exists()
    assert _graduation.load_graduated_skills("curve_fitting", skills_root=root) == []


# ------------------------------------------------------------ distill staging

def _stage(root, n, technique="xps"):
    return [_staging.stage_solution("curve_fitting", technique, {"note": f"EX{i}"}, root=root)
            for i in range(n)]


def test_two_consolidations_of_one_technique_distill_it_once(tmp_path):
    staging, skills = tmp_path / "staging", tmp_path / "skills"
    _stage(staging, 3)
    calls, results = [], []

    def consolidate():
        results.append(_staging.consolidate_technique(
            "curve_fitting", "xps", llm_call=_merging_model(calls),
            consolidation_template=FRESH, update_template=UPDATE,
            root=staging, skills_root=skills))

    _run_threads(consolidate, consolidate)
    statuses = sorted(r["status"] for r in results)
    assert statuses == ["error", "success"]
    assert "No staged solutions" in next(r for r in results if r["status"] == "error")["message"]
    assert len(calls) == 1
    assert _staging.list_staged("curve_fitting", root=staging) == []


def test_a_relabel_waits_for_a_consolidation_that_is_consuming_the_record(tmp_path):
    staging, skills = tmp_path / "staging", tmp_path / "skills"
    sid = _stage(staging, 1)[0]
    results = {}

    def consolidate():
        results["consolidate"] = _staging.consolidate_technique(
            "curve_fitting", "xps", llm_call=_merging_model(delay=0.4),
            consolidation_template=FRESH, update_template=UPDATE,
            root=staging, skills_root=skills)

    def relabel():
        time.sleep(0.1)                  # arrives while the model is answering
        results["relabel"] = _staging.relabel_staged("curve_fitting", sid, "eels", root=staging)

    _run_threads(consolidate, relabel)
    assert results["consolidate"]["status"] == "success"
    # The record was consumed first, so the relabel finds nothing rather than
    # re-creating a record that has already been distilled.
    assert results["relabel"]["status"] == "error"
    assert _staging.list_staged("curve_fitting", root=staging) == []


def test_an_upgrade_and_a_consolidation_of_one_skill_do_not_deadlock(tmp_path):
    """Both hold the domain's lock and then the skill's, in that order."""
    staging, skills = tmp_path / "staging", tmp_path / "skills"
    _graduation.graduate_to_skill_file(
        knowledge_entry={"note": "BASE"}, skill_name="auto_xps", domain="curve_fitting",
        llm_call=_merging_model(delay=0), fresh_template=FRESH, update_template=UPDATE,
        skills_root=skills)
    upgrade_ids = _stage(staging, 1, technique="other")
    _stage(staging, 2)

    def upgrade():
        _staging.upgrade_skill_from_staged(
            "curve_fitting", upgrade_ids, target_domain="curve_fitting", target_name="auto_xps",
            llm_call=_merging_model(), fresh_template=FRESH, update_template=UPDATE,
            root=staging, skills_root=skills)

    def consolidate():
        _staging.consolidate_technique(
            "curve_fitting", "xps", llm_call=_merging_model(),
            consolidation_template=FRESH, update_template=UPDATE,
            root=staging, skills_root=skills)

    _run_threads(upgrade, consolidate)
    assert _staging.list_staged("curve_fitting", root=staging) == []
    text = (skills / "curve_fitting" / "auto_xps" / "auto_xps.md").read_text()
    assert "BASE" in text


def test_removing_a_record_twice_counts_it_once(tmp_path):
    staging = tmp_path / "staging"
    ids = _stage(staging, 1)
    counts = []
    _run_threads(*[lambda: counts.append(_staging.remove_staged("curve_fitting", ids, root=staging))
                   for _ in range(4)])
    assert sorted(counts) == [0, 0, 0, 1]


# ------------------------------------------------------------ instrument homes

def _home_with_recipe(root, rid="r1"):
    home = ih.InstrumentHome({"id": "scope-1", "technique": "Raman"}, root=str(root))
    rdir = home.dir / "recipes" / rid
    (rdir / "anchor").mkdir(parents=True)
    (rdir / "anchor" / "script.py").write_text("print('fit')\n")
    (rdir / "anchor" / "analysis_results.json").write_text("{}")
    (rdir / "recipe.json").write_text(json.dumps({"recipe_id": rid, "modality": "curve", "uses": 0}))
    return home, rdir


def test_use_counts_from_several_processes_are_all_kept(tmp_path):
    home, rdir = _home_with_recipe(tmp_path)
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from scilink.live.instrument_home import InstrumentHome
        home = InstrumentHome({{"id": "scope-1"}}, root={str(tmp_path)!r})
        for _ in range(25):
            home.used("r1")
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script]) for _ in range(4)]
    assert all(p.wait(120) == 0 for p in procs)
    assert json.loads((rdir / "recipe.json").read_text())["uses"] == 100
    info = json.loads((home.dir / "instrument.json").read_text())
    assert info["id"] == "scope-1" and info["first_seen"]


def test_a_forget_waits_until_a_copy_out_is_complete(tmp_path, monkeypatch):
    home, rdir = _home_with_recipe(tmp_path)
    real_copytree = ih.shutil.copytree

    def slow_copytree(src, dst, *a, **kw):
        time.sleep(0.3)                  # the forget arrives in here
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(ih.shutil, "copytree", slow_copytree)
    dest = tmp_path / "run" / "recalled" / "r1"
    dest.parent.mkdir(parents=True)
    forgot = []

    def forget():
        time.sleep(0.1)
        forgot.append(ih.forget_recipe("scope-1", "r1", root=str(tmp_path)))

    _run_threads(lambda: home.copy_out(rdir / "anchor", dest), forget)
    assert (dest / "script.py").read_text() == "print('fit')\n"
    assert forgot == [True] and not rdir.exists()


def test_a_failed_copy_out_leaves_nothing_a_later_run_would_mistake_for_a_copy(tmp_path, monkeypatch):
    home, rdir = _home_with_recipe(tmp_path)
    dest = tmp_path / "run" / "recalled" / "r1"
    dest.parent.mkdir(parents=True)

    def broken_copytree(src, dst, *a, **kw):
        Path(dst).mkdir()
        (Path(dst) / "script.py").write_text("half")
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(ih.shutil, "copytree", broken_copytree)
        with pytest.raises(OSError):
            home.copy_out(rdir / "anchor", dest)
    assert not dest.exists()
    home.copy_out(rdir / "anchor", dest)
    assert (dest / "script.py").read_text() == "print('fit')\n"
    assert not any(p.name.endswith(".partial") for p in dest.parent.iterdir())


def test_forgetting_an_instrument_removes_its_folder_and_keeps_the_lock_outside_it(tmp_path):
    home, _ = _home_with_recipe(tmp_path)
    assert (tmp_path / ".locks" / "scope-1.lock").exists() and not list(home.dir.glob("*.lock"))
    assert [i["id"] for i in ih.known_instruments(str(tmp_path))] == ["scope-1"]
    assert ih.forget_instrument("scope-1", root=str(tmp_path))
    assert not home.dir.exists()
    from scilink.utils.file_lock import is_locked
    with ih._home_lock(tmp_path / "scope-1"):                          # the lock outlives the home
        assert is_locked(tmp_path / ".locks" / "scope-1")
    ih.InstrumentHome({"id": "scope-1"}, root=str(tmp_path))          # re-created cleanly
    assert [i["id"] for i in ih.known_instruments(str(tmp_path))] == ["scope-1"]


def test_a_reviewed_upgrade_refuses_a_skill_that_changed_during_the_review(tmp_path):
    """propose → (a worker's graduation lands) → apply: the apply must not
    overwrite the graduation. The hash is of the text the merge read, so a
    graduation during the model call is caught too."""
    staging, skills = tmp_path / "staging", tmp_path / "skills"
    _graduation.graduate_to_skill_file(
        knowledge_entry={"note": "BASE"}, skill_name="auto_xps", domain="curve_fitting",
        llm_call=_merging_model(delay=0), fresh_template=FRESH, update_template=UPDATE,
        skills_root=skills)
    ids = _stage(staging, 1, technique="other")
    path = skills / "curve_fitting" / "auto_xps" / "auto_xps.md"

    def model_while_another_graduates(prompt):
        # a graduation lands DURING the proposal's model call
        _graduation.graduate_to_skill_file(
            knowledge_entry={"note": "LANDED"}, skill_name="auto_xps", domain="curve_fitting",
            llm_call=_merging_model(delay=0), fresh_template=FRESH, update_template=UPDATE,
            skills_root=skills)
        return _merging_model(delay=0)(prompt)

    prop = _staging.propose_skill_upgrade(
        "curve_fitting", ids, target_domain="curve_fitting", target_name="auto_xps",
        llm_call=model_while_another_graduates, fresh_template=FRESH, update_template=UPDATE,
        root=staging, skills_root=skills)
    assert prop["status"] == "success" and prop["base_hash"]
    res = _staging.apply_skill_upgrade(
        "curve_fitting", prop["staged_ids"], target_domain="curve_fitting", target_name="auto_xps",
        proposed_content=prop["proposed_content"], base_hash=prop["base_hash"],
        root=staging, skills_root=skills)
    assert res["status"] == "error" and res.get("changed_since_proposal")
    assert "LANDED" in path.read_text()                       # the graduation survived
    assert _staging.list_staged("curve_fitting", root=staging)   # nothing consumed
    # proposed again against the current version, it applies
    prop2 = _staging.propose_skill_upgrade(
        "curve_fitting", ids, target_domain="curve_fitting", target_name="auto_xps",
        llm_call=_merging_model(delay=0), fresh_template=FRESH, update_template=UPDATE,
        root=staging, skills_root=skills)
    res2 = _staging.apply_skill_upgrade(
        "curve_fitting", prop2["staged_ids"], target_domain="curve_fitting", target_name="auto_xps",
        proposed_content=prop2["proposed_content"], base_hash=prop2["base_hash"],
        root=staging, skills_root=skills)
    assert res2["status"] == "success"
