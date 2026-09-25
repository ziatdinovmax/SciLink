"""Script-bank hardening: trust fixes + automatic aging.

The bank executes stored code with no LLM review on the verbatim path, so
before that path widens the bank has to (B1) remember failures, (B2) count
independent evidence rather than reruns, (B3) not get more lenient as image
metadata goes missing, (B4) treat a technique-skill mismatch as disqualifying,
(B5) refuse a record whose script no longer matches its stored hash, and
(B8) version its fingerprints. B14 archives records nobody uses so the bank
listing (CLI + UI) stays readable; an archive is reversible, unlike a prune.

No LLM calls anywhere.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.skills._shared import _script_bank as sb


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    monkeypatch.delenv("SCILINK_SCRIPT_BANK", raising=False)
    monkeypatch.delenv("SCILINK_BANK_AUTO_ARCHIVE", raising=False)
    monkeypatch.delenv("SCILINK_BANK_STALE_DAYS", raising=False)


def _curve(center=20.0, seed=0, n=2000):
    rs = np.random.RandomState(seed)
    x = np.linspace(0, 100, n)
    y = (5.0 * np.exp(-(x - center) ** 2 / 4) + 3.0 * np.exp(-(x - 60) ** 2 / 9)
         + rs.normal(0, 0.05, x.size))
    return x, y


def _fp(center=20.0, seed=0):
    return sb.curve_fingerprint(*_curve(center, seed), x_units="eV")


def _bank(script="print('a')", fp=None, session="s1", skills=None, **extra):
    return sb.add_record("curve_fitting", {
        "working_script": script,
        "data_fingerprint": fp if fp is not None else _fp(),
        "technique_signals": {"active_skills": skills or []},
        "measurement_context": {"technique": "xps"},
        "provenance": {"session": session},
        **extra,
    })["id"]


def _yesterday_sweep(domain="curve_fitting"):
    """Pretend the last unattended sweep ran two days ago."""
    stamp = sb._domain_dir(domain) / sb.ARCHIVE_DIRNAME / sb._SWEEP_STAMP
    stamp.write_text((datetime.now(timezone.utc) - timedelta(days=2))
                     .isoformat(timespec="seconds"))


def _rec(rid):
    return sb.get_record("curve_fitting", rid)


def _age(rid, days, domain="curve_fitting"):
    """Backdate a record's timestamps (aging rules read created/updated_at)."""
    f = sb._domain_dir(domain) / f"{rid}.json"
    rec = json.loads(f.read_text())
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    rec["created_at"] = rec["updated_at"] = stamp
    if rec.get("last_retrieved_at"):
        rec["last_retrieved_at"] = stamp
    # Rewriting the file must keep the integrity hash valid.
    f.write_text(json.dumps(rec))


# ──────────────────────────────────────────────────────────────
# B8 — fingerprint version
# ──────────────────────────────────────────────────────────────

class TestFingerprintVersion:
    def test_every_fingerprint_is_stamped(self):
        assert _fp()["v"] == sb.FINGERPRINT_VERSION
        assert sb.image_fingerprint(np.random.rand(64, 64))["v"] == sb.FINGERPRINT_VERSION
        cube = np.random.rand(8, 8, 40)
        assert sb.hyperspectral_fingerprint(cube)["v"] == sb.FINGERPRINT_VERSION

    def test_unversioned_legacy_record_still_matches(self):
        legacy = {k: v for k, v in _fp().items() if k != "v"}
        rid = _bank(fp=legacy)
        assert [m["record"]["id"] for m in sb.find_exemplar("curve_fitting", _fp(seed=1))] == [rid]

    def test_other_version_is_not_comparable(self):
        _bank(fp={**_fp(), "v": sb.FINGERPRINT_VERSION + 1})
        assert sb.find_exemplar("curve_fitting", _fp(seed=1)) == []


# ──────────────────────────────────────────────────────────────
# B5 — integrity
# ──────────────────────────────────────────────────────────────

class TestIntegrity:
    def test_intact_record_verifies(self):
        assert sb.verify_record(_rec(_bank())) is True

    def test_edited_script_is_refused(self):
        rid = _bank()
        f = sb._domain_dir("curve_fitting") / f"{rid}.json"
        rec = json.loads(f.read_text())
        rec["working_script"] = "import os; os.remove('data.csv')"
        f.write_text(json.dumps(rec))
        assert sb.verify_record(_rec(rid)) is False
        assert sb.find_exemplar("curve_fitting", _fp(seed=1)) == []

    def test_missing_hash_is_refused(self):
        rid = _bank()
        f = sb._domain_dir("curve_fitting") / f"{rid}.json"
        rec = json.loads(f.read_text())
        rec.pop("script_hash")
        f.write_text(json.dumps(rec))
        assert sb.verify_record(_rec(rid)) is False

    def test_trailing_whitespace_is_not_tampering(self):
        rid = _bank(script="x = 1\nprint(x)")
        f = sb._domain_dir("curve_fitting") / f"{rid}.json"
        rec = json.loads(f.read_text())
        rec["working_script"] = "x = 1   \nprint(x)\n\n"
        f.write_text(json.dumps(rec))
        assert sb.verify_record(_rec(rid)) is True


# ──────────────────────────────────────────────────────────────
# B2 — independent evidence
# ──────────────────────────────────────────────────────────────

class TestIndependentEvidence:
    def test_reruns_of_the_same_data_do_not_prove_a_record(self):
        fp = _fp()
        rid = _bank(fp=fp, session="s1")
        for s in ("s2", "s3", "s4"):
            _bank(fp=fp, session=s)            # same script, SAME data
        rec = _rec(rid)
        assert rec["stats"]["n_successes"] == 4   # runs are still counted
        assert sb.independent_successes(rec) == 1
        assert sb.is_proven(rec) is False

    def test_new_data_is_independent_evidence(self):
        rid = _bank(fp=_fp(seed=0), session="s1")
        _bank(fp=_fp(center=22, seed=1), session="s2")
        sb.record_success("curve_fitting", rid, session="s3",
                          fingerprint=_fp(center=24, seed=2))
        rec = _rec(rid)
        assert sb.independent_successes(rec) == 3
        assert sb.is_proven(rec) is True

    def test_adapted_success_is_not_verbatim_evidence(self):
        # Observed live: a two-peak script failed verbatim on three-peak data,
        # was edit-adapted to three peaks, passed QC, and the ORIGINAL record
        # was credited. Fine for graduation; not a licence to run it unreviewed.
        rid = _bank(fp=_fp(seed=0), session="s1")
        for i in (1, 2):
            sb.record_success("curve_fitting", rid, session=f"a{i}",
                              fingerprint=_fp(center=20 + i, seed=i), adapted=True)
        rec = _rec(rid)
        assert sb.independent_successes(rec) == 3 and sb.is_proven(rec)
        assert sb.independent_successes(rec, verbatim_only=True) == 1
        assert not sb.is_verbatim_proven(rec)
        for i in (3, 4):
            sb.record_success("curve_fitting", rid, session=f"v{i}",
                              fingerprint=_fp(center=20 + i, seed=i))
        assert sb.is_verbatim_proven(_rec(rid))
        [row] = sb.bank_summary("curve_fitting")
        assert row["n_independent"] == 5 and row["n_verbatim"] == 3

    def test_success_without_a_fingerprint_counts_once_per_session(self):
        rid = _bank(session="s1")
        sb.record_success("curve_fitting", rid, session="s2")
        sb.record_success("curve_fitting", rid, session="s2")   # same campaign
        assert sb.independent_successes(_rec(rid)) == 2

    def test_legacy_record_falls_back_to_distinct_sessions(self):
        rid = _bank()
        f = sb._domain_dir("curve_fitting") / f"{rid}.json"
        rec = json.loads(f.read_text())
        rec.pop("evidence", None)
        rec["stats"]["n_successes"] = 9
        rec["sessions"] = ["a", "b"]
        f.write_text(json.dumps(rec))
        assert sb.independent_successes(_rec(rid)) == 2

    def test_summary_and_promotion_use_independent_evidence(self):
        fp = _fp()
        rid = _bank(fp=fp)
        for s in ("s2", "s3"):
            _bank(fp=fp, session=s)
        [row] = sb.bank_summary("curve_fitting")
        assert row["proven"] is False and row["n_independent"] == 1
        # Nomination is always allowed, but it must not claim the star.
        sb.promote_to_staging("curve_fitting", rid, provenance="bank_proven")
        assert _rec(rid)["promoted_reason"] == "bank_nominated"


# ──────────────────────────────────────────────────────────────
# B1 — failures
# ──────────────────────────────────────────────────────────────

class TestFailures:
    def test_failure_is_recorded_with_reason(self):
        rid = _bank()
        sb.record_failure("curve_fitting", rid, "gate_failed", session="s9")
        rec = _rec(rid)
        assert rec["stats"]["n_failures"] == 1
        assert rec["last_failure"]["reason"] == "gate_failed"
        assert sb.verify_record(rec) is True      # bookkeeping keeps the hash valid

    def test_failing_record_loses_to_a_clean_twin(self):
        fp = _fp()
        bad = _bank(script="print('bad')", fp=fp)
        good = _bank(script="print('good')", fp=fp)
        for _ in range(3):
            sb.record_failure("curve_fitting", bad, "gate_failed")
        ranked = [m["record"]["id"]
                  for m in sb.find_exemplar("curve_fitting", _fp(seed=1), k=2)]
        assert ranked[0] == good

    def test_failures_never_raise(self):
        sb.record_failure("curve_fitting", "nope1234", "x")
        sb.record_failure("no_such_domain", "nope1234", "x")


# ──────────────────────────────────────────────────────────────
# B4 — technique skill as a hard filter
# ──────────────────────────────────────────────────────────────

class TestSkillFilter:
    def test_disjoint_skills_disqualify(self):
        _bank(skills=["raman"])
        q = _fp(seed=1)
        assert sb.find_exemplar("curve_fitting", q, active_skills=["ftir"]) == []
        assert len(sb.find_exemplar("curve_fitting", q, active_skills=["raman"])) == 1

    def test_overlap_is_enough(self):
        _bank(skills=["xps", "background_shirley"])
        assert len(sb.find_exemplar("curve_fitting", _fp(seed=1),
                                    active_skills=["xps"])) == 1

    def test_unknown_on_either_side_does_not_filter(self):
        _bank(skills=[])
        assert len(sb.find_exemplar("curve_fitting", _fp(seed=1),
                                    active_skills=["xps"])) == 1
        _bank(script="print('b')", skills=["xps"])
        assert len(sb.find_exemplar("curve_fitting", _fp(seed=1), k=5)) == 2


# ──────────────────────────────────────────────────────────────
# B3 — image similarity with missing terms
# ──────────────────────────────────────────────────────────────

class TestImageLeniency:
    FULL = {"kind": "image", "intensity": {"contrast": 0.30}, "edge_density": 0.10,
            "fft_periodicity": 8.0, "pixel_size_nm": 0.5}

    def test_full_agreement_is_unchanged(self):
        assert sb._image_similarity(self.FULL, dict(self.FULL)) == pytest.approx(1.0)

    def test_one_agreeing_term_cannot_carry_the_match(self):
        sparse = {"kind": "image", "intensity": {"contrast": 0.30}}
        assert sb._image_similarity(sparse, self.FULL) < sb._MIN_EXEMPLAR_SCORE

    def test_score_grows_with_the_number_of_agreeing_terms(self):
        two = {"kind": "image", "intensity": {"contrast": 0.30}, "edge_density": 0.10}
        one = {"kind": "image", "intensity": {"contrast": 0.30}}
        assert (sb._image_similarity(one, self.FULL)
                < sb._image_similarity(two, self.FULL)
                < sb._image_similarity(self.FULL, self.FULL))


# ──────────────────────────────────────────────────────────────
# B14 — aging / archive
# ──────────────────────────────────────────────────────────────

class TestAging:
    def test_fresh_records_are_never_stale(self):
        _bank()
        assert sb.stale_records("curve_fitting") == []

    def test_never_used_record_goes_stale(self):
        rid = _bank()
        _age(rid, 200)
        [s] = sb.stale_records("curve_fitting")
        assert s["id"] == rid and s["reason"] == "never_used"

    def test_retrieved_record_is_kept(self):
        rid = _bank()
        sb.mark_retrieved("curve_fitting", rid)
        _age(rid, 200)
        assert sb.stale_records("curve_fitting") == []

    def test_record_that_keeps_failing_goes_stale_even_when_young(self):
        rid = _bank()
        for _ in range(3):
            sb.mark_retrieved("curve_fitting", rid)
            sb.record_failure("curve_fitting", rid, "gate_failed")
        [s] = sb.stale_records("curve_fitting")
        assert s["reason"] == "never_succeeds"

    def test_proven_record_is_protected(self):
        rid = _bank(fp=_fp(seed=0))
        for i in (1, 2, 3):
            sb.record_success("curve_fitting", rid, session=f"s{i}",
                              fingerprint=_fp(center=20 + i, seed=i))
        _age(rid, 400)
        assert sb.stale_records("curve_fitting") == []

    def test_superseded_variant_goes_stale(self):
        fp = _fp(seed=0)
        proven = _bank(script="print('proven')", fp=fp)
        for i in (1, 2, 3):
            sb.record_success("curve_fitting", proven, session=f"s{i}",
                              fingerprint=_fp(center=20 + 0.1 * i, seed=i))
        sibling = _bank(script="print('sibling')", fp=_fp(seed=5))
        sb.mark_retrieved("curve_fitting", sibling)   # used once, long ago
        _age(sibling, 200)
        [s] = sb.stale_records("curve_fitting")
        assert s["id"] == sibling and s["reason"] == "superseded"

    def test_days_threshold_is_configurable(self, monkeypatch):
        rid = _bank()
        _age(rid, 10)
        assert sb.stale_records("curve_fitting") == []
        assert [s["id"] for s in sb.stale_records("curve_fitting", idle_days=5)] == [rid]
        monkeypatch.setenv("SCILINK_BANK_STALE_DAYS", "5")
        assert [s["id"] for s in sb.stale_records("curve_fitting")] == [rid]

    def test_archive_hides_and_restore_brings_back(self):
        rid = _bank()
        assert sb.archive_records("curve_fitting", [rid]) == 1
        assert sb.list_records("curve_fitting") == []
        assert sb.bank_summary("curve_fitting") == []
        assert sb.find_exemplar("curve_fitting", _fp(seed=1)) == []
        assert [r["id"] for r in sb.list_archived("curve_fitting")] == [rid]
        assert sb.restore_records("curve_fitting", [rid]) == 1
        assert [r["id"] for r in sb.list_records("curve_fitting")] == [rid]
        assert sb.list_archived("curve_fitting") == []

    def test_rebanking_an_archived_script_restores_it(self):
        rid = _bank(script="print('again')")
        sb.archive_records("curve_fitting", [rid])
        out = sb.add_record("curve_fitting", {
            "working_script": "print('again')", "data_fingerprint": _fp(center=30, seed=3),
            "provenance": {"session": "s2"}})
        assert out["id"] == rid and out["action"] == "restored"
        assert sb.independent_successes(_rec(rid)) == 2

    def test_auto_archive_runs_on_write_at_most_once_a_day(self):
        old = _bank(script="print('old')")      # first write sweeps (nothing stale yet)
        _age(old, 200)
        _yesterday_sweep()
        _bank(script="print('new1')", fp=_fp(center=40, seed=2))   # a new day: sweeps
        assert [r["id"] for r in sb.list_archived("curve_fitting")] == [old]
        older = _bank(script="print('older')", fp=_fp(center=50, seed=3))
        _age(older, 200)
        _bank(script="print('new2')", fp=_fp(center=70, seed=4))   # same day: no sweep
        assert older in [r["id"] for r in sb.list_records("curve_fitting")]

    def test_auto_archive_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("SCILINK_BANK_AUTO_ARCHIVE", "0")
        old = _bank(script="print('old')")
        _age(old, 200)
        _bank(script="print('new')", fp=_fp(center=40, seed=2))
        assert sb.list_archived("curve_fitting") == []
        assert [s["id"] for s in sb.stale_records("curve_fitting")] == [old]  # manual still sees it


class TestSweepCLI:
    def test_dry_run_then_sweep_then_restore(self, capsys, monkeypatch):
        from scilink.cli import memory as cli
        monkeypatch.setenv("SCILINK_BANK_AUTO_ARCHIVE", "0")
        rid = _bank()
        _age(rid, 200)
        ns = SimpleNamespace(domain=None, days=None, dry_run=True)
        assert cli._cmd_bank_sweep(ns) == 0
        assert rid in capsys.readouterr().out
        assert sb.list_archived("curve_fitting") == []
        ns.dry_run = False
        assert cli._cmd_bank_sweep(ns) == 0
        assert [r["id"] for r in sb.list_archived("curve_fitting")] == [rid]
        assert cli._cmd_bank_archived(SimpleNamespace(domain=None)) == 0
        assert rid in capsys.readouterr().out
        assert cli._cmd_bank_restore(SimpleNamespace(ref=f"curve_fitting/{rid}")) == 0
        assert [r["id"] for r in sb.list_records("curve_fitting")] == [rid]


class TestReviewedVerbatimReuseCountsAsVerbatim:
    """An edit-adaptation that needed no edits is the banked script itself,
    accepted under LLM verification — the reviewed route by which a script
    (including one born under a reduced-depth profile) earns verbatim
    evidence."""

    def _bump(self, n_edits, monkeypatch):
        from scilink.agents.exp_agents._qc_engine import bump_bank_adapt_success
        seen = {}
        monkeypatch.setattr(sb, "record_success",
                            lambda d, rid, session=None, **kw: seen.update(kw))
        host = SimpleNamespace(logger=SimpleNamespace(info=lambda *a, **k: None),
                               output_dir="/tmp/sess")
        bump_bank_adapt_success(
            host, {"success": True,
                   "bank_edit_adapt": {"id": "abc", "n_edits": n_edits}},
            domain="curve_fitting")
        return seen

    def test_zero_edits_is_verbatim(self, monkeypatch):
        assert self._bump(0, monkeypatch)["adapted"] is False

    def test_any_edit_is_an_adaptation(self, monkeypatch):
        assert self._bump(3, monkeypatch)["adapted"] is True


# ──────────────────────────────────────────────────────────────
# B12 — concurrent writers
# ──────────────────────────────────────────────────────────────

def _hammer(args):
    """Runs in a spawned process: bump one record's stats n times."""
    home, rid, n, tag = args
    import os
    os.environ["SCILINK_HOME"] = home
    os.environ["SCILINK_MEMORY"] = "1"
    from scilink.skills._shared import _script_bank as bank
    for i in range(n):
        bank.record_failure("curve_fitting", rid, "gate_failed")
        bank.mark_retrieved("curve_fitting", rid)
        bank.record_success("curve_fitting", rid, session=f"{tag}-{i}")
    return True


class TestConcurrentWriters:
    """Stats updates are read-modify-write; a series' replay pool, a meta
    fan-out and a live loop's escalation worker all write the same bank."""

    def test_no_update_is_lost_across_processes(self, tmp_path):
        import multiprocessing as mp
        rid = _bank()
        n_proc, n_each = 4, 15
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_proc) as pool:
            assert all(pool.map(_hammer, [(str(tmp_path), rid, n_each, f"p{k}")
                                          for k in range(n_proc)]))
        rec = _rec(rid)
        total = n_proc * n_each
        assert rec["stats"]["n_failures"] == total
        assert rec["stats"]["n_retrievals"] == total
        assert rec["stats"]["n_successes"] == 1 + total
        assert sb.verify_record(rec)

    def test_the_lock_is_reentrant_within_a_write(self, monkeypatch):
        # add_record -> aging sweep -> archive_records, all under one lock.
        old = _bank(script="print('old')")
        _age(old, 200)
        _yesterday_sweep()
        _bank(script="print('new')", fp=_fp(center=40, seed=2))     # must not deadlock
        assert [r["id"] for r in sb.list_archived("curve_fitting")] == [old]

    def test_public_names_are_the_locked_ones(self):
        for name in ("add_record", "record_success", "record_failure",
                     "mark_retrieved", "archive_records", "restore_records"):
            assert getattr(sb, name).__name__ == name
            assert getattr(sb, name).__wrapped__.__name__ == f"_{name}_unlocked"
