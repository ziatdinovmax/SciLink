"""Tests for #398 part 1: atomic publishes for the ~/.scilink store.

Engine-level tests for atomic_write_text plus a multi-process hammer
asserting a reader can never observe a torn file, and a store-level check
that script-bank record writes stay valid JSON under concurrency.
"""
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from scilink.utils.text_io import atomic_write_text


def test_basic_write_and_overwrite(tmp_path):
    p = tmp_path / "rec.json"
    atomic_write_text(p, '{"v": 1}')
    assert json.loads(p.read_text()) == {"v": 1}
    atomic_write_text(p, '{"v": 2}')
    assert json.loads(p.read_text()) == {"v": 2}
    # No temp-file droppings left behind.
    assert [f.name for f in tmp_path.iterdir()] == ["rec.json"]


def test_permissions_preserved_and_umask_honored(tmp_path):
    import stat

    p = tmp_path / "existing.json"
    p.write_text("{}")
    os.chmod(p, 0o644)
    atomic_write_text(p, '{"v": 1}')
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o644

    umask = os.umask(0)
    os.umask(umask)
    q = tmp_path / "fresh.json"
    atomic_write_text(q, '{"v": 1}')
    assert stat.S_IMODE(os.stat(q).st_mode) == 0o666 & ~umask


def test_unicode_roundtrip(tmp_path):
    p = tmp_path / "note.md"
    text = "α → β, Δ = −0.5 µm"
    atomic_write_text(p, text)
    assert p.read_text(encoding="utf-8") == text


def test_failed_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"

    class Boom:
        def __str__(self):
            raise RuntimeError("boom")

    try:
        atomic_write_text(p, Boom())  # type: ignore[arg-type]
    except Exception:
        pass
    assert list(tmp_path.iterdir()) == []


def _writer(path_str, wid, n_writes):
    # ~100 KB payloads widen the window a torn write would occupy.
    filler = "x" * 100_000
    for i in range(n_writes):
        atomic_write_text(path_str,
                          json.dumps({"writer": wid, "i": i, "fill": filler}))


def test_concurrent_writers_never_torn(tmp_path):
    """N processes hammer one file; every read must parse as a whole payload."""
    p = tmp_path / "shared.json"
    atomic_write_text(p, json.dumps({"writer": -1, "i": -1, "fill": ""}))

    n_writers, n_writes = 4, 150
    procs = [mp.Process(target=_writer, args=(str(p), w, n_writes))
             for w in range(n_writers)]
    for pr in procs:
        pr.start()

    reads = 0
    deadline = time.time() + 60
    while any(pr.is_alive() for pr in procs) and time.time() < deadline:
        rec = json.loads(p.read_text(encoding="utf-8"))  # must never raise
        assert set(rec) == {"writer", "i", "fill"}
        reads += 1
    for pr in procs:
        pr.join(timeout=30)
        assert pr.exitcode == 0

    final = json.loads(p.read_text(encoding="utf-8"))
    assert final["i"] == n_writes - 1  # some writer's last payload, whole
    assert reads > 0
    # Publishing left no temp files behind.
    assert [f.name for f in tmp_path.iterdir()] == ["shared.json"]


def _banker(root_str, session_id, script, n):
    from scilink.skills._shared import _script_bank as sb
    for _ in range(n):
        sb.add_record("curve_fitting", {
            "working_script": script,
            "provenance": {"session": session_id},
            "outcome": {"metric": "r2=0.99"},
        }, root=Path(root_str))


def test_script_bank_records_stay_parseable_under_concurrency(tmp_path, monkeypatch):
    """Concurrent add_record calls must never leave an unparseable record —
    the masked-data-loss mode where _find_by_hash skips a torn file forever.
    (Lost increments / duplicate records are the RMW race — #398 part 2.)"""
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    root = tmp_path / "bank"
    script = "import numpy as np\nprint('fit')\n"
    procs = [mp.Process(target=_banker, args=(str(root), f"s{w}", script, 25))
             for w in range(3)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=120)
        assert pr.exitcode == 0

    records = list((root / "curve_fitting").glob("*.json"))
    assert records, "no records were banked"
    for f in records:
        rec = json.loads(f.read_text(encoding="utf-8"))  # all parseable
        assert rec.get("script_hash")
    # No temp droppings in the domain dir.
    leftovers = [f.name for f in (root / "curve_fitting").iterdir()
                 if f.suffix == ".tmp"]
    assert leftovers == []
