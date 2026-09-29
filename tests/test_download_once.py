"""Shared-cache downloads happen once and are never half-visible.

Several analyses can reach the same missing model file at once: fan-out
branches, best-of-N candidates, series replays in spawned processes. SAM wrote
straight to its final path behind an ``os.path.exists`` check, so a second
worker could load a checkpoint that was still arriving; the DCNN ensemble went
through one fixed ``.part`` name, so two downloaders wrote into the same file.
These tests drive the real download path against a slow local HTTP server.
"""

import io
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scilink.utils.download import DownloadError, download_once

PAYLOAD = os.urandom(256 * 1024)


class _Server:
    def __init__(self, payload=PAYLOAD, chunk=16 * 1024, delay=0.02, truncate=False):
        self.payload, self.chunk, self.delay, self.truncate = payload, chunk, delay, truncate
        self.hits = 0
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with server._lock:
                    server.hits += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(server.payload)))
                self.end_headers()
                body = server.payload[: len(server.payload) // 2] if server.truncate else server.payload
                for i in range(0, len(body), server.chunk):
                    self.wfile.write(body[i:i + server.chunk])
                    self.wfile.flush()
                    time.sleep(server.delay)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/weights.bin"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


def _leftovers(d: Path):
    return [p.name for p in d.iterdir() if p.name.endswith(".part")]


def test_concurrent_threads_download_once_and_all_get_the_whole_file(server, tmp_path):
    dest = tmp_path / "models" / "sam_vit_b.pth"
    results, errors = [], []

    def worker():
        try:
            results.append(download_once(server.url, dest).read_bytes())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert server.hits == 1
    assert len(results) == 8 and all(r == PAYLOAD for r in results)
    assert _leftovers(dest.parent) == []


def test_concurrent_processes_download_once(server, tmp_path):
    """Spawned series replays are processes, not threads: the lock must hold
    across them too."""
    dest = tmp_path / "models" / "w.bin"
    code = ("import sys; from scilink.utils.download import download_once; "
            "p = download_once(sys.argv[1], sys.argv[2]); print(p.stat().st_size)")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(Path(__file__).resolve().parents[1])] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    procs = [subprocess.Popen([sys.executable, "-c", code, server.url, str(dest)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
             for _ in range(3)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0], [o[1].decode()[-500:] for o in outs]
    assert server.hits == 1
    assert all(o[0].strip() == str(len(PAYLOAD)).encode() for o in outs)
    assert dest.read_bytes() == PAYLOAD


def test_a_reader_never_sees_a_partial_file(server, tmp_path):
    dest = tmp_path / "w.bin"
    seen = []
    t = threading.Thread(target=download_once, args=(server.url, dest))
    t.start()
    while t.is_alive():
        if dest.exists():
            seen.append(dest.stat().st_size)
        time.sleep(0.005)
    t.join()
    assert all(size == len(PAYLOAD) for size in seen)
    assert dest.read_bytes() == PAYLOAD


def test_a_short_transfer_fails_and_leaves_nothing_behind(tmp_path):
    s = _Server(truncate=True)
    try:
        dest = tmp_path / "w.bin"
        with pytest.raises(DownloadError):
            download_once(s.url, dest)
        assert not dest.exists()
        assert _leftovers(tmp_path) == []
    finally:
        s.close()


def test_a_failed_download_releases_the_lock_for_the_next_try(server, tmp_path):
    dest = tmp_path / "w.bin"
    with pytest.raises(DownloadError):
        download_once("http://127.0.0.1:9/nothing", dest, timeout=2)
    assert download_once(server.url, dest).read_bytes() == PAYLOAD


def test_a_downloaded_file_gets_the_umask_mode_not_mkstemps_0600(server, tmp_path):
    dest = tmp_path / "w.bin"
    download_once(server.url, dest)
    umask = os.umask(0)
    os.umask(umask)
    assert dest.stat().st_mode & 0o777 == 0o666 & ~umask


def test_an_existing_file_is_not_fetched_again(server, tmp_path):
    dest = tmp_path / "w.bin"
    dest.write_bytes(b"cached")
    assert download_once(server.url, dest).read_bytes() == b"cached"
    assert server.hits == 0


def test_sam_checkpoint_is_fetched_once_by_concurrent_analyzers(server, tmp_path, monkeypatch):
    from scilink.skills._shared import particle_analyzer as pa
    monkeypatch.setitem(pa.ParticleAnalyzer._MODEL_URLS, "vit_b", server.url)
    monkeypatch.setenv("SCILINK_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("HOME", str(tmp_path))            # no legacy cache in the way
    paths = []
    threads = [threading.Thread(target=lambda: paths.append(
        pa.ParticleAnalyzer._ensure_checkpoint(None, "vit_b"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert server.hits == 1
    assert len(set(paths)) == 1 and Path(paths[0]).read_bytes() == PAYLOAD


# ── the DCNN ensemble: one download, and the folder appears whole ────────────

N_MEMBERS = 5


def _ensemble_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(N_MEMBERS):
            z.writestr(f"atomnet_ensemble/atomnet3_{i}.tar", os.urandom(64 * 1024))
    return buf.getvalue()


def test_concurrent_dcnn_requests_download_once_and_never_see_part_of_the_ensemble(tmp_path, monkeypatch):
    from scilink.skills._shared import atomistic_model_manager as mm
    from scilink.skills.image_analysis.atomic_stem import atomic_stem as tools

    s = _Server(payload=_ensemble_zip(), chunk=8 * 1024, delay=0.01)

    def slow_unzip(zip_path, out_dir, logger):
        # Extract one member at a time, slowly, so a reader has every chance
        # to catch a folder holding only some of them.
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                z.extract(member, out_dir)
                time.sleep(0.05)
        return True

    monkeypatch.setattr(tools, "unzip_file", slow_unzip)
    monkeypatch.setenv("SCILINK_MODELS", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    settings = {"dcnn_model_url": s.url}
    target = tmp_path / "models" / "dcnn_trained"
    partial_views, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            members = list(target.glob("**/atomnet3_*.tar")) if target.exists() else []
            if 0 < len(members) < N_MEMBERS:
                partial_views.append(len(members))
            time.sleep(0.002)

    watcher = threading.Thread(target=watch)
    watcher.start()
    results = []
    workers = [threading.Thread(target=lambda: results.append(
        mm.get_or_download_atomistic_model(settings, logging.getLogger("t")))) for _ in range(4)]
    try:
        for t in workers:
            t.start()
        for t in workers:
            t.join()
    finally:
        stop.set()
        watcher.join()
        s.close()
    assert s.hits == 1
    assert partial_views == []
    assert len(set(results)) == 1 and results[0] is not None
    assert len(list(Path(results[0]).glob("atomnet3_*.tar"))) == N_MEMBERS
    umask = os.umask(0)
    os.umask(umask)
    assert target.stat().st_mode & 0o777 == 0o777 & ~umask     # not mkdtemp's 0700
    leftovers = [p.name for p in (tmp_path / "models").iterdir() if p.name.startswith(".dcnn_trained.")
                 and not p.name.endswith(".lock")]
    assert leftovers == []


def test_an_incomplete_dcnn_folder_is_moved_aside_not_deleted(tmp_path, monkeypatch):
    from scilink.skills._shared import atomistic_model_manager as mm

    s = _Server(payload=_ensemble_zip(), delay=0)
    monkeypatch.setenv("SCILINK_MODELS", str(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models" / "dcnn_trained"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("mine")          # no model files: incomplete
    try:
        path = mm.get_or_download_atomistic_model({"dcnn_model_url": s.url}, logging.getLogger("t"))
    finally:
        s.close()
    assert len(list(Path(path).glob("atomnet3_*.tar"))) == N_MEMBERS
    kept = list((tmp_path / "models").glob(".dcnn_trained.*.stale/dcnn_trained/notes.txt"))
    assert len(kept) == 1 and kept[0].read_text() == "mine"


# ── review follow-ups ─────────────────────────────────────────────────────

def test_concurrent_downloads_never_touch_the_process_umask(tmp_path):
    """Reading the umask (umask(0) then restore) changed it for the whole
    process; threads racing through that window left files world-writable
    or the umask at 0. Files are now created with the umask applied by the
    kernel, so neither the downloads nor an unrelated writer can see it move."""
    import sys
    servers = [_Server(chunk=8 * 1024, delay=0.001) for _ in range(4)]
    before = os.umask(0o022)
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        stop = threading.Event()
        unrelated = []

        def writer():
            i = 0
            while not stop.is_set():
                f = tmp_path / "results" / f"r{i}.txt"
                f.parent.mkdir(exist_ok=True)
                f.write_text("x")
                unrelated.append(f)
                i += 1

        w = threading.Thread(target=writer)
        w.start()
        dests = [tmp_path / "cache" / f"w{i}.bin" for i in range(4)]
        ts = [threading.Thread(target=download_once, args=(s.url, d)) for s, d in zip(servers, dests)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        stop.set()
        w.join()
        assert os.umask(0o022) == 0o022                  # untouched
        assert all(d.stat().st_mode & 0o777 == 0o644 for d in dests)
        assert unrelated and all(f.stat().st_mode & 0o777 == 0o644 for f in unrelated)
    finally:
        sys.setswitchinterval(old_switch)
        os.umask(before)
        for s in servers:
            s.close()


def test_a_filesystem_without_locking_still_downloads(server, tmp_path, monkeypatch, caplog):
    """HPC mounts without flock raise ENOTSUP/ENOLCK: run unlocked (atomic
    publish still holds), warn once, never fail the model."""
    import errno
    import fcntl
    import logging as _logging
    from scilink.utils import file_lock
    real = fcntl.flock

    def unsupported(fh, op):
        if op & (fcntl.LOCK_EX | fcntl.LOCK_NB):
            raise OSError(errno.ENOTSUP, "Operation not supported")
        return real(fh, op)

    monkeypatch.setattr(fcntl, "flock", unsupported)
    file_lock._warned_unsupported.clear()
    with caplog.at_level(_logging.WARNING, logger="scilink.utils.file_lock"):
        a = download_once(server.url, tmp_path / "a.bin")
        b = download_once(server.url, tmp_path / "b.bin")
    assert a.read_bytes() == PAYLOAD and b.read_bytes() == PAYLOAD
    assert sum("not supported" in r.message for r in caplog.records) == 1


def test_the_windows_lock_serializes_across_handles(tmp_path, monkeypatch):
    """No fcntl: path_lock uses msvcrt.locking on byte 0 of the lock file.
    Simulated with a fake msvcrt whose locks conflict across handles."""
    import sys
    import types
    held, guard = {}, threading.Lock()

    def locking(fd, mode, nbytes):
        key = os.fstat(fd).st_ino
        with guard:
            if mode == fake.LK_UNLCK:
                held.pop(key, None)
                return
            if key in held:
                raise OSError(13, "locked")
            held[key] = fd

    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    from scilink.utils.file_lock import is_locked, path_lock
    inside, overlap = [0], []

    def work():
        with path_lock(tmp_path / "x"):
            inside[0] += 1
            overlap.append(inside[0])
            time.sleep(0.05)
            inside[0] -= 1

    ts = [threading.Thread(target=work) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert max(overlap) == 1
    with path_lock(tmp_path / "x"):
        assert is_locked(tmp_path / "x")
    assert not is_locked(tmp_path / "x")


def test_is_locked_sees_a_holder_in_another_process(tmp_path):
    from scilink.utils.file_lock import is_locked
    target = tmp_path / "kb"
    code = ("import sys, time; from scilink.utils.file_lock import path_lock\n"
            "with path_lock(sys.argv[1]):\n    print('held', flush=True); time.sleep(5)")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    p = subprocess.Popen([sys.executable, "-c", code, str(target)], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert is_locked(target)
    finally:
        p.kill()
        p.wait()
    assert not is_locked(target)                      # the kernel released it


def test_a_waiter_says_what_it_is_waiting_for(server, tmp_path, caplog):
    import logging as _logging
    from scilink.utils.file_lock import path_lock
    dest = tmp_path / "big.pth"
    release = threading.Event()

    def holder():
        with path_lock(dest):
            release.wait(5)

    h = threading.Thread(target=holder)
    h.start()
    time.sleep(0.1)
    with caplog.at_level(_logging.INFO, logger="scilink.utils.file_lock"):
        t = threading.Thread(target=download_once, args=(server.url, dest))
        t.start()
        time.sleep(0.3)
        release.set()
        t.join()
        h.join()
    assert any("Waiting for another process working on big.pth" in r.message for r in caplog.records)


def test_a_complete_ensemble_published_meanwhile_is_kept(tmp_path, monkeypatch):
    """Where locking is unavailable a second publisher used to move a
    complete ensemble aside (and leave a 770 MB .stale copy): it now keeps
    the one already there."""
    from scilink.skills._shared import atomistic_model_manager as mm
    from scilink.skills.image_analysis.atomic_stem import atomic_stem as tools
    out = tmp_path / "dcnn_trained"

    def fake_url(url, dest, logger):
        Path(dest).write_bytes(_ensemble_zip())
        # meanwhile another process publishes a complete ensemble
        (out / "atomnet_ensemble").mkdir(parents=True)
        for i in range(N_MEMBERS):
            (out / "atomnet_ensemble" / f"atomnet3_{i}.tar").write_bytes(b"theirs")
        return dest

    monkeypatch.setattr(mm, "_download_url", fake_url)
    assert mm._download_and_extract_model("gid", str(out), logging.getLogger("t"), url="https://x/y.zip")
    assert (out / "atomnet_ensemble" / "atomnet3_0.tar").read_bytes() == b"theirs"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dcnn_trained"]    # no .stale, no staging


def test_no_download_or_publish_path_touches_the_umask(server, tmp_path, monkeypatch):
    """The race above is two syscalls wide and rarely shows in a test; this
    pins the property itself: nothing on these paths calls os.umask."""
    from scilink.skills._shared import atomistic_model_manager as mm
    calls = []
    real = os.umask
    monkeypatch.setattr(os, "umask", lambda m: calls.append(m) or real(m))
    download_once(server.url, tmp_path / "a.bin")
    s = _Server(payload=_ensemble_zip(), delay=0)
    try:
        mm._download_and_extract_model("gid", str(tmp_path / "dcnn_trained"), logging.getLogger("t"), url=s.url)
    finally:
        s.close()
    assert calls == []
