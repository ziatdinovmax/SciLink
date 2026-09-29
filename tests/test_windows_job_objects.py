"""Windows: a script and everything it starts live in one Job object.

POSIX scripts run in their own process group and are ended as a group; on
Windows the equivalent is a Job object created with KILL_ON_JOB_CLOSE, so a
timeout or a Stop (TerminateJobObject) ends the whole tree, and SciLink dying
closes the handle and ends it too.

There is no Windows machine behind these tests. They check the structure
layout against the documented Win32 x64 sizes (identical on any LP64 host),
and drive the Win32 calls through a fake kernel32 whose terminate ends the
real (POSIX) process, so the executor's own control flow runs for real.
"""

import ctypes
import subprocess
import sys

import pytest

from scilink import executors as ex


def test_the_limit_structure_has_the_win32_x64_layout():
    info_t = ex._WindowsJob._limit_info_type()
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        pytest.skip("layout check is for 64-bit hosts")
    assert ctypes.sizeof(info_t) == 144                        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    basic = info_t._fields_[0][1]
    assert ctypes.sizeof(basic) == 64                          # JOBOBJECT_BASIC_LIMIT_INFORMATION
    assert basic.LimitFlags.offset == 16


class FakeKernel32:
    """Records the Win32 calls; TerminateJobObject ends the real processes."""

    def __init__(self, assign_ok=True):
        self.calls, self.assign_ok, self.jobs = [], assign_ok, {}

    def CreateJobObjectW(self, attrs, name):
        self.calls.append(("CreateJobObjectW",))
        h = 1000 + len(self.jobs)
        self.jobs[h] = []
        return h

    def SetInformationJobObject(self, h, klass, info_ref, size):
        flags = info_ref._obj.BasicLimitInformation.LimitFlags
        self.calls.append(("SetInformationJobObject", h, klass, flags, size))
        return 1

    def AssignProcessToJobObject(self, h, proc_handle):
        self.calls.append(("AssignProcessToJobObject", h, proc_handle))
        if self.assign_ok:
            self.jobs[h].append(proc_handle)
        return 1 if self.assign_ok else 0

    def TerminateJobObject(self, h, code):
        self.calls.append(("TerminateJobObject", h, code))
        for pid in self.jobs.get(h, []):
            FakeKernel32.procs[pid].kill()
        return 1

    def CloseHandle(self, h):
        self.calls.append(("CloseHandle", h))
        return 1

    procs = {}


@pytest.fixture
def windows(monkeypatch):
    fake = FakeKernel32()
    monkeypatch.setattr(ex, "_NEW_SESSION", False)
    monkeypatch.setattr(ex, "_WINDOWS_JOBS", True)
    monkeypatch.setattr(ex._WindowsJob, "_kernel32", fake)
    return fake


def _started():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    proc._handle = proc.pid                     # Popen's Win32 handle, stood in for by the pid
    FakeKernel32.procs[proc.pid] = proc
    return proc


def test_a_script_is_assigned_to_a_kill_on_close_job(windows):
    proc = _started()
    try:
        ex._mark_own_group(proc)
        names = [c[0] for c in windows.calls]
        assert names == ["CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject"]
        _, h, klass, flags, size = windows.calls[1]
        assert klass == 9 and flags == 0x2000 and size == ctypes.sizeof(ex._WindowsJob._limit_info_type())
        assert windows.calls[2] == ("AssignProcessToJobObject", h, proc.pid)
        assert proc._scilink_job.handle == h
    finally:
        proc.kill()
        proc.wait()


def test_a_timeout_or_stop_terminates_the_job_and_closes_it(windows):
    proc = _started()
    ex._mark_own_group(proc)
    h = proc._scilink_job.handle
    ex._kill_process_tree(proc, grace=0.1)
    assert ("TerminateJobObject", h, 1) in windows.calls
    assert windows.calls[-1] == ("CloseHandle", h)
    assert proc.poll() is not None


def test_leftovers_of_a_finished_script_end_with_its_job(windows):
    proc = _started()
    ex._mark_own_group(proc)
    h = proc._scilink_job.handle
    ex._end_leftover_group(proc)
    assert ("TerminateJobObject", h, 1) in windows.calls and ("CloseHandle", h) in windows.calls
    proc.wait(timeout=10)


def test_a_refused_assignment_falls_back_to_killing_the_process(monkeypatch):
    fake = FakeKernel32(assign_ok=False)
    monkeypatch.setattr(ex, "_NEW_SESSION", False)
    monkeypatch.setattr(ex, "_WINDOWS_JOBS", True)
    monkeypatch.setattr(ex._WindowsJob, "_kernel32", fake)
    proc = _started()
    ex._mark_own_group(proc)
    assert not hasattr(proc, "_scilink_job")
    assert ("CloseHandle", 1000) in fake.calls                 # the unused job is not leaked
    ex._kill_process_tree(proc, grace=0.1)                     # the process alone, as before jobs
    assert proc.poll() is not None


def test_posix_never_touches_the_win32_path(monkeypatch):
    monkeypatch.setattr(ex._WindowsJob, "_kernel32", None)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        ex._mark_own_group(proc)
        assert not hasattr(proc, "_scilink_job")
        assert ex._WindowsJob._kernel32 is None
    finally:
        ex._kill_process_tree(proc, grace=0.1)
