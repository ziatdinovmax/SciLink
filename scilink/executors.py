import json
import os
import sys
import subprocess
import re
import uuid
import tempfile
import logging
import signal
import threading
from typing import Optional

from .auth import get_api_key

DEFAULT_TIMEOUT = 600

# Global registry of active subprocesses, keyed by thread ID.
# Accessible from any thread so the UI stop handler can kill them.
# Re-entrant: the SIGHUP/SIGTERM handler below takes it on the main thread,
# which may be holding it (inside _register_subprocess) when the signal lands.
_active_subprocesses_lock = threading.RLock()
_active_subprocesses: dict[int, set[subprocess.Popen]] = {}


def _forget_inherited_subprocesses() -> None:
    """In a forked child the registry holds the PARENT's scripts; a SIGTERM
    to the child must not end them."""
    global _active_subprocesses_lock
    _active_subprocesses_lock = threading.RLock()
    _active_subprocesses.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_subprocesses)


def _register_subprocess(proc: subprocess.Popen) -> None:
    """Register a subprocess so it can be killed when the user clicks Stop."""
    tid = threading.get_ident()
    with _active_subprocesses_lock:
        _active_subprocesses.setdefault(tid, set()).add(proc)


def _unregister_subprocess(proc: subprocess.Popen) -> None:
    """Remove a subprocess from the registry."""
    tid = threading.get_ident()
    with _active_subprocesses_lock:
        procs = _active_subprocesses.get(tid)
        if procs:
            procs.discard(proc)
            if not procs:
                del _active_subprocesses[tid]


# A generated script runs in its own session and process group (POSIX), so a
# timeout or a Stop can end the whole tree it started: a loky or
# multiprocessing pool for per-pixel fits, a solver it shelled out to.
# Killing only the direct child left those grandchildren running, holding
# memory and CPU after the analysis had moved on.
_NEW_SESSION = os.name == "posix"


_WINDOWS_JOBS = os.name == "nt"


class _WindowsJob:
    """A Win32 Job object holding one script and everything it starts: the
    Windows counterpart of a POSIX process group.

    Created with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: terminating the job
    ends the whole tree, and if SciLink itself dies its handle closes and
    Windows ends the tree too (the job does on its own what the POSIX side
    needs signal handlers for). A process is assigned right after it starts;
    a child it creates in the first instants of interpreter startup, before
    the assignment, would escape, but a generated script starts nothing that
    early. Any failure (no ctypes, an older Windows refusing a nested job)
    leaves the process unassigned and the caller falls back to killing the
    process alone, the behaviour before jobs.

    Not run on real Windows in this repository's tests: they drive it
    through a fake ``kernel32`` and check the structure layout against the
    Win32 x64 sizes.
    """
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    _kernel32 = None            # tests inject a fake

    def __init__(self, handle):
        self.handle = handle

    @classmethod
    def _k32(cls):
        if cls._kernel32 is None:
            import ctypes
            from ctypes import wintypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            # Declared, or ctypes passes and returns a HANDLE as a 32-bit int.
            k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k.CreateJobObjectW.restype = wintypes.HANDLE
            k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                  ctypes.c_void_p, wintypes.DWORD]
            k.SetInformationJobObject.restype = wintypes.BOOL
            k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k.AssignProcessToJobObject.restype = wintypes.BOOL
            k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k.TerminateJobObject.restype = wintypes.BOOL
            k.CloseHandle.argtypes = [wintypes.HANDLE]
            k.CloseHandle.restype = wintypes.BOOL
            cls._kernel32 = k
        return cls._kernel32

    @staticmethod
    def _limit_info_type():
        import ctypes
        from ctypes import c_int64, c_size_t, c_uint32, c_ulonglong

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", c_int64), ("PerJobUserTimeLimit", c_int64),
                        ("LimitFlags", c_uint32), ("MinimumWorkingSetSize", c_size_t),
                        ("MaximumWorkingSetSize", c_size_t), ("ActiveProcessLimit", c_uint32),
                        ("Affinity", c_size_t), ("PriorityClass", c_uint32),
                        ("SchedulingClass", c_uint32)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", c_size_t),
                        ("JobMemoryLimit", c_size_t), ("PeakProcessMemoryUsed", c_size_t),
                        ("PeakJobMemoryUsed", c_size_t)]
        return JOBOBJECT_EXTENDED_LIMIT_INFORMATION

    @classmethod
    def attach(cls, proc: subprocess.Popen) -> "Optional[_WindowsJob]":
        import ctypes
        try:
            k = cls._k32()
            handle = k.CreateJobObjectW(None, None)
            if not handle:
                return None
            info = cls._limit_info_type()()
            info.BasicLimitInformation.LimitFlags = cls.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (not k.SetInformationJobObject(handle, cls.JobObjectExtendedLimitInformation,
                                              ctypes.byref(info), ctypes.sizeof(info))
                    or not k.AssignProcessToJobObject(handle, int(proc._handle))):
                k.CloseHandle(handle)
                return None
        except Exception:  # noqa: BLE001 - containment is best effort
            return None
        job = cls(handle)
        proc._scilink_job = job              # type: ignore[attr-defined]
        return job

    def terminate(self) -> None:
        if self.handle:
            try:
                self._k32().TerminateJobObject(self.handle, 1)
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self.handle:
            try:
                self._k32().CloseHandle(self.handle)
            except Exception:  # noqa: BLE001
                pass
            self.handle = None


def _mark_own_group(proc: subprocess.Popen) -> subprocess.Popen:
    """Contain ``proc`` and everything it starts: on POSIX record that it was
    started in a new session by us (its process group id is its pid by
    construction); on Windows assign it to a Job object."""
    if _NEW_SESSION:
        proc._scilink_own_group = True       # type: ignore[attr-defined]
    elif _WINDOWS_JOBS:
        _WindowsJob.attach(proc)
    return proc


def _group_of(proc: subprocess.Popen) -> "int | None":
    """The process group to signal for ``proc``, or None to signal it alone.

    A group we created is ``proc.pid``, known without asking: on macOS
    ``getpgid`` of a script that has already exited (a zombie whose helper
    still holds its pipes) raises ``ESRCH``, which is exactly the case where
    the group matters. The id cannot name someone else's group while the
    leader is unreaped or any member of the group is alive. For a process
    we did not start, the group is used only if it leads its own.
    """
    if not _NEW_SESSION:
        return None
    if getattr(proc, "_scilink_own_group", False):
        return proc.pid
    try:
        pgid = os.getpgid(proc.pid)
        if pgid == proc.pid and pgid != os.getpgrp():
            return pgid
    except OSError:              # already gone and reaped
        pass
    return None


def _end_leftover_group(proc: subprocess.Popen) -> None:
    """After a script exited on its own: end anything it left running in its
    group (a detached helper that does not hold its pipes). A generated
    script has no business leaving a daemon behind."""
    job = getattr(proc, "_scilink_job", None)
    if job is not None:
        job.terminate()
        job.close()
        return
    if _NEW_SESSION and getattr(proc, "_scilink_own_group", False):
        try:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:          # the group is already empty
            pass


def _kill_process_tree(proc: subprocess.Popen, grace: float = 2.0) -> None:
    """SIGTERM the script's process group, give it ``grace`` seconds, then
    SIGKILL whatever is left of the group.

    The group is signalled only when the process leads its own group;
    otherwise, and off POSIX, only the process itself is signalled, so a
    caller's own group is never hit.
    """
    job = getattr(proc, "_scilink_job", None)
    if job is not None:
        # Windows: the job ends the whole tree at once (TerminateProcess has
        # no graceful form to wait on).
        job.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        job.close()
        return
    group = _group_of(proc)
    # The descendants as they stand now: a grandchild that leads a session
    # of its own (a worker process's generated script, a replay child) is
    # not in this group, and its parent's SIGTERM handler — the only thing
    # that would end it — does not run while that parent sits in one long C
    # call. Whatever of the snapshot is still alive after the group is gone
    # is killed by pid.
    descendants = _descendants_of(proc.pid) if proc.poll() is None else []

    def send(sig):
        if group is not None:
            try:
                os.killpg(group, sig)
                return
            except OSError:      # the group is empty
                pass
        try:
            proc.send_signal(sig)
        except OSError:
            pass

    if proc.poll() is None:
        send(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    # Always: the leader may have exited on SIGTERM while its pool workers
    # ignored it.
    send(getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    _kill_survivors(descendants)


def _descendants_of(pid: int) -> list:
    """The process's descendants (``psutil``), or ``[]`` without it."""
    try:
        import psutil
        return psutil.Process(pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - no psutil, or the process already gone
        return []


def _kill_survivors(descendants: list) -> None:
    """SIGKILL what is left of a snapshot taken before the tree was
    signalled, and reap. A pid reused since the snapshot is recognised by
    psutil's creation-time check and left alone."""
    for p in descendants:
        try:
            if p.is_running() and p.status() != "zombie":
                p.kill()
        except Exception:  # noqa: BLE001 - gone, or not ours any more
            continue
    for p in descendants:
        try:
            p.wait(timeout=2)
        except Exception:  # noqa: BLE001
            continue


def _kill_all_registered() -> None:
    """At interpreter exit: a script in its own session no longer gets the
    terminal's SIGINT or SIGHUP, so end the ones still registered."""
    with _active_subprocesses_lock:
        procs = [p for ps in _active_subprocesses.values() for p in ps]
        _active_subprocesses.clear()
    for proc in procs:
        try:
            _kill_process_tree(proc, grace=0.5)
        except Exception:  # noqa: BLE001 - best effort at exit
            pass


import atexit as _atexit
_atexit.register(_kill_all_registered)


def _on_fatal_signal(signum, frame):  # noqa: ARG001 - signal handler signature
    """SIGHUP (the terminal closed, an SSH session dropped) and SIGTERM end
    the interpreter WITHOUT running atexit, and a script in its own session
    does not receive them. End the registered scripts, then die of the same
    signal as before."""
    try:
        _kill_all_registered()
    finally:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_fatal_signal_cleanup() -> None:
    """Only on the main thread (signal handlers can be set nowhere else) and
    only where nobody else handles the signal: a web server's or uvicorn's
    own handlers stay untouched, and they shut down through atexit."""
    if not _NEW_SESSION or threading.current_thread() is not threading.main_thread():
        return
    for name in ("SIGHUP", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            if signal.getsignal(sig) is signal.SIG_DFL:
                signal.signal(sig, _on_fatal_signal)
        except (ValueError, OSError):
            pass


_install_fatal_signal_cleanup()


def kill_subprocesses_for_thread(tid: int) -> None:
    """Terminate all subprocesses registered by a given thread — including
    subprocesses of fan-out worker threads registered (via
    ``log_context.register_worker``) as children of that thread, so a
    best-of-N candidate's running script is also stopped.

    Safe to call from any thread (e.g. the Streamlit UI thread).
    """
    from scilink.utils.log_context import effective_thread
    with _active_subprocesses_lock:
        target_tids = [t for t in _active_subprocesses
                       if t == tid or effective_thread(t) == tid]
        procs = [p for t in target_tids
                 for p in _active_subprocesses.pop(t, [])]
    for proc in procs:
        _kill_process_tree(proc)      # SIGTERM, a moment to clean up, SIGKILL

# Global cache for sandbox approval (shared across all agents in session)
_GLOBAL_SANDBOX_APPROVED: bool = False

# Description of what the LLM is instructed to do
LLM_EXECUTION_DESCRIPTION = """
WHAT THIS SYSTEM DOES:
  An LLM (Large Language Model) will generate and execute Python code on your
  machine to perform scientific data analysis. The LLM is instructed to:

  • Write Python scripts for curve fitting, spectral analysis, and data processing
  • Use scientific libraries: NumPy, SciPy, scikit-learn, matplotlib, pandas
  • Read input data files you provide (CSV, NPY, TXT, etc.)
  • Save output files (plots, results) to the designated output directory
  • Execute the generated code automatically without manual review

  The LLM is NOT instructed to:
  • Access the internet or make network requests
  • Modify system files or install software
  • Access files outside the working/output directories
  • Execute shell commands beyond running Python scripts

  However, AI-generated code can behave unexpectedly. A sandbox provides
  protection against unintended actions.
""".strip()


def is_in_colab():
    """Check for Google Colab environment."""
    if 'COLAB_GPU' in os.environ or 'GCE_METADATA_TIMEOUT' in os.environ:
        return True
    if 'google.colab' in sys.modules:
        return True
    return False


def check_security_sandbox_indicators(verbose=False):
    """Check for OS-level sandboxing indicators."""
    score = 0
    positive_indicators = []

    # Tier 1: High-Confidence Environments (Score: 10)
    if is_in_colab():
        score += 10
        positive_indicators.append("google_colab")
        if verbose:
            logging.info("High-Confidence Indicator: Google Colab environment detected.")
        return score, positive_indicators

    # Tier 2: Strong Indicators (Score: 5)
    if os.path.exists('/.dockerenv') or ('docker' in (open('/proc/1/cgroup').read() if os.path.exists('/proc/1/cgroup') else '')):
        score += 5
        positive_indicators.append("docker_container")
        if verbose:
            logging.info("Strong Indicator: Docker or container environment detected.")

    try:
        if sys.platform.startswith("linux"):
            result = subprocess.run(['systemd-detect-virt'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip() != 'none':
                score += 5
                positive_indicators.append(f"virtual_machine:{result.stdout.strip()}")
                if verbose:
                    logging.info(f"Strong Indicator: Virtual Machine detected ('{result.stdout.strip()}').")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Tier 3: Corroborating Evidence (Score: 2)
    try:
        mac = ':'.join(re.findall('..', f'{uuid.getnode():012x}'))
        vm_mac_prefixes = ["08:00:27", "00:05:69", "00:0c:29", "00:1c:14", "00:50:56"]
        if any(mac.lower().startswith(prefix) for prefix in vm_mac_prefixes):
            score += 2
            positive_indicators.append("vm_mac_address")
            if verbose:
                logging.info("Corroborating Indicator: VM-associated MAC address found.")
    except Exception:
        pass

    return score, list(set(positive_indicators))


def prompt_user_for_unsafe_execution(show_llm_description=True):
    """
    Prompt the user to decide whether to proceed without a sandbox.
    
    Returns:
        bool: True if user chooses to proceed, False to abort.
    """
    if not sys.stdin.isatty():
        logging.warning("Non-interactive environment detected. Cannot prompt user.")
        return False
    
    print("\n" + "=" * 74)
    print("⚠️  WARNING: NO SECURITY SANDBOX DETECTED ⚠️")
    print("=" * 74)
    
    if show_llm_description:
        print()
        print(LLM_EXECUTION_DESCRIPTION)
        print()
        print("-" * 74)
    
    print("""
WHY A SANDBOX IS RECOMMENDED:
  While the LLM is instructed to perform only safe operations, AI-generated
  code can sometimes behave unexpectedly due to:
  • Misinterpretation of instructions
  • Hallucinated or incorrect code patterns  
  • Edge cases in data that trigger unusual behavior

POTENTIAL RISKS WITHOUT A SANDBOX:
  • Accidental file modifications outside intended directories
  • High CPU/memory usage from inefficient generated code
  • Unexpected interactions with your Python environment

HOW TO RUN SAFELY:
  1. Docker (Recommended):  Run in a container using the provided Dockerfile
  2. Virtual Machine:       Use VMware, VirtualBox, or a cloud VM
  3. Google Colab:          Use Colab's free isolated environment

If you understand the risks and want to proceed anyway, you may continue.
""")
    print("=" * 74)
    
    while True:
        try:
            response = input("\n❓ Proceed WITHOUT sandbox protection? (y=yes, n=abort) [N]: ").strip().lower()
            if response in ('y', 'yes'):
                print()
                logging.warning("⚠️  User acknowledged risks and chose to proceed without sandbox.")
                return True
            elif response in ('n', 'no', ''):
                print()
                logging.info("User chose to abort. No code will be executed.")
                return False
            else:
                print("   Please enter 'y' to proceed or 'n' to abort.")
        except (EOFError, KeyboardInterrupt):
            print("\n\nAborted by user.")
            return False


def require_sandbox_approval(
    interactive: bool = True,
    allow_override: bool = True,
    context: str = "This operation"
) -> bool:
    """
    Check sandbox and get user approval if needed. 
    
    Results are cached globally so user is only prompted once per Python session,
    regardless of how many agents are created.
    
    Args:
        interactive: If True, prompt user when no sandbox detected
        allow_override: If True, respect UNSAFE_EXECUTION_OK env var
        context: Description of what will execute code (for user message)
    
    Returns:
        bool: True if execution is approved, False if user declined
        
    Raises:
        RuntimeError: If non-interactive and no sandbox/override
    """
    global _GLOBAL_SANDBOX_APPROVED
    
    # Check global cache first
    if _GLOBAL_SANDBOX_APPROVED:
        logging.info("✅ Sandbox approval already granted this session")
        return True
    
    # Check for environment variable override
    if allow_override and os.environ.get("UNSAFE_EXECUTION_OK", "false").lower() == "true":
        logging.warning("⚠️  Sandbox bypass via UNSAFE_EXECUTION_OK environment variable")
        _GLOBAL_SANDBOX_APPROVED = True
        return True
    
    # Check sandbox indicators
    score, indicators = check_security_sandbox_indicators(verbose=False)
    
    if score >= 4:
        friendly_name = indicators[0] if indicators else "sandbox"
        logging.info(f"✅ Sandbox detected ({friendly_name}) - code execution enabled")
        _GLOBAL_SANDBOX_APPROVED = True
        return True
    
    # No sandbox detected - need user approval
    if not interactive:
        raise RuntimeError(
            f"No sandbox detected and interactive=False. "
            f"Set UNSAFE_EXECUTION_OK=true or run in Docker/VM/Colab."
        )
    
    if not sys.stdin.isatty():
        logging.error("No sandbox detected and non-interactive terminal.")
        return False
    
    # Prompt user
    print("\n" + "=" * 74)
    print(f"⚠️  {context.upper()} REQUIRES CODE EXECUTION")
    print("=" * 74)
    print(LLM_EXECUTION_DESCRIPTION)
    
    approved = prompt_user_for_unsafe_execution(show_llm_description=False)
    
    if approved:
        _GLOBAL_SANDBOX_APPROVED = True
    
    return approved


def get_execution_description():
    """Return a description of what the LLM execution system does."""
    return LLM_EXECUTION_DESCRIPTION


# ── what a generated script may see and use ─────────────────────────
#
# A script runs as a child of the agent process. It used to inherit that
# process's whole environment, which is where the model's vendor keys, a
# Bedrock token, the web server's access token and the proxy key live. None of
# those is the script's business (the consent text says the model is not
# instructed to reach the network; the environment should not hand it the
# means either), so the child gets an allowlist: what Python, the scientific
# stack and SciLink's own tools legitimately read, plus what the executor was
# given explicitly. Anything secret-shaped is dropped even inside an allowed
# family. ``SCILINK_SANDBOX_ENV`` names extra variables to pass (comma
# separated) for a site that needs something not listed here.

_SANDBOX_ENV_NAMES = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LANGUAGE",
    "TMPDIR", "TEMP", "TMP", "DISPLAY", "PWD",
    "PYTHONPATH", "PYTHONHOME", "PYTHONHASHSEED", "PYTHONIOENCODING",
    "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONWARNINGS",
    "VIRTUAL_ENV", "MPLBACKEND", "MPLCONFIGDIR", "MP_API_KEY",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "MKLROOT", "CC", "CXX", "FC", "UNSAFE_EXECUTION_OK",
    # Windows: Python and the C runtime need these to start at all
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "HOMEDRIVE", "HOMEPATH",
    "USERNAME", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    # macOS
    "__CF_USER_TEXT_ENCODING",
})
_SANDBOX_ENV_PREFIXES = (
    "SCILINK_", "LC_", "XDG_", "CONDA_", "CUDA_", "PYTORCH_", "TORCH_", "HF_",
    "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "KMP_", "VECLIB_", "TF_", "JAX_",
    "NUMBA_",
    # the simulation toolchain a generated script may drive locally
    "ASE_", "VASP_", "LAMMPS_", "PMG_", "OPENMM_", "OMPI_", "MPICH_", "I_MPI_",
    "SLURM_", "PBS_",
)
# Never passed, whatever family they fall in.
_SANDBOX_ENV_DENY = frozenset({"SCILINK_API_KEY", "SCILINK_WEB_TOKEN", "HF_TOKEN"})
_SECRET_SUFFIXES = ("_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD", "_CREDENTIALS")


def sandbox_env(extra: "dict | None" = None, source: "dict | None" = None) -> dict:
    """The environment a generated script runs with: the allowlist above
    filtered from ``source`` (the process environment by default), plus
    ``extra`` verbatim (what the executor was handed explicitly, e.g. the
    Materials Project key)."""
    src = os.environ if source is None else source
    env: dict = {}
    for k, v in src.items():
        if k in _SANDBOX_ENV_DENY or k.upper().endswith(_SECRET_SUFFIXES):
            continue
        if k in _SANDBOX_ENV_NAMES or k.startswith(_SANDBOX_ENV_PREFIXES):
            env[k] = v
    for name in (src.get("SCILINK_SANDBOX_ENV") or "").split(","):
        name = name.strip()
        if name and name in src:
            env[name] = src[name]
    env.update({str(k): str(v) for k, v in (extra or {}).items() if v is not None})
    return env


def _sandbox_limits() -> "dict[str, int]":
    """Resource limits for a script's process, from the environment (megabytes;
    unset means unlimited): ``SCILINK_SANDBOX_MEM_MB`` (address space),
    ``SCILINK_SANDBOX_FILE_MB`` (largest file it may write),
    ``SCILINK_SANDBOX_PROCS`` (processes per user, a fork-bomb stop). Off by
    default: an address-space cap breaks CUDA and Metal, which reserve far
    more than they use, so a deployment sets what its container can take."""
    out = {}
    for var, key, scale in (("SCILINK_SANDBOX_MEM_MB", "RLIMIT_AS", 1 << 20),
                            ("SCILINK_SANDBOX_FILE_MB", "RLIMIT_FSIZE", 1 << 20),
                            ("SCILINK_SANDBOX_PROCS", "RLIMIT_NPROC", 1)):
        raw = os.environ.get(var)
        if raw:
            try:
                out[key] = int(float(raw) * scale)
            except ValueError:
                logging.warning(f"{var}={raw!r} is not a number; ignored")
    return out


def _sandbox_preexec():
    """``preexec_fn`` applying :func:`_sandbox_limits` in the child, or ``None``
    when there is nothing to apply (or no ``resource`` module: Windows)."""
    limits = _sandbox_limits()
    if not limits:
        return None
    try:
        import resource
    except ImportError:
        return None

    def apply():
        for key, value in limits.items():
            res = getattr(resource, key, None)
            if res is None:
                continue
            try:
                soft, hard = resource.getrlimit(res)
                cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
                resource.setrlimit(res, (cap, hard))
            except (ValueError, OSError):
                pass
    return apply


def _run_tracked(argv, *, timeout=None, input=None, text=True, cwd=None, env=None,
                 shell=False, preexec=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 stdin=None, encoding=None, errors=None, on_start=None):
    """``subprocess.run``'s shape for a process SciLink must be able to end:
    its own session and process group (a Job on Windows), registered for the
    user's Stop, and its whole tree killed on a timeout or an interrupt, so a
    grandchild (an MPI rank, a solver's helper) never outlives it (#685).
    Returns a ``CompletedProcess``; raises ``subprocess.TimeoutExpired`` on a
    timeout, as ``subprocess.run`` does.

    A worker's cancel (a budget, memory or coordinator cancel, or the turn's
    Stop) is checked before the process starts and after it ends: the kill
    that ends a running engine returns as an ordinary result, and without the
    check the worker went on to start its next engine run, which no one-shot
    kill covers.

    ``on_start(proc)`` is called once the process is registered, for a caller
    that watches it while it runs (a process worker's memory sampler); it
    must return quickly and never raise."""
    from scilink.utils.log_context import raise_if_cancelled
    raise_if_cancelled()
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else stdin,
                            stdout=stdout, stderr=stderr, text=text, cwd=cwd, env=env, shell=shell,
                            preexec_fn=preexec, start_new_session=_NEW_SESSION,
                            encoding=encoding, errors=errors)
    _mark_own_group(proc)
    _register_subprocess(proc)
    try:
        if on_start is not None:
            on_start(proc)
        try:
            out, err = proc.communicate(input=input, timeout=timeout)
            _end_leftover_group(proc)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc, grace=0.5)
            raise subprocess.TimeoutExpired(argv, timeout)
        except BaseException:
            _kill_process_tree(proc, grace=0.5)
            raise
    finally:
        _unregister_subprocess(proc)
    try:
        raise_if_cancelled()
    except BaseException as stop:
        # what the stopped process wrote travels with the stop, so a caller
        # can keep it beside the run (why was this run cut off?)
        try:
            stop.stdout, stop.stderr, stop.returncode = out, err, proc.returncode
        except Exception:  # noqa: BLE001 - an exception type without attributes
            pass
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def run_generated_script(script_path, *, timeout=None, args=None, cwd=None, extra_env=None):
    """Run a model-written Python script the way the analysis executor does
    (#685): ``sys.executable`` (never a ``python`` found on PATH), the
    sandboxed environment (no provider keys), the sandbox resource limits,
    Stop registration and a whole-tree kill. The caller has asked for sandbox
    consent. ``cwd`` defaults to the caller's working directory, so relative
    paths the script was given still resolve."""
    argv = [sys.executable, str(script_path), *[str(a) for a in (args or [])]]
    return _run_tracked(argv, timeout=timeout, cwd=cwd, env=sandbox_env(extra_env),
                        preexec=_sandbox_preexec())


def run_engine(cmd, *, timeout=None, cwd=None, env=None, shell=False, input=None, text=None,
               capture_output=False, check=False, stdin=None, stdout=None, stderr=None,
               encoding=None, errors=None):
    """Run an external engine (LAMMPS, AMBER tools, packmol, a training run)
    so the user's Stop reaches it and a timeout ends its whole tree (#685).
    A drop-in for ``subprocess.run`` at these call sites: the same keywords
    (``capture_output``, ``check``, ``stdin``, ``stdout``, ``stderr``,
    ``text``, ``input``), the same ``CompletedProcess`` and the same
    ``TimeoutExpired`` / ``CalledProcessError``. Engines keep the parent
    environment (licences, PATH); the sandbox allowlist is for generated code.
    An unknown keyword is a ``TypeError``, as in ``subprocess.run``, never
    silently dropped (``encoding`` and ``errors`` pass through)."""
    if capture_output:
        stdout = stderr = subprocess.PIPE
    proc = _run_tracked(cmd, timeout=timeout, cwd=cwd, env=env, shell=shell, input=input, text=bool(text),
                        stdout=stdout, stderr=stderr, stdin=stdin, encoding=encoding, errors=errors)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


class ScriptExecutor:
    """
    Executes Python scripts for scientific analysis.
    
    NOTE: Sandbox enforcement is handled at the agent level via 
    `require_sandbox_approval()`. This executor assumes the caller 
    has already verified it's safe to execute code.
    """
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, mp_api_key: str = None):
        self.timeout = timeout
        self.mp_api_key = mp_api_key or get_api_key('materials_project') or os.getenv("MP_API_KEY")
        
        logging.info(f"ScriptExecutor initialized (timeout: {self.timeout}s)")

    def execute_script(self, script_content: str, working_dir: str = None,
                       timeout: int | None = None) -> dict:
        """Execute a Python script.

        Thread-safe: the subprocess's CWD is set via ``Popen(cwd=...)`` and the
        caller's process CWD is never mutated, so concurrent calls from
        different threads do not race on a shared global.

        Args:
            script_content: Python source to execute.
            working_dir: Directory to use as the subprocess's CWD (and where
                the temp script file is written). Defaults to current CWD.
            timeout: Per-call timeout in seconds. When ``None`` (the default),
                uses ``self.timeout`` from construction. Callers that need an
                adaptive-timeout escalation pattern pass the override here
                without mutating shared executor state.
        """
        effective_timeout = self.timeout if timeout is None else timeout
        logging.info(f"   Executing Python script (timeout: {effective_timeout}s)...")

        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            script_dir = os.path.abspath(working_dir)
        else:
            script_dir = os.getcwd()

        temp_script_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=script_dir, encoding='utf-8') as tf:
                tf.write(script_content)
                temp_script_file = tf.name

            env = sandbox_env({"MP_API_KEY": self.mp_api_key} if self.mp_api_key else None)

            proc = subprocess.Popen(
                [sys.executable, os.path.basename(temp_script_file)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, cwd=script_dir,
                preexec_fn=_sandbox_preexec(),
                start_new_session=_NEW_SESSION,
            )
            _mark_own_group(proc)
            # Register so OutputCapture.kill_subprocesses() can terminate it.
            _register_subprocess(proc)
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                _end_leftover_group(proc)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc, grace=0.5)
                return {"status": "error", "message": f"Script execution timed out after {effective_timeout} seconds."}
            except BaseException:
                # Interrupted while waiting (Ctrl-C, a stop): the script is in
                # its own session and would otherwise outlive us.
                _kill_process_tree(proc, grace=0.5)
                raise
            finally:
                _unregister_subprocess(proc)

            logging.debug(f"STDOUT:\n{stdout}")
            logging.debug(f"STDERR:\n{stderr}")

            if proc.returncode == 0:
                return {"status": "success", "stdout": stdout, "stderr": stderr}
            elif proc.returncode == -getattr(signal, 'SIGTERM', 15) or proc.returncode == -getattr(signal, 'SIGKILL', 9):
                return {"status": "error", "message": "Script execution was stopped by the user."}
            else:
                error_msg = f"Script execution failed with return code {proc.returncode}.\nSTDERR:\n{stderr}"
                return {"status": "error", "message": error_msg}

        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}
        finally:
            if temp_script_file and os.path.exists(temp_script_file):
                try:
                    os.remove(temp_script_file)
                except OSError:
                    pass


class WarmScriptExecutor(ScriptExecutor):
    """Runs scripts in ONE long-lived interpreter instead of a fresh one each time.

    For a live loop's fast path, where the same approved script answers every
    frame: a fresh process pays the script's imports on every frame (measured on
    a real atomic-resolution recipe: 5 of 8 seconds were importing torch). Here
    they are paid once. What is kept from the cold executor:

    - each run has its own working directory, and the caller's is never touched;
    - the timeout is hard: the worker is killed and replaced, as a fresh process
      would be;
    - a chat turn's Stop reaches it (the worker is registered like any
      subprocess); the Live tab's Stop only sets its loop's event, so a replay
      already running there finishes or times out;
    - stdout / stderr are captured at the file-descriptor level, so the result
      has the same shape and the same success rule (exit code 0).

    What is NOT kept is a clean interpreter per run: module state survives from
    one run to the next (that is the point: imports, loaded model weights). So
    this is for replaying a script that has already been verified, not for
    trying generated code, and the worker is retired after ``max_runs`` runs.
    Any failure of the worker itself falls back to a cold run of the same script.
    """

    def __init__(self, timeout: int = DEFAULT_TIMEOUT, mp_api_key: str = None, max_runs: int = 500):
        super().__init__(timeout=timeout, mp_api_key=mp_api_key)
        self.max_runs = int(max_runs)
        self._proc: subprocess.Popen | None = None
        self._runs = 0
        self._lock = threading.Lock()
        self.cold_fallbacks = 0

    # -- worker lifecycle -------------------------------------------------
    def _start(self) -> bool:
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_warm_worker.py")
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                env=sandbox_env({"MP_API_KEY": self.mp_api_key} if self.mp_api_key else None),
                preexec_fn=_sandbox_preexec(), start_new_session=_NEW_SESSION)
            _mark_own_group(self._proc)
            ready = self._read_line(30.0)
            self._runs = 0
            return bool(ready and json.loads(ready).get("ready"))
        except Exception:  # noqa: BLE001 - no worker: the cold path still works
            self.close()
            return False

    def _read_line(self, timeout: float):
        """One line from the worker, or ``None`` on timeout / a dead worker."""
        import queue
        box: "queue.Queue" = queue.Queue(maxsize=1)
        proc = self._proc

        def reader():
            try:
                box.put(proc.stdout.readline())
            except Exception:  # noqa: BLE001
                box.put("")
        threading.Thread(target=reader, daemon=True).start()
        try:
            line = box.get(timeout=timeout)
        except queue.Empty:
            return None
        return line or None

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    _kill_process_tree(proc, grace=0.5)
            _end_leftover_group(proc)        # anything a replayed script left; the Windows job
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):  # noqa: D401 - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- execution --------------------------------------------------------
    def execute_script(self, script_content: str, working_dir: str = None,
                       timeout: int | None = None) -> dict:
        effective_timeout = self.timeout if timeout is None else timeout
        with self._lock:
            if self._proc is None or self._proc.poll() is not None or self._runs >= self.max_runs:
                self.close()
                if not self._start():
                    self.cold_fallbacks += 1
                    return super().execute_script(script_content, working_dir, timeout)
            script_dir = os.path.abspath(working_dir) if working_dir else os.getcwd()
            os.makedirs(script_dir, exist_ok=True)
            temp_script_file = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=script_dir,
                                                 encoding="utf-8") as tf:
                    tf.write(script_content)
                    temp_script_file = tf.name
                env = {"MP_API_KEY": self.mp_api_key} if self.mp_api_key else {}
                _register_subprocess(self._proc)
                try:
                    self._proc.stdin.write(json.dumps(
                        {"script": temp_script_file, "cwd": script_dir, "env": env}) + "\n")
                    self._proc.stdin.flush()
                    line = self._read_line(float(effective_timeout))
                finally:
                    _unregister_subprocess(self._proc)
                self._runs += 1
                if line is None:
                    dead = self._proc.poll() is not None
                    _kill_process_tree(self._proc, grace=0.5)
                    self.close()
                    if dead:                      # the worker died (a crash in native code, a Stop)
                        return {"status": "error", "message": "Script execution was stopped or the "
                                                              "interpreter running it died."}
                    return {"status": "error",
                            "message": f"Script execution timed out after {effective_timeout} seconds."}
                reply = json.loads(line)
                if reply.get("returncode") == 0:
                    return {"status": "success", "stdout": reply.get("stdout", ""),
                            "stderr": reply.get("stderr", "")}
                return {"status": "error",
                        "message": (f"Script execution failed with return code {reply.get('returncode')}."
                                    f"\nSTDERR:\n{reply.get('stderr', '')}")}
            except Exception as e:  # noqa: BLE001 - the worker failed, not the script: run it cold
                self.close()
                self.cold_fallbacks += 1
                logging.warning(f"warm executor failed ({e}); running this script in a fresh process")
                return super().execute_script(script_content, working_dir, timeout)
            finally:
                if temp_script_file and os.path.exists(temp_script_file):
                    try:
                        os.remove(temp_script_file)
                    except OSError:
                        pass


class SandboxTimeout(TimeoutError):
    """The sandbox's own limit fired (``ExecutionTimeout``), on either path —
    the SIGALRM handler on the main thread, the asynchronous injection into
    a worker thread. A ``TimeoutError`` a script raises itself is not one.
    The injected instance carries no message (``PyThreadState_SetAsyncExc``
    takes a class), so callers test the TYPE, never the text."""

    def __str__(self) -> str:
        return super().__str__() or "Code execution timed out (the sandbox's limit)."


class ExecutionTimeout:
    """Context manager that raises SandboxTimeout if exec() exceeds a time limit.

    Strategy:
    1. SIGALRM — used when running in the main thread on Unix (fast, reliable).
    2. ``ctypes.pythonapi.PyThreadState_SetAsyncExc`` — fallback for background
       threads (e.g., Streamlit UI).  A watchdog timer thread injects a
       ``TimeoutError`` into the target thread after *seconds* elapse.
    """

    def __init__(self, seconds: int = 300):
        self.seconds = seconds
        self._old_handler = None
        self._watchdog: threading.Timer | None = None
        self._target_tid: int | None = None
        self.fired = False                  # set when the limit went off, on either path

    def _handler(self, signum, frame):
        self.fired = True
        raise SandboxTimeout(
            f"Code execution timed out after {self.seconds}s. "
            "Consider vectorized operations or reducing iteration count."
        )

    def _can_use_sigalrm(self):
        return (
            hasattr(signal, 'SIGALRM')
            and threading.current_thread() is threading.main_thread()
        )

    def _inject_timeout(self):
        """Raise SandboxTimeout asynchronously in the target thread."""
        import ctypes
        tid = self._target_tid
        if tid is None:
            return
        self.fired = True
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(SandboxTimeout)
        )
        if ret == 0:
            logging.warning("ExecutionTimeout: target thread no longer exists")
        elif ret > 1:
            # Undo — more than one thread affected (should not happen)
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), None
            )

    def __enter__(self):
        if self._can_use_sigalrm():
            self._old_handler = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        else:
            self._target_tid = threading.get_ident()
            self._watchdog = threading.Timer(self.seconds, self._inject_timeout)
            self._watchdog.daemon = True
            self._watchdog.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._can_use_sigalrm():
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
        elif self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
            self._target_tid = None
        return False
