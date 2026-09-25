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

from .auth import get_api_key

DEFAULT_TIMEOUT = 600

# Global registry of active subprocesses, keyed by thread ID.
# Accessible from any thread so the UI stop handler can kill them.
_active_subprocesses_lock = threading.Lock()
_active_subprocesses: dict[int, set[subprocess.Popen]] = {}


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
        try:
            proc.terminate()          # SIGTERM first
            try:
                proc.wait(timeout=2)  # Give it a moment to clean up
            except subprocess.TimeoutExpired:
                proc.kill()           # SIGKILL if still alive
                proc.wait()
        except OSError:
            pass  # Already dead

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
    "FEFF_DIR", "FEFF_BIN",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "MKLROOT", "CC", "CXX", "FC", "UNSAFE_EXECUTION_OK",
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
            )
            # Register so OutputCapture.kill_subprocesses() can terminate it.
            _register_subprocess(proc)
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return {"status": "error", "message": f"Script execution timed out after {effective_timeout} seconds."}
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
    - the user's Stop reaches it (the worker is registered like any subprocess);
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
                preexec_fn=_sandbox_preexec())
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
                    proc.kill()
                    proc.wait(timeout=5)
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
                    self._proc.kill()
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


class ExecutionTimeout:
    """Context manager that raises TimeoutError if exec() exceeds a time limit.

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

    def _handler(self, signum, frame):
        raise TimeoutError(
            f"Code execution timed out after {self.seconds}s. "
            "Consider vectorized operations or reducing iteration count."
        )

    def _can_use_sigalrm(self):
        return (
            hasattr(signal, 'SIGALRM')
            and threading.current_thread() is threading.main_thread()
        )

    def _inject_timeout(self):
        """Raise TimeoutError asynchronously in the target thread."""
        import ctypes
        tid = self._target_tid
        if tid is None:
            return
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(TimeoutError)
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
