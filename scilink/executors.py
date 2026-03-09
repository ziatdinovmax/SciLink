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

DEFAULT_TIMEOUT = 300

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
    """Terminate all subprocesses registered by a given thread.

    Safe to call from any thread (e.g. the Streamlit UI thread).
    """
    with _active_subprocesses_lock:
        procs = list(_active_subprocesses.pop(tid, []))
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

    def execute_script(self, script_content: str, working_dir: str = None) -> dict:
        """Execute a Python script."""
        logging.info("Executing Python script...")
        
        original_cwd = os.getcwd()
        if working_dir:
            os.makedirs(working_dir, exist_ok=True)
            os.chdir(working_dir)

        temp_script_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=os.getcwd()) as tf:
                tf.write(script_content)
                temp_script_file = tf.name

            env = os.environ.copy()
            if self.mp_api_key:
                env['MP_API_KEY'] = self.mp_api_key

            proc = subprocess.Popen(
                ['python', os.path.basename(temp_script_file)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env,
            )
            # Register so OutputCapture.kill_subprocesses() can terminate it.
            _register_subprocess(proc)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return {"status": "error", "message": f"Script execution timed out after {self.timeout} seconds."}
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
            os.chdir(original_cwd)
            if temp_script_file and os.path.exists(temp_script_file):
                os.remove(temp_script_file)


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
