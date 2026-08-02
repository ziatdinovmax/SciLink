"""Capture stdout/stderr while still printing to the original streams."""

import io
import sys
import threading


class AgentStoppedError(Exception):
    """Raised inside the agent thread when the user clicks Stop."""


class TeeStream:
    """A stream that writes to both the original stream and a StringIO buffer."""

    def __init__(self, original: io.TextIOBase, buffer: io.StringIO,
                 stop_event: threading.Event | None = None):
        self._original = original
        self._buffer = buffer
        self._lock = threading.Lock()
        self._stop_event = stop_event

    def write(self, data: str) -> int:
        if self._stop_event is not None and self._stop_event.is_set():
            raise AgentStoppedError("Agent stopped by user")
        with self._lock:
            self._original.write(data)
            self._buffer.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()

    # Delegate attribute access so libraries checking .isatty() etc. still work.
    def __getattr__(self, name: str):
        return getattr(self._original, name)


class OutputCapture:
    """Context manager that captures stdout/stderr through a TeeStream.

    Usage::

        with OutputCapture() as cap:
            agent.chat("hello")
        print(cap.getvalue())

    Call ``cap.request_stop()`` from another thread to abort the agent
    on its next ``print()`` call.  This also terminates any Python
    subprocesses spawned by ``ScriptExecutor`` in the agent thread.
    """

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._old_stdout = None
        self._old_stderr = None
        self._stop_event = threading.Event()
        self._agent_thread_id: int | None = None

    def request_stop(self) -> None:
        """Signal the agent thread to abort on the next print() call.

        Also kills any active subprocesses spawned by ScriptExecutor
        in the agent thread so that long-running scripts are terminated
        immediately.
        """
        self._stop_event.set()
        # Kill any subprocess the agent thread is waiting on.
        if self._agent_thread_id is not None:
            try:
                from scilink.executors import kill_subprocesses_for_thread
                kill_subprocesses_for_thread(self._agent_thread_id)
            except Exception:
                pass  # Best-effort; stop event will still propagate via print()

    def __enter__(self) -> "OutputCapture":
        self._agent_thread_id = threading.get_ident()
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = TeeStream(self._old_stdout, self._buffer, self._stop_event)
        sys.stderr = TeeStream(self._old_stderr, self._buffer, self._stop_event)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def getvalue(self) -> str:
        return self._buffer.getvalue()
