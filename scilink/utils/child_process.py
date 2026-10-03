"""Run SciLink work in another process without re-running the caller's script.

``multiprocessing``'s spawn (and forkserver) start method re-imports the
caller's main script in every worker before the worker does anything else. A
script without an ``if __name__ == "__main__":`` guard then runs another copy
of itself in each worker — a new agent, model calls, writes into the same
session folder (#721) — and Python's own check fires only when that copy
tries to start a process of its own, which can be minutes of model calls
later. Two rules follow:

- SciLink never starts its own workers that way. ``run_in_child`` launches a
  fresh interpreter on this module, which imports the target function's
  module and nothing of the caller's; the payload goes over stdin and the
  result comes back through a private temporary file, so neither a credential
  in the payload nor the child's prints touch either channel.
- Every SciLink entry point refuses to start inside a spawn bootstrap
  (``refuse_in_spawn_bootstrap``), which covers the pools SciLink does not
  own: an unguarded script that starts a spawn pool of its own stops at the
  first agent, orchestrator or model it builds in a worker, before any model
  call or file write.
"""
from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import Any

_MODULE = "scilink.utils.child_process"


class SpawnBootstrapError(RuntimeError):
    """SciLink was started while a worker process was re-importing a script."""


def refuse_in_spawn_bootstrap(what: str = "SciLink") -> None:
    """Raise when called while a spawned/forkserver worker is re-importing
    the main script. ``_inheriting`` is the flag Python's own bootstrapping
    check reads: set for the whole re-import, absent in a normal run and in
    a worker once its bootstrap is done (a pool's task may run SciLink)."""
    proc = multiprocessing.current_process()
    if getattr(proc, "_inheriting", False):
        raise SpawnBootstrapError(
            f"{what} was started while a worker process ({proc.name}) was "
            "re-importing your script: the code that runs SciLink executes "
            "at import time, so every worker process would run another copy "
            "of it. Put that code under `if __name__ == \"__main__\":`.")


class ChildLost(RuntimeError):
    """The child process ended without returning a result (killed, crashed,
    or it could not start the target) — a failure of the process, not of
    the work it was given."""

    def __init__(self, target: str, detail: str):
        super().__init__(f"{target}: {detail}")
        self.target = target
        self.detail = detail


def _exit_reason(returncode: int) -> str:
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        hint = (" — commonly the operating system's out-of-memory killer"
                if -returncode == getattr(signal, "SIGKILL", 9) else "")
        return f"killed by {name}{hint}"
    return f"exited with code {returncode} without returning a result"


def run_in_child(target: str, payload: Any) -> Any:
    """Call ``target`` (``"package.module:function"``) with ``payload`` in a
    fresh interpreter and return what it returns. The child sees the
    parent's ``sys.path`` and environment and inherits its working directory
    and console. Raises ``ChildLost`` when no result comes back — including
    when the target itself raised, whose traceback is in the detail."""
    fd, result_path = tempfile.mkstemp(prefix="scilink_child_", suffix=".pkl")
    os.close(fd)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(os.path.abspath(p) for p in sys.path if p)
    try:
        proc = subprocess.Popen([sys.executable, "-m", _MODULE, target, result_path],
                                stdin=subprocess.PIPE, env=env)
        try:
            proc.stdin.write(pickle.dumps(payload))
            proc.stdin.close()
        except BrokenPipeError:
            pass                        # the child died first; its exit says why
        returncode = proc.wait()
        try:
            with open(result_path, "rb") as f:
                outcome, value = pickle.load(f)
        except Exception:  # noqa: BLE001 - empty or torn: nothing came back
            raise ChildLost(target, _exit_reason(returncode)) from None
        if outcome != "returned":
            raise ChildLost(target, f"the call raised in the child:\n{value}")
        return value
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def _child_main(target: str, result_path: str) -> int:
    try:
        payload = pickle.load(sys.stdin.buffer)
        module, _, name = target.partition(":")
        outcome = ("returned", getattr(importlib.import_module(module), name)(payload))
        blob = pickle.dumps(outcome)
    except BaseException:  # noqa: BLE001 - report everything to the parent
        blob = pickle.dumps(("raised", traceback.format_exc()))
    with open(result_path, "wb") as f:
        f.write(blob)
    return 0


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv[1], sys.argv[2]))
