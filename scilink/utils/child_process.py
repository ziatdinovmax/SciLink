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

The child is a process SciLink must be able to end (stage 4 of the swarm
work): it runs through the executor's tracked runner, so it leads its own
session, is registered to the waiting thread for a Stop and a cancel, and
its whole tree is killed with it. A caller that needs the child's memory —
the swarm's measured peak per item class — uses ``run_child``, which also
reports what the child's tree held while it ran.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

_MODULE = "scilink.utils.child_process"
#: How often a child's tree is sampled for its resident memory.
SAMPLE_INTERVAL_S = 0.5


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
    the work it was given. ``reason`` is one line (the exit, or the
    exception the target raised); ``detail`` adds the child's traceback;
    ``returncode`` is the process's exit status (negative: the signal that
    killed it), ``None`` when the target raised."""

    def __init__(self, target: str, reason: str, detail: str = "",
                 returncode: Optional[int] = None):
        super().__init__(f"{target}: {reason}")
        self.target = target
        self.reason = reason
        self.detail = detail or reason
        self.returncode = returncode

    @property
    def killed(self) -> bool:
        """Ended by SIGKILL with nothing returned: the memory guard, the
        operating system's out-of-memory killer, or a kill by hand."""
        return self.returncode is not None and -self.returncode == getattr(signal, "SIGKILL", 9)


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


@dataclass
class ChildOutcome:
    """What ``run_child`` returns: the target's value and what the child
    cost. ``peak_rss_bytes`` is the most its process tree held at one time
    while it ran (sampled), or its own ``ru_maxrss`` when sampling was not
    possible; ``None`` when neither could be measured."""
    value: Any
    returncode: int
    peak_rss_bytes: Optional[float] = None


class TreeSampler:
    """The resident memory of a process and its descendants, sampled while
    it runs (``psutil``; inert without it). The SUM over the tree at one
    instant is what the work holds on the machine — an agent's arrays beside
    its running script's — where ``ru_maxrss`` is one process's own peak.
    ``watch(current, peak)`` is called after every sample, for a guard that
    wants to know what a worker holds right now."""

    def __init__(self, watch: Optional[Callable[[float, float], None]] = None,
                 interval_s: float = SAMPLE_INTERVAL_S):
        self.peak: Optional[float] = None
        self.current: Optional[float] = None
        self._watch = watch
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._root = None

    def start(self, proc: subprocess.Popen) -> None:
        try:
            import psutil
            self._root = psutil.Process(proc.pid)
        except Exception:  # noqa: BLE001 - no psutil, or the child already gone
            return
        self._thread = threading.Thread(target=self._run, name="scilink-rss-sampler", daemon=True)
        self._thread.start()

    def _sample(self) -> Optional[float]:
        try:
            procs = [self._root] + self._root.children(recursive=True)
        except Exception:  # noqa: BLE001 - the tree is gone
            return None
        total = 0.0
        for p in procs:
            try:
                total += float(p.memory_info().rss)
            except Exception:  # noqa: BLE001 - a process ended between the two calls
                continue
        return total

    def _run(self) -> None:
        while True:
            rss = self._sample()
            if rss is not None:
                self.current = rss
                self.peak = rss if self.peak is None else max(self.peak, rss)
                if self._watch is not None:
                    try:
                        self._watch(rss, self.peak)
                    except Exception:  # noqa: BLE001 - a watcher never stops the sampler
                        pass
            if self._stop.wait(self._interval):
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def run_child(target: str, payload: Any, *,
              watch: Optional[Callable[[float, float], None]] = None) -> ChildOutcome:
    """Call ``target`` (``"package.module:function"``) with ``payload`` in a
    fresh interpreter and return what it returned, with what the child cost.

    The child sees the parent's ``sys.path`` — exactly, as spawn passes it:
    ``-P`` keeps ``-m`` from putting the working directory first, where a
    folder holding a ``scilink/`` (another checkout) or a ``signal.py`` would
    shadow the parent's — and its environment, working directory and
    console. The path travels as ``PYTHONPATH`` for the child's start only:
    the target runs with the caller's own ``PYTHONPATH``, so the scripts it
    executes see what a serial run's do.

    The child runs through the executor's tracked runner: it leads its own
    session, the calling thread's Stop or cancel kills its whole tree, and a
    cancel that lands while it runs raises the thread's stop
    (``AgentStoppedError``) once the child is gone. Raises ``ChildLost``
    when no result comes back — including when the target itself raised (the
    exception is the reason, the traceback the detail).
    """
    from scilink.executors import _run_tracked
    fd, result_path = tempfile.mkstemp(prefix="scilink_child_", suffix=".pkl")
    os.close(fd)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(os.path.abspath(p or os.curdir) for p in sys.path)
    caller_pythonpath = json.dumps(os.environ.get("PYTHONPATH"))
    sampler = TreeSampler(watch)
    try:
        try:
            proc = _run_tracked([sys.executable, "-P", "-m", _MODULE, target, result_path,
                                 caller_pythonpath],
                                input=pickle.dumps(payload), text=False, stdout=None, stderr=None,
                                env=env, on_start=sampler.start)
        finally:
            sampler.stop()
        returncode = proc.returncode
        try:
            with open(result_path, "rb") as f:
                envelope = pickle.load(f)
        except Exception:  # noqa: BLE001 - empty or torn: nothing came back
            raise ChildLost(target, _exit_reason(returncode), returncode=returncode) from None
        outcome, value, info = envelope
        if outcome != "returned":
            last = next((ln.strip() for ln in reversed(value.splitlines()) if ln.strip()),
                        "an exception")
            raise ChildLost(target, f"raised in the child: {last}",
                            f"raised in the child:\n{value}")
        peaks = [p for p in (sampler.peak, info.get("ru_maxrss_bytes")) if p]
        return ChildOutcome(value, returncode, max(peaks) if peaks else None)
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def run_in_child(target: str, payload: Any) -> Any:
    """``run_child`` for a caller that wants the target's value only."""
    return run_child(target, payload).value


def _ru_maxrss_bytes() -> Optional[float]:
    """The most this process, or any one of its waited children, held:
    bytes on macOS, kilobytes elsewhere."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    unit = 1.0 if sys.platform == "darwin" else 1024.0
    try:
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except Exception:  # noqa: BLE001
        return None
    return float(max(own, kids)) * unit


def _child_main(target: str, result_path: str, caller_pythonpath: str) -> int:
    pythonpath = json.loads(caller_pythonpath)
    if pythonpath is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = pythonpath
    try:
        payload = pickle.load(sys.stdin.buffer)
        module, _, name = target.partition(":")
        value = getattr(importlib.import_module(module), name)(payload)
        envelope = ("returned", value, {"ru_maxrss_bytes": _ru_maxrss_bytes()})
        blob = pickle.dumps(envelope)
    except BaseException:  # noqa: BLE001 - report everything to the parent
        blob = pickle.dumps(("raised", traceback.format_exc(), {}))
    with open(result_path, "wb") as f:
        f.write(blob)
    return 0


if __name__ == "__main__":
    sys.exit(_child_main(*sys.argv[1:4]))
