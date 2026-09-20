"""One chat turn on the terminal — the web runner's ``_run_turn``, rendered
live instead of streamed.

The agent runs on a worker thread inside a ``RoutedCapture`` that does not
echo to the console (the shell renders the buffer itself). The main thread
polls the buffer, feeds the renderer, answers parked questions through the
widgets, and turns Ctrl+C into the same stop the web's ■ button sends: the
capture's stop event, which lands on the agent's next print as
``AgentStoppedError``, plus a kill of its subprocesses.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import logging
import os
import select
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from scilink import hitl as _hitl
from scilink import tracing
from scilink.server.hitl_channel import PendingQuestion
from scilink.server.runner import quiet_third_party_loggers, turn_log_handler
from scilink.server.stdout_router import RoutedCapture
from scilink.ui.narration import current_activity
from scilink.ui.output_capture import AgentStoppedError

from .channel import ShellChannel

_POLL_S = 0.1
_STOP_GRACE_S = 3.0
_CTRL_O = "\x0f"


class KeyWatcher:
    """Reads single keys from the terminal while a turn runs (cbreak mode:
    no line buffering, no echo, Ctrl+C still a signal). Other keys are
    discarded — the prompt is not active during a turn."""

    def __init__(self) -> None:
        self._fd = None
        self._saved = None

    def __enter__(self):
        try:
            import termios
            import tty
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                self._cbreak()
        except Exception:  # noqa: BLE001 - no raw keys on this terminal
            self._fd = None
        return self

    def __exit__(self, *exc):
        self.restore()

    def restore(self) -> None:
        if self._fd is not None and self._saved is not None:
            import termios
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001
                pass

    def _cbreak(self) -> None:
        """cbreak, minus IEXTEN: with it on, the line discipline eats ^O as
        its 'discard output' key (and toggles output flushing) before we
        can read it — prompt_toolkit's raw mode clears the flag the same way."""
        import termios
        import tty
        tty.setcbreak(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[3] &= ~termios.IEXTEN
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)

    def reenter(self) -> None:
        if self._fd is not None:
            try:
                self._cbreak()
            except Exception:  # noqa: BLE001
                pass

    def read_key(self):
        """The next pending key, or None (never blocks)."""
        if self._fd is None:
            return None
        try:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return None
            return os.read(self._fd, 1).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            return None


@contextlib.contextmanager
def quiet_console_logging():
    """Silence the agents' own console log handlers (basicConfig on the
    real stderr, which the router does not see) so the terminal shows only
    what the shell renders; the turn handler still captures every record."""
    root = logging.getLogger()
    muted = [h for h in root.handlers
             if isinstance(h, logging.StreamHandler) and h.level < logging.ERROR]
    saved = [(h, h.level) for h in muted]
    for h in muted:
        h.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for h, level in saved:
            h.setLevel(level)


@dataclass
class TurnState:
    """What the channel and the loop share (the web ``TurnState``'s subset)."""
    stopped: bool = False
    interrupted: bool = False   # set by the SIGINT handler; the loop acts on it
    pending_question: Optional[PendingQuestion] = None
    result: Optional[str] = None
    error: Optional[str] = None
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class TurnResult:
    result: Optional[str]
    error: Optional[str]
    stopped: bool
    log: str
    elapsed_s: float
    tokens: dict


def run_turn(agent: Any, user_input: str, *, session_dir: str, renderer,
             ask_question: Callable[[dict], str],
             checkpoint: bool = True,
             on_tick: Optional[Callable[[], None]] = None,
             read_key: Optional[Callable[[], Optional[str]]] = None) -> TurnResult:
    """Run ``agent.chat(user_input)`` and render it; returns when the turn
    ends (answer, error or stop).

    ``ask_question(presented)`` answers a parked question (the widgets);
    raising ``KeyboardInterrupt`` from it stops the turn. ``on_tick`` runs
    on every poll — tests use it to inject an interrupt. ``read_key``
    supplies pending keys (Ctrl+O toggles verbose mid-turn); by default a
    ``KeyWatcher`` on the terminal.
    """
    turn = TurnState()
    cap = RoutedCapture(echo_console=False)
    channel = ShellChannel(turn, cap, session_dir)
    before = tracing.llm_counters()
    t0 = time.monotonic()

    def _shell_input(prompt: str = "") -> str:
        # Raw input() from third-party code or the sandbox consent prompt
        # on the agent thread: park it like any other question.
        return channel.ask(_hitl.FeedbackRequest(prompt=prompt))

    quiet = quiet_console_logging()

    def worker() -> None:
        original_input = builtins.input
        root = logging.getLogger()
        handler = turn_log_handler(cap, threading.get_ident())
        quiet_third_party_loggers()
        # Mute the agents' console handlers BEFORE adding the turn's own
        # capture handler, which is also a stream handler and must stay at
        # INFO: the codegen / verification milestones are log records.
        quiet.__enter__()
        root.addHandler(handler)
        try:
            builtins.input = _shell_input
            _hitl.set_thread_channel(channel)
            with cap:
                result = agent.chat(user_input)
            if not turn.stopped:
                turn.result = result if result is not None else ""
        except AgentStoppedError:
            pass
        except Exception as exc:  # noqa: BLE001 - reported, never raised past the turn
            if not turn.stopped:
                turn.error = f"{type(exc).__name__}: {exc}"
        finally:
            builtins.input = original_input
            _hitl.set_thread_channel(None)
            root.removeHandler(handler)
            quiet.__exit__(None, None, None)
            turn.done.set()

    thread = threading.Thread(target=worker, daemon=True, name="scilink-turn")
    thread.start()
    renderer.begin_turn()
    sent = 0

    def _stop() -> None:
        turn.stopped = True
        cap.request_stop()
        pending = turn.pending_question
        if pending is not None and not pending.event.is_set():
            pending.response = ""
            pending.event.set()
        turn.done.wait(_STOP_GRACE_S)

    # Ctrl+C: a handler that only sets a flag, read by the poll loop. A
    # KeyboardInterrupt raised inside rich's or threading's internals can
    # land at an awkward point (observed live: an interrupt lost while a
    # long analysis ran); the flag cannot be swallowed. The exception path
    # below stays as the fallback (not on the main thread, or a prompt).
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        def _on_sigint(signum, frame):
            turn.interrupted = True
        try:
            previous_handler = signal.signal(signal.SIGINT, _on_sigint)
        except (ValueError, OSError):
            previous_handler = None
    watcher = KeyWatcher() if read_key is None else None
    if watcher is not None:
        watcher.__enter__()
        read_key = watcher.read_key
    try:
        while True:
            finished = turn.done.wait(_POLL_S)
            if on_tick is not None:
                on_tick()
            if turn.interrupted:
                _stop()
                break
            key = read_key()
            while key is not None:
                if key == _CTRL_O:
                    renderer.set_verbose(not renderer.verbose)
                key = read_key()
            buf = cap.getvalue()
            if len(buf) > sent:
                renderer.feed(buf[sent:])
                sent = len(buf)
                renderer.set_activity(current_activity(buf[-6000:]))
            pending = turn.pending_question
            if pending is not None and not pending.event.is_set():
                renderer.pause()
                if watcher is not None:
                    watcher.restore()      # the prompt owns the terminal now
                try:
                    # prompt_toolkit owns SIGINT while the prompt runs and
                    # raises KeyboardInterrupt; our handler is restored after.
                    answer = ask_question(pending.presented)
                except (KeyboardInterrupt, EOFError):
                    _stop()
                    break
                finally:
                    renderer.resume()
                    if watcher is not None:
                        watcher.reenter()
                pending.response = answer
                pending.event.set()
                continue
            if finished:
                break
    except KeyboardInterrupt:
        _stop()
    finally:
        if watcher is not None:
            watcher.restore()
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except (ValueError, OSError):
                pass

    log = cap.getvalue()
    if len(log) > sent and not turn.stopped:
        renderer.feed(log[sent:])
    tokens = tracing.counters_delta(before)
    renderer.end_turn(result=turn.result, error=turn.error, stopped=turn.stopped,
                      stop_message=getattr(renderer, "stop_message", "Stopped by user."),
                      tokens=tokens)

    if checkpoint and not turn.stopped and (turn.result is not None or turn.error is not None):
        save_checkpoint_quietly(agent)

    return TurnResult(result=turn.result, error=turn.error, stopped=turn.stopped,
                      log=log, elapsed_s=time.monotonic() - t0, tokens=tokens)


def save_checkpoint_quietly(agent: Any) -> Optional[str]:
    """Per-turn checkpoint, like the web runner — the orchestrators print
    while saving, so their output is swallowed here."""
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            if hasattr(agent, "save_checkpoint"):
                return agent.save_checkpoint()
            if hasattr(agent, "_auto_checkpoint"):
                agent._auto_checkpoint()
    except Exception as exc:  # noqa: BLE001 - never break the turn
        logging.warning(f"Per-turn checkpoint failed: {exc}")
    return None
