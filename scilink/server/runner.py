"""One chat turn in a background thread — port of the Streamlit turn runner
(``_run_agent_chat``, scilink/ui/app.py:563-646) plus the completion handling
the Streamlit monitor fragment does (app.py:986-1050), which here runs at the
end of the same thread since no UI rerun is needed.

Additions over the Streamlit original:
- a log-watcher thread that diffs the OutputCapture buffer (~2×/s) and emits
  incremental ``log`` SSE events, replacing the 1s full-buffer poll;
- explicit ``status`` / ``assistant_message`` events at the transitions the
  Streamlit fragment inferred from ChatTask fields.

The stop path is identical: print/log-driven ``AgentStoppedError`` (a
BaseException) + subprocess kill; no cancellation token.
"""

from __future__ import annotations

import builtins
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from scilink import hitl as _hitl
from scilink.ui.output_capture import AgentStoppedError, OutputCapture

from .hitl_channel import HTTPChannel, PendingQuestion

_LOG_POLL_S = 0.5


@dataclass
class TurnState:
    """Server-side twin of the Streamlit ChatTask (scilink/ui/state.py:28)."""

    thread: Optional[threading.Thread] = None
    is_running: bool = False
    stopped: bool = False
    result: Optional[str] = None
    error: Optional[str] = None
    verbose_log: str = ""
    live_capture: Optional[OutputCapture] = None
    pending_question: Optional[PendingQuestion] = None
    done: threading.Event = field(default_factory=threading.Event)


def start_turn(session, user_input: str) -> TurnState:
    """Create a TurnState and launch the agent thread (daemon, like Streamlit)."""
    turn = TurnState(is_running=True)
    session.turn = turn
    session.chat_messages.append({"role": "user", "content": user_input})
    session.events.emit("status", {"status": "running"})
    t = threading.Thread(target=_run_turn, args=(session, turn, user_input),
                         daemon=True)
    turn.thread = t
    t.start()
    return turn


def request_stop(session) -> bool:
    """Port of the Streamlit Stop handler (app.py:1350-1379)."""
    turn = session.turn
    if turn is None or not turn.is_running:
        return False
    turn.stopped = True
    turn.is_running = False
    if turn.live_capture:
        turn.live_capture.request_stop()
        turn.verbose_log = turn.live_capture.getvalue()
    turn.live_capture = None
    pending = turn.pending_question
    if pending is not None:
        pending.response = ""
        pending.event.set()
        turn.pending_question = None
    stop_label = ("Planning stopped by user." if session.mode == "plan"
                  else "Analysis stopped by user.")
    message = {"role": "assistant", "content": stop_label,
               "verbose": turn.verbose_log}
    session.chat_messages.append(message)
    session.events.emit("assistant_message", message)
    session.events.emit("status", {"status": "idle"})
    return True


def _watch_log(session, turn, cap) -> None:
    """Emit incremental ``log`` events while the turn runs."""
    sent = 0
    while turn.is_running and turn.live_capture is cap:
        try:
            buf = cap.getvalue()
        except Exception:
            break
        if len(buf) > sent:
            session.events.emit("log", {"chunk": buf[sent:]})
            sent = len(buf)
        turn.done.wait(_LOG_POLL_S)
    try:
        buf = cap.getvalue()
        if len(buf) > sent:
            session.events.emit("log", {"chunk": buf[sent:]})
    except Exception:
        pass


def _run_turn(session, turn: TurnState, user_input: str) -> None:
    agent = session.agent
    original_input = builtins.input

    cap = OutputCapture()
    turn.live_capture = cap
    channel = HTTPChannel(turn, cap, session)

    def _http_input(prompt: str = "") -> str:
        # Raw-input fallback for third-party code / agent worker threads that
        # bypass the thread-local channel (app.py:576-580).
        return channel.ask(_hitl.FeedbackRequest(prompt=prompt))

    log_handler = logging.StreamHandler(cap._buffer)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    _this_thread = threading.get_ident()
    from scilink.utils.log_context import effective_thread, keep_in_fanout_panel

    def _panel_filter(record):
        if effective_thread(record.thread) != _this_thread:
            return False
        if cap.stop_requested:
            raise AgentStoppedError("Agent stopped by user")
        return keep_in_fanout_panel(
            record.thread, record.getMessage(), record.levelno)

    log_handler.addFilter(_panel_filter)
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    for _lib in ("urllib3", "httpx", "httpcore", "google", "openai",
                 "anthropic", "matplotlib", "PIL", "fsspec", "asyncio",
                 "grpc", "absl", "uvicorn"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    root_logger.addHandler(log_handler)

    watcher = threading.Thread(target=_watch_log, args=(session, turn, cap),
                               daemon=True)
    watcher.start()

    try:
        builtins.input = _http_input
        _hitl.set_thread_channel(channel)
        with cap:
            result = agent.chat(user_input)
        if not turn.stopped:
            turn.result = result
            turn.verbose_log = cap.getvalue()
    except AgentStoppedError:
        pass  # Expected — user clicked Stop; thread exits cleanly.
    except Exception as exc:
        if not turn.stopped:
            turn.error = str(exc)
            turn.verbose_log = cap.getvalue()
    finally:
        builtins.input = original_input
        _hitl.set_thread_channel(None)
        root_logger.removeHandler(log_handler)
        turn.live_capture = None
        if not turn.stopped:
            turn.is_running = False
        turn.done.set()

    if not turn.stopped and (turn.result is not None or turn.error is not None):
        _complete_turn(session, turn)


def _complete_turn(session, turn: TurnState) -> None:
    """Port of the Streamlit completion branch (app.py:986-1050)."""
    content = turn.result if turn.result is not None else f"Error: {turn.error}"
    # Strip markdown image tags with local file paths — images are attached
    # separately from the artifact sweep.
    content = re.sub(r"!\[[^\]]*\]\([^)]+\)\n?", "", content).strip()
    artifacts = session.tracker.sweep_turn(session.autonomy)
    message = {
        "role": "assistant",
        "content": content,
        "images": artifacts["images"],
        "html_reports": artifacts["html_reports"],
        "md_reports": artifacts["md_reports"],
        "verbose": turn.verbose_log or "",
    }
    session.chat_messages.append(message)
    session.events.emit("assistant_message", message)
    if turn.error is not None:
        session.events.emit("error", {"message": turn.error})

    # Per-turn checkpoint (short sessions were otherwise unresumable).
    try:
        agent = session.agent
        if hasattr(agent, "save_checkpoint"):
            agent.save_checkpoint()
        elif hasattr(agent, "_auto_checkpoint"):
            agent._auto_checkpoint()
    except Exception as exc:  # noqa: BLE001 - never break the turn
        logging.warning(f"Per-turn checkpoint failed: {exc}")

    # One-shot LLM auto-titling from the first exchange; a user-set name is
    # never overwritten and failures fall back silently to the timestamp.
    try:
        from scilink.ui.session_meta import (
            generate_session_title, load_session_name, save_session_name)
        if session.session_dir and not load_session_name(session.session_dir):
            first_user = next(
                (m["content"] for m in session.chat_messages
                 if m["role"] == "user"), "")
            title = generate_session_title(
                getattr(session.agent, "model", None), first_user, content)
            if title and save_session_name(session.session_dir, title,
                                           named_by="agent"):
                session.events.emit("session_named", {"name": title})
    except Exception:  # noqa: BLE001 - naming must never break a turn
        pass

    session.events.emit("status", {"status": "idle"})
