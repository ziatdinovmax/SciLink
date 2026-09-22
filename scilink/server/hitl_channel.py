"""hitl.FeedbackChannel that parks a question for a front-end to answer.

Modeled on the Streamlit ``_UIChannel`` (scilink/ui/app.py:515-560) and the
MCP server's ``_MCPChannel`` job-parking pattern: ``ask`` presents the
question (``presenter.present_question``), parks it on the turn and blocks
the agent thread on a ``threading.Event`` until the front-end answers (or
stops the turn). Carries a metadata auto-reply cache so a repeated
dataset-description prompt for the same file answers itself.

``ParkingChannel`` is the surface-neutral part; ``HTTPChannel`` adds the
SSE events the React frontend listens for, and the terminal shell's
channel adds nothing (its main loop watches ``turn.pending_question``).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from scilink.ui.output_capture import AgentStoppedError

from .presenter import present_question


@dataclass
class PendingQuestion:
    hreq: Any                       # scilink.hitl.FeedbackRequest
    presented: Dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None


class ParkingChannel:
    """One per turn. ``turn`` carries ``stopped`` and ``pending_question``;
    ``cap`` is the turn's capture (its buffer is the question's context);
    ``session_dir`` locates preview images and code files."""

    def __init__(self, turn, cap, session_dir: str) -> None:
        self._turn = turn
        self._cap = cap
        self._session_dir = session_dir
        self._metadata_cache: Dict[str, str] = {}

    def _metadata_key(self, hreq, context: str) -> Optional[str]:
        # Port of _UIChannel._metadata_key (app.py:529-537).
        if hreq.kind == "dataset_description" and hreq.origin.get("filename"):
            return str(hreq.origin["filename"])
        if "Context" in hreq.prompt and "MISSING METADATA" in context:
            m = re.search(r"MISSING METADATA FOR:\s*(.+)", context)
            if m:
                return m.group(1).strip()
        return None

    # Hooks for a surface that announces the question somewhere.
    def _published(self, pending: PendingQuestion) -> None:
        pass

    def _cleared(self, pending: PendingQuestion) -> None:
        pass

    def ask(self, hreq) -> str:
        turn = self._turn
        if turn.stopped:
            raise AgentStoppedError("Agent stopped by user")
        context = self._cap.getvalue()
        key = self._metadata_key(hreq, context)
        if key is not None and key in self._metadata_cache:
            return self._metadata_cache[key]

        presented = present_question(hreq, context, self._session_dir)
        pending = PendingQuestion(hreq=hreq, presented=presented)
        turn.pending_question = pending
        self._published(pending)

        pending.event.wait()

        turn.pending_question = None
        self._cleared(pending)
        if turn.stopped:
            raise AgentStoppedError("Agent stopped by user")
        response = pending.response or ""
        if key is not None:
            self._metadata_cache[key] = response
        return response


class HTTPChannel(ParkingChannel):
    """The web backend's channel: the parked question and the status
    transitions travel to the browser as SSE events."""

    def __init__(self, turn, cap, session) -> None:
        super().__init__(turn, cap, session.session_dir)
        self._session = session

    def _published(self, pending: PendingQuestion) -> None:
        self._session.events.emit("question", pending.presented)
        self._session.events.emit("status", {"status": "awaiting_input"})

    def _cleared(self, pending: PendingQuestion) -> None:
        self._session.events.emit("question_cleared",
                                  {"request_id": pending.hreq.id})
        if not self._turn.stopped:
            self._session.events.emit("status", {"status": "running"})
