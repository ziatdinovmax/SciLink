"""hitl.FeedbackChannel that parks questions for the HTTP frontend.

Modeled on the Streamlit ``_UIChannel`` (scilink/ui/app.py:515-560) and the
MCP server's ``_MCPChannel`` job-parking pattern: ``ask`` publishes the
presented question as an SSE event and blocks the agent thread on a
``threading.Event`` until ``POST /sessions/{id}/feedback`` (or stop) fires
it. Carries the same metadata auto-reply cache so a repeated
dataset-description prompt for the same file answers itself.
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


class HTTPChannel:
    """One per turn; ``turn`` is the TurnState the runner owns."""

    def __init__(self, turn, cap, session) -> None:
        self._turn = turn
        self._cap = cap
        self._session = session
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

    def ask(self, hreq) -> str:
        turn = self._turn
        if turn.stopped:
            raise AgentStoppedError("Agent stopped by user")
        context = self._cap.getvalue()
        key = self._metadata_key(hreq, context)
        if key is not None and key in self._metadata_cache:
            return self._metadata_cache[key]

        presented = present_question(hreq, context, self._session.session_dir)
        pending = PendingQuestion(hreq=hreq, presented=presented)
        turn.pending_question = pending
        self._session.events.emit("question", presented)
        self._session.events.emit("status", {"status": "awaiting_input"})

        pending.event.wait()

        turn.pending_question = None
        self._session.events.emit("question_cleared",
                                  {"request_id": hreq.id})
        if turn.stopped:
            raise AgentStoppedError("Agent stopped by user")
        self._session.events.emit("status", {"status": "running"})
        response = pending.response or ""
        if key is not None:
            self._metadata_cache[key] = response
        return response
