"""Session state initialization and ChatTask dataclass for the Streamlit UI."""

import threading
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st


@dataclass
class FeedbackRequest:
    """A pending input() call from the agent thread waiting for the user."""
    prompt: str = ""
    context: str = ""  # stdout captured before the input() call
    response: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)


@dataclass
class ChatTask:
    """Tracks a background agent.chat() call."""
    thread: object = None
    result: Optional[str] = None
    error: Optional[str] = None
    is_running: bool = False
    verbose_log: str = ""
    feedback_request: Optional[FeedbackRequest] = None
    live_capture: object = None  # OutputCapture instance for real-time reading
    stopped: bool = False  # True when the user requests cancellation


def init_session_state() -> None:
    """Set default values for every key the UI relies on."""
    defaults = {
        "agent": None,
        "agent_initialized": False,
        "agent_config": {},
        "session_dir": None,
        "chat_messages": [],
        "chat_task": ChatTask(),
        "uploaded_data_path": None,
        "uploaded_metadata_path": None,
        "uploaded_sidecar_metadata": False,
        "known_images": set(),
        "_processed_uploads": set(),
        "pending_auto_examine": None,
        "pending_auto_load_metadata": None,
        "cfg_consent": False,
        "selected_preview_file": None,
        # Mode selection (None until chosen on welcome screen)
        "app_mode": None,
        # Planning mode state
        "planning_objective": "",
        "uploaded_knowledge_paths": [],
        "uploaded_code_paths": [],
        "uploaded_planning_data_paths": [],
        # Theme
        "theme_mode": "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
