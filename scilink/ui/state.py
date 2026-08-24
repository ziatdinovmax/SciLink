"""Session state initialization and ChatTask dataclass for the Streamlit UI."""

import threading
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st


@dataclass
class FeedbackRequest:
    """A pending human-feedback request from the agent thread.

    ``kind``/``options``/``origin`` mirror the structured fields of
    ``scilink.hitl.FeedbackRequest`` when the prompt arrives through the
    chokepoint; a stray raw ``input()`` (third-party code) produces the
    defaults, and the widget chooser falls back to prompt/context sniffing.
    """
    prompt: str = ""
    context: str = ""  # stdout captured before the prompt
    response: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)
    kind: str = ""
    options: Optional[list] = None
    origin: dict = field(default_factory=dict)


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
        # Mode selection. Initialized to the real default ("meta" — mission
        # control) rather than None: the sidebar renders BEFORE the main
        # body's None->meta assignment, so a None here made the first page
        # load hide the mode-gated sidebar sections (embedding model/API key
        # for plan+meta, the meta's two-level autonomy list) and point
        # resume-session discovery at the analyze fallback.
        "app_mode": "meta",
        # Planning mode state
        "planning_objective": "",
        # Meta mode state
        "meta_objective": "",
        "uploaded_meta_paths": [],
        "uploaded_knowledge_paths": [],
        "uploaded_code_paths": [],
        "uploaded_planning_data_paths": [],
        # ── HPC agent workflow ──
        "hpc_sim_agent": None,              # MDSimulationAgent instance
        "hpc_gen_task": None,               # background generation tracker
        "hpc_gen_result": None,             # result dict from generate_simulation()
        "hpc_workflow_dir": None,           # local temp dir for generated files
        "hpc_workflow_script": None,        # editable script content (review step)
        "hpc_workflow_phase": "configure",      # configure|review|monitoring|results
        "hpc_mon_known_files": set(),           # remote files already seen
        "hpc_mon_downloaded_images": {},         # {remote_path: bytes}
        "hpc_remote_origins": {},           # {local_filename: remote_path}
        "hpc_env_probe": None,              # HPCEnvironment from probe_remote()
        # Theme
        "theme_mode": "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
