"""The words SciLink's two chat surfaces share.

The React web UI and the terminal shell narrate the same agents, ask the
same questions and name the same things. This module is the one place those
words live: mode names and blurbs, autonomy levels, status badges, the
consent sentence, the stop messages, the "Enter accepts" hint and the
activity labels shown while a turn runs.

Python consumers import it. The frontend reads the generated
``webui/src/vocabulary.ts`` (``scripts/gen_vocabulary.py``), and
``tests/test_vocabulary_sync.py`` keeps the two identical, so a label
changed here without regenerating fails the suite instead of drifting.

No imports on purpose: the shell loads this before anything heavy.
"""

from __future__ import annotations

import copy

# Invisible marker (U+2063) the orchestrators append to a 💭 / 🤖 line when
# the printing agent is a meta-delegated specialist, so a capture that strips
# ANSI can still colour the specialist's narration differently.
THOUGHT_MARK = "⁣"

# ── Modes ────────────────────────────────────────────────────────
# ``label`` is the legacy Streamlit tab label; ``name`` / ``emoji`` / ``blurb``
# are what the React welcome screen and the shell banner show.
# ``session_prefix`` names new session directories; ``legacy_prefixes`` are
# older CLI spellings that discovery still lists for resume.
MODES = {
    "meta": {
        "key": "meta",
        "name": "Mission Control",
        "emoji": "🎛️",
        "label": "🧪 📋 ⚛️",
        "description": "Routes your research goal to the Analyze & Plan specialists",
        "blurb": ("Describe a goal — the meta-agent routes work to the analysis "
                  "and planning specialists and fuses the results"),
        "placeholder": "Message mission control...",
        "session_prefix": "meta_session",
        "legacy_prefixes": [],
        "autonomy_options": ["autopilot", "autonomous"],
        "stop_message": "Analysis stopped by user.",
    },
    "analyze": {
        "key": "analyze",
        "name": "Analyze",
        "emoji": "🔬",
        "label": "Analyze",
        "description": "Multi-modal data analysis",
        "blurb": ("Interpret experimental data — images, spectra, hyperspectral "
                  "cubes — with agentic fitting and reports"),
        "placeholder": "Message the analysis agent...",
        "session_prefix": "analysis_session",
        "legacy_prefixes": [],
        "autonomy_options": ["co-pilot", "autopilot", "autonomous"],
        "stop_message": "Analysis stopped by user.",
    },
    "plan": {
        "key": "plan",
        "name": "Plan",
        "emoji": "📋",
        "label": "Plan",
        "description": "Experimental design & optimization",
        "blurb": ("Design experiments and optimization campaigns, grounded in "
                  "your papers, code, and data"),
        "placeholder": "Message the planning agent...",
        "session_prefix": "planning_session",
        "legacy_prefixes": ["campaign_session"],
        "autonomy_options": ["co-pilot", "autopilot", "autonomous"],
        "stop_message": "Planning stopped by user.",
    },
    "simulate": {
        "key": "simulate",
        "name": "Simulate",
        "emoji": "⚛️",
        "label": "Simulate",
        "description": "Submit and monitor DFT/MD simulations",
        "blurb": "Build structures and run DFT/MD simulations end to end",
        "placeholder": "Message the simulation agent...",
        "session_prefix": "simulation_session",
        "legacy_prefixes": ["simulate_session"],
        "autonomy_options": ["co-pilot", "autopilot", "autonomous"],
        "stop_message": "Simulation stopped by user.",
    },
}

# ── Session status ───────────────────────────────────────────────
STATUS_LABELS = {
    "idle": "idle",
    "running": "running",
    "awaiting_input": "awaiting your input",
}
STATUS_BADGES = {
    "idle": "⚪ idle",
    "running": "🟢 running",
    "awaiting_input": "🟠 awaiting your input",
}

# Spinner label when the narration has no recognizable milestone yet.
DEFAULT_ACTIVITY = "Agent is working…"

# ── Names of things ──────────────────────────────────────────────
NAMES = {
    "meta": "Mission control",
    "specialist": "Specialist",
    "delegation": "Delegation",
    "candidate": "Candidate",
    "judge_pick": "Judge's pick",
    "branch": "Branch",
    "turn": "Turn",
    "verbose_section": "Verbose output",
    "verbose_toggle": "Show verbose output",
    "stop": "Stop agent",
    "start_session": "Start Session",
    "resume_session": "Resume Session",
    "resume_past": "Resume past session",
}

# ── Sentences ────────────────────────────────────────────────────
# The code-execution consent: the web checkbox and the shell's startup
# confirmation show the same sentence.
CONSENT_TEXT = ("I understand that the agent will execute generated "
                "Python code on my machine")

# How both surfaces tell the user that the empty answer accepts as-is:
# ``{accept}`` is the presenter's accept label ("Approve plan", ...).
ENTER_ACCEPTS_HINT = "Enter = {accept}"

# Meta → specialist handoff banners (line prefixes in the narration).
HANDOFF_PREFIXES = [
    "🧪 Delegating to",
    "📋 Delegating to",
    "⚛️ Delegating to",
    "🧬 Fusing delegations",
]

# ── Activity labels ──────────────────────────────────────────────
# Templates for the one-line "what is it doing now" derived from the
# narration tail (``scilink.ui.narration.current_activity`` and its TS twin).
# The regexes live in code — the two dialects differ — but the words live
# here, so the label a milestone produces is the same on both surfaces.
ACTIVITY_LABELS = {
    "writing_response": "Writing response…",
    "waiting_for": "Waiting for the {who}…",
    "waiting_for_input": "Waiting for your input…",
    "wrapping_up": "Wrapping up…",
    "escalating": "Escalating to {n} parallel candidates…",
    "writing_code": "Writing analysis code…",
    "writing_code_attempt": "Writing analysis code (attempt {n})…",
    "executing_code": "Executing analysis code…",
    "executing_code_attempt": "Executing analysis code (attempt {n})…",
    "executing_script": "Executing analysis script…",
    "executing_script_attempt": "Executing analysis script (attempt {n})…",
    "visual_qc": "Visual QC · {target}",
    "reviewing": "Reviewing {target}",
    "verification": "Verification {i}/{n}…",
    "verification_annealing": "Verification {i}/{n} · annealing level {level}…",
    "correcting": "Correcting the script (attempt {n})…",
    "applying_feedback": "Applying your feedback to the script…",
    "delegating": "Delegating to {target}",
    "analyzing": "Analyzing {target}",
    "attempt": "Attempt {n} · {label}",
    "candidate": "Candidate {n} · {label}",
}


# ── Helpers ──────────────────────────────────────────────────────

def mode_keys() -> list:
    return list(MODES)


def mode(key: str) -> dict:
    """The mode record, or a KeyError naming the valid keys."""
    try:
        return MODES[key]
    except KeyError:
        raise KeyError(f"unknown mode {key!r}; one of {', '.join(MODES)}") from None


def autonomy_options(key: str) -> list:
    return list(mode(key)["autonomy_options"])


def stop_message(key: str) -> str:
    return MODES.get(key, MODES["analyze"])["stop_message"]


def session_prefixes(key: str) -> list:
    """Canonical prefix first, then the legacy spellings discovery still lists."""
    m = mode(key)
    return [m["session_prefix"], *m["legacy_prefixes"]]


def as_json() -> dict:
    """Everything the frontend needs, JSON-shaped (deep-copied)."""
    return copy.deepcopy({
        "modes": MODES,
        "status_labels": STATUS_LABELS,
        "status_badges": STATUS_BADGES,
        "default_activity": DEFAULT_ACTIVITY,
        "names": NAMES,
        "consent_text": CONSENT_TEXT,
        "enter_accepts_hint": ENTER_ACCEPTS_HINT,
        "thought_mark": THOUGHT_MARK,
        "handoff_prefixes": HANDOFF_PREFIXES,
        "activity_labels": ACTIVITY_LABELS,
    })
