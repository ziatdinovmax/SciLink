"""Shared constants and defaults for the SciLink Streamlit UI."""

from pathlib import Path

MODEL_OPTIONS = [
    "gemini-3.1-pro-preview",
    "claude-opus-4-6",
    "gpt-5.2",
]

EMBEDDING_MODEL_OPTIONS = [
    "gemini-embedding-001",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

# ── Mode registry ────────────────────────────────────────────────
APP_MODES = [
    {"key": "analyze", "label": "Analyze", "description": "Multi-modal data analysis"},
    {"key": "plan",    "label": "Plan",    "description": "Experimental design & optimization"},
    # {"key": "simulate", "label": "Simulate", "description": "MD/DFT simulations"},
]

SESSION_DIR_PREFIXES = {
    "analyze": "analysis_session",
    "plan": "planning_session",
}

# ── File extensions ──────────────────────────────────────────────
SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
)

SUPPORTED_METADATA_EXTENSIONS = (".json", ".txt")

SUPPORTED_KNOWLEDGE_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".xlsx", ".tsv")
SUPPORTED_CODE_EXTENSIONS = (".py", ".txt", ".md", ".json", ".yaml", ".yml")
SUPPORTED_PLANNING_DATA_EXTENSIONS = (".csv", ".xlsx", ".tsv", ".txt", ".npy", ".json")

AVATAR_USER = str(Path(__file__).resolve().parent / "assets" / "avatar_user.svg")
AVATAR_AGENT = str(Path(__file__).resolve().parent / "assets" / "avatar_agent.svg")
