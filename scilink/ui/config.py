"""Shared constants and defaults for the SciLink Streamlit UI."""

from pathlib import Path

MODEL_OPTIONS = [
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
]

EMBEDDING_MODEL_OPTIONS = [
    "gemini-embedding-001",
    "text-embedding-3-small",
    "text-embedding-3-large",
]

# ── Mode registry ────────────────────────────────────────────────
APP_MODES = [
    {"key": "meta",    "label": "🧪 📋 ⚛️",  "beta": True, "description": "Routes your research goal to the Analyze & Plan specialists"},
    {"key": "analyze", "label": "Analyze", "description": "Multi-modal data analysis"},
    {"key": "plan",    "label": "Plan",    "description": "Experimental design & optimization"},
    {"key": "simulate", "label": "Simulate", "description": "Submit and monitor DFT/MD simulations"},
    {"key": "live",    "label": "Live",     "beta": True, "description": "Real-time monitoring of an in-progress measurement"},
]

SESSION_DIR_PREFIXES = {
    "meta": "meta_session",
    "analyze": "analysis_session",
    "plan": "planning_session",
    "simulate": "simulation_session",
    "live": "live_session",
}

# ── File extensions ──────────────────────────────────────────────
SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
)

# Vendor formats that SciLink itself cannot read; surfaced in the upload
# whitelist only when an MCP server exposing a compatible reader is
# connected (see ``extra_data_extensions_for``).
VENDOR_DATA_EXTENSIONS = (
    ".dm3", ".dm4", ".emd", ".ndata",
    ".ibw", ".ardf",
    ".gwy", ".gsf",
    ".spe", ".spc", ".spx",
    ".asc", ".dat",
    ".h5", ".hdf5",
)


def extra_data_extensions_for(agent) -> tuple:
    """Return upload extensions enabled by external readers on *agent*.

    Recognises the SciFiReaders MCP server via its ``read_scifireaders_file``
    tool. Returns an empty tuple if no compatible reader server is
    connected, so the uploader does not advertise formats SciLink cannot
    actually handle.
    """
    if agent is None:
        return ()
    conns = getattr(agent, "_mcp_connections", None) or {}
    for conn in conns.values():
        for schema in getattr(conn, "tool_schemas", []) or []:
            if schema.get("function", {}).get("name") == "read_scifireaders_file":
                return VENDOR_DATA_EXTENSIONS
    return ()


SUPPORTED_METADATA_EXTENSIONS = (".json", ".txt")

SUPPORTED_KNOWLEDGE_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".xlsx", ".tsv", ".json")
SUPPORTED_CODE_EXTENSIONS = (".py", ".txt", ".md", ".json", ".yaml", ".yml")
SUPPORTED_PLANNING_DATA_EXTENSIONS = (".csv", ".xlsx", ".tsv", ".txt", ".npy", ".json")

# Explore (meta) mode accepts everything the specialist modes accept — one
# combined uploader; the meta-agent routes each file to the right child.
SUPPORTED_META_EXTENSIONS = tuple(sorted(set(
    SUPPORTED_DATA_EXTENSIONS
    + SUPPORTED_METADATA_EXTENSIONS
    + SUPPORTED_KNOWLEDGE_EXTENSIONS
    + SUPPORTED_CODE_EXTENSIONS
)))

AVATAR_USER = str(Path(__file__).resolve().parent / "assets" / "avatar_user.svg")
AVATAR_AGENT = str(Path(__file__).resolve().parent / "assets" / "avatar_agent.svg")
