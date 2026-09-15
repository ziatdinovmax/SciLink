"""Shared constants and defaults for the SciLink Streamlit UI."""

from pathlib import Path
from typing import Dict, Optional, Tuple

from .. import auth

MODEL_OPTIONS = [
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    # Amazon Bedrock (Claude Opus 4.8) via the US geo cross-region inference
    # profile (exact ID from the AWS model card; Opus 4.8 has no date stamp and
    # no version suffix). Invoke-able only through an inference profile, hence
    # the ``us.`` prefix. For EU/JP/AU keys swap the prefix:
    # ``eu.``/``jp.``/``au.``. (Base model ID: anthropic.claude-opus-4-8.)
    "bedrock/us.anthropic.claude-opus-4-8",
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
]

SESSION_DIR_PREFIXES = {
    "meta": "meta_session",
    "analyze": "analysis_session",
    "plan": "planning_session",
    "simulate": "simulation_session",
}

# ── File extensions ──────────────────────────────────────────────
SUPPORTED_DATA_EXTENSIONS = (
    ".tif", ".tiff", ".png", ".jpg", ".npy", ".csv", ".txt", ".tsv", ".xlsx",
    ".h5", ".hdf5", ".nxs",
)

# Vendor formats that SciLink itself cannot read; surfaced in the upload
# whitelist only when an MCP server exposing a compatible reader is
# connected (see ``extra_data_extensions_for``).
VENDOR_DATA_EXTENSIONS = (
    ".dm3", ".dm4", ".emd", ".ndata", ".ndata1", ".ndata2",
    ".ibw", ".ardf",
    ".gwy", ".gsf",
    ".spe", ".spc", ".spx",
    ".asc", ".dat",
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


# ── Credential prefill resolution ────────────────────────────────

def resolve_prefill(
    model: str, existing_base_url: str = ""
) -> Dict[str, Tuple[str, Optional[str]]]:
    """Resolve which environment variables prefill the credential form fields.

    Pure function — no Streamlit / session_state — so the precedence rules are
    unit-testable in isolation. The sidebar wraps this to seed widget state and
    render captions.

    Returns ``{field: (value, env_var_name_or_None)}`` for fields
    ``api_key``, ``base_url``, ``fh``, ``mp``. ``env_var_name`` is the variable
    a value came from (for a "✓ from X" caption), or ``None`` when nothing was
    detected for that field.

    A key is prefilled only when the environment variable that *correctly*
    corresponds to the current selection is set — never by borrowing an
    unrelated vendor's key. If the matching variable is absent the field is
    left empty (the user sets the right env var or types a key). Proxy-vs-vendor
    correctness is also enforced by the backend (``require_vendor_credentials``).
    Main-key precedence:

      1. Proxy pair — ``SCILINK_API_KEY`` when a base URL is available
         (``SCILINK_BASE_URL`` env or one already entered): a fully-configured
         proxy deployment.
      2. The vendor key matching the selected model's provider
         (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` …).
      3. ``SCILINK_API_KEY`` on its own — a proxy deployment whose base URL will
         be supplied separately (the sidebar warns that one is still needed).
      4. Otherwise empty — no mismatched key is substituted.

    ``base_url`` prefills from ``SCILINK_BASE_URL`` when set; FutureHouse /
    Materials Project keys come from their own env vars, independent of the
    model.
    """
    proxy_key = auth.get_internal_proxy_key()
    proxy_url = auth.get_internal_proxy_base_url()
    base_available = bool(proxy_url or existing_base_url)

    base: Tuple[str, Optional[str]] = (
        (proxy_url, auth.INTERNAL_PROXY_BASE_URL) if proxy_url else ("", None)
    )

    provider_kv = auth.find_env_var_for_model(model)

    if proxy_key and base_available:
        api: Tuple[str, Optional[str]] = (proxy_key, auth.INTERNAL_PROXY_KEY)
    elif provider_kv:
        api = (provider_kv[1], provider_kv[0])
    elif proxy_key:
        api = (proxy_key, auth.INTERNAL_PROXY_KEY)
    else:
        api = ("", None)

    fh = auth.find_env_var("futurehouse")
    mp = auth.find_env_var("materials_project")

    return {
        "api_key": api,
        "base_url": base,
        "fh": (fh[1], fh[0]) if fh else ("", None),
        "mp": (mp[1], mp[0]) if mp else ("", None),
    }


def resolve_embedding_prefill(
    embedding_model: str,
) -> Tuple[str, Optional[str]]:
    """Resolve which env var prefills the embedding API key field.

    Mirrors the main key's provider-matching heuristic but without the proxy
    machinery (there is no embedding-specific proxy concept): the embedding
    model name is mapped to a provider via ``infer_provider`` (handles
    ``text-embedding-*`` → OpenAI and ``gemini-embedding-*`` → Google) and the
    matching env var, if set, fills the field. Otherwise empty.

    Returns ``(value, env_var_name_or_None)`` — same shape as the entries in
    :func:`resolve_prefill`. The sidebar wraps this and feeds it through
    :func:`reconcile_autofill` so switching the embedding model refreshes the
    field without clobbering a value the user typed.
    """
    if not embedding_model:
        return ("", None)
    kv = auth.find_env_var_for_model(embedding_model)
    return (kv[1], kv[0]) if kv else ("", None)


def reconcile_autofill(
    current: Optional[str], prev_autofill: Optional[str], resolved: str
) -> Tuple[str, str]:
    """Reconcile an auto-prefilled field when the resolved value may have changed.

    Pure function — no Streamlit. Used for the main API-key field, whose
    resolved value depends on the selected model's provider: switching the model
    to another vendor should refresh the key, but only when the user has not
    typed their own value over the prefill.

    Args:
        current: the field's current value (``None`` if it has never existed).
        prev_autofill: the value we last auto-filled into the field.
        resolved: the value the env resolution now yields for the current model.

    Returns ``(value, autofill)`` to store. The field is refreshed to
    ``resolved`` when it still holds what we last auto-filled (``prev_autofill``)
    — i.e. it has not been hand-edited — or when it has never been set. A
    user-edited (or deliberately cleared) field is left untouched.
    """
    if current is None or current == (prev_autofill or ""):
        return resolved, resolved
    return current, (prev_autofill or "")
