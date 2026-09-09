"""Request bodies for the web API. Responses are plain dicts (documented at
their endpoints); requests get pydantic validation."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    mode: str                       # meta | analyze | plan
    model: str
    autonomy: str                   # co-pilot | autopilot | autonomous
    consent: bool = False           # "agent executes generated code" checkbox
    api_key: str = ""
    base_url: str = ""
    provider_fields: Dict[str, str] = {}   # e.g. {"region": "us-east-1"}
    fh_api_key: str = ""
    mp_api_key: str = ""
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    objective: str = ""             # plan mode: research objective
    resume_dir: Optional[str] = None  # session dir NAME to resume, not a path


class SendMessageRequest(BaseModel):
    content: str


class FeedbackResponseRequest(BaseModel):
    request_id: str
    response: str = ""


class RenameSessionRequest(BaseModel):
    name: str


class FolderCheckRequest(BaseModel):
    """Local folder paths the user pasted into a hero form."""

    paths: list[str]


class PlanDirsRequest(BaseModel):
    """Point the planning agent's resource dirs at existing local folders
    (KB indexes key on source paths, so stable folders are reused across
    sessions instead of rebuilt)."""

    knowledge: Optional[str] = None
    code: Optional[str] = None
    data: Optional[str] = None


class LoginRequest(BaseModel):
    """Access token → session cookie (POST /auth/login)."""

    token: str



class MCPConnectRequest(BaseModel):
    """Connect an MCP server to the session's agent (POST /mcp)."""

    name: str
    transport: str = "stdio"            # stdio | sse | http
    command: str = ""                   # stdio: "npx -y @scope/server /path"
    url: str = ""                       # sse / http
    headers: Optional[Dict[str, str]] = None


# ── persistent memory ────────────────────────────────────────────────

class MemoryEnabledRequest(BaseModel):
    enabled: bool


class MemoryEditRequest(BaseModel):
    content: str


class MemoryIdsRequest(BaseModel):
    ids: List[str]
    technique: Optional[str] = None


class MemoryConsolidateRequest(BaseModel):
    ids: List[str]
    label: str
    session_id: str                 # the live session whose model distills


class MemoryUpgradeRequest(BaseModel):
    ids: List[str]
    target_domain: str
    target_name: str
    session_id: str


class MemoryApplyRequest(BaseModel):
    ids: List[str]
    target_domain: str
    target_name: str
    content: str
    fork_builtin: bool = False


class MemoryCheckRequest(BaseModel):
    existing: str
    proposed: str
