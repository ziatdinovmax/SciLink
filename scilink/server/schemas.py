"""Request bodies for the web API. Responses are plain dicts (documented at
their endpoints); requests get pydantic validation."""

from __future__ import annotations

from typing import Dict, Optional

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
