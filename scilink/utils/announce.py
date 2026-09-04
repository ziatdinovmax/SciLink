"""Once-per-process LLM-backend announcements.

A meta session constructs many agents in a row, and every one of them
announcing "using LiteLLM: <model>" turned session startup into a wall of
identical lines. The backend/model pair is announced at INFO the first time
it is seen in the process and at DEBUG afterwards — the information stays
available without the repetition.
"""

from __future__ import annotations

import logging
from typing import Optional, Set, Tuple

_seen: Set[Tuple[str, str]] = set()


def announce_litellm(role: str, model: str, *, embeddings: bool = False,
                     logger: Optional[logging.Logger] = None) -> None:
    log = logger or logging.getLogger(__name__)
    suffix = " for embeddings" if embeddings else ""
    line = f"🌐 {role} using LiteLLM{suffix}: {model}"
    key = ("embeddings" if embeddings else "model", model)
    if key in _seen:
        log.debug(line)
    else:
        _seen.add(key)
        log.info(line)
