"""Feed images returned by a tool back into the orchestrator LLM's context.

The chat orchestrators run a manual function-calling loop: a tool returns a
JSON string, which is appended as a ``{"role": "tool", "content": <str>}``
message. A tool that packaged an image (``preview_image``, a reconciled-series
figure) therefore handed the model a base64 blob *as text* — which the model
cannot see. (Verified: on ``bedrock/…claude`` the model answers "CANNOT SEE
IMAGE" for a plain-string tool result, and reads the image correctly when the
same bytes are delivered as an ``image_url`` content block.)

This module converts an image-bearing tool result into a multimodal
``tool`` message (a list of a text part + one image part per image), which
providers that support images in tool results (Anthropic/Claude incl. Bedrock,
Gemini) render as real vision.

Three deliberate no-regression properties:

1. When images are not allowed for the active provider, or the result is not
   JSON, or it carries no recognised image field, the returned message is
   **byte-for-byte the plain-string message the loop built before** — every
   non-image tool is unaffected.
2. The big base64 blob is stripped from the text part (replaced with a short
   marker) so the model sees the structured fields *and* the image without the
   text being bloated by the encoded bytes.
3. :func:`sanitize_history_images` collapses any multimodal message back to a
   plain string before it is persisted, so ``chat_history.json`` keeps its
   original shape (no base64 on disk) and a restored session is provider-safe.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Provider substrings whose tool-result messages accept image content blocks.
# Bedrock Claude model ids contain "anthropic"/"claude"; direct Anthropic
# contains "claude"; Gemini/Vertex-Gemini contain "gemini". OpenAI ids
# (gpt-*, o1/o3, azure/*) match none, so they keep the plain-string path.
_IMAGE_TOOL_PROVIDER_KEYS = ("claude", "anthropic", "gemini")

# Tool-result JSON keys that may carry base64 image payload(s).
_IMAGE_KEYS_SINGLE = ("image_base64",)
_IMAGE_KEYS_LIST = ("images_base64",)


def provider_supports_tool_image(model: Optional[str]) -> bool:
    """True if the provider behind ``model`` renders images inside a tool
    result. Conservative allow-list — an unrecognised provider returns False,
    so the caller keeps the current plain-string behaviour (no regression)."""
    m = (model or "").lower()
    return any(k in m for k in _IMAGE_TOOL_PROVIDER_KEYS)


def _mime_of(b64: str) -> str:
    """Guess the image mime from the base64 magic prefix so the data URL is
    correct without the producing tool having to declare it. JPEG base64
    begins '/9j/', PNG begins 'iVBOR'; default to PNG."""
    return "image/jpeg" if b64.startswith("/9j/") else "image/png"


def _cap_b64(b64: str) -> str:
    """Downscale an over-large image before it is embedded, reusing the same cap
    the agent-level vision path applies (``_MAX_IMAGE_DIM`` px, downscaled to
    PNG). A super-high-res image would otherwise blow past provider limits
    (Anthropic rejects very large / >5 MB images) or waste tokens. Best-effort:
    any failure (no PIL, undecodable, already small) returns the input
    unchanged, so capping never breaks a tool result."""
    try:
        import base64
        from ..wrappers.litellm_wrapper import _cap_image_bytes
        raw = base64.b64decode(b64)
        capped = _cap_image_bytes(raw)
        if capped is raw or capped == raw:
            return b64                       # common case: unchanged
        return base64.b64encode(capped).decode()
    except Exception:
        return b64


def _extract_images(result_str: str):
    """Pull base64 images out of a JSON tool-result string.

    Returns ``(images, text)`` where ``images`` is a list of ``(mime, b64)``
    and ``text`` is the JSON with the base64 fields removed (a marker left in
    their place). Returns ``([], result_str)`` unchanged when the result is not
    JSON or carries no image field — the caller then keeps the plain string."""
    try:
        data = json.loads(result_str)
    except (ValueError, TypeError):
        return [], result_str
    if not isinstance(data, dict):
        return [], result_str

    b64s: list[str] = []
    for k in _IMAGE_KEYS_SINGLE:
        v = data.pop(k, None)
        if isinstance(v, str) and v:
            b64s.append(v)
    for k in _IMAGE_KEYS_LIST:
        v = data.pop(k, None)
        if isinstance(v, list):
            b64s.extend(x for x in v if isinstance(x, str) and x)

    if not b64s:
        return [], result_str

    data["_images_attached"] = (
        f"{len(b64s)} image(s) delivered to you as image content in this "
        "message — view them directly.")
    # Cap over-large images so a super-high-res result can't exceed provider
    # limits; recompute the mime AFTER capping (a downscale re-encodes to PNG).
    b64s = [_cap_b64(b) for b in b64s]
    images = [(_mime_of(b), b) for b in b64s]
    return images, json.dumps(data)


def build_tool_message(tool_call_id: str, result_str: str, *,
                       allow_image: bool) -> dict[str, Any]:
    """Build the ``tool`` message for the loop. Multimodal (text + image parts)
    when ``allow_image`` and the result carries an image; otherwise the exact
    plain-string message the loop used before."""
    if not allow_image:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
    images, text = _extract_images(result_str)
    if not images:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result_str}
    content: list[dict] = [{"type": "text", "text": text}]
    for mime, b64 in images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def sanitize_history_images(messages: list) -> list:
    """Collapse any multimodal message content back to a plain string for
    persistence: keep the text parts, drop image parts (leaving a marker). No-op
    for the plain-string messages that make up all non-image traffic, so saved
    history keeps its original shape and carries no base64."""
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            texts = [p.get("text", "") for p in c
                     if isinstance(p, dict) and p.get("type") == "text"]
            had_img = any(isinstance(p, dict) and p.get("type") == "image_url"
                          for p in c)
            joined = " ".join(t for t in texts if t)
            if had_img:
                joined = (joined + " [image omitted from saved history]").strip()
            out.append({**m, "content": joined})
        else:
            out.append(m)
    return out
