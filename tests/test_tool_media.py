"""Tests for feeding tool-returned images to the orchestrator LLM
(scilink.utils.tool_media).

The load-bearing property is NO REGRESSION: every non-image tool result, and
every result on a provider that does not render tool-result images, must yield
the byte-for-byte plain-string message the loop built before. Only a
JSON result carrying a recognised base64 image, on an image-capable provider,
becomes a multimodal message — and even then the base64 never reaches disk.
"""

from __future__ import annotations

import json

from scilink.utils.tool_media import (build_tool_message,
                                       provider_supports_tool_image,
                                       sanitize_history_images)


def test_provider_guard_allowlist():
    for ok in ("bedrock/us.anthropic.claude-opus-4-8", "anthropic/claude-3-5",
               "gemini/gemini-2.0", "vertex_ai/claude-3"):
        assert provider_supports_tool_image(ok), ok
    for no in ("gpt-4o", "azure/gpt-4", "o3-mini", "", None):
        assert not provider_supports_tool_image(no), no


def test_non_image_result_is_unchanged_both_flags():
    plain = json.dumps({"status": "success", "value": 42})
    for allow in (True, False):
        m = build_tool_message("c1", plain, allow_image=allow)
        assert m == {"role": "tool", "tool_call_id": "c1", "content": plain}


def test_non_json_result_is_unchanged():
    m = build_tool_message("c1", "plain text, not json", allow_image=True)
    assert m["content"] == "plain text, not json"


def test_image_result_disabled_keeps_base64_plain_string():
    img = json.dumps({"status": "ok", "image_base64": "iVBORdata"})
    m = build_tool_message("c1", img, allow_image=False)
    assert m["content"] == img          # untouched on a non-capable provider


def test_image_result_enabled_becomes_multimodal_without_bloating_text():
    img = json.dumps({"status": "ok", "shape": [2, 2], "image_base64": "iVBORdata"})
    m = build_tool_message("c1", img, allow_image=True)
    assert isinstance(m["content"], list)
    text = m["content"][0]
    assert text["type"] == "text"
    assert "image_base64" not in text["text"]      # blob stripped from text
    assert "shape" in text["text"]                 # structured fields kept
    img_part = m["content"][1]
    assert img_part["type"] == "image_url"
    assert img_part["image_url"]["url"] == "data:image/png;base64,iVBORdata"


def test_mime_detected_from_magic_prefix():
    m = build_tool_message("c1", json.dumps({"image_base64": "/9j/jpeg"}),
                           allow_image=True)
    assert m["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_list_of_images():
    m = build_tool_message("c1", json.dumps({"images_base64": ["iVBORa", "iVBORb"]}),
                           allow_image=True)
    assert sum(1 for p in m["content"] if p["type"] == "image_url") == 2


def _png_b64(w, h):
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _dims(data_url):
    import base64
    import io
    from PIL import Image
    b64 = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def test_oversized_image_is_downscaled():
    # a super-high-res image must be capped before embedding so it cannot blow
    # past provider size limits; the long side lands at the shared cap.
    big = _png_b64(12000, 9000)
    m = build_tool_message("c1", json.dumps({"image_base64": big}), allow_image=True)
    w, h = _dims(m["content"][1]["image_url"]["url"])
    assert max(w, h) <= 8000
    assert (w, h) != (12000, 9000)


def test_normal_image_passes_through_byte_identical():
    from scilink.utils.tool_media import _cap_b64
    ok = _png_b64(1650, 1350)          # ~reconcile-figure size
    assert _cap_b64(ok) == ok


def test_cap_never_breaks_on_garbage():
    from scilink.utils.tool_media import _cap_b64
    assert _cap_b64("not-valid-base64-image!!") == "not-valid-base64-image!!"


def test_sanitize_collapses_multimodal_and_strips_base64():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            build_tool_message("c1", json.dumps({"image_base64": "iVBORx"}),
                               allow_image=True)]
    out = sanitize_history_images(msgs)
    assert out[0] == msgs[0] and out[1] == msgs[1]          # plain msgs untouched
    tool = out[2]
    assert isinstance(tool["content"], str)
    assert "iVBORx" not in tool["content"]                  # no base64 on disk
    assert "image omitted" in tool["content"]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "c1"  # structure kept


def test_sanitize_is_noop_for_plain_messages():
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": "{\"ok\":1}"},
            {"role": "assistant", "content": "done"}]
    assert sanitize_history_images(msgs) == msgs
