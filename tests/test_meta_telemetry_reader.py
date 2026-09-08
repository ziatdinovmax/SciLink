"""The telemetry reader's tool-sequence extraction: a failed tool call keeps
its error message, and an image-bearing (multimodal) tool message is
classified by its text part rather than reported as ok."""
import json

from scilink.agents.meta_agent.telemetry import _extract_tool_calls


def _call(i, name, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_tool_sequence_keeps_error_messages_and_reads_multimodal_results():
    err = {"status": "error", "error": "FileNotFoundError: no such file 'x.npy'"}
    bad_args = {"status": "error", "message": "Invalid JSON in tool arguments"}
    messages = [
        _call(1, "read_file", {"path": "x.npy"}),
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(err)},
        _call(2, "view_image", {"path": "a.png"}),
        # multimodal result: text part carries the tool JSON, image part follows
        {"role": "tool", "tool_call_id": "c2", "content": [
            {"type": "text", "text": json.dumps({"status": "error", "error": "unreadable image"})},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        _call(3, "inspect_uploads", {"path": "."}),
        {"role": "tool", "tool_call_id": "c3", "content": json.dumps(bad_args)},
        _call(4, "delegate_to_analysis", {"task": "t"}),   # still running: no result
    ]
    seq = _extract_tool_calls(messages)
    assert [c["tool"] for c in seq] == ["read_file", "view_image", "inspect_uploads",
                                        "delegate_to_analysis"]
    assert seq[0]["status"] == "error" and seq[0]["result"]["error"].startswith("FileNotFoundError")
    assert seq[1]["status"] == "error" and seq[1]["result"] == {"status": "error", "error": "unreadable image"}
    assert seq[2]["status"] == "error" and seq[2]["result"]["message"] == bad_args["message"]
    assert seq[3]["status"] == "pending" and seq[3]["result"] is None
