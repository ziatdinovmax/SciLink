"""Regression test for #244: the image trend-codegen path must use the
compile-checked parse_codegen_response (like the curve trend controller has
since #240), so a non-compilable or truncated script is rejected+retried
rather than executed.

Offline — the LLM is stubbed.

  python -m pytest tests/test_image_trend_codegen_parse.py -v
"""
import json
import logging
import types

from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    ConditionalImageTrendController,
)

GOOD_SCRIPT = "import json\nprint('trend ok')\n"
BROKEN_SCRIPT = "def f(:\n    pass"  # does not compile


def _controller(fake_text, finish_reason=None):
    obj = ConditionalImageTrendController.__new__(ConditionalImageTrendController)
    obj.logger = logging.getLogger("test_image_trend_parse")

    class _Model:
        def generate_content(self, contents, generation_config=None,
                             safety_settings=None):
            resp = types.SimpleNamespace(text=fake_text, raw_text=fake_text)
            if finish_reason is not None:
                resp.candidates = [
                    types.SimpleNamespace(finish_reason=finish_reason)
                ]
            return resp

    obj.model = _Model()
    obj.generation_config = None
    obj.safety_settings = None
    return obj


def _state():
    return {"series_results": [
        {"success": True, "index": 0, "name": "a.npy"},
        {"success": True, "index": 1, "name": "b.npy"},
    ], "series_metadata": {"variable": "T"}, "flagged_images": []}


def test_json_response_parses_with_side_fields():
    tc = _controller(json.dumps({
        "analysis_approach": "trend dashboard",
        "script": GOOD_SCRIPT,
    }))
    out = tc._generate_trend_script(_state())
    assert out and out["script"].strip() == GOOD_SCRIPT.strip()
    assert out.get("analysis_approach") == "trend dashboard"


def test_fenced_response_parses():
    tc = _controller(f"Here is the script:\n```python\n{GOOD_SCRIPT}```\n")
    out = tc._generate_trend_script(_state())
    assert out and "print('trend ok')" in out["script"]


def test_noncompilable_script_rejected():
    tc = _controller(json.dumps({"script": BROKEN_SCRIPT}))
    assert tc._generate_trend_script(_state()) is None


def test_truncated_response_rejected():
    # finish_reason == 0 is the wrapper's finish_reason=length marker.
    tc = _controller('{"script": "print(', finish_reason=0)
    assert tc._generate_trend_script(_state()) is None


def test_correct_script_returns_fix_and_rejects_garbage():
    tc = _controller(json.dumps({"diagnosis": "bad import",
                                 "script": GOOD_SCRIPT}))
    assert tc._correct_script(BROKEN_SCRIPT, "SyntaxError", 1).strip() == \
        GOOD_SCRIPT.strip()

    tc = _controller(json.dumps({"diagnosis": "x", "script": BROKEN_SCRIPT}))
    assert tc._correct_script(BROKEN_SCRIPT, "SyntaxError", 1) is None
