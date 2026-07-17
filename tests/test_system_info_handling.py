"""_handle_system_info input handling (base analysis agent).

Covers the contract that free-text metadata is PRESERVED, not silently
dropped — a regression guard for the 17O skill-selection miss where a
descriptive string was treated as a missing file path and discarded.
"""
import json
import logging

from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent


class _Stub:
    """Minimal carrier — _handle_system_info only touches self.logger."""
    logger = logging.getLogger("test_system_info")


_handle = BaseAnalysisAgent._handle_system_info.__get__(_Stub())


def test_none_returns_empty():
    assert _handle(None) == {}


def test_flat_dict_passthrough():
    si = {"technique": "XPS", "sample": "TiO2"}
    assert _handle(si) == si


def test_nested_system_info_extracted():
    nested = {"technique": "NMR", "nucleus": "17O"}
    assert _handle({"system_info": nested, "other": 1}) == nested


def test_free_text_preserved_as_description():
    # The regression: a descriptive string is NOT a file, so it must be kept
    # (so skill selection and downstream prompts still see it), not dropped.
    text = "17O NMR of D2O: chemical shift (ppm) vs intensity."
    assert _handle(text) == {"description": text}


def test_long_free_text_preserved():
    # A multi-KB string would overflow a path (OSError/ValueError) — still text.
    text = "Raman spectrum. " * 500
    assert _handle(text) == {"description": text}


def test_valid_json_file_loaded(tmp_path):
    p = tmp_path / "meta.json"
    si = {"technique": "EPR", "g": 2.0}
    p.write_text(json.dumps(si))
    assert _handle(str(p)) == si


def test_json_file_with_nested_key(tmp_path):
    p = tmp_path / "meta.json"
    nested = {"technique": "XRD"}
    p.write_text(json.dumps({"system_info": nested}))
    assert _handle(str(p)) == nested


def test_existing_non_json_file_is_error_not_text(tmp_path):
    # An existing file that fails to parse as JSON is a genuine error -> {},
    # NOT silently wrapped as description (we'd be wrapping the path string).
    p = tmp_path / "notjson.txt"
    p.write_text("this is not json {{{")
    assert _handle(str(p)) == {}
