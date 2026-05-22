"""Tests for live_codegen: LLM-generated reading_fn save/load/validate.

The LLM call itself is mocked (no real API key needed). The
save-and-load + validation paths are exercised against synthetic
script source so we cover happy paths and every failure mode.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from scilink.agents.exp_agents.live_codegen import (
    generate_reading_script,
    save_and_load_reading_script,
    validate_reading_script,
)
from scilink.agents.exp_agents.live_data_sources import LatestData
from scilink.agents.exp_agents.live_types import LiveReadingResult


# --- save_and_load -----------------------------------------------------------

_GOOD_SCRIPT = '''
import time
from scilink.agents.exp_agents.live_types import LiveReadingResult

def reading_fn(latest_data, session_state, skill_state):
    text = (latest_data.text or "")
    n_lines = text.count("\\n")
    return LiveReadingResult(
        timestamp=time.time(),
        primary_metric=float(n_lines),
        metric_name="line_count",
        verdict="accept" if n_lines > 5 else "marginal",
        detected_features=[],
        notes=f"{n_lines} lines",
    )
'''

_BAD_SYNTAX = '''
def reading_fn(latest_data, session_state, skill_state):
    return  # this is fine actually let me break it
def $$$invalid_python(:
    pass
'''

_NO_READING_FN = '''
def some_other_function():
    return 42
'''


def test_save_and_load_good_script(tmp_path):
    fn = save_and_load_reading_script(_GOOD_SCRIPT, tmp_path, version=1)
    assert callable(fn)
    # The file was written
    assert (tmp_path / "reading_script_v1.py").is_file()
    # And it actually works
    data = LatestData(timestamp=time.time(), source_kind="test",
                       text="a\nb\nc\nd\ne\nf\n")
    result = fn(data, {}, {})
    assert isinstance(result, LiveReadingResult)
    assert result.primary_metric == 6.0
    assert result.verdict == "accept"


def test_save_and_load_syntax_error_returns_none(tmp_path):
    fn = save_and_load_reading_script(_BAD_SYNTAX, tmp_path, version=1)
    assert fn is None


def test_save_and_load_missing_reading_fn_returns_none(tmp_path):
    fn = save_and_load_reading_script(_NO_READING_FN, tmp_path, version=1)
    assert fn is None


def test_save_and_load_creates_session_dir_if_missing(tmp_path):
    target = tmp_path / "new" / "session"
    assert not target.exists()
    fn = save_and_load_reading_script(_GOOD_SCRIPT, target, version=1)
    assert callable(fn)
    assert target.is_dir()


def test_save_and_load_uses_version_in_filename(tmp_path):
    save_and_load_reading_script(_GOOD_SCRIPT, tmp_path, version=1)
    save_and_load_reading_script(_GOOD_SCRIPT, tmp_path, version=5)
    assert (tmp_path / "reading_script_v1.py").is_file()
    assert (tmp_path / "reading_script_v5.py").is_file()


# --- validate ----------------------------------------------------------------

def _mk_history(n: int = 3) -> list[LatestData]:
    return [
        LatestData(timestamp=time.time() - n + i, source_kind="test",
                    text="x\n" * (i + 2))
        for i in range(n)
    ]


def test_validate_passes_on_well_behaved_fn(tmp_path):
    fn = save_and_load_reading_script(_GOOD_SCRIPT, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, _mk_history(3))
    assert ok, reason


def test_validate_passes_on_empty_history(tmp_path):
    """Empty history → conservative pass (script imported cleanly)."""
    fn = save_and_load_reading_script(_GOOD_SCRIPT, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, [])
    assert ok


def test_validate_fails_when_fn_raises(tmp_path):
    raising_script = '''
def reading_fn(latest_data, session_state, skill_state):
    raise ValueError("oops")
'''
    fn = save_and_load_reading_script(raising_script, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, _mk_history(2))
    assert not ok
    assert "raised" in reason.lower()
    assert "ValueError" in reason


def test_validate_fails_when_fn_returns_wrong_type(tmp_path):
    bad_return = '''
def reading_fn(latest_data, session_state, skill_state):
    return {"not": "a", "LiveReadingResult": 1}
'''
    fn = save_and_load_reading_script(bad_return, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, _mk_history(2))
    assert not ok
    assert "LiveReadingResult" in reason


def test_validate_fails_on_invalid_verdict(tmp_path):
    bad_verdict = '''
import time
from scilink.agents.exp_agents.live_types import LiveReadingResult

def reading_fn(latest_data, session_state, skill_state):
    return LiveReadingResult(
        timestamp=time.time(),
        primary_metric=1.0,
        metric_name="x",
        verdict="not_a_real_verdict",
        detected_features=[],
        notes="",
    )
'''
    fn = save_and_load_reading_script(bad_verdict, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, _mk_history(2))
    assert not ok
    assert "verdict" in reason.lower()


def test_validate_fails_when_fn_returns_wrong_metric_type(tmp_path):
    """primary_metric must be int/float, not str."""
    bad_metric = '''
import time
from scilink.agents.exp_agents.live_types import LiveReadingResult

def reading_fn(latest_data, session_state, skill_state):
    return LiveReadingResult(
        timestamp=time.time(),
        primary_metric="not a number",
        metric_name="x",
        verdict="marginal",
        detected_features=[],
        notes="",
    )
'''
    fn = save_and_load_reading_script(bad_metric, tmp_path, version=1)
    ok, reason = validate_reading_script(fn, _mk_history(2))
    assert not ok
    assert "primary_metric" in reason


# --- generate (LLM mocked) ---------------------------------------------------


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def test_generate_strips_code_fences():
    """LLMs sometimes wrap in ```python ... ``` even when told not to."""
    fake_resp = _FakeResp(
        "```python\n" + _GOOD_SCRIPT + "\n```\n"
    )
    with patch("litellm.completion", return_value=fake_resp):
        source = generate_reading_script(
            description="some experiment", additional_guidance=None,
            skill_context=None, model="claude-haiku", api_key="k",
        )
    assert source is not None
    assert "def reading_fn" in source
    assert "```" not in source


def test_generate_rejects_response_without_reading_fn():
    fake_resp = _FakeResp("Sure! Here is some code:\n\ndef other(): pass\n")
    with patch("litellm.completion", return_value=fake_resp):
        source = generate_reading_script(
            description="experiment", additional_guidance=None,
            skill_context=None, model="m", api_key="k",
        )
    assert source is None


def test_generate_returns_none_on_llm_failure():
    with patch("litellm.completion", side_effect=RuntimeError("API down")):
        source = generate_reading_script(
            description="experiment", additional_guidance=None,
            skill_context=None, model="m", api_key="k",
        )
    assert source is None
