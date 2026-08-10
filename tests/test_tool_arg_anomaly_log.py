"""Capture the raw provider output when a required argument goes missing.

The cause of these omissions is still unknown. Two candidate explanations
were ruled out — the output-token ceiling (injected explicitly on the
bedrock/anthropic paths, far above the arguments involved) and tail
truncation (the observed calls were valid JSON missing only the FIRST
declared parameter while later-declared ones arrived) — but the evidence
needed to identify the real cause is destroyed before anyone can look at
it: by the time the guard runs it holds parsed kwargs, not the response.

So record it. `finish_reason` is the decisive field: "length" means the
response really was cut; anything else rules truncation out for that call.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent)


def _orch(tmp):
    o = PlanningOrchestratorAgent.__new__(PlanningOrchestratorAgent)
    o.base_dir = Path(tmp)
    o.model_name = "bedrock/us.anthropic.claude-opus-4-8"
    return o


def _call(raw_args, name="write_technical_document"):
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=raw_args))


def _missing_result(tool="write_technical_document", missing=("request",)):
    return json.dumps({"status": "error", "tool": tool,
                       "error_kind": "missing_required_arguments",
                       "missing": list(missing), "message": "..."})


def _log(tmp):
    p = Path(tmp) / "tool_call_diagnostics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_records_the_raw_arguments_and_finish_reason():
    """The live shape: valid JSON, later-declared args present, no request."""
    tmp = tempfile.mkdtemp()
    raw = '{"filename": "memo.md", "title": "T", "use_literature": false}'
    _orch(tmp)._record_tool_arg_anomaly(
        _call(raw), _missing_result(),
        finish_reason="tool_calls",
        response=SimpleNamespace(usage=SimpleNamespace(
            completion_tokens=812, prompt_tokens=41000)))

    rec, = _log(tmp)
    assert rec["arguments_raw"] == raw, "the raw string is the whole point"
    assert rec["finish_reason"] == "tool_calls", "decides the truncation question"
    assert rec["arguments_parse_ok"] is True
    assert rec["arguments_keys"] == ["filename", "title", "use_literature"]
    assert rec["missing"] == ["request"]
    assert rec["completion_tokens"] == 812
    assert rec["model"].startswith("bedrock/")
    assert rec["arguments_len"] == len(raw)


def test_a_genuinely_truncated_call_is_distinguishable():
    """If truncation IS the cause, the record must show it plainly."""
    tmp = tempfile.mkdtemp()
    _orch(tmp)._record_tool_arg_anomaly(
        _call('{"request": "half a brie'), _missing_result(),
        finish_reason="length", response=None)

    rec, = _log(tmp)
    assert rec["finish_reason"] == "length"
    assert rec["arguments_parse_ok"] is False
    assert rec["arguments_keys"] is None


def test_successful_calls_write_nothing():
    """No anomaly, no file — this must not become a per-call trace log."""
    tmp = tempfile.mkdtemp()
    o = _orch(tmp)
    o._record_tool_arg_anomaly(_call('{"request": "x"}'),
                               json.dumps({"status": "success", "path": "p"}))
    o._record_tool_arg_anomaly(_call('{}'),
                               json.dumps({"status": "error",
                                           "message": "some other failure"}))
    assert not (Path(tmp) / "tool_call_diagnostics.jsonl").exists()


def test_occurrences_accumulate():
    """JSONL so a second occurrence does not overwrite the first."""
    tmp = tempfile.mkdtemp()
    o = _orch(tmp)
    for i in range(3):
        o._record_tool_arg_anomaly(_call('{"filename": "m%d.md"}' % i),
                                   _missing_result())
    assert len(_log(tmp)) == 3


def test_a_runaway_argument_cannot_fill_the_disk():
    tmp = tempfile.mkdtemp()
    o = _orch(tmp)
    huge = '{"request": "' + "x" * 200000 + '"}'
    o._record_tool_arg_anomaly(_call(huge), _missing_result())

    rec, = _log(tmp)
    assert rec["arguments_truncated_in_log"] is True
    assert len(rec["arguments_raw"]) == o._RAW_ARGS_CAP
    assert rec["arguments_len"] == len(huge), "true length still recorded exactly"


def test_recording_never_breaks_the_turn():
    """A diagnostic that kills the turn it is diagnosing is worse than none."""
    tmp = tempfile.mkdtemp()
    o = _orch(tmp)
    o.base_dir = Path("/proc/nonexistent-and-unwritable")  # force a write error
    o._record_tool_arg_anomaly(_call('{"filename": "m.md"}'), _missing_result())

    o2 = _orch(tmp)
    o2._record_tool_arg_anomaly(_call(None), _missing_result())      # bad raw
    o2._record_tool_arg_anomaly(_call('{}'), "not json at all")      # bad result
    o2._record_tool_arg_anomaly(_call('{}'), None)                   # no result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
