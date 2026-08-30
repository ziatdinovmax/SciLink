"""Offline tests for the session event log (#462 Phase 0 + pure readers).

  python -m pytest tests/test_session_events.py -v
"""
import json
import threading

import scilink.session_events as se
from scilink.session_events import (
    append_event, get_thread_event_log, read_events, search_events,
    set_thread_event_log, use_event_log,
)


def _lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def setup_function(_):
    set_thread_event_log(None)
    se._next_n.clear()


# ---------------------------------------------------------------- writer

def test_unbound_is_noop(tmp_path):
    append_event("run_analysis", {"x": 1}, '{"status": "success"}')
    assert list(tmp_path.iterdir()) == []


def test_bounded_fields_and_files(tmp_path):
    log = tmp_path / "events.jsonl"
    set_thread_event_log(log)
    huge = "y" * 10_000
    result = json.dumps({
        "status": "success", "message": "R²=0.991, 3 peaks",
        "files_produced": ["a.png", "b.csv"], "report_path": "rep.md",
        "blob": huge,
    })
    append_event("run_analysis", {"data_path": "spec.csv", "note": huge}, result)
    (rec,) = _lines(log)
    assert rec["n"] == 1 and rec["tool"] == "run_analysis"
    assert len(rec["args"]) <= se.ARGS_MAX_CHARS
    assert len(rec["summary"]) <= se.SUMMARY_MAX_CHARS
    assert rec["status"] == "success"
    assert "R²=0.991" in rec["summary"]          # UTF-8, not \\u escapes
    assert rec["files"] == ["a.png", "b.csv", "rep.md"]
    assert huge[:500] not in json.dumps(rec)     # nothing unbounded leaked


def test_long_paths_keep_their_tail(tmp_path):
    log = tmp_path / "e.jsonl"
    set_thread_event_log(log)
    deep = "/very/long/session/dir/" + "sub/" * 60 + "fit_overlay.png"
    append_event("run_analysis", {}, json.dumps(
        {"status": "success", "files_produced": [deep]}))
    (rec,) = _lines(log)
    (f,) = rec["files"]
    assert len(f) <= 160 and f.endswith("fit_overlay.png")


def test_redaction():
    gist = se._gist_args({"api_key": "sk-123", "GEMINI_TOKEN": "t",
                          "path": "f.csv"})
    assert "sk-123" not in gist and "api_key=<redacted>" in gist
    assert "GEMINI_TOKEN=<redacted>" in gist and "path=f.csv" in gist


def test_non_json_result_and_error_status(tmp_path):
    log = tmp_path / "e.jsonl"
    set_thread_event_log(log)
    append_event("view_document", {}, "plain text " * 100)
    append_event("run_analysis", {}, json.dumps({"status": "error",
                                                 "message": "boom"}))
    a, b = _lines(log)
    assert a["status"] == "success" and a["summary"].startswith("plain text")
    assert b["status"] == "error" and "boom" in b["summary"]
    assert (a["n"], b["n"]) == (1, 2)


def test_counter_resumes_from_existing_file(tmp_path):
    log = tmp_path / "e.jsonl"
    log.write_text('{"n": 1}\n{"n": 2}\n', encoding="utf-8")
    set_thread_event_log(log)
    append_event("t", {}, "{}")
    assert _lines(log)[-1]["n"] == 3


def test_writer_never_raises(tmp_path, monkeypatch):
    set_thread_event_log(tmp_path / "e.jsonl")
    monkeypatch.setattr(se, "_gist_result",
                        lambda r: (_ for _ in ()).throw(RuntimeError("x")))
    append_event("t", {}, "{}")  # must not raise


def test_use_event_log_restores(tmp_path):
    set_thread_event_log(tmp_path / "outer.jsonl")
    with use_event_log(tmp_path / "inner.jsonl"):
        assert get_thread_event_log().endswith("inner.jsonl")
        append_event("t", {}, "{}")
    assert get_thread_event_log().endswith("outer.jsonl")
    assert (tmp_path / "inner.jsonl").exists()


def test_binding_is_thread_local(tmp_path):
    set_thread_event_log(tmp_path / "main.jsonl")
    seen = {}

    def worker():
        seen["binding"] = get_thread_event_log()
        append_event("t", {}, "{}")  # unbound in this thread -> no-op

    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert seen["binding"] is None
    assert not (tmp_path / "main.jsonl").exists()


# ---------------------------------------------------------------- readers

def _seed(tmp_path, n=30):
    log = tmp_path / "events.jsonl"
    set_thread_event_log(log)
    for i in range(1, n + 1):
        tool = "run_analysis" if i % 3 == 0 else "save_file"
        append_event(tool, {"i": i}, json.dumps(
            {"status": "success", "message": f"step {i} T={300 + i}K"}))
    return log


def test_search_substring_and_regex(tmp_path):
    log = _seed(tmp_path)
    out = search_events(log, "run_analysis", max_hits=50)
    assert out["total_matches"] == 10 and out["returned"] == 10
    out = search_events(log, r"T=3[01]\dK", max_hits=50)
    assert out["total_matches"] == 19  # T=301K..T=319K
    # Invalid regex degrades to substring, not an exception.
    out = search_events(log, "T=310K (", max_hits=5)
    assert out["total_matches"] == 0


def test_search_caps_return_most_recent(tmp_path):
    log = _seed(tmp_path)
    out = search_events(log, "save_file", max_hits=3)
    assert out["total_matches"] == 20 and out["returned"] == 3
    ns = [h["n"] for h in out["hits"]]
    assert ns == sorted(ns) and ns[-1] == 29  # last non-multiple-of-3 index


def test_search_max_hits_hard_cap(tmp_path):
    log = _seed(tmp_path)
    out = search_events(log, "step", max_hits=500)
    assert out["returned"] <= se.SEARCH_MAX_HITS


def test_read_events_range_and_caps(tmp_path):
    log = _seed(tmp_path)
    out = read_events(log, 5, 8)
    assert [e["n"] for e in out["events"]] == [5, 6, 7, 8]
    out = read_events(log, 1, 500)          # range clamp
    assert out["returned"] <= se.READ_MAX_EVENTS
    out = read_events(log, 8, 5)            # inverted range tolerated
    assert [e["n"] for e in out["events"]] == [5, 6, 7, 8]


def test_malformed_lines_skipped(tmp_path):
    log = tmp_path / "e.jsonl"
    log.write_text('{"n": 1, "tool": "a"}\nNOT JSON\n{"n": 2, "tool": "b"}\n',
                   encoding="utf-8")
    out = search_events(log, "tool", max_hits=10)
    assert out["total_matches"] == 2


def test_search_missing_file_honest_null(tmp_path):
    out = search_events(tmp_path / "nope.jsonl", "x", max_hits=5)
    assert out == {"total_matches": 0, "returned": 0, "hits": []}


# ----------------------------------------------------- chokepoint smoke

def test_execute_tool_logs_event(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator_tools import (
        MetaOrchestratorTools)
    t = MetaOrchestratorTools.__new__(MetaOrchestratorTools)
    t.functions_map = {"f": lambda **kw: json.dumps(
        {"status": "success", "message": "ok"})}
    t.tools_schema = [{"name": "f",
                       "input_schema": {"properties": {}, "required": []}}]
    log = tmp_path / "events.jsonl"
    with use_event_log(log):
        MetaOrchestratorTools.execute_tool(t, "f", x=1)
        MetaOrchestratorTools.execute_tool(t, "missing_tool")
    a, b = _lines(log)
    assert a["tool"] == "f" and a["status"] == "success" and "x=1" in a["args"]
    assert b["tool"] == "missing_tool" and b["status"] == "error"
