"""UI Stop-button semantics: a user stop must not be swallowed.

The Stop button sets an event that makes the agent thread's next print()
(or, via the UI log-handler filter, its next logging call) raise
``AgentStoppedError``. The agent stack is full of ``except Exception``
recovery handlers (tool dispatch, retry loops, the orchestrator chat()
catch-all), so the error must be a ``BaseException`` — otherwise the stop
is converted into a tool-error string fed back to the LLM and the run
continues (the original bug).
"""
import io
import logging
import subprocess
import sys
import threading
import time

from scilink.ui.output_capture import AgentStoppedError, OutputCapture, TeeStream


def test_agent_stopped_error_is_base_exception():
    assert issubclass(AgentStoppedError, BaseException)
    assert not issubclass(AgentStoppedError, Exception)


def test_stop_survives_except_exception_recovery_handlers():
    """The exact swallow pattern that caused the bug: agent-side code wraps
    work in ``except Exception`` and keeps going. The stop must escape it."""
    stop = threading.Event()
    stop.set()
    stream = TeeStream(io.StringIO(), io.StringIO(), stop)

    def agent_like_loop():
        for _ in range(3):  # tool-dispatch retry loop
            try:
                stream.write("narration\n")  # agent print() lands here
            except Exception:
                continue  # pre-fix: stop swallowed here, run continued

    try:
        agent_like_loop()
    except AgentStoppedError:
        return
    raise AssertionError("stop was swallowed by an except-Exception handler")


def test_teestream_does_not_write_after_stop():
    buf = io.StringIO()
    stop = threading.Event()
    stream = TeeStream(io.StringIO(), buf, stop)
    stream.write("before\n")
    stop.set()
    try:
        stream.write("after\n")
    except AgentStoppedError:
        pass
    assert buf.getvalue() == "before\n"


def test_stop_requested_property():
    cap = OutputCapture()
    assert cap.stop_requested is False
    cap.request_stop()
    assert cap.stop_requested is True


def test_logging_filter_raise_propagates_to_emitter():
    """The UI attaches a handler whose filter raises on stop; logging must
    propagate that out of the logger.info() call (filters, unlike emit
    errors, are not swallowed by logging's error handling)."""
    cap = OutputCapture()
    logger = logging.getLogger("test_ui_stop_filterprop")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(io.StringIO())

    def _filter(record):
        if cap.stop_requested:
            raise AgentStoppedError("Agent stopped by user")
        return True

    handler.addFilter(_filter)
    logger.addHandler(handler)
    try:
        logger.info("fine before stop")
        cap.request_stop()
        try:
            logger.info("must raise")
        except AgentStoppedError:
            return
        raise AssertionError("logging path did not propagate the stop")
    finally:
        logger.removeHandler(handler)


def test_kill_subprocesses_includes_registered_fanout_workers():
    """Stop must also kill subprocesses spawned by best-of-N candidate
    worker threads (registered via log_context.register_worker), not just
    the chat thread's own."""
    from scilink.executors import (_register_subprocess,
                                   kill_subprocesses_for_thread)
    from scilink.utils.log_context import register_worker, unregister_worker

    parent_tid = threading.get_ident()
    proc_holder = {}
    started = threading.Event()

    def worker():
        register_worker(parent_tid, "cand_00", prefix=True)
        try:
            proc = subprocess.Popen([sys.executable, "-c",
                                     "import time; time.sleep(60)"])
            proc_holder["proc"] = proc
            _register_subprocess(proc)
            started.set()
            proc.wait()
        finally:
            unregister_worker()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert started.wait(timeout=10)
    kill_subprocesses_for_thread(parent_tid)  # what request_stop() calls
    t.join(timeout=10)
    assert not t.is_alive(), "worker's subprocess was not killed on stop"
    assert proc_holder["proc"].returncode is not None


def _tc(tcid, name="run_analysis"):
    return {"id": tcid, "type": "function",
            "function": {"name": name, "arguments": "{}"}}


def test_repair_inserts_synthetic_result_for_dangling_tool_call():
    """The post-stop resume bug: history ends with assistant tool_calls and
    no tool result — Bedrock rejects the next turn. Repair must insert a
    synthetic 'interrupted' tool result."""
    from scilink.utils.tool_media import repair_dangling_tool_calls
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "analyze this"},
        {"role": "assistant", "content": None, "tool_calls": [_tc("t1")]},
    ]
    out = repair_dangling_tool_calls(msgs)
    assert len(out) == 4
    assert out[3]["role"] == "tool" and out[3]["tool_call_id"] == "t1"
    assert "interrupted" in out[3]["content"].lower()


def test_repair_fills_only_missing_of_parallel_tool_calls():
    from scilink.utils.tool_media import repair_dangling_tool_calls
    msgs = [
        {"role": "assistant", "content": None,
         "tool_calls": [_tc("t1"), _tc("t2")]},
        {"role": "tool", "tool_call_id": "t1", "content": "done"},
    ]
    out = repair_dangling_tool_calls(msgs)
    assert [m.get("tool_call_id") for m in out[1:]] == ["t1", "t2"]
    assert out[1]["content"] == "done"
    assert "interrupted" in out[2]["content"].lower()


def test_repair_drops_orphan_tool_messages():
    """History trimming can slice a pair apart the other way: a tool result
    whose assistant tool_calls message was trimmed away."""
    from scilink.utils.tool_media import repair_dangling_tool_calls
    msgs = [
        {"role": "tool", "tool_call_id": "gone", "content": "orphan"},
        {"role": "user", "content": "hi"},
    ]
    out = repair_dangling_tool_calls(msgs)
    assert out == [{"role": "user", "content": "hi"}]


def test_repair_noop_on_healthy_history():
    from scilink.utils.tool_media import repair_dangling_tool_calls
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "analyze"},
        {"role": "assistant", "content": "thinking", "tool_calls": [_tc("t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
        {"role": "assistant", "content": "answer"},
    ]
    assert repair_dangling_tool_calls(msgs) == msgs


def test_agent_chat_catchall_does_not_eat_stop():
    """The orchestrator's chat() wraps everything in ``except Exception``
    and returns an error string; a stop must pass through it instead."""

    def chat_like(fn):
        try:
            return fn()
        except Exception as e:  # the real catch-all at chat()'s top level
            return f"Error: {e}"

    def raises_stop():
        raise AgentStoppedError("Agent stopped by user")

    try:
        result = chat_like(raises_stop)
    except AgentStoppedError:
        return
    raise AssertionError(f"chat() catch-all swallowed the stop: {result!r}")
