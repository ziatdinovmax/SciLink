"""Phase-0 chokepoint tests: scilink.hitl routing + the no-raw-input gate."""

import builtins
import re
import threading
from pathlib import Path

import pytest

from scilink import hitl
from scilink.hitl import (
    ConsoleChannel,
    FeedbackRequest,
    request_human_feedback,
    set_default_channel,
    set_thread_channel,
    use_channel,
)

SCILINK_ROOT = Path(hitl.__file__).parent


class StubChannel:
    def __init__(self, answer=""):
        self.answer = answer
        self.requests = []

    def ask(self, req):
        self.requests.append(req)
        return self.answer


@pytest.fixture(autouse=True)
def _reset_channels():
    yield
    set_default_channel(None)
    set_thread_channel(None)


# ---------------------------------------------------------------- core API

def test_console_channel_calls_builtins_input_with_exact_prompt(monkeypatch):
    seen = {}

    def fake_input(prompt=""):
        seen["prompt"] = prompt
        return "  hello  "

    monkeypatch.setattr(builtins, "input", fake_input)
    answer = request_human_feedback("\n🤔 Your feedback (or Enter to accept): ")
    # Raw return — no stripping in the chokepoint (call sites strip).
    assert answer == "  hello  "
    assert seen["prompt"] == "\n🤔 Your feedback (or Enter to accept): "


def test_console_channel_resolves_input_at_call_time(monkeypatch):
    """A front-end that monkeypatches builtins.input (the Streamlit UI)
    must keep intercepting prompts after the chokepoint migration."""
    monkeypatch.setattr(builtins, "input", lambda prompt="": "patched")
    assert ConsoleChannel().ask(FeedbackRequest(prompt="x")) == "patched"


def test_request_metadata_reaches_channel():
    stub = StubChannel(answer="ok")
    set_default_channel(stub)
    out = request_human_feedback(
        "p: ", kind="bestofn_select", options=["1", "2"], default="",
        context="ctx", origin={"stage": "s"},
    )
    assert out == "ok"
    (req,) = stub.requests
    assert req.kind == "bestofn_select"
    assert req.options == ["1", "2"]
    assert req.context == "ctx"
    assert req.origin == {"stage": "s"}
    assert req.id.startswith("q_")


def test_thread_local_channel_wins_and_clears():
    default = StubChannel(answer="default")
    local = StubChannel(answer="local")
    set_default_channel(default)
    set_thread_channel(local)
    assert request_human_feedback("p") == "local"
    set_thread_channel(None)
    assert request_human_feedback("p") == "default"


def test_use_channel_scopes_and_restores():
    outer = StubChannel(answer="outer")
    inner = StubChannel(answer="inner")
    set_thread_channel(outer)
    with use_channel(inner):
        assert request_human_feedback("p") == "inner"
    assert request_human_feedback("p") == "outer"


def test_thread_isolation():
    set_default_channel(StubChannel(answer="main-default"))
    results = {}

    def worker():
        set_thread_channel(StubChannel(answer="worker-local"))
        results["worker"] = request_human_feedback("p")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert results["worker"] == "worker-local"
    assert request_human_feedback("p") == "main-default"


def test_channel_exceptions_propagate():
    class EOFChannel:
        def ask(self, req):
            raise EOFError

    set_default_channel(EOFChannel())
    with pytest.raises(EOFError):
        request_human_feedback("p")


# ------------------------------------------------------- converted call sites

def test_get_user_feedback_routes_through_chokepoint():
    from scilink.agents.planning_agents import user_interface

    stub = StubChannel(answer="")
    set_default_channel(stub)
    assert user_interface.get_user_feedback() is None
    (req,) = stub.requests
    assert req.kind == "approve_or_revise"

    stub2 = StubChannel(answer="revise it")
    set_default_channel(stub2)
    assert user_interface.get_user_feedback() == "revise it"


def test_get_dataset_description_routes_and_keeps_eof_fallback():
    from scilink.agents.planning_agents import user_interface

    stub = StubChannel(answer=" my dataset ")
    set_default_channel(stub)
    assert user_interface.get_dataset_description("f.csv") == "my dataset"
    (req,) = stub.requests
    assert req.kind == "dataset_description"
    assert req.origin["filename"] == "f.csv"

    class EOFChannel:
        def ask(self, req):
            raise EOFError

    set_default_channel(EOFChannel())
    assert user_interface.get_dataset_description("f.csv") == ""


# ------------------------------------------------------------- the grep gate

# Files allowed to call input() directly: CLI bootstrap/REPL prompts,
# destructive-op confirms, and the TTY-gated sandbox consent.
_ALLOWED_RAW_INPUT = {
    "cli/plan.py", "cli/analyze.py", "cli/simulate.py", "cli/meta.py",
    "cli/memory.py", "cli/kb.py", "executors.py", "hitl.py",
}

_INPUT_CALL = re.compile(r"(?<![\w.])input\s*\(")


def _strip_strings_and_comments(source: str) -> str:
    import io
    import tokenize

    skip = {tokenize.STRING, tokenize.COMMENT}
    # Python 3.12+ tokenizes f-string text separately; skip it too so prose
    # like "3 input(s):" inside an f-string doesn't trip the gate.
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            skip.add(getattr(tokenize, name))
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in skip:
                continue
            out.append(tok.string)
    except tokenize.TokenizeError:
        return source
    return " ".join(out)


def test_no_raw_input_outside_allowed_files():
    offenders = []
    for py in SCILINK_ROOT.rglob("*.py"):
        rel = py.relative_to(SCILINK_ROOT).as_posix()
        if rel in _ALLOWED_RAW_INPUT or rel.startswith("ui/"):
            continue
        code = _strip_strings_and_comments(py.read_text(encoding="utf-8"))
        if _INPUT_CALL.search(code):
            offenders.append(rel)
    assert offenders == [], (
        f"raw input() call in {offenders} — route through "
        "scilink.hitl.request_human_feedback instead"
    )
