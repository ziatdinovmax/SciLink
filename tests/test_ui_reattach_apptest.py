"""End-to-end reattach through Streamlit's AppTest harness.

Simulates the real failure: an agent turn is running in a background thread
when the browser session is lost. A NEW AppTest run (fresh session_state,
same process) must show the reattach banner; clicking it must hand the new
session the live task, whose completion then lands in the chat.
"""
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from scilink.ui import registry  # noqa: E402
from scilink.ui.state import ChatTask  # noqa: E402

APP = "scilink/ui/app.py"
TIMEOUT = 60


class _StubAgent:
    model = "stub-model"
    base_dir = "."


def _seed_running_entry(tmp_path):
    registry._SESSIONS.clear()
    task = ChatTask(is_running=True)
    sdir = tmp_path / "meta_session_20990101_000000"
    sdir.mkdir()
    registry.sync_from({
        "session_dir": str(sdir), "agent": _StubAgent(),
        "agent_config": {"model": "stub-model", "mode": "autonomous"},
        "chat_messages": [{"role": "user", "content": "long job please"}],
        "chat_task": task, "known_images": set(), "app_mode": "meta",
    })
    return sdir, task


def test_banner_lists_live_session_and_reattaches(tmp_path):
    sdir, task = _seed_running_entry(tmp_path)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    # Fresh session (disconnected browser): welcome screen with the banner
    assert not at.session_state["agent_initialized"]
    assert any("live session" in str(i.value) for i in at.info)
    btns = [b for b in at.button if b.key == f"reattach_{sdir}"]
    assert btns, "reattach button not rendered"

    at2 = btns[0].click().run()
    assert at2.session_state["agent_initialized"] is True
    assert at2.session_state["session_dir"] == str(sdir)
    assert at2.session_state["chat_task"] is task
    assert at2.session_state["chat_messages"][0]["content"] == "long job please"


def test_completion_after_reattach_lands_in_chat(tmp_path):
    sdir, task = _seed_running_entry(tmp_path)
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at = [b for b in at.button if b.key == f"reattach_{sdir}"][0].click().run()

    # the "background thread" finishes while attached
    task.result = "the long answer"
    task.verbose_log = "log"
    task.is_running = False
    time.sleep(0.1)
    at = at.run()   # monitor fragment consumes the completion

    msgs = at.session_state["chat_messages"]
    assert any(m.get("role") == "assistant" and m.get("content") == "the long answer"
               for m in msgs)
    fresh = at.session_state["chat_task"]
    assert fresh is not task and not fresh.is_running

    # duplicate-consumption guard: a second observer holding the old task
    at.session_state["chat_task"] = task
    at = at.run()
    n = sum(1 for m in at.session_state["chat_messages"]
            if m.get("role") == "assistant" and m.get("content") == "the long answer")
    assert n == 1


def test_no_banner_without_live_sessions():
    registry._SESSIONS.clear()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not any("live session" in str(i.value) for i in at.info)
