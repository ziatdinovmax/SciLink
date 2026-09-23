"""``RoutedCapture(echo_console=False)`` keeps a turn's output in its buffer
only — the terminal shell renders the buffer itself, so the console tee
would print every line twice."""

import io
import sys
import threading

from scilink.server import stdout_router
from scilink.server.stdout_router import RoutedCapture


def _run_in_thread(fn):
    t = threading.Thread(target=fn)
    t.start()
    t.join()


def test_echo_off_writes_buffer_only(monkeypatch):
    console = io.StringIO()
    # Install the router over a fake console, then restore afterwards.
    monkeypatch.setattr(sys, "stdout", console)
    stdout_router.install()
    cap = RoutedCapture(echo_console=False)

    def turn():
        with cap:
            print("only in the buffer")

    _run_in_thread(turn)
    assert "only in the buffer" in cap.getvalue()
    assert console.getvalue() == ""


def test_echo_default_still_tees(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    stdout_router.install()
    cap = RoutedCapture()

    def turn():
        with cap:
            print("in both")

    _run_in_thread(turn)
    assert "in both" in cap.getvalue()
    assert "in both" in console.getvalue()


def test_unrouted_thread_passes_through(monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    stdout_router.install()
    cap = RoutedCapture(echo_console=False)
    with cap:
        pass  # registered and released on this thread
    print("after the turn")
    assert "after the turn" in console.getvalue()
    assert cap.getvalue() == ""
