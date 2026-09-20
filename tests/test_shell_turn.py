"""One shell turn: narration rendered, the question parked and answered,
the answer printed once, the stop path, the error path."""

import io

import pytest
from rich.console import Console

from scilink.cli.shell.render import Renderer
from scilink.cli.shell.turn import run_turn

from shell_fakes import FakeOrchestrator


def _console():
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=100, highlight=False), buf


def _run(agent, tmp_path, *, verbose=False, answer="", on_tick=None):
    console, buf = _console()
    renderer = Renderer(console, verbose=verbose)
    renderer.stop_message = "Analysis stopped by user."
    asked = []

    def ask_question(presented):
        asked.append(presented)
        return answer

    res = run_turn(agent, "fit the spectrum", session_dir=str(tmp_path),
                   renderer=renderer, ask_question=ask_question, on_tick=on_tick)
    return res, buf.getvalue(), asked


def test_turn_renders_narration_and_answer_once(tmp_path):
    agent = FakeOrchestrator(str(tmp_path))
    res, out, asked = _run(agent, tmp_path)
    assert res.error is None and not res.stopped
    assert "delegate_to_analysis" in out            # tool call shown
    assert "Deciding which specialist fits" in out   # thought shown
    assert "the data looks like a spectrum" in out   # thought continuation
    assert "Waiting for meta-orchestrator" not in out # verbose line hidden
    assert "Executing generated code" not in out     # plain line hidden
    assert out.count("42") == 1                      # the answer, rendered once
    assert "Result" in out
    assert len(asked) == 1
    assert asked[0]["widget"] == "generic"
    assert asked[0]["labels"]["accept"] == "Approve plan"
    assert agent.answers == [""]
    assert agent.checkpoints == 1                    # per-turn checkpoint
    assert "Checkpoint saved" not in out             # ... quietly
    assert "REQUESTING FEEDBACK" in res.log          # the raw log is kept


def test_verbose_shows_hidden_lines(tmp_path):
    agent = FakeOrchestrator(str(tmp_path))
    _, out, _ = _run(agent, tmp_path, verbose=True)
    assert "Waiting for meta-orchestrator" in out
    assert "Executing generated code" in out


def test_self_printed_answer_is_not_duplicated(tmp_path):
    agent = FakeOrchestrator(str(tmp_path), self_prints_answer=True)
    _, out, _ = _run(agent, tmp_path, verbose=True)
    assert out.count("42") == 1
    assert "🤖 Agent:" not in out


def test_question_answer_reaches_the_agent(tmp_path):
    agent = FakeOrchestrator(str(tmp_path))
    _run(agent, tmp_path, answer="tighten the window")
    assert agent.answers == ["tighten the window"]


def test_error_is_reported_not_raised(tmp_path):
    agent = FakeOrchestrator(str(tmp_path), raise_error=True)
    res, out, _ = _run(agent, tmp_path)
    assert res.error == "RuntimeError: boom"
    assert "Error:" in out and "boom" in out
    assert agent.checkpoints == 1


def test_ctrl_c_stops_the_turn(tmp_path):
    agent = FakeOrchestrator(str(tmp_path), block_until_stopped=True)
    ticks = {"n": 0}

    def on_tick():
        ticks["n"] += 1
        if ticks["n"] == 3:
            raise KeyboardInterrupt

    res, out, _ = _run(agent, tmp_path, on_tick=on_tick)
    assert res.stopped
    assert "Analysis stopped by user." in out
    assert agent.checkpoints == 0                    # no checkpoint on a stop


def test_accounting_line_present(tmp_path):
    agent = FakeOrchestrator(str(tmp_path), ask=False)
    res, out, _ = _run(agent, tmp_path)
    assert "· " in out and out.rstrip().splitlines()[-1].endswith("s")   # accounting is the last text line
    assert isinstance(res.tokens, dict) and "calls" in res.tokens


def test_ctrl_c_inside_the_question_stops(tmp_path):
    agent = FakeOrchestrator(str(tmp_path))
    console, buf = _console()
    renderer = Renderer(console)
    renderer.stop_message = "Analysis stopped by user."

    def ask_question(presented):
        raise KeyboardInterrupt

    res = run_turn(agent, "x", session_dir=str(tmp_path), renderer=renderer,
                   ask_question=ask_question)
    assert res.stopped
    assert "stopped by user" in buf.getvalue()


def test_live_status_row_does_not_hijack_the_capture(tmp_path):
    """With the status row active (a terminal console) the agent thread's
    prints must still land in the capture and render — rich's Live would
    otherwise redirect sys.stdout to its own proxy (observed live: tool-call
    and handoff lines vanished after a delegation's first lines)."""
    import time as _time

    class MetaLike(FakeOrchestrator):
        def chat(self, text):
            print("  🔧 Calling tool: delegate_to_analysis")
            _time.sleep(0.3)                       # the Live has started by now
            print("  \x1b[1;33m🧪 Delegating to analysis specialist: examine\x1b[0m")
            print("\n\x1b[1;33m🤖\u2063 Analysis specialist:\x1b[0m")
            print("The file is a 512x512 image.")
            print("  🔧 Calling tool: summarize_session_state")
            return "## Result\n\nThe answer is **42**."

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, highlight=False)
    renderer = Renderer(console)
    renderer.stop_message = "x"
    res = run_turn(MetaLike(str(tmp_path), ask=False), "go", session_dir=str(tmp_path),
                   renderer=renderer, ask_question=lambda q: "")
    assert renderer._live is None                  # stopped at the end
    for needle in ("Delegating to analysis specialist", "Analysis specialist:",
                   "512x512", "summarize_session_state"):
        assert needle in res.log, needle           # captured
    import re as _re
    shown = _re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buf.getvalue())
    for needle in ("delegate_to_analysis", "Delegating to analysis specialist",
                   "Analysis specialist:", "512x512", "summarize_session_state", "42"):
        assert needle in shown, needle             # rendered


def test_real_sigint_stops_the_turn(tmp_path):
    """A real SIGINT (what the terminal sends for Ctrl+C) stops the turn
    through the flag-setting handler, without relying on where the
    KeyboardInterrupt would have surfaced."""
    import os, signal
    agent = FakeOrchestrator(str(tmp_path), block_until_stopped=True)
    ticks = {"n": 0}

    def on_tick():
        ticks["n"] += 1
        if ticks["n"] == 3:
            os.kill(os.getpid(), signal.SIGINT)

    console, buf = _console()
    renderer = Renderer(console)
    renderer.stop_message = "Analysis stopped by user."
    res = run_turn(agent, "x", session_dir=str(tmp_path), renderer=renderer,
                   ask_question=lambda q: "", on_tick=on_tick)
    assert res.stopped and "stopped by user" in buf.getvalue()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored


def test_ctrl_o_mid_turn_reveals_hidden_lines(tmp_path):
    """Ctrl+O during a turn turns verbose on: the lines hidden so far are
    replayed and later verbose lines stream; a second Ctrl+O turns it off."""
    import time as _time

    class Slow(FakeOrchestrator):
        def chat(self, text):
            print("  ⏳ Waiting for meta-orchestrator response ...")   # verbose
            _time.sleep(0.35)
            print("second verbose line")
            _time.sleep(0.35)
            print("third verbose line")
            return "done"

    ticks = {"n": 0}

    def read_key():
        # on once the first verbose line has been hidden (~0.2s), off at ~0.5s
        ticks["n"] += 1
        return "\x0f" if ticks["n"] in (3, 6) else None

    console, buf = _console()
    renderer = Renderer(console)
    renderer.stop_message = "x"
    run_turn(Slow(str(tmp_path), ask=False), "go", session_dir=str(tmp_path),
             renderer=renderer, ask_question=lambda q: "", read_key=read_key)
    out = buf.getvalue()
    assert "1 lines hidden so far" in out and "Waiting for meta-orchestrator" in out   # replayed
    assert "second verbose line" in out                                               # streamed while on
    assert "third verbose line" not in out                                            # hidden again
    assert "Verbose output: on" in out and "Verbose output: off" in out


def test_log_records_reach_the_capture_and_the_activity_label(tmp_path):
    """The analysis agents narrate their inner loop through logging, not
    print: those records must land in the capture (verbose lines) and drive
    the activity label even while the agents' own console handlers are muted."""
    import logging

    class Logging(FakeOrchestrator):
        def chat(self, text):
            logging.getLogger("SomeController").info("   Executing Python script (timeout: 600s)...")
            logging.getLogger("SomeController").info("   Verification 2/7 (annealing level 1)...")
            logging.getLogger("SomeController").warning("⚠️ Literature Analysis disabled")
            return "done"

    console_handler = logging.StreamHandler(io.StringIO())   # the agents' basicConfig handler
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    try:
        labels = []
        console, buf = _console()
        renderer = Renderer(console)
        renderer.stop_message = "x"
        renderer.set_activity = lambda label: labels.append(label)
        res = run_turn(Logging(str(tmp_path), ask=False), "go", session_dir=str(tmp_path),
                       renderer=renderer, ask_question=lambda q: "")
    finally:
        logging.getLogger().removeHandler(console_handler)
    assert "Verification 2/7" in res.log and "Executing Python script" in res.log
    from scilink.ui.narration import current_activity
    # The reader labels the captured records (all three arrived in one poll,
    # so only the last label was pushed live).
    assert current_activity(res.log.split("⚠")[0]) == "Verification 2/7 · annealing level 1…"
    assert labels[-1] == "⚠️ Literature Analysis disabled"
    assert "Literature Analysis disabled" in buf.getvalue()      # a warning is shown
    assert "Executing Python script" not in console_handler.stream.getvalue()   # console muted
    assert console_handler.level == logging.INFO                  # ... and restored


def test_live_region_toggle_and_commit(tmp_path):
    """Terminal mode: verbose lines are absent from the region while off,
    appear once Ctrl+O flips the flag (region redraw), and the turn's block
    is committed once, in the verbosity in force, when the turn ends."""
    import re as _re
    import time as _time

    class Slow(FakeOrchestrator):
        def chat(self, text):
            print("  🔧 Calling tool: delegate_to_analysis")
            print("  ⏳ Waiting for meta-orchestrator response ...")   # verbose
            _time.sleep(0.4)
            print("second verbose line")
            _time.sleep(0.3)
            return "done"

    ticks = {"n": 0}

    def read_key():
        ticks["n"] += 1
        return "\x0f" if ticks["n"] == 3 else None      # on at ~0.3s, stays on

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=30, highlight=False)
    renderer = Renderer(console)
    renderer.stop_message = "x"
    res = run_turn(Slow(str(tmp_path), ask=False), "go", session_dir=str(tmp_path),
                   renderer=renderer, ask_question=lambda q: "", read_key=read_key)
    out = _re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buf.getvalue())
    first_frame = out[: out.index("Waiting for")] if "Waiting for" in out else out
    assert "delegate_to_analysis" in first_frame        # visible line drawn before the toggle
    assert "Waiting for meta-orchestrator" in out         # verbose line drawn after the toggle
    assert "second verbose line" in out
    assert renderer.verbose is True
    # The committed block (after the region is gone) holds every line once.
    committed = out[out.rindex("delegate_to_analysis"):]
    assert committed.count("Waiting for meta-orchestrator") == 1
    assert committed.count("second verbose line") == 1


def test_resume_after_a_question_does_not_erase_the_panel(tmp_path):
    """After a question the live region restarts below the prompt: the first
    frame must not move the cursor up over the panel (a reused Live would,
    by the height of its last frame before the pause)."""
    import re as _re
    import time as _time

    class Asking(FakeOrchestrator):
        def chat(self, text):
            for i in range(6):
                print(f"  🔧 Calling tool: step_{i}")      # a tall region before the question
            _time.sleep(0.4)
            print("=" * 40)
            print("🙋 REQUESTING FEEDBACK")
            answer = hitl.request_human_feedback("Review the plan:", kind="review_plan", default="")
            _time.sleep(0.3)
            print("  🔧 Calling tool: after_question")
            return "done"

    from scilink import hitl
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, highlight=False)
    renderer = Renderer(console)
    renderer.stop_message = "x"

    def ask(presented):
        console.print("PANEL LINE")           # what a question widget prints
        return ""

    run_turn(Asking(str(tmp_path)), "go", session_dir=str(tmp_path), renderer=renderer,
             ask_question=ask)
    out = buf.getvalue()
    after = out[out.index("PANEL LINE"):]
    # Cursor-up sequences after the panel may only be as tall as the frames
    # the restarted region actually drew (spinner row + at most 1 new line).
    ups = [int(m or 1) for m in _re.findall(r"\x1b\[(\d*)A", after)]
    assert all(n <= 2 for n in ups), ups
