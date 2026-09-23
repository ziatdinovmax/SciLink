"""The shell end to end over the fake adapter: bootstrap, a seeded turn,
slash commands (core + aliases), and quit — driven through a pipe input."""

import io

import pytest
from prompt_toolkit.input import create_pipe_input
from contextlib import ExitStack
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from scilink.cli.shell.commands import Registry, core_commands
from scilink.cli.shell.shell import Shell

from shell_fakes import FakeAdapter, make_args


def _run_shell(tmp_path, monkeypatch, keys, argv=(), **orch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    adapter = FakeAdapter(**orch)
    args = make_args(adapter, ["--yes", *argv])
    stack = ExitStack()
    pipe = stack.enter_context(create_pipe_input())
    pipe.send_text(keys)
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100, highlight=False)
    shell = Shell(adapter, args, console=console, pt_input=pipe, pt_output=DummyOutput())
    with stack:
        code = shell.run()
    return code, buf.getvalue(), shell


def test_registry_aliases_and_dispatch():
    reg = Registry()
    reg.extend(core_commands())
    assert reg.get("/state").name == "/status"
    assert reg.get("/autonomy").name == "/mode"
    assert reg.get("/exit").name == "/quit"
    assert reg.get("/nope") is None
    assert "/help" in reg.names()

    class S:
        console = Console(file=io.StringIO(), force_terminal=False)
        commands = reg
        renderer = type("R", (), {"verbose": False})()

    assert reg.dispatch(S, "/verbose") is True
    assert S.renderer.verbose is True
    assert reg.dispatch(S, "hello") is False


def test_shell_status_mode_and_quit(tmp_path, monkeypatch):
    code, out, shell = _run_shell(
        tmp_path, monkeypatch,
        "/status\r/mode autonomous\r/state\r/mode nonsense\r/help\r/cost\r/quit\r",
        ask=False)
    assert code == 0
    assert "Mission Control" in out            # banner + status from the vocabulary
    assert "Fake field" in out                 # adapter status fields
    assert "autonomy set to autonomous" in out
    assert "Unknown autonomy level" in out
    assert "/help" in out and "/status" in out # help table
    assert "This session:" in out              # /cost
    assert "Session saved" in out
    assert shell.agent.checkpoints >= 1        # quit saves a checkpoint
    assert (tmp_path / "home" / "history" / "meta.txt").exists()


def test_seeded_turn_then_chat_turn(tmp_path, monkeypatch):
    code, out, shell = _run_shell(
        tmp_path, monkeypatch, "fit it again\r\r/quit\r", argv=["--message", "start here"],
        ask=False)
    assert code == 0
    assert "start here" in out                 # the seeded turn is echoed
    assert out.count("The answer to") == 2     # two answers, each once
    assert shell.agent.message_count == 2
    assert shell.totals["calls"] == 0          # no real LLM calls in the fake


def test_question_answered_from_the_prompt(tmp_path, monkeypatch):
    # The turn parks a question; the next line of input answers it (Enter = accept).
    code, out, shell = _run_shell(tmp_path, monkeypatch, "go\r\r/quit\r")
    assert code == 0
    assert "Enter = Approve plan" in out
    assert shell.agent.answers == [""]


def test_unknown_command_and_session_dir(tmp_path, monkeypatch):
    sdir = tmp_path / "my_session"
    code, out, shell = _run_shell(tmp_path, monkeypatch, "/bogus\r/quit\r",
                                  argv=["--session-dir", str(sdir)], ask=False)
    assert code == 0
    assert "Unknown command" in out
    assert shell.session_dir == sdir and sdir.exists()


def test_ctrl_d_quits(tmp_path, monkeypatch):
    code, out, _ = _run_shell(tmp_path, monkeypatch, "\x04", ask=False)
    assert code == 0 and "Session saved" in out


def test_context_usage_matches_the_orchestrators_trim_rule():
    from scilink.cli.shell.shell import context_usage

    class Agent:
        MAX_HISTORY_MESSAGES = 100
        TRIM_HYSTERESIS = 20
        messages = [{"role": "system"}] + [{"role": "user"}] * 29

    assert context_usage(Agent()) == (30, 120)
    assert context_usage(object()) is None


def test_arrows_move_between_lines_of_a_multiline_draft(tmp_path, monkeypatch):
    """Alt+Enter adds a line; Up moves the cursor to the line above (history
    is browsed only from the first line), so the draft can be edited."""
    keys = "a\x1b\rb\x1b[AX\r/quit\r"     # a, Alt+Enter, b, Up, X, Enter -> "aX\nb"
    code, out, shell = _run_shell(tmp_path, monkeypatch, keys, ask=False)
    assert code == 0
    assert "The answer to aX" in out.replace("\n", " ") or "aX" in out


def test_exit_prints_the_resume_command_and_registers(tmp_path, monkeypatch):
    from scilink import sessions as S
    code, out, shell = _run_shell(tmp_path, monkeypatch, "/quit\r", ask=False)
    assert f"--resume {shell.session_dir.name}" in out
    assert S.resolve_session(shell.session_dir.name, "meta") == shell.session_dir.resolve()
    sdir = tmp_path / "elsewhere" / "s"
    code, out, shell = _run_shell(tmp_path, monkeypatch, "/quit\r",
                                  argv=["--session-dir", str(sdir)], ask=False)
    assert "--resume s" in out                       # indexed, so the id is enough
    assert S.resolve_session("s", "meta") == sdir.resolve()


def test_status_bar_fits_a_narrow_terminal(tmp_path, monkeypatch):
    """The bar drops the model, then the session name, before it would push
    the version off the right edge."""
    from prompt_toolkit.formatted_text import to_plain_text
    monkeypatch.chdir(tmp_path)
    adapter = FakeAdapter()
    args = make_args(adapter, ["--yes", "--model", "bedrock/us.anthropic.claude-opus-4-8"])
    console = Console(file=io.StringIO(), force_terminal=False, width=80)
    shell = Shell(adapter, args, console=console, pt_output=DummyOutput())
    shell.agent = adapter.build(args, None, tmp_path / "meta_session_20260920_120000",
                                restore=False, extras={})
    shell.session_dir = tmp_path / "meta_session_20260920_120000"
    lines = to_plain_text(shell._toolbar()).split("\n")
    assert lines[-1].endswith("scilink " + __import__("scilink.cli.shell.shell", fromlist=["x"]).scilink_version() + " ")
    assert len(lines[-1]) <= 80 - 4
    assert "autopilot" in lines[-1] and "verbose off" in lines[-1]
    assert "bedrock" not in lines[-1]           # the model was dropped first


def test_name_command_sets_the_session_name_and_the_picker_shows_it(tmp_path, monkeypatch):
    from scilink import sessions as S
    from scilink.ui.session_meta import load_session_name
    code, out, shell = _run_shell(tmp_path, monkeypatch, "/name Grain sizes in 304 steel\r/status\r/quit\r",
                                  ask=False)
    assert code == 0 and "session named" in out
    assert load_session_name(shell.session_dir) == "Grain sizes in 304 steel"
    import re as _re
    assert _re.search(r"Name\s+Grain sizes in 304 steel", out)               # /status shows it
    listed = S.list_sessions("meta", root=tmp_path)
    assert listed[0]["label"].startswith("Grain sizes in 304 steel")       # the picker label


def test_queued_messages_run_after_the_turn(tmp_path, monkeypatch):
    """A message queued mid-turn runs next, echoed at the prompt; a slash
    command in the queue is dispatched; a draft waits in the prompt."""
    from scilink.cli.shell import shell as shell_mod
    from scilink.cli.shell.turn import TurnResult
    real = shell_mod.run_turn
    calls = []

    def fake_run_turn(agent, text, **kw):
        calls.append(text)
        res = real(agent, text, **kw)
        if text == "first":
            res.queued, res.draft = ["second", "/status"], "third"
        return res

    monkeypatch.setattr(shell_mod, "run_turn", fake_run_turn)
    code, out, shell = _run_shell(tmp_path, monkeypatch, "first\r\r/quit\r", ask=False)
    assert code == 0
    assert calls == ["first", "second", "third"]      # the draft was prefilled, sent by the next Enter
    assert "❯ second" in out and "Fake field" in out  # echoed; /status ran from the queue


def test_after_ctrl_c_the_queue_goes_back_to_the_prompt(tmp_path, monkeypatch):
    from scilink.cli.shell import shell as shell_mod
    real = shell_mod.run_turn
    calls = []

    def fake_run_turn(agent, text, **kw):
        calls.append(text)
        res = real(agent, text, **kw)
        if text == "first":
            res.stopped, res.queued, res.draft = True, ["second"], "more"
        return res

    monkeypatch.setattr(shell_mod, "run_turn", fake_run_turn)
    code, out, shell = _run_shell(tmp_path, monkeypatch, "first\r\r/quit\r", ask=False)
    assert calls == ["first", "second\nmore"]         # nothing ran on its own; the prompt held it
