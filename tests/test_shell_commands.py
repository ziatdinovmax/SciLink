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
