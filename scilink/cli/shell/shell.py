"""The prompt loop.

One prompt_toolkit session (history, slash-command and path completion,
Alt+Enter for a newline, Ctrl+O to toggle verbose narration) over one
mode adapter. A line starting with ``/`` is a command; anything else is a
turn. Ctrl+C at the prompt clears the line; during a turn it stops the
turn (``turn.py``); Ctrl+D or ``/quit`` saves a checkpoint and exits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console

from scilink.skills.loader import scilink_home
from scilink.ui import vocabulary as V

from . import bootstrap
from .channel import Widgets, ask_secret
from .commands import Registry, core_commands
from .render import Renderer
from .sessions import pick_session
from .turn import quiet_console_logging, run_turn, save_checkpoint_quietly


class Shell:
    def __init__(self, adapter, args, *, console: Optional[Console] = None,
                 pt_input=None, pt_output=None) -> None:
        self.adapter = adapter
        self.args = args
        self.mode = adapter.key
        self.console = console or Console(highlight=False)
        self._pt_kwargs: Dict[str, Any] = {}
        if pt_input is not None:
            self._pt_kwargs["input"] = pt_input
        if pt_output is not None:
            self._pt_kwargs["output"] = pt_output
        self.commands = Registry()
        self.commands.extend(core_commands())
        self.commands.extend(adapter.extra_commands())
        self.renderer = Renderer(self.console, verbose=bool(getattr(args, "verbose", False)))
        self.renderer.stop_message = V.stop_message(self.mode)
        self.agent = None
        self.session_dir: Optional[Path] = None
        self.creds: Optional[bootstrap.Credentials] = None
        self.extras: Dict[str, Any] = {}
        self.totals = {"calls": 0, "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
        self._quit = False
        # A plain session for bootstrap questions and the HITL widgets; the
        # chat session (history, completion, bindings) is built in run().
        self._plain_session = PromptSession(**self._pt_kwargs)
        self.prompt_session = self._plain_session
        self.widgets = Widgets(self.console, self._plain_session)

    # ── bootstrap ──────────────────────────────────────────────

    def ask(self, prompt: str, *, secret: bool = False, default: str = "") -> str:
        return ask_secret(self._plain_session, prompt, secret=secret, default=default)

    def model_label(self) -> str:
        model = getattr(self.args, "model", "?")
        if self.creds is None:
            return model
        return f"{model} via {self.creds.base_url}" if self.creds.base_url \
            else f"{model} ({self.creds.provider})"

    def _choose_session_dir(self):
        """(restore, path) from --session-dir / --resume / --restore."""
        args = self.args
        resume = getattr(args, "resume", None)
        restore = bool(getattr(args, "restore", False))
        if getattr(args, "session_dir", None):
            return restore or bool(resume), Path(args.session_dir)
        if isinstance(resume, str):
            return True, Path(resume)
        if resume is True or restore:
            pick = pick_session(self.console, self._plain_session, Path.cwd(), self.mode)
            if pick:
                return True, Path(pick)
            self.console.print("[yellow]Starting a new session instead.[/]")
        return False, bootstrap.new_session_dir(self.mode)

    def _bootstrap(self) -> None:
        args = self.args
        self.creds = bootstrap.resolve_llm_credentials(
            args.model, getattr(args, "base_url", None), getattr(args, "api_key", None), self.ask)
        restore, session_dir = self._choose_session_dir()
        session_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir
        self.widgets.session_dir = str(session_dir)
        bootstrap.ensure_sandbox_consent(assume_yes=bool(getattr(args, "yes", False)),
                                         ask=self.ask, console=self.console)
        if restore:
            setattr(args, "restore", True)
        self.extras = self.adapter.prepare(args, self.ask)
        with self.console.status("[dim]Initializing agent…[/]", spinner="dots"), \
                quiet_console_logging():
            self.agent = self.adapter.build(args, self.creds, session_dir,
                                            restore=restore, extras=self.extras)
        bootstrap.register_extras(
            self.agent,
            skill_files=getattr(args, "skill_files", None) or (),
            tool_files=getattr(args, "tool_files", None) or (),
            agent_files=getattr(args, "agent_files", None) or (),
            mcp_entries=getattr(args, "mcp_servers", None) or (),
            console=self.console)

    def resume(self, target: str) -> None:
        """/resume <id>: rebuild the agent from that session's checkpoint."""
        path = Path(target)
        if not path.exists():
            self.console.print(f"[red]No such session directory:[/] {path}")
            return
        setattr(self.args, "restore", True)
        with self.console.status("[dim]Restoring session…[/]", spinner="dots"), \
                quiet_console_logging():
            self.agent = self.adapter.build(self.args, self.creds, path, restore=True,
                                            extras=self.extras)
        self.session_dir = path
        self.widgets.session_dir = str(path)
        self.console.print(f"[green]✓[/] resumed [bold]{path.name}[/]")

    def _banner(self) -> None:
        m = V.mode(self.mode)
        # Two spaces after the emoji: the mode glyphs are double-width and
        # many terminals measure them as one cell, so text would collide.
        self.console.print(f"\n[bold]{m['emoji']}  SciLink {m['name']}[/]  "
                           f"[dim]{self.model_label()} · {self.session_dir}[/]")
        self.console.print(f"[dim]{m['blurb']}. Type /help for commands.[/]\n")

    # ── turns ──────────────────────────────────────────────────

    def turn(self, text: str) -> None:
        res = run_turn(self.agent, text, session_dir=str(self.session_dir),
                       renderer=self.renderer, ask_question=self.widgets.ask)
        for k in self.totals:
            self.totals[k] += res.tokens.get(k, 0)
        if res.error and not self.renderer.verbose:
            self.console.print("[dim]Ctrl+O or /verbose shows the full narration.[/]")

    # ── the loop ───────────────────────────────────────────────

    def _toolbar(self):
        m = V.mode(self.mode)
        autonomy = self.adapter.get_autonomy(self.agent) if self.agent is not None else "?"
        verbose = "verbose on" if self.renderer.verbose else "verbose off"
        # A blank line above the status line, and no reverse-video bar.
        return HTML(f"\n {m['emoji']}  <b>{m['name']}</b> · {autonomy} · "
                    f"{self.session_dir.name if self.session_dir else ''} · "
                    f"{getattr(self.args, 'model', '')} · {verbose}")

    def _completer(self) -> NestedCompleter:
        options: Dict[str, Any] = {}
        for c in self.commands.commands():
            sub = None
            if c.completes_paths:
                sub = PathCompleter(expanduser=True)
            elif c.name == "/mode":
                sub = WordCompleter(V.autonomy_options(self.mode))
            elif c.name == "/memory":
                sub = WordCompleter(["on", "off"])
            for name in (c.name, *c.aliases):
                options[name] = sub
        return NestedCompleter.from_nested_dict(options)

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-o")
        def _toggle_verbose(event):
            self.renderer.verbose = not self.renderer.verbose
            event.app.invalidate()

        @kb.add("enter")
        def _submit(event):
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        return kb

    def _history(self) -> Optional[FileHistory]:
        try:
            d = scilink_home() / "history"
            d.mkdir(parents=True, exist_ok=True)
            return FileHistory(str(d / f"{self.mode}.txt"))
        except OSError:
            return None

    def request_quit(self) -> None:
        self._quit = True

    def _loop(self) -> None:
        placeholder = V.mode(self.mode)["placeholder"]
        self.prompt_session = PromptSession(
            history=self._history(), completer=self._completer(),
            key_bindings=self._bindings(), multiline=True, prompt_continuation="  ",
            bottom_toolbar=self._toolbar, complete_while_typing=False,
            style=Style.from_dict({"bottom-toolbar": "noreverse",
                                   "bottom-toolbar.text": "noreverse fg:ansibrightblack"}),
            **self._pt_kwargs)
        while not self._quit:
            try:
                text = self.prompt_session.prompt(
                    HTML("<b><ansicyan>❯</ansicyan></b> "),
                    placeholder=HTML(f"<i><ansibrightblack>{placeholder}</ansibrightblack></i>"))
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.request_quit()
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not self.commands.dispatch(self, text):
                    self.console.print(f"[red]Unknown command[/] {text.split()[0]} — /help lists them")
                continue
            self.turn(text)

    def _shutdown(self) -> None:
        if self.agent is not None:
            save_checkpoint_quietly(self.agent)
        self.console.print(f"\n👋 Session saved: [dim]{self.session_dir}[/]")

    def run(self) -> int:
        try:
            self._bootstrap()
        except bootstrap.BootstrapError as e:
            self.console.print(f"[red]{e}[/]")
            return 2
        except (EOFError, KeyboardInterrupt):
            self.console.print("\n[dim]cancelled[/]")
            return 130
        self._banner()
        for text in self.adapter.initial_turns(self.args, self.agent):
            self.console.print(f"[bold cyan]❯[/] {text}")
            self.turn(text)
            if self._quit:
                break
        self._loop()
        self._shutdown()
        return 0


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
