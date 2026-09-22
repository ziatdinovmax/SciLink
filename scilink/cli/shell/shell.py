"""The prompt loop.

One prompt_toolkit session (history, slash-command and path completion,
Alt+Enter for a newline, Ctrl+O to toggle verbose narration) over one
mode adapter. A line starting with ``/`` is a command; anything else is a
turn. Ctrl+C at the prompt clears the line; during a turn it stops the
turn (``turn.py``); Ctrl+D or ``/quit`` saves a checkpoint and exits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from scilink.sessions import register_session, resolve_session, touch_session
from scilink.skills.loader import scilink_home
from scilink.ui.session_meta import generate_session_title, load_session_name, save_session_name
from scilink.ui import vocabulary as V

from . import bootstrap
from .channel import Widgets, ask_secret
from .commands import Registry, core_commands
from .render import Renderer
from .sessions import pick_session
from .turn import quiet_console_logging, run_turn, save_checkpoint_quietly


def scilink_version() -> str:
    """The running version: from the source checkout's pyproject when the
    package runs from one (an editable install's metadata goes stale),
    else from the installed metadata."""
    import re
    import scilink
    pyproject = Path(scilink.__file__).resolve().parent.parent / "pyproject.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version
        return version("scilink")
    except Exception:  # noqa: BLE001
        return "dev"


def context_usage(agent) -> Optional[tuple]:
    """(messages, limit) of the agent's chat history against the point where
    the orchestrators trim it — the same test all four use
    (``len(messages) > MAX_HISTORY_MESSAGES + TRIM_HYSTERESIS``). None when
    the agent does not expose those."""
    msgs = getattr(agent, "messages", None)
    limit = getattr(agent, "MAX_HISTORY_MESSAGES", None)
    if not isinstance(msgs, list) or not isinstance(limit, int):
        return None
    return len(msgs), limit + int(getattr(agent, "TRIM_HYSTERESIS", 0) or 0)


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
        self._prefill = ""      # text for the next prompt (a draft typed mid-turn)
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
            found = resolve_session(resume, self.mode)
            if found is None:
                raise bootstrap.BootstrapError(
                    f"No session {resume!r} here or in the index; /sessions lists them.")
            return True, found
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
        register_session(session_dir, self.mode)
        bootstrap.register_extras(
            self.agent,
            skill_files=getattr(args, "skill_files", None) or (),
            tool_files=getattr(args, "tool_files", None) or (),
            agent_files=getattr(args, "agent_files", None) or (),
            mcp_entries=getattr(args, "mcp_servers", None) or (),
            console=self.console)

    def resume(self, target: str) -> None:
        """/resume <id>: rebuild the agent from that session's checkpoint."""
        path = resolve_session(target, self.mode)
        if path is None:
            self.console.print(f"[red]No such session:[/] {target} — /sessions lists them")
            return
        setattr(self.args, "restore", True)
        with self.console.status("[dim]Restoring session…[/]", spinner="dots"), \
                quiet_console_logging():
            self.agent = self.adapter.build(self.args, self.creds, path, restore=True,
                                            extras=self.extras)
        self.session_dir = path
        self.widgets.session_dir = str(path)
        register_session(path, self.mode)
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
        if res.result and not res.stopped:
            self._auto_title(text, res.result)
        if res.error and not self.renderer.verbose:
            self.console.print("[dim]Ctrl+O or /verbose shows the full narration.[/]")
        # What was typed while the turn ran: queued messages run next, in
        # order (each shown as if typed at the prompt); an unfinished draft
        # waits in the prompt. After Ctrl+C nothing runs on its own — the
        # queue goes back into the prompt for the user to decide.
        # (Only set when there is something: a queued turn that ran from
        # here must not wipe the draft the outer turn left for the prompt.)
        if res.stopped:
            held = "\n".join(res.queued + ([res.draft] if res.draft else []))
            if held:
                self._prefill = held
            return
        if res.draft:
            self._prefill = res.draft
        for queued in res.queued:
            if self._quit:
                break
            self.console.print(f"[bold cyan]❯[/] {queued}")
            self.submit(queued)

    def submit(self, text: str) -> None:
        """A line from the prompt (or the queue): a slash command or a turn."""
        if text.startswith("/"):
            if not self.commands.dispatch(self, text):
                self.console.print(f"[red]Unknown command[/] {text.split()[0]} — /help lists them")
            return
        self.turn(text)

    def _auto_title(self, first_user: str, first_reply: str) -> None:
        """Name the session from the first exchange, as the web UI does (one
        small model call); a name the user set is never overwritten."""
        if self.session_dir is None or load_session_name(self.session_dir):
            return
        try:
            title = generate_session_title(getattr(self.agent, "model", None),
                                           first_user, first_reply)
        except Exception:  # noqa: BLE001 - naming must never break a turn
            title = None
        if title and save_session_name(self.session_dir, title, named_by="agent"):
            touch_session(self.session_dir)
            self.console.print(Text(f"  session named: {title}  (/name changes it)", style="dim"))

    def set_name(self, name: str) -> bool:
        ok = save_session_name(self.session_dir, name, named_by="user")
        if ok:
            touch_session(self.session_dir)
        return ok

    # ── the loop ───────────────────────────────────────────────

    def _toolbar(self):
        """The bar under the message line: a rule, then the session state on
        the left and the running version on the right. The mode is not
        repeated here — the placeholder already names it."""
        autonomy = self.adapter.get_autonomy(self.agent) if self.agent is not None else "?"
        verbose = "verbose on" if self.renderer.verbose else "verbose off"
        usage = context_usage(self.agent)
        parts = [autonomy, self.session_dir.name if self.session_dir else "",
                 getattr(self.args, "model", ""), verbose]
        if usage:
            parts.append(f"context {100 * usage[0] // usage[1]}%")
        right = f"scilink {scilink_version()} "
        try:
            width = get_app().output.get_size().columns
        except Exception:  # noqa: BLE001
            width = self.console.size.width
        # A few columns short of the width: prompt_toolkit's renderer never
        # writes the last column and wraps a line that reaches it, which
        # pushed the status line (with the version) out of the toolbar.
        usable = max(20, width - 4)
        # On a narrow terminal the model name goes first, then the session,
        # so the autonomy, verbose state, context and version always fit.
        left = " " + " · ".join(p for p in parts if p)
        for drop in (2, 1):
            if len(left) + len(right) + 1 <= usable:
                break
            parts[drop] = ""
            left = " " + " · ".join(p for p in parts if p)
        gap = max(1, usable - len(left) - len(right))
        rule = "\u2500" * usable
        return HTML(f"{rule}\n{left}{' ' * gap}{right}")

    def _rule(self) -> None:
        """The bar above the message line."""
        self.console.print(Text("\u2500" * self.console.size.width, style="grey35"))

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
            # No lines reserved under the input for the completion menu: the
            # bar sits right under the message line and the input grows as
            # the text does; the menu pops over the space below when used.
            reserve_space_for_menu=0,
            style=Style.from_dict({"bottom-toolbar": "noreverse fg:#5c5c5c",
                                   "bottom-toolbar.text": "noreverse fg:#8a8a8a"}),
            **self._pt_kwargs)
        while not self._quit:
            self._rule()
            prefill, self._prefill = self._prefill, ""
            try:
                text = self.prompt_session.prompt(
                    HTML("<b><ansicyan>❯</ansicyan></b> "), default=prefill,
                    placeholder=HTML(f"<i><ansibrightblack>{placeholder}</ansibrightblack></i>"))
            except KeyboardInterrupt:
                continue
            except EOFError:
                self.request_quit()
                break
            text = text.strip()
            if not text:
                continue
            self.submit(text)

    def resume_command(self) -> str:
        """The command that resumes this session from anywhere: the session
        is in the central index, so its id is enough."""
        launcher = os.environ.get("SCILINK_ARGV0", "scilink")
        base = launcher if self.mode == "meta" else f"{launcher} {self.mode}"
        return f"{base} --resume {Path(self.session_dir).name}"

    def _shutdown(self) -> None:
        if self.agent is not None:
            save_checkpoint_quietly(self.agent)
            touch_session(self.session_dir)
        self.console.print(f"\n👋 Session saved: [dim]{self.session_dir}[/]")
        self.console.print(f"   Resume it with: [bold]{self.resume_command()}[/]")

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
