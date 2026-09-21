"""Human-in-the-loop for the terminal: the channel and the question widgets.

``ShellChannel`` is the web ``HTTPChannel``'s sibling: it parks the
question on the turn (``turn.pending_question``) and blocks the agent
thread; the shell's main loop notices, renders it with ``Widgets.ask`` and
answers. The widgets follow the presenter's vocabulary — the same seven
shapes and the same button labels the React FeedbackPanel shows — so a
user who has seen one surface recognises the other. Every widget shows
the "Enter = <accept>" convention the web panel shows as a caption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from prompt_toolkit.application import Application, run_in_terminal
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from rich.console import Group
from rich.rule import Rule
from rich.text import Text

from scilink.server.hitl_channel import ParkingChannel
from scilink.ui import vocabulary as V


class ShellChannel(ParkingChannel):
    """Parks the question; the shell's turn loop answers it."""


class AutoAcceptChannel:
    """Headless: every question gets its default (accept as-is)."""

    def ask(self, req) -> str:
        return req.default or ""


def enter_hint(labels: Dict[str, str], key: str = "accept") -> str:
    accept = labels.get(key) or ""
    return V.ENTER_ACCEPTS_HINT.format(accept=accept) if accept else ""


def choose(options, default: int, *, prompt_session, console_width: int = 120,
           esc_label: str = "stop", key_bindings=None):
    """An arrow-key picker in place of typing a number: ``options`` is a list
    of (value, label, note); up/down move the highlight (starting on
    ``default``), Enter chooses, a digit or the option's key letter jumps,
    Esc / Ctrl+C raise KeyboardInterrupt (stop the turn). ``key_bindings``
    adds bindings (a question's Ctrl+O). Returns the value."""
    state = {"i": max(0, min(default, len(options) - 1))}
    keys = {}
    for n, (value, label, note) in enumerate(options):
        keys[str(n + 1)] = n
        first = (label or "").strip()[:1].lower()
        if first and first not in keys:
            keys[first] = n

    def render():
        out = []
        width = max(30, console_width - 6)
        for n, (value, label, note) in enumerate(options):
            cur = n == state["i"]
            text = f"{n + 1}  {label}"
            if len(text) > width:
                text = text[: width - 1] + "…"
            marker = "❯ " if cur else "  "
            style = "class:pick.current" if cur else ""
            out.append((style, marker + text))
            if note:
                out.append(("class:pick.note", f"  {note}"))
            out.append(("", "\n"))
        out.append(("class:pick.hint", f"  ↑↓ move · Enter choose · Esc {esc_label}"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        state["i"] = (state["i"] - 1) % len(options)

    @kb.add("down")
    def _down(event):
        state["i"] = (state["i"] + 1) % len(options)

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=state["i"])

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    @kb.add("c-d")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt())

    for key, index in keys.items():
        def _jump(event, index=index):
            event.app.exit(result=index)
        kb.add(key)(_jump)

    from prompt_toolkit.styles import Style
    if key_bindings is not None:
        kb = merge_key_bindings([key_bindings, kb])
    app = Application(
        layout=Layout(Window(FormattedTextControl(render, focusable=True),
                             wrap_lines=False)),
        key_bindings=kb, full_screen=False, mouse_support=False,
        style=Style.from_dict({"pick.current": "bold ansicyan",
                               "pick.note": "ansigreen",
                               "pick.hint": "fg:ansibrightblack"}),
        input=prompt_session.input, output=prompt_session.output)
    chosen = app.run()
    return options[chosen][0]


class Widgets:
    """Renders a presented question and reads the answer.

    ``prompt_session`` is a prompt_toolkit ``PromptSession`` (tests pass one
    built on a pipe input); ``console`` is the rich console the narration
    renders on.
    """

    def __init__(self, console: Console, prompt_session, session_dir: str = "") -> None:
        self.console = console
        self.session = prompt_session
        self.session_dir = session_dir
        self._overflow: list = []   # the question's earlier lines cut from the panel

    # ── helpers ────────────────────────────────────────────────

    def _bindings(self) -> KeyBindings:
        """Ctrl+O at a question prompt (or a picker) shows the lines the panel
        cut, above the prompt, and leaves the prompt open (prompt_toolkit's
        default for the key accepted the line — observed live as an
        unintended plan approval; the picker had no binding at all, so the
        panel's "Ctrl+O shows them" pointer did nothing there)."""
        kb = KeyBindings()
        overflow = self._overflow
        console = self.console

        @kb.add("c-o")
        def _show_earlier(event):
            def _print():
                if overflow:
                    console.print(Panel(Text("\n".join(overflow)),
                                        title=f"[bold yellow]{len(overflow)} earlier lines[/]",
                                        border_style="yellow"))
                    overflow.clear()
                else:
                    console.print(Text("  (nothing more to show for this question)", style="dim"))
            run_in_terminal(_print)

        return kb

    def _read(self, prompt: str, default: str = "") -> str:
        # A yellow question mark, so a question is told apart from the chat prompt.
        text = self.session.prompt(HTML(f"<ansiyellow><b>?</b></ansiyellow> {prompt}"),
                                   key_bindings=self._bindings())
        return text if text.strip() else default

    _CONTEXT_MAX_LINES = 120

    def _show_context(self, q: Dict[str, Any]) -> None:
        # What is under review — the plan / result / code the agent printed
        # before asking (the presenter's ``context_display``, the same block
        # the web panel shows). Those lines are "plain" narration, hidden
        # unless verbose, so the question must carry them. The agent's own
        # prompt line follows.
        prompt = (q.get("prompt") or "").strip()
        context = (q.get("context_display") or "").strip()
        parts = []
        self._overflow.clear()
        if context:
            lines = context.split("\n")
            if len(lines) > self._CONTEXT_MAX_LINES:
                self._overflow.extend(lines[:-self._CONTEXT_MAX_LINES])
                lines = lines[-self._CONTEXT_MAX_LINES:]
                parts.append(Text(f"… {len(self._overflow)} earlier lines (Ctrl+O shows them)",
                                  style="dim"))
            parts.append(Text("\n".join(lines)))
        if prompt:
            if parts:
                parts.append(Rule(style="yellow"))
            parts.append(Text(prompt, style="bold"))
        if parts:
            self.console.print(Panel(Group(*parts), title="[bold yellow]Question[/]",
                                     border_style="yellow"))
        for rel in q.get("preview_images") or []:
            cap = (q.get("candidate_captions") or {}).get(Path(rel).name)
            label = f" ({cap})" if cap else ""
            path = Path(self.session_dir, rel) if self.session_dir else Path(rel)
            self.console.print(f"  [dim]figure:[/] {path}{label}")
        for f in q.get("code_files") or []:
            self.console.print(Panel(Syntax(f.get("content", ""), "python",
                                            line_numbers=False, word_wrap=True),
                                     title=f"📄 {f.get('name')}", border_style="dim"))

    def _choose(self, options, default: int) -> str:
        return choose(options, default, prompt_session=self.session,
                      console_width=self.console.size.width,
                      key_bindings=self._bindings())

    # ── widgets ────────────────────────────────────────────────

    def ask(self, q: Dict[str, Any]) -> str:
        widget = q.get("widget", "generic")
        labels = q.get("labels") or {}
        self._show_context(q)
        if widget == "keep_revert":
            return self._choose([("keep", labels.get("keep", "Keep"), ""),
                                 ("", labels.get("revert", "Revert"), "")], default=1)
        if widget == "fanout_confirm":
            f = q.get("fanout") or {}
            self.console.print("[bold]🔀 Launch parallel multi-dataset analysis?[/]")
            if f.get("verdict"):
                self.console.print(f"  [bold]Complementarity:[/] {f['verdict']}")
            if f.get("join_axis"):
                self.console.print(f"  [bold]Join axis:[/] {f['join_axis']}")
            if f.get("branches"):
                self.console.print("  [bold]Branches[/] — run concurrently, each seeing the others as auxiliary:")
                for b in f["branches"]:
                    self.console.print(f"    • {b}")
            if f.get("rationale"):
                self.console.print(f"  [bold]Why:[/] {f['rationale']}")
            self.console.print("  [dim]Branches run autonomously — no per-branch approval pauses.[/]")
            return self._choose([("no", labels.get("cancel", "Cancel"), ""),
                                 ("y", labels.get("confirm", "Launch"), "")], default=0)
        if widget in ("bestofn", "plan_candidates"):
            pick = q.get("judge_pick")
            cands = q.get("candidates") or []
            self.console.print(f"[bold]{labels.get('select')}[/]")
            options = [(str(c.get("idx")), str(c.get("label")),
                        f"← {V.NAMES['judge_pick']}" if c.get("idx") == pick else "")
                       for c in cands]
            default = next((n for n, c in enumerate(cands) if c.get("idx") == pick), 0)
            chosen = self._choose(options, default=default)
            # The empty answer is "accept the judge's pick", as on the web.
            return "" if chosen == str(pick) else chosen
        # generic / dataset_description / code_review
        hint = enter_hint(labels)
        self.console.print(f"[bold]{labels.get('input', 'Your feedback (optional):')}[/]"
                           + (f"  [dim]{hint}[/]" if hint else ""))
        return self._read("").strip()


def ask_secret(prompt_session, prompt: str, *, secret: bool = False,
               default: str = "") -> str:
    """The bootstrap ``ask``: one prompt_toolkit prompt, hidden when secret."""
    try:
        text = prompt_session.prompt(prompt, is_password=secret)
    except (EOFError, KeyboardInterrupt):
        return default
    return text if text else default
