"""Rendering a turn on the terminal.

Claude Code's split, in rich: finished content lives in the terminal's
scrollback (write-once), the turn still running is one live region that is
redrawn on every change. The region holds the tail of the current turn's
narration that fits the screen plus the status row (spinner · activity ·
elapsed · key hints). "Verbose" is a flag on that region: Ctrl+O flips it
and the region redraws with the hidden lines shown or gone, so nothing has
to be un-printed. A question commits the lines so far to scrollback before
the prompt takes the terminal; the end of the turn commits the whole block
once, in the verbosity in force, then the answer as markdown and one dim
line of accounting.

Without a terminal (piped output, tests) there is no region: visible lines
stream as they arrive and turning verbose on replays the hidden ones.

The agents' own printed answer block (the unmarked ``🤖`` header and its
body) is not echoed — the return value is the answer, rendered once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Heading, Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from scilink.ui import vocabulary as V
from scilink.ui.narration import LineClassifier

# Styles per line kind (rich markup); the web's tokens.css is the twin.
_STYLES = {
    "tool_call": "dim",
    "thought": "italic cyan",
    "thought_specialist": "italic dark_orange3",
    "answer_specialist": "dark_orange3",
    "handoff": "bold gold3",
    "fanout": "bold",
    "warning": "yellow",
    "checkpoint": "dim",
    "files": "dim",
    "memory": "dim",
    "verbose": "grey50",
}


class _LeftHeading(Heading):
    """Chat-style headings: left-aligned, no box around h1 (rich centres
    headings and frames h1 by default, which reads oddly under a prompt)."""

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            text.stylize("bold")
        yield text


class ChatMarkdown(Markdown):
    elements = dict(Markdown.elements, heading_open=_LeftHeading)


@dataclass
class _Entry:
    text: Text
    verbose: bool          # hidden unless verbose is on
    printed: bool = False  # streamed already (no-terminal mode)


class Renderer:
    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._live: Optional[Live] = None
        self._activity: Optional[str] = None
        self._t0 = 0.0
        self._partial = ""
        self._classifier = LineClassifier()
        self._paused = False
        self._entries: List[_Entry] = []   # the current turn's narration
        self._committed = 0                # entries already in scrollback

    # ── the live region ────────────────────────────────────────

    def _row(self) -> Table:
        elapsed = time.monotonic() - self._t0
        hint = "ctrl+o hide verbose" if self.verbose else "ctrl+o verbose"
        # One screen line, always: the region's height is what keeps the
        # redraw in place, so the label is truncated rather than wrapped.
        grid = Table.grid(padding=(0, 1))
        grid.add_row(Spinner("dots", style="cyan"),
                     Text(self._activity or V.DEFAULT_ACTIVITY, style="cyan",
                          no_wrap=True, overflow="ellipsis"),
                     Text(f"{elapsed:.0f}s", style="dim", no_wrap=True),
                     Text(f"({hint} · ctrl+c stop)", style="dim", no_wrap=True,
                          overflow="ellipsis"))
        return grid

    def _visible(self, entries) -> List[Text]:
        return [e.text for e in entries if not e.verbose or self.verbose]

    def _height(self, text: Text) -> int:
        """Screen lines the text takes at the current width (it wraps)."""
        return max(1, len(text.wrap(self.console, self.console.size.width)))

    def _view(self) -> Group:
        """What the live region shows: the tail of the uncommitted narration
        that fits above the status row — measured in SCREEN lines, since a
        long thought wraps. A region taller than the terminal cannot be
        redrawn in place: rich re-prints it below itself on every refresh
        (observed live as the same lines repeating and the screen scrolling)."""
        lines = self._visible(self._entries[self._committed:])
        room = max(2, self.console.size.height - 4)   # status row + margin
        tail: List[Text] = []
        used = 0
        for t in reversed(lines):
            h = self._height(t)
            if used + h > room - 1:      # keep a line for the "earlier" note
                break
            tail.insert(0, t)
            used += h
        if len(tail) < len(lines):
            tail.insert(0, Text(f"  … {len(lines) - len(tail)} earlier lines "
                                "(shown when the turn ends)", style="dim"))
        return Group(*tail, self._row())

    def begin_turn(self) -> None:
        self.console.print()   # a gap under the submitted line
        self._t0 = time.monotonic()
        self._activity = None
        self._partial = ""
        self._classifier = LineClassifier()
        self._entries = []
        self._committed = 0
        if self.console.is_terminal:
            self._start_live()

    def _start_live(self) -> None:
        """A fresh Live each time: a restarted Live remembers its last frame's
        height and moves the cursor up by it on the first refresh, erasing
        whatever was printed in between (the question panel, observed live).
        No stdout/stderr redirection: rich's default proxy would hijack the
        agent thread's prints away from the capture (they arrive laced with
        cursor-control sequences and never reach the classifier); the
        narration is rendered from the capture buffer by the main thread."""
        self._live = Live(self._view(), console=self.console, transient=True,
                          refresh_per_second=8, get_renderable=self._view,
                          redirect_stdout=False, redirect_stderr=False,
                          vertical_overflow="crop")
        self._live.start()

    def set_activity(self, label: Optional[str]) -> None:
        self._activity = label

    def _commit(self) -> None:
        """Print the uncommitted visible lines into scrollback (terminal mode)."""
        for t in self._visible(self._entries[self._committed:]):
            self.console.print(t)
        self._committed = len(self._entries)

    def pause(self) -> None:
        """A prompt takes the terminal: commit what the region showed, stop it."""
        if self._live is not None and not self._paused:
            self._live.stop()
            self._commit()
            self._paused = True

    def resume(self) -> None:
        if self._live is not None and self._paused:
            self._start_live()
            self._paused = False

    # ── narration ──────────────────────────────────────────────

    def feed(self, chunk: str) -> None:
        self._partial += chunk
        *lines, self._partial = self._partial.split("\n")
        for raw in lines:
            self._render_line(raw)

    def _entry_for(self, raw: str) -> Optional[_Entry]:
        ln = self._classifier.push(raw)
        if ln.kind in ("answer_header", "answer_body"):
            if not ln.specialist:
                return None  # the return value is rendered once, at the end
            text = ln.text if ln.kind == "answer_header" else raw.replace(V.THOUGHT_MARK, "")
            return _Entry(Text("  " + text, style=_STYLES["answer_specialist"]), verbose=False)
        if ln.kind == "blank":
            return None
        if ln.kind == "thought":
            style = _STYLES["thought_specialist" if ln.specialist else "thought"]
        elif ln.kind in _STYLES:
            style = _STYLES[ln.kind]
        else:
            style = _STYLES["verbose"]
        return _Entry(Text("  " + ln.text, style=style), verbose=ln.verbose)

    def _render_line(self, raw: str) -> None:
        entry = self._entry_for(raw)
        if entry is None:
            return
        self._entries.append(entry)
        if self._live is None and (not entry.verbose or self.verbose):
            self.console.print(entry.text)     # no region: stream
            entry.printed = True

    def set_verbose(self, on: bool) -> None:
        """Toggle mid-turn (Ctrl+O). With a live region the region simply
        redraws; without one, turning verbose on replays the hidden lines."""
        if on == self.verbose:
            return
        self.verbose = on
        if self._live is not None:
            self._live.refresh()
            return
        if on:
            hidden = [e for e in self._entries if e.verbose and not e.printed]
            if hidden:
                self.console.print(Text(f"  ── {len(hidden)} lines hidden so far ──", style="dim"))
                for e in hidden:
                    self.console.print(e.text)
                    e.printed = True
        self.console.print(Text(f"  ── {V.NAMES['verbose_section']}: {'on' if on else 'off'} ──",
                                style="dim"))

    # ── end of turn ────────────────────────────────────────────

    def end_turn(self, *, result: Optional[str], error: Optional[str],
                 stopped: bool, stop_message: str, tokens: Optional[dict] = None) -> None:
        if self._partial.strip():
            self._render_line(self._partial)
        self._partial = ""
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._paused = False
            self._commit()            # the whole block, once, as it stands now
        elapsed = time.monotonic() - self._t0
        if stopped:
            self.console.print(f"\n[yellow]■ {stop_message}[/]")
        elif error is not None:
            self.console.print(f"\n[red]Error:[/] {error}")
        elif result:
            self.console.print()
            self.console.print(ChatMarkdown(result))
        self.console.print(Text(self.accounting_line(tokens, elapsed), style="dim"))
        self.console.print()   # breathing room before the next prompt

    @staticmethod
    def accounting_line(tokens: Optional[dict], elapsed: float) -> str:
        parts = []
        if tokens and tokens.get("calls"):
            parts.append(f"{int(tokens['calls'])} LLM call{'s' if tokens['calls'] != 1 else ''}")
            p, c = int(tokens.get("prompt_tokens", 0)), int(tokens.get("completion_tokens", 0))
            if p or c:
                parts.append(f"{p:,} tokens in · {c:,} out")
        parts.append(f"{elapsed:.0f}s")
        return "· " + " · ".join(parts)
