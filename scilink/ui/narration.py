"""Reading the agents' narration.

The orchestrators tell the user what they are doing by printing: a tool
call, a line of reasoning, a delegation banner, a warning, a checkpoint.
Both chat surfaces capture that stream and need the same two readings of
it — *what kind of line is this* (so a terminal can dim a tool call and
italicise a thought, and the web log pane can colour them) and *what is
the agent doing right now* (the one-line activity label next to the
spinner). This module is the Python source of both; the frontend's
``webui/src/narration.ts`` is its twin, and ``tests/fixtures/
narration_activity.json`` pins the two to the same answers.

Nothing here parses for control flow: the classification is presentation
only, and a line it does not recognise is simply ``plain``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .vocabulary import ACTIVITY_LABELS, HANDOFF_PREFIXES, THOUGHT_MARK

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RULE_RE = re.compile(r"^[-=_*—─═]+$")
_CAND_RE = re.compile(r"^\[cand[_-]?0*(\d+)\]\s*(.*)$")
# The atom emoji may or may not carry its variation selector (U+FE0F).
_HANDOFF_RE = re.compile(
    r"^(?:\U0001F9EA|\U0001F4CB|⚛️?)\s+Delegating to"
    r"|^\U0001F9EC Fusing delegations")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


# ── Line classification ──────────────────────────────────────────

# Kinds shown by default in the terminal; everything else is verbose.
VISIBLE_KINDS = frozenset({
    "tool_call", "thought", "answer_header", "answer_body", "handoff",
    "fanout", "warning", "checkpoint", "files", "memory",
})


@dataclass(frozen=True)
class Line:
    kind: str
    text: str                  # ANSI- and mark-stripped, indentation removed
    specialist: bool = False   # printed by a meta-delegated specialist
    verbose: bool = True       # hidden unless verbose output is shown


class LineClassifier:
    """Stateful line classifier: a 💭 line opens a thought whose
    continuation lines are indented five spaces; a 🤖 line opens an answer
    that runs until the next recognisable kind."""

    def __init__(self) -> None:
        self._in_thought = False
        self._thought_specialist = False
        self._in_answer = False
        self._answer_specialist = False

    def _reset(self) -> None:
        self._in_thought = False
        self._in_answer = False

    def push(self, raw: str) -> Line:
        clean = strip_ansi(raw.rstrip("\n"))
        specialist = THOUGHT_MARK in clean
        clean = clean.replace(THOUGHT_MARK, "")
        s = clean.strip()

        if s.startswith("🤖"):
            self._in_thought = False
            self._in_answer = True
            self._answer_specialist = specialist
            return Line("answer_header", s, specialist, verbose=False)
        if _HANDOFF_RE.match(s):
            self._reset()
            return Line("handoff", s, verbose=False)
        if s.startswith("💭"):
            self._in_thought = True
            self._in_answer = False
            self._thought_specialist = specialist
            return Line("thought", s, specialist, verbose=False)
        if self._in_thought and clean.startswith("     "):
            return Line("thought", s, self._thought_specialist, verbose=False)
        self._in_thought = False

        kind = None
        if s.startswith("🔧 Calling tool:"):
            kind = "tool_call"
        elif s.startswith("⏳"):
            kind = "waiting"
        elif s.startswith("⚠"):
            kind = "warning"
        elif s.startswith("💾"):
            kind = "checkpoint"
        elif s.startswith(("📂", "📁")):
            kind = "files"
        elif s.startswith("🧠 Memory:"):
            kind = "memory"
        elif s.startswith("🔀"):
            kind = "fanout"
        elif s.startswith(("🔄", "✅", "⚡", "🙋", "📊", "🧠", "📚", "🖼", "📄", "🗑")):
            kind = "bookkeeping"     # the agents' own housekeeping / tool banners
        elif s.startswith("Human feedback enabled"):
            kind = "bookkeeping"
        elif _CAND_RE.match(s):
            kind = "candidate"
        elif _RULE_RE.match(s):
            kind = "rule"
        if kind is not None:
            self._in_answer = False
            return Line(kind, s, verbose=kind not in VISIBLE_KINDS)

        if self._in_answer:
            return Line("answer_body", clean, self._answer_specialist,
                        verbose=False)
        if not s:
            return Line("blank", "")
        return Line("plain", s)


def classify(text: str) -> list:
    """Classify every line of a narration chunk (stateless convenience)."""
    c = LineClassifier()
    return [c.push(ln) for ln in text.split("\n")]


# ── Current activity ─────────────────────────────────────────────

def _clip(s: str, n: int = 110) -> str:
    return s[: n - 1] + "…" if len(s) > n else s


def _pretty(s: str) -> str:
    """'ImagePlanningController' -> 'Image Planning'."""
    s = re.sub(r"Controller$", "", s)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s).strip()


_L = ACTIVITY_LABELS

_ACTION_VERBS = (
    "Executing|Generating|Running|Loading|Refitting|Preparing|Searching|"
    "Creating|Initializing|Hiring|Connecting|Refining|Reasoning|Processing|"
    "Saving|Verifying|Updating|Building|Synthesizing|Retrieving|Querying|"
    "Inspecting|Reviewing|Drafting|Writing|Analyzing"
)
_ACTION_RE = re.compile(
    r"^(?:(?:LLM Step|Tool):\s*)?((?:" + _ACTION_VERBS + r")\b.+)$")


def current_activity(log: str) -> Optional[str]:
    """One line saying what the agent is doing now, from the narration
    tail — the middle ground between a bare spinner and the full log.
    Backward scan: the most recent milestone wins. ``None`` when nothing in
    the tail is recognisable (the caller shows ``DEFAULT_ACTIVITY``)."""
    lines = strip_ansi(log[-6000:]).split("\n")
    for raw in reversed(lines):
        line = raw.replace(THOUGHT_MARK, "").strip()
        if not line:
            continue
        # Bare rules frame headers in the narration; the "--- title ---"
        # pattern below would read one as a title of "-".
        if _RULE_RE.match(line):
            continue
        # Best-of-N candidates narrate as "[cand_NN] <milestone>": strip the
        # tag, classify the milestone, prefix the candidate back on.
        cand = None
        cm = _CAND_RE.match(line)
        if cm:
            cand = cm.group(1)
            line = cm.group(2).strip()
            if not line:
                continue

        def tag(s: str) -> str:
            return _L["candidate"].format(n=cand, label=s) if cand else s

        if re.match(r"^Candidate\s+\d+\s+finished\s+\(\d+/\d+\)", line):
            return tag(_clip(line))
        m = re.search(r"escalating to (\d+) candidates", line, re.I)
        if m:
            return tag(_L["escalating"].format(n=m.group(1)))
        if line.startswith("🤖"):
            return tag(_L["writing_response"])
        m = re.match(r"^⏳\s*Waiting for (.+?) response", line)
        if m:
            return tag(_L["waiting_for"].format(who=m.group(1)))
        if line.startswith("💭"):
            return tag(_clip(line))
        if any(line.startswith(p) for p in HANDOFF_PREFIXES):
            return tag(_clip(line))
        m = re.search(r"STEP\s+\d+:\s*(\S.*)$", line)
        if m:
            return tag(_pretty(m.group(1)))
        m = re.match(r"^-{2,}\s*(.+?)\s*-{2,}$", line)
        if m:
            return tag(_clip(m.group(1)))
        m = re.match(r"^(?:[^\w\s]\s*)?Analyzing:\s*(.+)$", line)
        if m:
            return tag(_clip(_L["analyzing"].format(target=m.group(1))))
        m = re.match(r"^(?:[^\w\s]\s*)?Delegating to\s+(.+)$", line)
        if m:
            return tag(_clip(_L["delegating"].format(target=m.group(1))))
        # Inside the analysis agents: the shared codegen / QC loop.
        m = re.search(r"\(Attempt\s+(\d+)\)\s+Asking LLM to write code", line)
        if m:
            return tag(_L["writing_code_attempt"].format(n=m.group(1)))
        if "Asking LLM to write code" in line:
            return tag(_L["writing_code"])
        if "Executing generated code" in line:
            return tag(_L["executing_code"])
        m = re.search(r"Executing (?:Python )?script\s*\(attempt\s+(\d+)\)", line)
        if m:
            return tag(_L["executing_script_attempt"].format(n=m.group(1)))
        if re.search(r"Executing (?:Python )?script", line):
            return tag(_L["executing_script"])
        m = re.search(r"Execution attempt\s+(\d+)", line)
        if m:
            return tag(_L["executing_code_attempt"].format(n=m.group(1)))
        m = re.search(r"Performing Visual QC on\s+(.+?)\.*$", line)
        if m:
            return tag(_clip(_L["visual_qc"].format(target=m.group(1))))
        m = re.search(r"Combined review on\s+(.+?)\.*$", line)
        if m:
            return tag(_clip(_L["reviewing"].format(target=m.group(1))))
        m = re.match(r"^Verification\s+(\d+)/(\d+)(?:\s*\(annealing level\s+(\d+)\))?",
                     line)
        if m:
            i, n, level = m.groups()
            if level and level != "0":
                return tag(_L["verification_annealing"].format(i=i, n=n, level=level))
            return tag(_L["verification"].format(i=i, n=n))
        m = re.search(r"Attempting script correction \(attempt\s+(\d+)\)", line)
        if m:
            return tag(_L["correcting"].format(n=m.group(1)))
        if "Applying user feedback to existing script" in line:
            return tag(_L["applying_feedback"])
        if re.match(r"^Best-of-\d+", line):
            return tag(_clip(line))
        # Outcome / warning milestones read well as transient status too —
        # but not a section heading ("✅ Quality Criteria:"), whose content
        # follows on the next lines.
        if (line.startswith("✅") or line.startswith("⚠️")) and not line.endswith(":"):
            return tag(_clip(line))
        # Generic fallbacks — the many phrasing variants of action lines.
        # ``bare`` drops a leading emoji / symbol ("🧠 LLM Step: …").
        bare = re.sub(r"^[^\w]+\s*", "", line)
        # Human-in-the-loop pauses: the narration ends on the prompt banner.
        if re.match(r"^(REQUESTING FEEDBACK|SELECT A PLAN CANDIDATE|CODE REVIEW REQUIRED)\b",
                    bare):
            return tag(_L["waiting_for_input"])
        if re.match(r"^FILES PRODUCED THIS TURN\b", bare):
            return tag(_L["wrapping_up"])
        m = re.match(r"^-{2,}\s*(.+?)\s*-{2,}$", bare)
        if m:
            return tag(_clip(m.group(1)))
        m = re.match(r"^Attempt\s+(\d+(?:/\d+)?):\s*(.+)$", bare)
        if m:
            return tag(_clip(_L["attempt"].format(n=m.group(1), label=m.group(2))))
        m = _ACTION_RE.match(bare)
        if m:
            return tag(_clip(m.group(1)))
    return None
