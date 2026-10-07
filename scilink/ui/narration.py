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
# A swarm item's lines carry its label, "[XRD 300 K] …", inserted at the
# line start by the item's own stream (fanout._ThreadStopStream) or the
# process worker's relay; never a candidate tag, which has its own rule.
# The writers put the label through ``worker_tag`` first, so what they
# write is always what this reads.
WORKER_TAG_MAX = 48
_WORKER_RE = re.compile(r"^\[(?!cand[_-]?\d)([^\[\]\n]{1,48})\]\s?(.*)$", re.S)
# The swarm and fan-out coordinators' own lines, by their wording — the
# launch, an item held for memory or the breaker, the guard, a rerun, a
# budget, a refusal, the heartbeat, an item finished. An agent's own line
# that opens with the same emoji ("⏱ Nobody answered the review in time",
# "🔁 Diagram render error") is not one.
_COORDINATOR_RE = re.compile(
    r"^(?:🐝 |"
    r"⏸\s+holding (?:branch )?'|"
    r"🧯 (?:free memory is low|'[^']*' ran out of memory)|"
    r"🔁 (?:running (?:branch )?'[^']*' again|Resuming \d+ fan-out branch)|"
    r"⛔ (?:not started:|reaction refused|not running '[^']*' again)|"
    r"⏱️?\s+(?:swarm item|analysis branch|resumed branch|raw-instrument branch\(es\)) |"
    r"⏳ (?:\d+ swarm item\(s\) still running|\d+ of \d+ parallel analyses still|"
    r"\d+ resumed branch\(es\) still|waiting for the cancelled (?:worker|branch) to end)|"
    r"✅ (?:swarm item|analysis branch|resumed branch) finished)")


def worker_tag(label: object) -> str:
    """The tag a writer puts before a swarm item's lines, from the item's
    label as the model wrote it: one line, no brackets, at most
    ``WORKER_TAG_MAX`` characters (cut with an ellipsis), never shaped like
    a candidate tag — so the reader (``_WORKER_RE``) always accepts it."""
    text = re.sub(r"\s+", " ", str(label or "").replace("[", "(").replace("]", ")")).strip()
    if not text:
        text = "item"
    if re.match(r"^cand[_-]?\d", text, re.I):
        text = "item " + text
    if len(text) > WORKER_TAG_MAX:
        text = text[:WORKER_TAG_MAX - 1].rstrip() + "…"
    return text
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
    text: str                  # ANSI- and mark-stripped, indentation and worker tag removed
    specialist: bool = False   # printed by a meta-delegated specialist
    verbose: bool = True       # hidden unless verbose output is shown
    worker: Optional[str] = None   # the swarm item whose line this is ("[label] …")


def split_worker_tag(clean: str) -> tuple:
    """``("XRD 300 K", "💭 …")`` for a swarm item's tagged line, else
    ``(None, clean)``. The tag sits before the indentation the line had, so
    a thought's continuation keeps its five spaces after the tag."""
    m = _WORKER_RE.match(clean.lstrip())
    if not m:
        return None, clean
    return m.group(1), m.group(2)


class _Continuation:
    """What a 💭 or 🤖 line opened and has not closed — per writer: the
    meta's stream and each swarm item's are interleaved, and an indented
    line of item B must never read as the continuation of item A's
    thought."""
    __slots__ = ("in_thought", "thought_specialist", "in_answer", "answer_specialist")

    def __init__(self) -> None:
        self.in_thought = False
        self.thought_specialist = False
        self.in_answer = False
        self.answer_specialist = False


class LineClassifier:
    """Stateful line classifier: a 💭 line opens a thought whose
    continuation lines are indented five spaces; a 🤖 line opens an answer
    that runs until the next recognisable kind. The state is kept per
    worker (the meta's own lines and each swarm item's)."""

    def __init__(self) -> None:
        self._by_worker: dict = {}

    def _state(self, worker: Optional[str]) -> _Continuation:
        st = self._by_worker.get(worker)
        if st is None:
            st = self._by_worker[worker] = _Continuation()
        return st

    def push(self, raw: str) -> Line:
        clean = strip_ansi(raw.rstrip("\n"))
        specialist = THOUGHT_MARK in clean
        clean = clean.replace(THOUGHT_MARK, "")
        worker, clean = split_worker_tag(clean)
        s = clean.strip()
        st = self._state(worker)

        def line(kind, text, spec=False, verbose=True):
            return Line(kind, text, spec, verbose, worker)

        if s.startswith("🤖"):
            st.in_thought = False
            st.in_answer = True
            st.answer_specialist = specialist
            return line("answer_header", s, specialist, verbose=False)
        if _HANDOFF_RE.match(s):
            st.in_thought = st.in_answer = False
            return line("handoff", s, verbose=False)
        if s.startswith("💭"):
            st.in_thought = True
            st.in_answer = False
            st.thought_specialist = specialist
            return line("thought", s, specialist, verbose=False)
        if st.in_thought and clean.startswith("     "):
            return line("thought", s, st.thought_specialist, verbose=False)
        st.in_thought = False

        kind = None
        if s.startswith("🔧 Calling tool:"):
            kind = "tool_call"
        elif s.startswith("🔀") or _COORDINATOR_RE.match(s):
            kind = "fanout"              # the coordinators' own lines, by their wording
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
        elif s.startswith(("🔄", "✅", "⚡", "🙋", "📊", "🧠", "📚", "🖼", "📄", "🗑")):
            kind = "bookkeeping"     # the agents' own housekeeping / tool banners
        elif s.startswith("Human feedback enabled"):
            kind = "bookkeeping"
        elif _CAND_RE.match(s):
            kind = "candidate"
        elif _RULE_RE.match(s):
            kind = "rule"
        if kind is not None:
            st.in_answer = False
            return line(kind, s, verbose=kind not in VISIBLE_KINDS)

        if st.in_answer:
            return line("answer_body", clean, st.answer_specialist,
                        verbose=False)
        if not s:
            return line("blank", "")
        return line("plain", s)


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
        # A swarm item's line carries its label, "[XRD 300 K] <milestone>":
        # strip it, classify the milestone, prefix the label back on — so the
        # activity says WHOSE stage this is when several items run at once.
        worker, line = split_worker_tag(line)
        line = line.strip()
        if not line:
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
            s = _L["candidate"].format(n=cand, label=s) if cand else s
            return _L["worker"].format(worker=worker, label=s) if worker else s

        # The coordinators' own milestones (a swarm's, a fan-out's).
        m = re.match(r"^🐝 .*?(\d+) item\(s\)(?:, up to (\d+) at a time)?", line)
        if m:
            return tag(_L["swarm_started"].format(n=m.group(1)))
        m = re.match(r"^⏳ (\d+) swarm item\(s\) still running", line)
        if m:
            return tag(_L["swarm_running"].format(n=m.group(1)))
        m = re.match(r"^⏸\s+holding (?:branch )?'([^']+)'", line)
        if m:
            return tag(_clip(_L["swarm_holding"].format(label=m.group(1))))
        m = re.match(r"^🧯 .*?cancelling (?:branch )?'([^']+)'", line)
        if m:
            return tag(_clip(_L["swarm_guard"].format(label=m.group(1))))
        m = re.match(r"^🔁 running (?:branch )?'([^']+)' again", line)
        if m:
            return tag(_clip(_L["swarm_rerun"].format(label=m.group(1))))
        m = re.match(r"^✅ swarm item finished: (.+?) \((\w+)\)", line)
        if m:
            return tag(_clip(_L["swarm_item_done"].format(label=m.group(1), status=m.group(2))))

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
