"""Fixture-label schema for the critic-validator A/B experiment.

A *fixture* is one input file (an INCAR for VASP, a LAMMPS input script
for LAMMPS) plus the ground-truth list of issues we expect a critic to
find. Variants are scored against this ground truth.

Schema lives here so all four sources (generator, planted, controls,
breakage) and all critic variants emit / consume the same shape.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional


Severity = Literal["error", "warning", "info"]
Category = Literal[
    "syntax_typo",          # malformed tag/command name (ISPN, pair_styl)
    "malformed_value",      # syntactically wrong value (ENCUT=abc)
    "wrong_choice_for_system",  # tag/value valid but physically wrong here
    "missing_required",     # required tag/command absent
    "redundant",            # extra / contradictory directive
]


@dataclass(frozen=True)
class Issue:
    """One ground-truth issue in a fixture, or one reported issue by a variant.

    `locator` is engine-neutral free-form:
      VASP:  "tag:ISPN"  or  "tag:ENCUT"
      LAMMPS: "command:pair_style"  or  "line:42"  or  "command:fix:nve"

    `fix` is optional. If present, applying it should produce the canonical
    corrected fixture. Schema for fix payload is engine-specific (e.g.,
    {"tag": "ISPIN", "value": "2"} for VASP).
    """
    locator: str
    severity: Severity
    category: Category
    message: str
    fix: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["fix"] is None:
            d.pop("fix")
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Issue":
        return cls(
            locator=d["locator"],
            severity=d["severity"],
            category=d["category"],
            message=d["message"],
            fix=d.get("fix"),
        )


@dataclass
class FixtureLabel:
    """One labeled fixture row in ``labels.jsonl``.

    Required fields are populated when the fixture is created. The label
    fields (``true_issues``, ``labeled_by``, ``labeled_at``) are filled
    during the labeling pass.
    """
    id: str                              # globally unique: <engine>_<prompt>_<NNN>
    engine: Literal["vasp", "lammps"]
    source: Literal["from_generator", "planted", "controls", "breakage"]
    prompt_id: str                       # references the prompt registry
    prompt_text: str                     # frozen at fixture-creation time
    system_name: str                     # benchmark.systems entry
    fixture_path: str                    # relative to critic_experiment/
    true_issues: List[Issue] = field(default_factory=list)
    canonical_fix_path: Optional[str] = None  # relative path to corrected fixture
    labeled_by: Optional[str] = None
    labeled_at: Optional[str] = None     # ISO date string
    notes: str = ""

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["true_issues"] = [i.to_json() if isinstance(i, Issue)
                            else Issue(**i).to_json()
                            for i in d["true_issues"]]
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "FixtureLabel":
        return cls(
            id=d["id"],
            engine=d["engine"],
            source=d["source"],
            prompt_id=d["prompt_id"],
            prompt_text=d["prompt_text"],
            system_name=d["system_name"],
            fixture_path=d["fixture_path"],
            true_issues=[Issue.from_json(i) for i in d.get("true_issues", [])],
            canonical_fix_path=d.get("canonical_fix_path"),
            labeled_by=d.get("labeled_by"),
            labeled_at=d.get("labeled_at"),
            notes=d.get("notes", ""),
        )


def load_labels(path: str | Path) -> List[FixtureLabel]:
    """Load all FixtureLabel rows from a JSONL file. Empty file → []."""
    p = Path(path)
    if not p.exists():
        return []
    rows: List[FixtureLabel] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(FixtureLabel.from_json(json.loads(line)))
    return rows


def append_labels(path: str | Path, rows: Iterable[FixtureLabel]) -> None:
    """Append FixtureLabel rows to a JSONL file. Creates the file if absent."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), separators=(",", ":")) + "\n")


def rewrite_labels(path: str | Path, rows: Iterable[FixtureLabel]) -> None:
    """Overwrite a JSONL file with the given rows (use for in-place label edits)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), separators=(",", ":")) + "\n")
    tmp.replace(p)
