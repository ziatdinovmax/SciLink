"""Golden harness for the analysis-agent QC loops (issue #327, phase 0).

Runs the full agents end-to-end with a *scripted* mock LLM (real sandbox
script execution, canned model responses) and pins, as goldens:

  1. every prompt sent to the model, in order (normalized),
  2. the compiled result dict,
  3. the per-item ``quality_history``,
  4. the saved scripts.

The engine-extraction phases of the plan (``analysis_qc_unification_plan.md``)
must reproduce these goldens byte-identically.

Regenerate goldens with ``QC_GOLDEN_UPDATE=1 pytest tests/qc_golden``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

GOLDEN_DIR = Path(__file__).parent / "goldens"


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """One prompt→response routing rule.

    ``marker`` is a literal substring that identifies the call type (prefer a
    distinctive phrase imported from ``instruct.py`` so the marker tracks the
    prompt).  ``responses`` is consumed once per matching call, in order; if
    the rule keeps matching after exhaustion, the last response repeats when
    ``repeat_last`` is set, otherwise the harness fails loudly.  A response may
    be a ``str``, a dict (JSON-encoded), or a callable ``(call_no, text) -> str``.
    """

    name: str
    marker: str | tuple[str, ...]
    responses: list = field(default_factory=list)
    repeat_last: bool = False
    _used: int = 0

    def matches(self, text: str) -> bool:
        markers = self.marker if isinstance(self.marker, tuple) else (self.marker,)
        return all(m in text for m in markers)

    def next_response(self, text: str) -> str:
        if self._used >= len(self.responses):
            if self.repeat_last and self.responses:
                resp = self.responses[-1]
            else:
                raise AssertionError(
                    f"ScriptedModel rule '{self.name}' exhausted after "
                    f"{self._used} uses (add responses or set repeat_last)"
                )
        else:
            resp = self.responses[self._used]
        self._used += 1
        if callable(resp):
            resp = resp(self._used, text)
        if isinstance(resp, dict):
            resp = json.dumps(resp)
        return resp


class ScriptedModel:
    """Drop-in for the LLM wrappers: routes prompts to canned responses.

    Matching is first-rule-wins in list order, so put more specific rules
    first.  Every call is recorded (normalized) in ``self.calls`` as
    ``{"rule": name, "prompt": normalized_text}``.
    """

    def __init__(self, rules: list[Rule], normalizer: Callable[[str], str] | None = None):
        self.rules = rules
        self.calls: list[dict] = []
        self._normalizer = normalizer or (lambda s: s)

    # -- the wrapper contract -------------------------------------------------
    def generate_content(self, contents=None, generation_config=None,
                         safety_settings=None, **kwargs):
        text = flatten_prompt(contents)
        rule = next((r for r in self.rules if r.matches(text)), None)
        if rule is None:
            snippet = text[:800]
            raise AssertionError(
                f"ScriptedModel: no rule matched prompt (first 800 chars):\n{snippet}"
            )
        self.calls.append({"rule": rule.name, "prompt": self._normalizer(text)})
        return SimpleNamespace(text=rule.next_response(text))

    # -- assertions -----------------------------------------------------------
    def assert_all_rules_used(self, expected: Iterable[str] | None = None):
        used = {c["rule"] for c in self.calls}
        if expected is not None:
            missing = set(expected) - used
            assert not missing, f"Expected rules never fired: {sorted(missing)}"


def flatten_prompt(contents) -> str:
    """Normalize a prompt-parts list into one text blob.

    Non-text parts (inline images) become stable placeholders so goldens do
    not depend on matplotlib/JPEG byte output.
    """
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    parts = []
    for part in contents:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and "mime_type" in part:
            parts.append(f"<inline-media:{part['mime_type']}>")
        else:
            parts.append(f"<part:{type(part).__name__}>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Normalization (paths, volatile values) for golden stability
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"(\d+\.\d{7,})")


def make_normalizer(volatile_paths: dict[str, str]) -> Callable[[str], str]:
    """Return a text normalizer replacing volatile absolute paths.

    ``volatile_paths`` maps real path prefix -> stable token, e.g.
    ``{str(tmp_out): "<OUTDIR>", str(data_dir): "<DATADIR>"}``.
    Longest prefixes are applied first so nested dirs normalize correctly.
    Long floats are truncated to 6 decimals to absorb platform noise in
    printed statistics.
    """

    ordered = sorted(volatile_paths.items(), key=lambda kv: -len(kv[0]))

    def norm(text: str) -> str:
        for real, token in ordered:
            text = text.replace(real, token)
        text = _FLOAT_RE.sub(lambda m: m.group(1)[: m.group(1).index(".") + 7], text)
        return text

    return norm


def normalize_obj(obj: Any, norm: Callable[[str], str]) -> Any:
    """Recursively normalize a result structure for golden comparison.

    bytes -> placeholder, Path -> normalized str, floats rounded to 6 dp,
    numpy scalars/arrays -> plain python.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: normalize_obj(v, norm) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_obj(v, norm) for v in obj]
    if isinstance(obj, bytes):
        return "<bytes>"
    if isinstance(obj, Path):
        return norm(str(obj))
    if isinstance(obj, str):
        return norm(obj)
    if isinstance(obj, np.ndarray):
        return normalize_obj(obj.tolist(), norm)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 6)
    return obj


# ---------------------------------------------------------------------------
# Golden compare
# ---------------------------------------------------------------------------

def check_golden(name: str, payload: Any) -> None:
    """Compare ``payload`` against the stored golden ``goldens/<name>.json``.

    With ``QC_GOLDEN_UPDATE=1`` the golden is (re)written instead.  The
    comparison is exact on the canonical JSON serialization.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)

    if os.environ.get("QC_GOLDEN_UPDATE") == "1":
        path.write_text(canonical + "\n", encoding="utf-8")
        return

    assert path.exists(), (
        f"Golden {path} missing — run QC_GOLDEN_UPDATE=1 pytest to create it"
    )
    stored = path.read_text(encoding="utf-8").rstrip("\n")
    if stored != canonical:
        # Produce a readable first-divergence message instead of a wall diff.
        s_lines, c_lines = stored.splitlines(), canonical.splitlines()
        for i, (a, b) in enumerate(zip(s_lines, c_lines)):
            if a != b:
                raise AssertionError(
                    f"Golden mismatch for '{name}' at line {i + 1}:\n"
                    f"  golden : {a}\n  actual : {b}\n"
                    f"(run QC_GOLDEN_UPDATE=1 to regenerate if intended)"
                )
        raise AssertionError(
            f"Golden mismatch for '{name}': length differs "
            f"({len(s_lines)} vs {len(c_lines)} lines)"
        )


def collect_saved_scripts(output_dir: Path, norm: Callable[[str], str]) -> dict[str, str]:
    """Read every saved script under <output_dir>/scripts for golden pinning."""
    scripts_dir = output_dir / "scripts"
    if not scripts_dir.is_dir():
        return {}
    return {
        p.name: norm(p.read_text(encoding="utf-8"))
        for p in sorted(scripts_dir.glob("*.py"))
    }
