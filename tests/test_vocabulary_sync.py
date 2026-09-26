"""The generated ``webui/src/vocabulary.ts`` must equal ``vocabulary.as_json()``
— run ``python scripts/gen_vocabulary.py`` after editing the Python module."""

import json
from pathlib import Path

from scilink.ui import vocabulary

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "webui" / "src" / "vocabulary.ts"


def _parse_ts() -> dict:
    text = TS.read_text(encoding="utf-8")
    start = text.index("export const VOCAB = ") + len("export const VOCAB = ")
    end = text.rindex(" as const;")
    return json.loads(text[start:end])


def test_generated_ts_matches_python():
    assert _parse_ts() == vocabulary.as_json(), (
        "webui/src/vocabulary.ts is stale: run python scripts/gen_vocabulary.py")


def test_mode_records_are_complete():
    keys = {"key", "name", "emoji", "label", "description", "blurb", "placeholder",
            "session_prefix", "legacy_prefixes", "autonomy_options", "stop_message"}
    for k, m in vocabulary.MODES.items():
        assert set(m) == keys, k
        assert m["key"] == k
        assert m["autonomy_options"][-1] == "autonomous"


def test_helpers():
    assert vocabulary.session_prefixes("plan") == ["planning_session", "campaign_session"]
    assert vocabulary.stop_message("plan") == "Planning stopped by user."
    assert vocabulary.stop_message("nonsense") == "Analysis stopped by user."
    assert vocabulary.autonomy_options("meta") == ["autopilot", "autonomous"]


def test_every_gate_kind_has_a_widget():
    """Every ``kind=`` a gate passes to request_human_feedback is presented
    by the kind table once it carries a subject — a new kind must be added
    there, not sniffed from its prompt."""
    import re
    kinds = set()
    for path in (ROOT / "scilink").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"request_human_feedback\((?:.|\n){0,600}?\)", text):
            kinds.update(re.findall(r'kind="([a-z_]+)"', m.group(0)))
    assert kinds, "no gates found"
    missing = kinds - set(vocabulary.QUESTION_WIDGETS)
    assert not missing, f"gate kinds without a widget: {sorted(missing)}"


def test_question_labels_layering():
    base = vocabulary.question_labels("keep_or_revert")
    assert base["keep"] == "Keep user-guided fit"
    reopen = vocabulary.question_labels("keep_or_revert", "plan_reopen")
    assert reopen["keep"] == "Adopt the revision" and reopen["submit"] == "Adopt with changes"
    assert vocabulary.question_labels("nonsense") == vocabulary.QUESTION_LABELS["default"]
    assert vocabulary.question_widget("nonsense") == "generic"
    # the pickers' accept label names the pick once the presenter fills it
    assert "{pick}" in vocabulary.question_labels("bestofn_select")["accept"]
