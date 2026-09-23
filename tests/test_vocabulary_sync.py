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
