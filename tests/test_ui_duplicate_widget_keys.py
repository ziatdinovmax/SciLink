"""UI: a re-embedded document must not collide on widget keys.

Live crash (StreamlitDuplicateElementKey) after a white paper was revised
in place: the file surfaces in a LATER chat message under the SAME path, and
the download-button key was path-only.
"""
import ast
import re
from pathlib import Path

SRC = Path("scilink/ui/app.py").read_text()


def test_download_keys_are_message_scoped():
    for fam in ("dl_html_", "dl_md_"):
        keys = re.findall(rf'key=f"{fam}([^"]*)"', SRC)
        assert keys, fam
        for k in keys:
            assert k.startswith("{_mi}_"), f"{fam}{k} is not message-scoped"


def test_the_message_loop_provides_that_index():
    assert "for _mi, msg in enumerate(st.session_state.chat_messages):" in SRC
    assert "for msg in st.session_state.chat_messages:" not in SRC


def test_same_path_in_two_messages_yields_distinct_keys():
    """The exact live scenario: authored in one turn, revised in a later one."""
    path = "/s/delegations/02/white_paper_class2.md"
    keys = {f"dl_md_{mi}_{path}" for mi in (3, 7)}
    assert len(keys) == 2


def test_app_still_parses():
    ast.parse(SRC)
