"""The shared file I/O engine (#481): one reader / writer for every mode.

Behavior-preserving extraction of the planning / analysis / simulation
copies, plus the two guards the issue asked for: a cap on pretty-printed
JSON (windowed like text) and backup-on-overwrite for save_file.
"""

import json
from pathlib import Path

import pytest

from scilink.utils.file_io import (
    read_documents_combined, read_file_content, window_lines, write_text_file)

BODY = ([f"line {i}\n" for i in range(1, 501)]
        + ["\n", "## References\n", "[1] Boettiger et al. 2013\n",
           "[2] Dakos et al. 2024\n"])


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "paper.md"
    p.write_text("".join(BODY))
    return p


# ── windowing ────────────────────────────────────────────────────

def test_head_default_and_truncation_notice(doc):
    out = read_file_content(doc)
    assert out["status"] == "success" and out["mode"] == "head"
    assert out["shown_lines"] == "1-200" and out["truncated"] is True
    assert out["total_lines"] == 504
    assert "tail=true" in out["content"] and "search=" in out["content"]
    assert "## References" not in out["content"]


def test_tail_offset_search(doc):
    t = read_file_content(doc, tail=True, max_lines=10)
    assert t["mode"] == "tail" and t["shown_lines"] == "495-504"
    assert "[2] Dakos et al. 2024" in t["content"] and "earlier lines not shown" in t["content"]
    o = read_file_content(doc, offset=498, max_lines=3)
    assert o["shown_lines"] == "498-500" and "line 498" in o["content"]
    assert "Sections:" not in o["content"]       # one heading: no outline (needs 2+)
    s = read_file_content(doc, search=r"^##\s*References")
    assert s["mode"] == "search" and s["matches"] == 1 and s["match_lines"] == [502]
    assert "@@ line 502" in s["content"]
    none = read_file_content(doc, search="Acknowledgements")
    assert none["matches"] == 0 and "(no matches)" in none["content"]
    broad = read_file_content(doc, search="line")
    assert broad["matches"] == 500 and len(broad["match_lines"]) == 40
    assert "showing the first 40" in broad["content"]
    bad = read_file_content(doc, search="[unclosed")
    assert bad["status"] == "error" and "Invalid search pattern" in bad["message"]


def test_short_file_whole_and_full_read_stems(tmp_path):
    small = tmp_path / "small.md"
    small.write_text("one\ntwo\n")
    out = read_file_content(small)
    assert out["truncated"] is False and out["content"] == "one\ntwo\n"
    assert out["shown_lines"] == "1-2"
    lit = tmp_path / "literature_search_1.md"
    lit.write_text("".join(f"row {i}\n" for i in range(1000)))
    whole = read_file_content(lit, full_read_stems=("literature_search",))
    assert whole["truncated"] is False and whole["total_lines"] == 1000
    # ...but offset / tail still window it, and the char cap still applies
    assert read_file_content(lit, full_read_stems=("literature_search",), tail=True)["truncated"] is True
    assert read_file_content(lit, full_read_stems=("literature_search",),
                             full_read_max_chars=100)["truncated"] is True


def test_missing_and_oversized(tmp_path):
    assert read_file_content(tmp_path / "nope.txt")["status"] == "error"
    big = tmp_path / "big.log"
    big.write_bytes(b"x" * (6 * 1024 * 1024))
    assert "too large" in read_file_content(big)["message"]
    assert read_file_content(big, text_cap_mb=10)["status"] == "success"


# ── JSON: the cap #481 asked for ─────────────────────────────────

def test_small_json_is_returned_whole_pretty_printed(tmp_path):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"a": 1, "b": [1, 2]}))
    out = read_file_content(p)
    assert out["status"] == "success" and out["truncated"] is False
    assert '"a": 1' in out["content"] and out["content"].startswith("{\n")


def test_large_json_is_windowed_not_dumped(tmp_path):
    p = tmp_path / "big.json"
    p.write_text(json.dumps({f"key_{i}": list(range(20)) for i in range(400)}))
    out = read_file_content(p)
    assert out["truncated"] is True and out["shown_lines"] == "1-200"
    assert out["total_lines"] > 8000
    assert "TRUNCATED READ" in out["content"]
    found = read_file_content(p, search="key_399")
    assert found["matches"] == 1
    assert len(read_file_content(p, max_lines=50)["content"]) < 5000


# ── tabular + documents ──────────────────────────────────────────

def test_csv_preview(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("a,b\n" + "".join(f"{i},{i*2}\n" for i in range(300)))
    out = read_file_content(p)
    assert out["status"] == "success" and out["content"].startswith("Shape: 300 rows × 2 columns")
    assert "showing first 100 rows" in out["content"]


def test_pdf_is_extracted_and_windowed(tmp_path):
    pytest.importorskip("fitz")
    pytest.importorskip("markdown_it")
    from scilink.utils.md_to_pdf import markdown_to_pdf
    md = tmp_path / "prop.md"
    md.write_text("# Proposal\n\n" + "".join(f"Aim {i}: MARKER-{i}. " for i in range(1, 30))
                  + "\n\n## References\n\n[1] MARKER-REF Boettiger 2013\n")
    pdf = markdown_to_pdf(md, tmp_path / "prop.pdf")
    out = read_file_content(pdf)
    assert "%PDF" not in out["content"] and "MARKER-1" in out["content"]
    assert out["extracted"] == "pdf" and out["n_pages"] >= 1
    assert read_file_content(pdf, search="MARKER-REF")["matches"] == 1
    docs = read_documents_combined([str(pdf), str(md), str(tmp_path / "gone.pdf")])
    assert docs["status"] == "success" and docs["n_documents"] == 2
    assert docs["errors"] == ["Not a file: " + str(tmp_path / "gone.pdf")]
    assert "## prop.pdf" in docs["text"] and "## prop.md" in docs["text"]


def test_read_documents_relative_and_empty(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    out = read_documents_combined(["notes.md"], base_dir=tmp_path)
    assert out["n_documents"] == 1 and "hello" in out["text"]
    assert read_documents_combined([])["status"] == "error"
    assert read_documents_combined(["nope.md"], base_dir=tmp_path)["status"] == "error"
    cap = read_documents_combined(["notes.md"], base_dir=tmp_path, max_chars=3)
    assert cap["combined_truncated"] is True and len(cap["text"]) == 3


# ── write: traversal-proof, append, backup-on-overwrite ──────────

def test_write_append_and_backup(tmp_path):
    r = write_text_file(tmp_path, "../../escape.txt", "x", subfolder="../protocols")
    assert r["status"] == "success" and r["created"] is True and r["overwritten"] is False
    assert (tmp_path / "protocols" / "escape.txt").read_text() == "x"
    assert not (tmp_path.parent.parent / "escape.txt").exists()
    r = write_text_file(tmp_path, "escape.txt", "more", subfolder="protocols", append=True)
    assert r["status"] == "success" and r["created"] is False
    assert (tmp_path / "protocols" / "escape.txt").read_text() == "xmore"
    # overwrite: previous content is preserved in a counter-suffixed sibling
    r = write_text_file(tmp_path, "escape.txt", "v2", subfolder="protocols")
    assert r["overwritten"] is True
    assert Path(r["backup"]).name == "escape.before_overwrite.txt"
    assert Path(r["backup"]).read_text() == "xmore"
    r = write_text_file(tmp_path, "escape.txt", "v3", subfolder="protocols")
    assert Path(r["backup"]).name == "escape.before_overwrite.2.txt"
    assert Path(r["backup"]).read_text() == "v2"
    assert (tmp_path / "protocols" / "escape.txt").read_text() == "v3"
    # opt out
    r = write_text_file(tmp_path, "escape.txt", "v4", subfolder="protocols",
                        backup_on_overwrite=False)
    assert "backup" not in r and r["overwritten"] is True
    assert write_text_file(tmp_path, "/", "x")["status"] == "error"
    assert write_text_file(tmp_path, "", "x")["status"] == "error"


def test_window_lines_direct_and_outline():
    out = window_lines(["a\n", "b\n", "c\n"], max_lines=2)
    assert out["truncated"] and out["shown_lines"] == "1-2"
    assert window_lines(["a\n"], max_lines=2)["truncated"] is False
    lines = ["# One\n"] + ["x\n"] * 300 + ["## Two\n", "y\n"]
    out = window_lines(lines, max_lines=10)
    assert "Sections: One @ line 1 · Two @ line 302" in out["content"]
