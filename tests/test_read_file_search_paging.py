"""#644 — a search that hits the cap must say so, say where the rest is,
and be pageable; the tool descriptions must not claim exhaustiveness.

Live: a broad pattern over a 267-page DOE PDF returned its first 40 hits
(all front matter) with a soft note, while the tool description promised
"every matching line"; the agent cited the document as consulted.
"""
import json
import re
from pathlib import Path

import pytest

from scilink.utils.file_io import (SEARCH_HIT_CAP, read_file_content,
                                   window_lines)

ROOT = Path(__file__).resolve().parents[1]
TOOL_FILES = [
    "scilink/agents/planning_agents/orchestrator_tools.py",
    "scilink/agents/exp_agents/analysis_orchestrator_tools.py",
    "scilink/agents/sim_agents/simulation_orchestrator_tools.py",
    "scilink/agents/meta_agent/meta_orchestrator_tools.py",
]


def _lines(n=1000, every=5):
    # a "hit" every `every` lines, so 200 hits spread evenly over the file
    return [(f"hit {i}\n" if i % every == 0 else f"filler {i}\n") for i in range(n)]


def test_first_page_is_truncated_with_a_next_page_and_a_density_map():
    out = window_lines(_lines(), search="hit")
    assert out["matches"] == 200 and out["shown"] == SEARCH_HIT_CAP
    assert out["match_offset"] == 1 and out["truncated"] is True
    assert out["next_match_offset"] == SEARCH_HIT_CAP + 1
    assert out["match_lines"][0] == 1 and out["match_lines"][-1] == 196
    assert "TRUNCATED: 160 more match(es), up to line 996" in out["content"]
    assert "match_offset=41" in out["content"]
    assert len(out["match_density"]) == 10
    assert all(d["matches"] == 20 for d in out["match_density"])
    assert "Where the matches are" in out["content"]


def test_paging_walks_every_match_exactly_once():
    lines = _lines()
    seen, off, pages = [], 1, 0
    while off is not None:
        out = window_lines(lines, search="hit", match_offset=off)
        seen += out["match_lines"]
        off = out["next_match_offset"]
        pages += 1
    assert pages == 5 and seen == [i + 1 for i in range(0, 1000, 5)]
    last = window_lines(lines, search="hit", match_offset=161)
    assert last["truncated"] is False and last["next_match_offset"] is None
    assert "TRUNCATED" not in last["content"]


def test_paging_past_the_end_is_explained_not_empty():
    out = window_lines(_lines(), search="hit", match_offset=999)
    assert out["shown"] == 0 and out["matches"] == 200
    assert "past the last match (#200, line 996)" in out["content"]


def test_a_search_under_the_cap_is_untruncated_and_has_no_density_noise():
    out = window_lines(_lines(), search="hit 99")
    assert out["truncated"] is False and out["next_match_offset"] is None
    assert out["match_density"] == [] and "Where the matches" not in out["content"]


def test_match_offset_rides_through_read_file_content(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("".join(_lines()))
    out = read_file_content(f, search="hit", match_offset=41)
    assert out["status"] == "success" and out["match_offset"] == 41
    assert out["match_lines"][0] == 201


@pytest.mark.parametrize("rel", TOOL_FILES)
def test_every_orchestrator_describes_the_cap_and_paging(rel):
    src = (ROOT / rel).read_text()
    i = src.index('name="read_file"')
    # join the implicitly concatenated string literals so a phrase that
    # wraps across source lines is matched as the model sees it
    reg = re.sub(r'"\s*\n\s*"', "", src[i:i + 7000])
    assert "Returns every matching line" not in reg
    assert "up to 40 matching lines" in reg
    assert "next_match_offset" in reg and '"match_offset"' in reg
    assert "has NOT read the document" in reg


@pytest.mark.parametrize("rel", TOOL_FILES)
def test_every_read_file_wrapper_forwards_match_offset(rel):
    src = (ROOT / rel).read_text()
    i = src.index("def read_file(")
    body = src[i:i + 2500]
    assert "match_offset: int = None" in body
    assert "match_offset=match_offset" in body
