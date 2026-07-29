"""read_file: answering questions about the END and the CONTENT of a file.

Live: an agent needed to confirm a 35 KB white paper closed with a
References section. read_file read only from the top, and its truncation
notice named what was missing without naming any way to get it — so the
question was unanswerable at every parameter value the model could pass. It
asked five times, then gave up and reasoned from an earlier tool result.
"""

import json
from types import SimpleNamespace

import pytest

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools

# 500 body lines, a blank, the heading, two entries = 504 lines.
BODY = ([f"line {i}\n" for i in range(1, 501)]
        + ["\n", "## References\n", "[1] Boettiger et al. 2013\n",
           "[2] Dakos et al. 2024\n"])


@pytest.fixture
def read(tmp_path):
    doc = tmp_path / "paper.md"
    doc.write_text("".join(BODY))
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map, t.openai_schemas = {}, []
    t.orch = SimpleNamespace(base_dir=tmp_path, _active_output_subdir=None,
                             planner=SimpleNamespace())
    t._resolve_data_path = lambda p: (str(p), None)
    t._register_tool = lambda func, name, **kw: t.functions_map.setdefault(name, func)
    OrchestratorTools._register_all_tools(t)
    fn = t.functions_map["read_file"]
    return lambda **kw: json.loads(fn(file_path=str(doc), **kw))


def test_head_is_still_the_default(read):
    out = read()
    assert out["mode"] == "head" and out["shown_lines"] == "1-200"
    assert "line 1\n" in out["content"]
    assert "## References" not in out["content"]


def test_truncation_notice_names_the_way_out(read):
    """A notice that says only what is missing turns a long file into a dead
    end — which is exactly what happened."""
    out = read()
    assert out["truncated"] is True
    assert out["total_lines"] == 504
    assert "tail=true" in out["content"]
    assert "search=" in out["content"]


def test_tail_answers_how_the_document_ends(read):
    out = read(tail=True, max_lines=10)
    assert out["mode"] == "tail" and out["shown_lines"] == "495-504"
    assert "## References" in out["content"]
    assert "[2] Dakos et al. 2024" in out["content"]
    # and says what it skipped, without pretending to be the whole file
    assert "earlier lines not shown" in out["content"]


def test_search_answers_presence_definitively(read):
    out = read(search=r"^##\s*References")
    assert out["mode"] == "search" and out["matches"] == 1
    assert out["match_lines"] == [502]
    assert "## References" in out["content"]


def test_search_reports_absence_rather_than_nothing(read):
    out = read(search="Acknowledgements")
    assert out["matches"] == 0 and "(no matches)" in out["content"]
    assert out["total_lines"] == 504


def test_search_returns_context_and_line_numbers(read):
    out = read(search="Boettiger")
    assert out["match_lines"] == [503]
    assert "@@ line 503" in out["content"]
    assert "## References" in out["content"], "one line of context either side"


def test_search_is_capped_so_a_broad_pattern_cannot_dump_the_file(read):
    out = read(search="line")
    assert out["matches"] == 500
    assert len(out["match_lines"]) == 40
    assert "showing the first 40" in out["content"]
    assert len(out["content"]) < 6000


def test_a_bad_pattern_is_reported_not_raised(read):
    out = read(search="[unclosed")
    assert out["status"] == "error" and "Invalid search pattern" in out["message"]


def test_short_files_are_returned_whole(tmp_path):
    doc = tmp_path / "small.md"
    doc.write_text("one\ntwo\n")
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map, t.openai_schemas = {}, []
    t.orch = SimpleNamespace(base_dir=tmp_path, _active_output_subdir=None,
                             planner=SimpleNamespace())
    t._resolve_data_path = lambda p: (str(p), None)
    t._register_tool = lambda func, name, **kw: t.functions_map.setdefault(name, func)
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["read_file"](file_path=str(doc)))
    assert out["truncated"] is False and out["content"] == "one\ntwo\n"
    assert out["shown_lines"] == "1-2"


def test_json_reading_is_untouched(tmp_path):
    doc = tmp_path / "plan.json"
    doc.write_text(json.dumps({"a": 1}))
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map, t.openai_schemas = {}, []
    t.orch = SimpleNamespace(base_dir=tmp_path, _active_output_subdir=None,
                             planner=SimpleNamespace())
    t._resolve_data_path = lambda p: (str(p), None)
    t._register_tool = lambda func, name, **kw: t.functions_map.setdefault(name, func)
    OrchestratorTools._register_all_tools(t)
    out = json.loads(t.functions_map["read_file"](file_path=str(doc)))
    assert out["status"] == "success" and '"a": 1' in out["content"]


def test_the_schema_tells_the_model_not_to_re_read(read):
    from pathlib import Path
    src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
    i = src.index('name="read_file"')
    desc = src[i:i + 2600]
    assert "do not read it repeatedly" in desc
    assert "tail=true" in desc and "search=" in desc
