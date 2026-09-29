"""The meta's prompt names the knowledge base that is attached.

`_kb_note` listed only DETACHED knowledge bases and returned "" once one was
attached, so the meta's model was never told a KB was attached, although the
meta passes it to its planning child. Live, asked for a plan that needed facts
recorded in the attached KB, the meta searched uploads and session state and
never delegated to planning.
"""

import json
from pathlib import Path

import pytest

from scilink.agents.meta_agent.meta_orchestrator import (
    MetaMode, MetaOrchestratorAgent, get_system_prompt)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _store_kb(home: Path, name="lab-kb", sources=("lab.md",), description=""):
    d = home / "knowledge_bases" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"name": name, "embedding_model": "m", "sources": list(sources),
         "description": description}))
    return d


def _meta(knowledge_dir=None):
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m.knowledge_dir = knowledge_dir
    m._shared_kb_candidate = None
    m.meta_mode = MetaMode.AUTONOMOUS
    m.messages = [{"role": "system", "content": "stale"}]
    return m


def test_an_attached_store_kb_is_named_with_its_sources(home):
    kb = _store_kb(home, description="lab records for batch A7")
    note = _meta(kb)._kb_note()
    assert "KNOWLEDGE BASE (attached)" in note
    assert "'lab-kb'" in note and "lab records for batch A7" in note and "lab.md" in note
    assert "delegate_to_planning" in note
    assert "NOT attached" not in note


def test_an_attached_plain_directory_is_named_by_path(home, tmp_path):
    d = tmp_path / "my_kb"
    d.mkdir()
    (d / "default_kb_docs.sources.json").write_text(json.dumps([{"path": "x", "files": ["notes.pdf"]}]))
    note = _meta(d)._kb_note()
    assert str(d) in note and "notes.pdf" in note and "(attached)" in note


def test_detached_kbs_are_still_listed_when_none_is_attached(home):
    _store_kb(home, name="other-kb", sources=("paper.pdf",))
    note = _meta(None)._kb_note()
    assert "KNOWLEDGE BASES (detached)" in note and "'other-kb'" in note
    assert "(attached)" not in note


def test_attaching_moves_the_prompt_from_detached_to_attached(home):
    kb = _store_kb(home)
    m = _meta(None)
    m.set_meta_mode(MetaMode.AUTONOMOUS)
    assert "NOT attached" in m.messages[0]["content"]
    m.knowledge_dir = kb                      # what attach_knowledge_dir sets
    m.set_meta_mode(m.meta_mode)              # and how it refreshes the prompt
    prompt = m.messages[0]["content"]
    assert prompt.startswith(get_system_prompt(MetaMode.AUTONOMOUS))
    assert "(attached)" in prompt and "'lab-kb'" in prompt and "NOT attached" not in prompt
