"""create_knowledge_base: a chat-side `scilink kb create`.

A session that indexed an uploaded report kept the index in its own folder,
and the next session started from nothing: no chat tool could save a new KB
to the store (only `add_to_knowledge_base`, which needs an existing one), so
reuse needed the terminal. The tool builds a named KB in the store, leaves it
detached, and refuses on a shared server.
"""

import json

import pytest

from scilink.agents.meta_agent.meta_orchestrator import (
    MetaMode, MetaOrchestratorAgent)
from scilink.agents.meta_agent.meta_orchestrator_tools import MetaOrchestratorTools
from scilink.knowledge import kb_store


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SCILINK_SHARED_STORE", raising=False)
    return tmp_path / "home"


def _meta(embedding_model=None):
    m = MetaOrchestratorAgent.__new__(MetaOrchestratorAgent)
    m.knowledge_dir = None
    m._shared_kb_candidate = None
    m.meta_mode = MetaMode.AUTONOMOUS
    m.messages = [{"role": "system", "content": "stale"}]
    m.embedding_model = embedding_model
    m.embedding_api_key = m.embedding_base_url = m.base_url = None
    m._children = {}
    return m


def _doc(tmp_path, name="report.md", text="Magnesium is rated critical."):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_create_builds_a_keyword_only_kb_without_an_embedding_model(home, tmp_path):
    m = _meta()
    manifest = m.create_kb_from_documents(
        "doe-report", [str(_doc(tmp_path))], description="DOE criticality")
    assert manifest["name"] == "doe-report"
    assert manifest["embedding_model"] is None and not manifest.get("n_vectors")
    assert manifest["n_chunks"] >= 1 and manifest["sources"] == ["report.md"]
    assert kb_store.read_manifest(kb_store.kb_path("doe-report")) is not None
    # Not attached, but offered: the detached list in the prompt names it.
    assert m.knowledge_dir is None
    assert "'doe-report'" in m.messages[0]["content"]
    assert "DOE criticality" in m.messages[0]["content"]


def test_create_embeds_with_the_sessions_model(home, tmp_path, monkeypatch):
    seen = {}

    def fake_build(target, embedding_model, api_key, base_url, *a, **k):
        seen.update(model=embedding_model, key=api_key, base_url=base_url)
        for suffix in (".faiss", ".json", ".sources.json"):
            (target / "default_kb_docs").with_suffix(suffix).write_text("[]")
        return 3, 3

    monkeypatch.setattr(kb_store, "_build_index_into", fake_build)
    m = _meta(embedding_model="bedrock/amazon.titan-embed-text-v2:0")
    m.embedding_api_key, m.embedding_base_url = "k", "https://embed.example"
    manifest = m.create_kb_from_documents("pw", [str(_doc(tmp_path))])
    assert seen == {"model": "bedrock/amazon.titan-embed-text-v2:0", "key": "k",
                    "base_url": "https://embed.example"}
    assert manifest["embedding_model"] == "bedrock/amazon.titan-embed-text-v2:0"


def test_an_existing_name_is_refused(home, tmp_path):
    m = _meta()
    m.create_kb_from_documents("pw", [str(_doc(tmp_path))])
    with pytest.raises(FileExistsError):
        m.create_kb_from_documents("pw", [str(_doc(tmp_path, "other.md"))])


def test_a_shared_server_refuses_create_and_add(home, tmp_path, monkeypatch):
    m = _meta()
    m.create_kb_from_documents("pw", [str(_doc(tmp_path))])
    monkeypatch.setenv("SCILINK_SHARED_STORE", "1")
    with pytest.raises(PermissionError, match="shared server"):
        m.create_kb_from_documents("pw2", [str(_doc(tmp_path, "b.md"))])
    with pytest.raises(PermissionError, match="shared server"):
        m.add_documents_to_kb([str(_doc(tmp_path, "c.md"))], name="pw")
    assert kb_store.read_manifest(kb_store.kb_path("pw2")) is None


def test_the_tool_reports_the_build_and_errors_as_json(home, tmp_path):
    m = _meta()
    tools = MetaOrchestratorTools(m)
    assert "create_knowledge_base" in tools.functions_map
    out = json.loads(tools.functions_map["create_knowledge_base"](
        name="pw", paths=str(_doc(tmp_path))))
    assert out["status"] == "success" and out["knowledge_base"] == "pw"
    assert out["documents"] == 1 and out["attached"] is False
    assert "keyword" in out["note"].lower()
    again = json.loads(tools.functions_map["create_knowledge_base"](
        name="pw", paths=[str(_doc(tmp_path))]))
    assert again["status"] == "error" and "already exists" in again["message"]
