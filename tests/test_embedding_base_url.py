"""An embedding base URL routes ONLY the embeddings: the chat model keeps
its own route (vendor via LiteLLM, or the main proxy), the knowledge-base
clients point at the embedding endpoint with the embedding key (the main
key when none is given), and the old rules hold when it is absent."""
import logging

import pytest

from scilink.agents.planning_agents.planning_agent import PlanningAgent
from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
from scilink.wrappers.openai_wrapper import OpenAIAsGenerativeModel
from scilink.wrappers.openai_wrapper_embeddings import OpenAIAsEmbeddingModel
from scilink.wrappers.litellm_wrapper import LiteLLMEmbeddingModel


def _agent(tmp_path, **kw):
    return PlanningAgent(model_name="bedrock/us.anthropic.claude-opus-4-8", kb_base_path=str(tmp_path / "kb" / "k"),
                         output_dir=str(tmp_path), embedding_model="text-embedding-3-small", **kw)


def test_embedding_base_url_with_a_direct_chat_model(tmp_path, caplog):
    a = _agent(tmp_path, api_key="main", embedding_api_key="emb-key", embedding_base_url="http://emb.local/v1")
    assert isinstance(a.model, LiteLLMGenerativeModel)                       # chat: direct
    kb = a.kb_docs.embedding_client
    assert isinstance(kb, OpenAIAsEmbeddingModel) and str(kb.client.base_url).startswith("http://emb.local/v1")
    assert kb.client.api_key == "emb-key" and kb.model == "text-embedding-3-small"
    assert a.kb_code.embedding_client.client.api_key == "emb-key"


def test_embedding_base_url_alongside_a_main_proxy(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    a = _agent(tmp_path, api_key="proxy-key", base_url="http://chat.local/v1",
               embedding_api_key="emb-key", embedding_base_url="http://emb.local/v1")
    assert isinstance(a.model, OpenAIAsGenerativeModel) and str(a.model.client.base_url).startswith("http://chat.local/v1")
    kb = a.kb_docs.embedding_client
    assert str(kb.client.base_url).startswith("http://emb.local/v1") and kb.client.api_key == "emb-key"
    assert "ignored for internal proxy" not in caplog.text                   # the embedding key is USED


def test_embedding_base_url_without_a_key_falls_back_to_the_main_key(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    a = _agent(tmp_path, api_key="main", embedding_base_url="http://emb.local/v1")
    assert a.kb_docs.embedding_client.client.api_key == "main"
    assert "using the main api_key for the embedding endpoint" in caplog.text


def test_without_an_embedding_base_url_the_old_rules_hold(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    a = _agent(tmp_path, api_key="proxy-key", base_url="http://chat.local/v1", embedding_api_key="emb-key")
    kb = a.kb_docs.embedding_client
    assert isinstance(kb, OpenAIAsEmbeddingModel) and str(kb.client.base_url).startswith("http://chat.local/v1")
    assert kb.client.api_key == "proxy-key" and "ignored for internal proxy" in caplog.text
    a = _agent(tmp_path, api_key="main", embedding_api_key="emb-key")
    assert isinstance(a.kb_docs.embedding_client, LiteLLMEmbeddingModel)


def test_orchestrators_forward_the_embedding_base_url(tmp_path):
    from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent, AutonomyLevel
    o = PlanningOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="main", model_name="bedrock/us.anthropic.claude-opus-4-8",
                                  embedding_model="text-embedding-3-small", embedding_api_key="emb-key",
                                  embedding_base_url="http://emb.local/v1", autonomy_level=AutonomyLevel.CO_PILOT)
    assert o.embedding_base_url == "http://emb.local/v1"
    assert str(o.planner.kb_docs.embedding_client.client.base_url).startswith("http://emb.local/v1")
    from scilink.agents.meta_agent.meta_orchestrator import MetaOrchestratorAgent, MetaMode
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "m"), api_key="main", model_name="bedrock/us.anthropic.claude-opus-4-8",
                              embedding_model="text-embedding-3-small", embedding_api_key="emb-key",
                              embedding_base_url="http://emb.local/v1", meta_mode=MetaMode.AUTONOMOUS)
    assert m.embedding_base_url == "http://emb.local/v1" and m.embedding_api_key == "emb-key"
