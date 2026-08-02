"""
API Wrappers for Multi-Backend LLM Support

Two deployment modes:
1. Internal (Incubator) - OpenAI-compatible proxy endpoint
   → Uses OpenAIAsGenerativeModel, OpenAIAsEmbeddingModel
   
2. Public - Direct provider access via LiteLLM
   → Uses LiteLLMGenerativeModel, LiteLLMEmbeddingModel
   → Supports: Google, OpenAI, Anthropic, Cohere, and 100+ providers

Usage:
    # Internal proxy
    from wrappers import OpenAIAsGenerativeModel
    model = OpenAIAsGenerativeModel("model-name", api_key="...", base_url="https://proxy/v1")
    
    # Public (multi-provider)
    from wrappers import LiteLLMGenerativeModel
    model = LiteLLMGenerativeModel("gemini/gemini-2.0-flash", api_key="...")
"""

# Internal/Incubator - OpenAI-compatible endpoints
from .openai_wrapper import OpenAIAsGenerativeModel
from .openai_wrapper_embeddings import OpenAIAsEmbeddingModel

# Public - Multi-provider via LiteLLM
try:
    from .litellm_wrapper import (
        LiteLLMGenerativeModel,
        LiteLLMEmbeddingModel,
        LiteLLMChatSession,
        LITELLM_AVAILABLE,
    )
except ImportError:
    LiteLLMGenerativeModel = None
    LiteLLMEmbeddingModel = None
    LiteLLMChatSession = None
    LITELLM_AVAILABLE = False

__all__ = [
    # Internal proxy
    'OpenAIAsGenerativeModel',
    'OpenAIAsEmbeddingModel',
    # Public multi-provider
    'LiteLLMGenerativeModel',
    'LiteLLMEmbeddingModel',
    'LiteLLMChatSession',
    'LITELLM_AVAILABLE',
]