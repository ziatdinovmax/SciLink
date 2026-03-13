"""
Novita provider wrapper using OpenAI-compatible API.

Novita provides an OpenAI-compatible API endpoint at:
    https://api.novita.ai/openai

This wrapper extends OpenAIAsGenerativeModel to hardcode the Novita endpoint
while maintaining the same interface contract:
    response = model.generate_content(prompt_parts)
    text = response.text

Usage:
    from wrappers import NovitaGenerativeModel
    model = NovitaGenerativeModel("deepseek/deepseek-v3.2", api_key="...")
    response = model.generate_content(["Hello!"])
    print(response.text)
"""

import openai

from .openai_wrapper import OpenAIAsGenerativeModel

# Novita's OpenAI-compatible endpoint
NOVITA_BASE_URL = "https://api.novita.ai/openai"

# Default model IDs for Novita
# Use '/' separator as per Novita's naming convention
DEFAULT_MODELS = [
    "deepseek/deepseek-v3.2",  # Default model
    "zai-org/glm-5",  # GLM-5 from Zhipu AI
    "minimax/minimax-m2.5",  # MiniMax M2.5
]


class NovitaGenerativeModel(OpenAIAsGenerativeModel):
    """
    Unified LLM interface backed by Novita's OpenAI-compatible API.

    Novita is an OpenAI-compatible provider that supports multiple models
    through a unified endpoint. This wrapper configures the Novita endpoint
    while maintaining the same interface as OpenAIAsGenerativeModel.

    All SciLink agents interact with LLMs through a single contract:
        response = model.generate_content(prompt_parts)
        text = response.text

    This wrapper implements that contract for Novita, using the OpenAI-compatible
    Chat Completions API.

    Input:  A flat list of mixed-content parts — strings and image dicts
            ({mime_type, data}) — so callers never deal with provider-specific
            message/role nesting.
    Output: A SimpleNamespace with .text and .candidates, giving callers a
            uniform accessor regardless of backend.

    Usage:
        # Novita with default model
        model = NovitaGenerativeModel(api_key="...")
        response = model.generate_content(["Hello!"])
        print(response.text)

        # Novita with specific model
        model = NovitaGenerativeModel("deepseek/deepseek-v3.2", api_key="...")
        response = model.generate_content(["Analyze this..."])
        print(response.text)

    Model IDs (use '/' separator):
        - deepseek/deepseek-v3.2 (default)
        - zai-org/glm-5
        - minimax/minimax-m2.5
    """

    def __init__(
        self,
        model: str = DEFAULT_MODELS[0],
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        """
        Initialize Novita model with OpenAI-compatible endpoint.

        Args:
            model: Model ID (default: deepseek/deepseek-v3.2). Use '/' separator.
            api_key: Novita API key (NOVITA_API_KEY environment variable)
            base_url: Override Novita endpoint (rarely needed)
            timeout: Request timeout in seconds (default: 300)
        """
        # Use Novita's default endpoint unless overridden
        effective_base_url = base_url or NOVITA_BASE_URL

        super().__init__(
            model=model, api_key=api_key, base_url=effective_base_url, timeout=timeout
        )
