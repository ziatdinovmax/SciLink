import openai
from typing import List

from openai._utils import maybe_transform
from openai.types import embedding_create_params
from openai.types.create_embedding_response import CreateEmbeddingResponse
from ..utils.child_process import refuse_in_spawn_bootstrap

class OpenAIAsEmbeddingModel:
    """
    Mimics Google's genai.embed_content function using an OpenAI-compatible API.
    """
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        # Retries are SciLink's (call_with_retries): the SDK's would sleep
        # inside the in-flight slot.
        refuse_in_spawn_bootstrap(type(self).__name__)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model

    def embed_content(self, model: str, content: List[str], task_type: str = None, title: str = None, **kwargs) -> dict:
        """
        Generates embeddings by calling the internal `_post` method
        """

        params = {
            "model": self.model,
            "input": content,
        }

        from .litellm_wrapper import call_with_retries
        response = call_with_retries(lambda: self.client.embeddings._post(
            "/embeddings",
            body=maybe_transform(params, embedding_create_params.EmbeddingCreateParams),
            cast_to=CreateEmbeddingResponse,
        ), None, model=self.model)

        if not response.data:
            raise ValueError(f"Invalid response from embedding API: 'data' key is empty or missing. Response: {response}")

        all_embedding_vectors = [item.embedding for item in response.data]

        return {'embedding': all_embedding_vectors}