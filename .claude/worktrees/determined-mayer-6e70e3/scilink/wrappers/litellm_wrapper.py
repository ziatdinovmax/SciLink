"""
Unified LLM gateway via LiteLLM.

All SciLink agents interact with LLMs through a single contract:

    response = model.generate_content(prompt_parts)
    text = response.text

This module implements that contract for 100+ providers (Google, OpenAI,
Anthropic, Cohere, Ollama, Azure, etc.) by routing through LiteLLM. Agent
and controller code stays completely provider-agnostic — the only
provider-specific logic lives here.

The interface originated from Google's Generative AI SDK but has become
a self-standing abstraction: a flat list of mixed-content parts in,
a simple .text accessor out. See OpenAIAsGenerativeModel for the direct
OpenAI-only equivalent.

Classes:
    LiteLLMGenerativeModel  — Text/vision generation (the primary wrapper)
    LiteLLMChatSession      — Multi-turn chat with history management
    LiteLLMEmbeddingModel   — Embedding generation with size-aware batching

Replaces the earlier single-provider wrappers:
    google_wrapper.py, google_wrapper_embeddings.py, llama_wrapper.py
"""

import os
import io
import re
import base64
import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


# OpenAI reasoning-class models (gpt-5*, o1*, o3*, o4*) only accept the default
# temperature/top_p. Sending custom values raises "Unsupported value" errors.
_REASONING_MODEL_RE = re.compile(r'^(gpt-5|o[134])(?!\d)', re.IGNORECASE)


def _is_openai_reasoning_model(model: str) -> bool:
    if not model:
        return False
    name = model.split('/', 1)[-1]
    return bool(_REASONING_MODEL_RE.match(name))

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None


def _ensure_gemini_api_key():
    """
    Ensure GEMINI_API_KEY is set for LiteLLM.
    Falls back to GOOGLE_API_KEY for backward compatibility.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            os.environ["GEMINI_API_KEY"] = google_key

def _normalize_model_name(model: str) -> str:
    """
    Auto-detect and add the LiteLLM provider prefix if not already present.

    LiteLLM uses prefixes (``gemini/``, ``openai/``, ``anthropic/``) to route
    requests to the correct provider. This helper infers the prefix from
    common model name patterns so callers can pass bare names like
    ``"gemini-2.0-flash"`` or ``"claude-sonnet-4-20250514"``.
    """
    if not model:
        return model
        
    if '/' in model:
        # Already has prefix
        return model
    
    model_lower = model.lower()
    
    # Google Gemini models
    if 'gemini' in model_lower:
        return f"gemini/{model}"
    
    # OpenAI models
    if model_lower.startswith(('gpt-', 'o1-', 'o3-', 'text-embedding', 'davinci', 'curie', 'babbage', 'ada')):
        return f"openai/{model}"
    
    # Anthropic models
    if model_lower.startswith('claude'):
        return f"anthropic/{model}"
    
    # Default: return as-is, let LiteLLM figure it out
    return model

def _check_litellm():
    """Raise ImportError if LiteLLM is not available."""
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "LiteLLM is required for public deployments. "
            "Install with: pip install litellm"
        )
    
    _ensure_gemini_api_key()
    
    # Suppress verbose logging
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    
    litellm.suppress_debug_info = True


class LiteLLMGenerativeModel:
    """
    Unified LLM interface backed by LiteLLM, supporting 100+ providers.

    This is the primary model wrapper used throughout SciLink. Every agent
    controller calls ``model.generate_content(prompt_parts)`` and reads
    ``response.text`` — this class translates that contract into the
    appropriate provider API call and back.

    What it handles so callers don't have to:
        - Provider routing via model name prefixes (auto-detected)
        - Prompt format translation (flat parts list → chat messages)
        - Image encoding (bytes/PIL → base64 data URLs)
        - Generation config mapping (max_output_tokens → max_tokens, etc.)
        - Response normalization (.text, .candidates, tool calls)
        - JSON extraction from markdown fences and preamble text

    Usage::

        # Google Gemini
        model = LiteLLMGenerativeModel("gemini/gemini-2.0-flash", api_key="...")

        # OpenAI
        model = LiteLLMGenerativeModel("gpt-4o", api_key="...")

        # Anthropic
        model = LiteLLMGenerativeModel("claude-sonnet-4-20250514", api_key="...")

        # Local Ollama
        model = LiteLLMGenerativeModel("ollama/llama3")

        # All use the same interface:
        response = model.generate_content(["Hello!", image_dict])
        print(response.text)

    Model Name Format:
        Provider prefixes are auto-detected from common model name patterns,
        but can be specified explicitly:

        - Google:    ``"gemini/gemini-2.0-flash"``
        - OpenAI:    ``"gpt-4o"`` or ``"openai/gpt-4o"``
        - Anthropic: ``"claude-sonnet-4-20250514"`` or ``"anthropic/claude-sonnet-4-20250514"``
        - Azure:     ``"azure/deployment-name"``
        - Ollama:    ``"ollama/llama3"``

        See: https://docs.litellm.ai/docs/providers
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List] = None,
        timeout: Optional[int] = 1200,
    ):
        """
        Initialize the LiteLLM model.

        Args:
            model: LiteLLM model string (e.g., "gemini/gemini-2.0-flash")
            api_key: API key for the provider
            base_url: Optional custom API base URL
            system_instruction: Default system message
            tools: Default tools for function calling
            timeout: Request timeout in seconds (default: 1200)
        """
        _check_litellm()

        self.model = _normalize_model_name(model)
        self.api_key = api_key
        self.base_url = base_url
        self.system_instruction = system_instruction
        self.tools = tools
        self.timeout = timeout
    
    def generate_content(
        self,
        contents: Union[str, List],
        generation_config: Any = None,
        safety_settings: Any = None,
        stream: bool = False,
        tools: Optional[List] = None,
    ):
        """
        Generate a response from mixed-content prompt parts.

        This is the single method all SciLink controllers call. It accepts
        the internal prompt format (a flat list of strings and image dicts),
        translates it to the provider's API, and returns a normalized response.

        Args:
            contents: Prompt string or list of parts — strings, PIL Images,
                and/or {mime_type, data} image dicts. The flat-list format
                keeps call sites simple; provider-specific message nesting
                is handled internally.
            generation_config: Optional config object or dict. Attributes are
                mapped to provider parameters (e.g., max_output_tokens → max_tokens).
            safety_settings: Accepted for interface compatibility; ignored.
            stream: If True, return a streaming iterator of partial responses.
            tools: Tool definitions for function calling (overrides defaults).

        Returns:
            SimpleNamespace with:
                .text — the model's text output (JSON fences stripped)
                .candidates — list of candidates, each with .content and
                    .finish_reason (int: 1=stop, 0=length, 2=tool_calls)
        """
        messages = self._build_messages(contents)
        params = self._build_params(generation_config, tools or self.tools)
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                api_base=self.base_url,
                stream=stream,
                timeout=self.timeout,
                **params
            )
            
            if stream:
                return self._handle_stream(response)
            
            return self._to_legacy_response(response)
            
        except Exception as e:
            logging.error(f"LiteLLM generation error: {e}")
            raise
    
    def _handle_stream(self, stream):
        """Yield legacy-format responses from stream."""
        for chunk in stream:
            yield self._to_legacy_response(chunk, is_stream=True)
    
    def start_chat(
        self,
        history: Optional[List] = None,
        enable_automatic_function_calling: bool = False
    ) -> 'LiteLLMChatSession':
        """
        Start a chat session.
        
        Args:
            history: Optional conversation history
            enable_automatic_function_calling: Enable auto function calling
        
        Returns:
            LiteLLMChatSession instance
        """
        return LiteLLMChatSession(
            model=self,
            history=history,
            enable_afc=enable_automatic_function_calling
        )
    
    def _build_messages(self, contents: Union[str, List]) -> List[Dict]:
        """Convert contents to LiteLLM/OpenAI message format."""
        messages = []
        
        if self.system_instruction:
            messages.append({"role": "system", "content": self.system_instruction})
        
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            user_content = self._convert_parts(contents)
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": str(contents)})
        
        return messages
    
    def _convert_parts(self, parts: List) -> Union[str, List[Dict]]:
        """
        Convert a flat list of prompt parts to chat-API content format.

        Handles strings, PIL Images, and {mime_type, data} image dicts.
        When all parts are text-only, joins them into a single string
        for maximum provider compatibility.
        """
        converted = []
        
        for part in parts:
            if isinstance(part, str):
                converted.append({"type": "text", "text": part})
            
            elif Image and isinstance(part, Image.Image):
                buf = io.BytesIO()
                part.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                converted.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            
            elif isinstance(part, dict):
                if "mime_type" in part and "data" in part:
                    data = part["data"]
                    if isinstance(data, bytes):
                        b64 = base64.b64encode(data).decode("utf-8")
                    else:
                        b64 = data
                    converted.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{part['mime_type']};base64,{b64}"}
                    })
                elif "text" in part:
                    converted.append({"type": "text", "text": part["text"]})
                else:
                    converted.append({"type": "text", "text": str(part)})
            
            else:
                converted.append({"type": "text", "text": str(part)})
        
        # Simplify if only text parts
        if all(p.get("type") == "text" for p in converted):
            return " ".join(p["text"] for p in converted)
        
        return converted
    
    def _build_params(self, generation_config: Any, tools: Optional[List]) -> Dict:
        """Build LiteLLM parameters from generation config."""
        params = {}
        
        if generation_config:
            if isinstance(generation_config, dict):
                cfg = generation_config
            elif hasattr(generation_config, '__dict__'):
                cfg = vars(generation_config)
            else:
                cfg = {}
            
            mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "max_output_tokens": "max_tokens",
                "stop_sequences": "stop",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
            }

            reasoning = _is_openai_reasoning_model(self.model)
            for old, new in mapping.items():
                if reasoning and old in ("temperature", "top_p"):
                    continue
                val = cfg.get(old)
                if val is not None:
                    params[new] = val
        
        if tools:
            params["tools"] = tools
        
        return params
    
    def _to_legacy_response(self, response, is_stream: bool = False):
        """Normalize a LiteLLM/OpenAI response into the unified format (.text, .candidates)."""
        candidates = []
        
        choices = getattr(response, "choices", None) or []
        
        if not choices:
            logging.warning(f"LiteLLM response has no choices")
            return SimpleNamespace(
                text="",
                candidates=[SimpleNamespace(
                    content=SimpleNamespace(parts=[], role="model"),
                    finish_reason=1
                )]
            )
        
        for choice in choices:
            parts = []
            
            message = getattr(choice, "message", None)
            if message is None:
                message = getattr(choice, "delta", None)
            
            text = ""
            if message:
                content = getattr(message, "content", None)
                if content:
                    text = self._clean_text(str(content))
            
            tool_calls = getattr(message, "tool_calls", None) if message else None
            if tool_calls:
                for tc in tool_calls:
                    func = getattr(tc, "function", None)
                    if func:
                        try:
                            args = json.loads(getattr(func, "arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        parts.append(SimpleNamespace(
                            function_call=SimpleNamespace(
                                name=getattr(func, "name", ""),
                                args=args
                            ),
                            text=None
                        ))
            
            if text:
                parts.append(SimpleNamespace(text=text, function_call=None))
            
            finish_reason = getattr(choice, "finish_reason", "stop")
            finish_map = {"stop": 1, "length": 0, "tool_calls": 2, "content_filter": 3}
            fr_int = finish_map.get(finish_reason, 1) if finish_reason else 1
            
            content = SimpleNamespace(parts=parts, role="model")
            candidates.append(SimpleNamespace(content=content, finish_reason=fr_int))
        
        first_text = ""
        if candidates and candidates[0].content.parts:
            for part in candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    first_text += part.text
        
        return SimpleNamespace(text=first_text, candidates=candidates)
    
    def _clean_text(self, text: str) -> str:
        """
        Extract JSON from LLM response so agents can parse .text directly.

        Many LLMs wrap JSON in markdown fences or preamble text. This method
        strips that away, trying (in order): ```json``` blocks, generic
        ``` blocks, and raw JSON objects embedded in prose.
        """
        if not text:
            return ""
        
        text = text.strip()
        
        import re
        
        # Pattern 1: ```json ... ``` anywhere in text
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1).strip()
        
        # Pattern 2: ``` ... ``` anywhere in text
        code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
        if code_match:
            extracted = code_match.group(1).strip()
            if extracted.startswith('{') or extracted.startswith('['):
                return extracted
        
        # Pattern 3: Raw JSON object anywhere in text
        json_obj_match = re.search(r'(\{[\s\S]*\})', text)
        if json_obj_match:
            potential = json_obj_match.group(1)
            if '"' in potential and ':' in potential:
                return potential
        
        return text


class LiteLLMChatSession:
    """
    Multi-turn chat session with automatic history management.

    Wraps LiteLLMGenerativeModel to maintain a running message list,
    appending user/assistant turns after each exchange. Provides the
    same send_message() / history interface used by agents that need
    conversational context (e.g., iterative refinement loops).
    """
    
    def __init__(
        self,
        model: LiteLLMGenerativeModel,
        history: Optional[List] = None,
        enable_afc: bool = False
    ):
        self._model = model
        self._history = self._convert_history(history) if history else []
        self._enable_afc = enable_afc
    
    def _convert_history(self, legacy_history: List) -> List[Dict]:
        """Convert legacy history format to messages."""
        messages = []
        for entry in legacy_history:
            if isinstance(entry, dict):
                role = entry.get("role", "user")
                parts = entry.get("parts", [])
            else:
                role = getattr(entry, "role", "user")
                parts = getattr(entry, "parts", [])
            
            text_parts = []
            for part in parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
                else:
                    text_parts.append(str(part))
            
            messages.append({"role": role, "content": " ".join(text_parts)})
        
        return messages
    
    @property
    def history(self) -> List:
        """Return conversation history."""
        return self._history
    
    def send_message(
        self,
        content: Union[str, List],
        generation_config: Any = None,
        stream: bool = False
    ):
        """
        Send message and update history.
        
        Args:
            content: Message content (string or list of parts)
            generation_config: Optional generation config
            stream: If True, return streaming iterator
        
        Returns:
            Response object with .text and .candidates
        """
        messages = []
        
        if self._model.system_instruction:
            messages.append({"role": "system", "content": self._model.system_instruction})
        
        messages.extend(self._history)
        
        if isinstance(content, str):
            user_content = content
        else:
            user_content = self._model._convert_parts(content)
        
        messages.append({"role": "user", "content": user_content})
        
        params = self._model._build_params(generation_config, self._model.tools)
        
        response = litellm.completion(
            model=self._model.model,
            messages=messages,
            api_key=self._model.api_key,
            api_base=self._model.base_url,
            stream=stream,
            **params
        )
        
        if stream:
            return self._handle_stream(response, user_content)
        
        self._history.append({"role": "user", "content": user_content})
        
        legacy_response = self._model._to_legacy_response(response)
        
        if legacy_response.text:
            self._history.append({"role": "assistant", "content": legacy_response.text})
        
        return legacy_response
    
    def _handle_stream(self, stream, user_content):
        """Handle streaming response."""
        full_text = ""
        for chunk in stream:
            legacy_chunk = self._model._to_legacy_response(chunk, is_stream=True)
            full_text += legacy_chunk.text
            yield legacy_chunk
        
        self._history.append({"role": "user", "content": user_content})
        self._history.append({"role": "assistant", "content": full_text})


class LiteLLMEmbeddingModel:
    """
    Unified embedding interface backed by LiteLLM.

    Provides a single embed_content() method that works across providers
    (Google, OpenAI, Cohere, etc.) with automatic handling of payload
    size limits:

    - Size-aware batching: splits inputs to stay under ~30 MB per request
    - Count limiting: caps batches at 100 items
    - Automatic retry: on payload-too-large errors, recursively halves
      the batch and retries each half

    Usage::

        embedder = LiteLLMEmbeddingModel("text-embedding-3-small", api_key="...")
        response = embedder.embed_content(content=["Hello", "World"])
        vectors = response["embedding"]  # [[...], [...]]
    """
    
    # Max payload size ~30MB to stay under 40MB limit with overhead
    MAX_BATCH_BYTES = 30 * 1024 * 1024
    # Also limit by count to avoid other API limits
    MAX_BATCH_SIZE = 100
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the embedding model.
        
        Args:
            model: Default model name (can be overridden in embed_content)
            api_key: API key for the provider
            base_url: Optional custom API base URL
        """
        _check_litellm()
        
        self.model = _normalize_model_name(model) if model else None
        self.api_key = api_key
        self.base_url = base_url
    
    def embed_content(
        self,
        model: Optional[str] = None,
        content: Union[str, List[str]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings with automatic size-aware batching.
        
        Args:
            model: Model name (overrides default)
            content: Text or list of texts to embed
            task_type: Ignored (kept for interface compatibility)
        
        Returns:
            Dict with 'embedding' key containing vector(s)
        """
        effective_model = model or self.model
        if not effective_model:
            raise ValueError("Model must be specified")
        
        effective_model = _normalize_model_name(effective_model)
        
        is_single = isinstance(content, str)
        inputs = [content] if is_single else content
        
        try:
            # Create size-aware batches to avoid payload limits
            batches = self._create_size_aware_batches(inputs)
            
            all_embeddings = []
            for batch_idx, batch in enumerate(batches):
                try:
                    response = litellm.embedding(
                        model=effective_model,
                        input=batch,
                        api_key=self.api_key,
                        api_base=self.base_url
                    )
                    
                    batch_embeddings = [item["embedding"] for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for payload size errors
                    if "payload size" in error_str or "too large" in error_str or "exceeds the limit" in error_str:
                        logging.warning(f"Batch {batch_idx + 1} too large, splitting and retrying...")
                        # Recursively embed with smaller batches
                        sub_embeddings = self._embed_with_retry(effective_model, batch)
                        all_embeddings.extend(sub_embeddings)
                    else:
                        raise
            
            if is_single:
                return {"embedding": all_embeddings[0]}
            return {"embedding": all_embeddings}
            
        except Exception as e:
            logging.error(f"LiteLLM embedding error: {e}")
            raise
    
    def _create_size_aware_batches(self, inputs: List[str]) -> List[List[str]]:
        """
        Split inputs into batches that won't exceed payload limits.
        
        Uses both byte size and count limits to stay within API constraints.
        """
        batches = []
        current_batch = []
        current_size = 0
        
        for text in inputs:
            text_size = len(text.encode('utf-8'))
            
            # Check if adding this text would exceed limits
            would_exceed_size = current_size + text_size > self.MAX_BATCH_BYTES
            would_exceed_count = len(current_batch) >= self.MAX_BATCH_SIZE
            
            if (would_exceed_size or would_exceed_count) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            
            # Handle individual texts that are too large
            if text_size > self.MAX_BATCH_BYTES:
                logging.warning(
                    f"Single text too large ({text_size:,} bytes), truncating to fit limit..."
                )
                # Truncate to fit - estimate conservatively
                max_chars = (self.MAX_BATCH_BYTES * 3) // 4  # Account for UTF-8 overhead
                text = text[:max_chars]
                text_size = len(text.encode('utf-8'))
            
            current_batch.append(text)
            current_size += text_size
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _embed_with_retry(
        self, 
        model: str, 
        texts: List[str], 
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Recursively embed texts, splitting on size errors.
        
        If a batch fails due to size, split it in half and retry.
        """
        if not texts:
            return []
        
        # Base case: single text
        if len(texts) == 1:
            try:
                response = litellm.embedding(
                    model=model,
                    input=texts,
                    api_key=self.api_key,
                    api_base=self.base_url
                )
                return [item["embedding"] for item in response.data]
            except Exception as e:
                error_str = str(e).lower()
                if "payload size" in error_str or "too large" in error_str:
                    # Single text is too large - truncate it
                    logging.warning("Single text still too large, truncating further...")
                    truncated = texts[0][:len(texts[0]) // 2]
                    response = litellm.embedding(
                        model=model,
                        input=[truncated],
                        api_key=self.api_key,
                        api_base=self.base_url
                    )
                    return [item["embedding"] for item in response.data]
                raise
        
        # Try the full batch first
        try:
            response = litellm.embedding(
                model=model,
                input=texts,
                api_key=self.api_key,
                api_base=self.base_url
            )
            return [item["embedding"] for item in response.data]
        except Exception as e:
            error_str = str(e).lower()
            if "payload size" in error_str or "too large" in error_str or "exceeds the limit" in error_str:
                # Split in half and retry each half
                mid = len(texts) // 2
                left_half = texts[:mid]
                right_half = texts[mid:]
                
                logging.info(f"Splitting batch of {len(texts)} into {len(left_half)} + {len(right_half)}")
                
                left_embeddings = self._embed_with_retry(model, left_half, max_retries)
                right_embeddings = self._embed_with_retry(model, right_half, max_retries)
                
                return left_embeddings + right_embeddings
            raise