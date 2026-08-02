import io
import re
import base64
from types import SimpleNamespace
from PIL import Image
import openai


# OpenAI reasoning-class models (gpt-5*, o1*, o3*, o4*) only accept the default
# temperature/top_p. Sending custom values raises "Unsupported value" errors.
_REASONING_MODEL_RE = re.compile(r'^(gpt-5|o[134])(?!\d)', re.IGNORECASE)


def _is_openai_reasoning_model(model: str) -> bool:
    if not model:
        return False
    name = model.split('/', 1)[-1]
    return bool(_REASONING_MODEL_RE.match(name))


class OpenAIAsGenerativeModel:
    """
    Unified LLM interface backed by an OpenAI-compatible Chat Completions API.

    All SciLink agents interact with LLMs through a single contract:
        response = model.generate_content(prompt_parts)
        text = response.text

    This wrapper implements that contract for OpenAI and any OpenAI-compatible
    endpoint (vLLM, Ollama, Azure, etc.), translating between the internal
    prompt format and the Chat Completions API.

    Input:  A flat list of mixed-content parts — strings and image dicts
            ({mime_type, data}) — so callers never deal with provider-specific
            message/role nesting.
    Output: A SimpleNamespace with .text and .candidates, giving callers a
            uniform accessor regardless of backend.

    The interface originated from Google's Generative AI SDK but has become a
    provider-agnostic abstraction in its own right. See LiteLLMGenerativeModel
    for the multi-provider equivalent.
    """

    # Default timeout (seconds) sent to the server via extra_body.
    # LiteLLM proxies respect this to override their own request_timeout.
    DEFAULT_TIMEOUT = 300

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: int | None = None):
        # Store attributes for access by orchestrator
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # Works with OpenAI and any OpenAI-compatible endpoint
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                    timeout=self.timeout)

    # ---------------------- public API ----------------------
    def generate_content(self, contents, generation_config=None, safety_settings=None):
        """
        Generate a response from mixed-content prompt parts.

        Args:
            contents: List of prompt parts — strings and/or image dicts
                ({mime_type: str, data: bytes}). This flat-list format keeps
                call sites simple: controllers just append text and images
                without worrying about provider-specific message structure.
            generation_config: Optional config object. Attributes are mapped
                to OpenAI parameters (e.g., max_output_tokens → max_tokens).
            safety_settings: Accepted for interface compatibility; ignored.

        Returns:
            SimpleNamespace with:
                .text — the model's text output (JSON fences stripped)
                .candidates — list of candidate responses
        """
        messages = self._prompt_parser(contents)
        params = self._map_gen_config(generation_config)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            extra_body={"timeout": self.timeout},
            **params,
        )

        # Build Gemini-like response
        finish_map = {"stop": 1, "length": 0, "tool_calls": 2, "content_filter": 3}
        candidates = []
        for ch in resp.choices:
            text = (ch.message.content or "")
            text = self._fix_json_format(text)
            fr = finish_map.get(ch.finish_reason, -1)
            candidates.append(SimpleNamespace(content=text, finish_reason=fr))

        first_text = candidates[0].content if candidates else ""
        final_response = SimpleNamespace(text=first_text, candidates=candidates)
        return final_response

    # ---------------------- helpers ----------------------
    def _fix_json_format(self, response_text: str) -> str:
        """Strip markdown code fences so agents can parse raw JSON from .text."""
        if "```" in response_text:
            response_text = response_text.replace("```", "")
        # be conservative: only strip leading 'json' fences, not legitimate content
        if response_text.lstrip().startswith("json"):
            response_text = response_text.lstrip()[4:]
        return response_text

    def _map_gen_config(self, cfg):
        """Translate generation config attributes to OpenAI parameter names."""
        if not cfg:
            return {}

        out = {}
        reasoning = _is_openai_reasoning_model(self.model)
        if not reasoning and getattr(cfg, "temperature", None) is not None:
            out["temperature"] = cfg.temperature
        if not reasoning and getattr(cfg, "top_p", None) is not None:
            out["top_p"] = cfg.top_p
        if getattr(cfg, "max_output_tokens", None) is not None:
            out["max_tokens"] = cfg.max_output_tokens
        if getattr(cfg, "presence_penalty", None) is not None:
            out["presence_penalty"] = cfg.presence_penalty
        if getattr(cfg, "frequency_penalty", None) is not None:
            out["frequency_penalty"] = cfg.frequency_penalty
        if getattr(cfg, "stop_sequences", None):
            out["stop"] = cfg.stop_sequences

        return out

    def _to_data_url(self, mime_type: str, data_bytes: bytes) -> str:
        b64 = base64.b64encode(data_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    def _pil_to_data_url(self, img: Image.Image, mime_type: str = "image/png") -> str:
        buf = io.BytesIO()
        fmt = "PNG" if mime_type.lower().endswith("png") else "JPEG"
        img.save(buf, format=fmt)
        return self._to_data_url(mime_type, buf.getvalue())

    def _prompt_parser(self, genai_parts):
        """
        Convert a flat list of prompt parts to OpenAI chat message format.

        Handles strings, PIL Images, and {mime_type, data} image dicts.
        When the prompt is text-only with a single part, emits a plain string
        message for maximum API compatibility (some endpoints don't support
        the structured content-block format).
        """
        # Handle case where a single string is passed directly
        if isinstance(genai_parts, str):
            return [{"role": "user", "content": genai_parts}]
        
        user_content = []
        has_images = False

        for part in genai_parts:
            if isinstance(part, str):
                user_content.append({"type": "text", "text": part})
                continue

            if isinstance(part, Image.Image):
                url = self._pil_to_data_url(part, "image/png")
                user_content.append({"type": "image_url", "image_url": {"url": url}})
                has_images = True
                continue

            if isinstance(part, dict):
                mime = part.get("mime_type", "")
                data = part.get("data", None)

                if isinstance(data, (bytes, bytearray)) and mime.startswith("image/"):
                    try:
                        url = self._to_data_url(mime, data)
                        user_content.append({"type": "image_url", "image_url": {"url": url}})
                        has_images = True
                        continue
                    except Exception:
                        pass

                user_content.append({"type": "text", "text": str(part)})
                continue

            user_content.append({"type": "text", "text": str(part)})

        # Only simplify if: no images AND exactly one text part
        # This is the most conservative change
        text_parts = [p for p in user_content if p.get("type") == "text"]
        
        if not has_images and len(text_parts) == 1:
            return [{"role": "user", "content": text_parts[0]["text"]}]
        
        # Otherwise keep structured format (original behavior)
        return [{"role": "user", "content": user_content}]