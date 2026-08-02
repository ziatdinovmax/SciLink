import io
import re
import base64
from types import SimpleNamespace
from PIL import Image
import openai
import json


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
    Pretends to be a Google GenerativeModel while using an OpenAI-compatible Chat Completions API.
    - Accepts 'contents' as a list of Google-style prompt parts (strings or dicts with {mime_type, data})
    - Returns an object with .text and .candidates (each candidate has .content and .finish_reason int)
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        # Works with OpenAI and any OpenAI-compatible endpoint 
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # ---------------------- public API ----------------------
    def generate_content(self, contents, generation_config=None, safety_settings=None, tools=None):
        """
        contents: list of prompt parts (e.g., ["hello", {"mime_type": "image/png", "data": b"..."}])
        generation_config: optional dict; common keys mapped: temperature, top_p, max_output_tokens, stop,
                           presence_penalty, frequency_penalty
        safety_settings: ignored for OpenAI backends (kept for interface compatibility)
        """
        messages = self._prompt_parser(contents)
        params = self._map_gen_config(generation_config)
        oa_tools = genai_tools_to_openai_tools(tools)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            tools=oa_tools,       
            **params,
        )

        if not any(getattr(getattr(ch, "message", None), "tool_calls", None) for ch in resp.choices): print('!!'*20)
        # Build Gemini-like response
        finish_map = {"stop": 1, "length": 0, "tool_calls": 2, "content_filter": 3}

        candidates = []
        for ch in resp.choices:
            parts = []

            # --- tool calls -> Gemini-style function_call parts (if any) ---
            tool_called = False
            if getattr(ch.message, "tool_calls", None):
                for tc in ch.message.tool_calls:
                    name = getattr(tc.function, "name", None)
                    args_json = getattr(tc.function, "arguments", "{}")
                    try:
                        args = json.loads(args_json)
                    except Exception:
                        # keep raw if not valid JSON
                        args = {"_raw": args_json}
                    parts.append(SimpleNamespace(function_call=SimpleNamespace(name=name, args=args)))
                    tool_called = True

            # --- text -> Gemini-style text part (if any) ---
            text = (ch.message.content or "")
            text = self._fix_json_format(text)
            if text.strip():
                parts.append(SimpleNamespace(text=text))

            # --- finish_reason: mark TOOL_CALL when any function_call part exists ---
            if tool_called:
                fr_out = "TOOL_CALL"          # satisfies your parser's whitelist
            else:
                fr_out = finish_map.get(ch.finish_reason, -1)

            # --- wrap into Gemini-like content object with .parts ---
            content = SimpleNamespace(parts=parts)
            candidates.append(SimpleNamespace(content=content, finish_reason=fr_out))

        first_text = candidates[0].content if candidates else ""
        final_response = SimpleNamespace(text=first_text, candidates=candidates)
        return final_response

    # ---------------------- helpers ----------------------
    def _fix_json_format(self, response_text: str) -> str:
        # Trim ``` and stray "json" tags for downstream JSON parsing convenience
        if "```" in response_text:
            response_text = response_text.replace("```", "")
        # be conservative: only strip leading 'json' fences, not legitimate content
        if response_text.lstrip().startswith("json"):
            response_text = response_text.lstrip()[4:]
        return response_text

    def _map_gen_config(self, cfg):
        if not cfg:
            return {}
        # Map Gemini-style keys to OpenAI params where reasonable
        out = {}
        reasoning = _is_openai_reasoning_model(self.model)
        if not reasoning and "temperature" in cfg:
            out["temperature"] = cfg["temperature"]
        if not reasoning and "top_p" in cfg:
            out["top_p"] = cfg["top_p"]
        if "max_output_tokens" in cfg:
            out["max_tokens"] = cfg["max_output_tokens"]
        if "presence_penalty" in cfg:
            out["presence_penalty"] = cfg["presence_penalty"]
        if "frequency_penalty" in cfg:
            out["frequency_penalty"] = cfg["frequency_penalty"]
        if "stop_sequences" in cfg:
            out["stop"] = cfg["stop_sequences"]
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
        Convert Google-style prompt parts to OpenAI chat format:
          messages = [
            {"role": "user",
             "content": [
               {"type": "text", "text": "..."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." }}
             ]}
          ]
        """
        user_content = []

        for part in genai_parts:
            # plain text
            if isinstance(part, str):
                user_content.append({"type": "text", "text": part})
                continue

            # PIL Image
            if isinstance(part, Image.Image):
                url = self._pil_to_data_url(part, "image/png")
                user_content.append({"type": "image_url", "image_url": {"url": url}})
                continue

            # dict parts: expect {"mime_type": "...", "data": b"..."} for images
            if isinstance(part, dict):
                mime = part.get("mime_type", "")
                data = part.get("data", None)

                if isinstance(data, (bytes, bytearray)) and mime.startswith("image/"):
                    try:
                        # if data is raw bytes, use directly
                        url = self._to_data_url(mime, data)
                        user_content.append({"type": "image_url", "image_url": {"url": url}})
                        continue
                    except Exception as _:
                        pass

                # fallback: stringify any other dict
                user_content.append({"type": "text", "text": str(part)})
                continue

            # everything else -> string
            user_content.append({"type": "text", "text": str(part)})

        return [{"role": "user", "content": user_content}]

def genai_tools_to_openai_tools(genai_tools, *, debug=False):
    from collections.abc import Mapping
    """
    Convert Google GenAI Tool / FunctionDeclaration objects to OpenAI Chat tools.
    Robust to:
      - protobuf map containers (MessageMapContainer)
      - typed SDK objects with attributes
      - dicts
      - lists of key/value entries
    """

    def _get(obj, *names, default=None):
        for n in names:
            if hasattr(obj, n):
                try:
                    return getattr(obj, n)
                except Exception:
                    pass
            if isinstance(obj, dict) and n in obj:
                return obj[n]
        return default

    def _norm_type(t):
        if t is None:
            return None
        if isinstance(t, (list, tuple)):
            return [_norm_type(x) for x in t]
        name = getattr(t, "name", t)  # enums like Type.OBJECT
        return str(name).lower().replace("type.", "").strip("_")

    def _to_mapping_like(pr):
        """
        Normalize a 'properties' container to a dict[str, SchemaLike].
        Accepts:
          - dict
          - Mapping
          - objects with .items()
          - iterable of entries having .key/.value or {'key':..., 'value':...}
        """
        if pr is None:
            return {}
        if isinstance(pr, dict):
            return pr
        if isinstance(pr, Mapping):
            # protobuf map containers are Mappings
            return dict(pr.items())

        # has an .items() but not a Mapping
        it = getattr(pr, "items", None)
        if callable(it):
            try:
                return dict(pr.items())
            except Exception:
                pass

        # iterable of entries with key/value
        try:
            out = {}
            for entry in pr:
                k = getattr(entry, "key", None)
                if k is None and isinstance(entry, dict):
                    k = entry.get("key")
                v = getattr(entry, "value", None)
                if v is None and isinstance(entry, dict):
                    v = entry.get("value")
                if k is not None:
                    out[k] = v
            if out:
                return out
        except Exception:
            pass

        return {}

    def _schema_to_jsonschema(s):
        if s is None:
            return {"type": "object", "properties": {}}

        ty  = _norm_type(_get(s, "type_", "type"))
        dsc = _get(s, "description")
        enm = _get(s, "enum")
        itm = _get(s, "items")
        prw = _get(s, "properties")   # raw properties container
        req = _get(s, "required")

        out = {}
        if ty:
            out["type"] = ty
        if dsc:
            out["description"] = dsc
        if enm:
            out["enum"] = list(enm)

        # OBJECT: expand properties
        if (ty or "object") == "object":
            props = _to_mapping_like(prw)
            if props:
                out["properties"] = {k: _schema_to_jsonschema(v) for k, v in props.items()}
            else:
                out.setdefault("properties", {})

        # ARRAY: only then include items
        if ty == "array":
            out["items"] = _schema_to_jsonschema(itm) if itm is not None else {"type": "object"}

        if req:
            out["required"] = [req] if isinstance(req, str) else list(req)

        if "type" not in out:
            out["type"] = "object"
            out.setdefault("properties", {})

        return out

    def _fd_to_openai(fd):
        name = _get(fd, "name")
        desc = _get(fd, "description") or ""
        params = _schema_to_jsonschema(_get(fd, "parameters"))
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params,
            },
        }

    def _extract_function_declarations(item):
        fds = _get(item, "function_declarations")
        if fds:
            return list(fds)
        # direct FunctionDeclaration-like
        if _get(item, "name") and _get(item, "parameters") is not None:
            return [item]
        if isinstance(item, dict) and "name" in item:
            return [item]
        return []

    out = []
    for it in genai_tools or []:
        for fd in _extract_function_declarations(it):
            out.append(_fd_to_openai(fd))

    if debug:
        # Print a concise introspection for the first tool
        try:
            t0 = genai_tools[0]
            p0 = _get(_get(_extract_function_declarations(t0)[0], "parameters"), "properties")
            print("[DEBUG] genai properties type:", type(p0))
            if hasattr(p0, "items"):
                print("[DEBUG] genai properties keys:", list(k for k, _ in list(p0.items())[:5]))
            else:
                print("[DEBUG] genai properties sample repr:", repr(p0)[:200])
        except Exception as e:
            print("[DEBUG] introspection error:", e)

    return out


