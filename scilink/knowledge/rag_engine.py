"""Shared RAG engine — retrieval + generation over a KnowledgeBase.

This is the generic core consumed by every mode. Domain-specific
orchestration (planning's hypothesis/code generation, plan verification,
refinement) lives with the agent that owns it and calls into ``run_rag`` /
``retrieve_context`` here.

Imports nothing from ``scilink.agents`` — instructions are passed in by the
caller.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image as PIL_Image


def parse_json_from_response(resp) -> "Tuple[Optional[Dict[str, Any]], Optional[str]]":
    """
    Robustly extracts and parses JSON from an LLM response object.

    Handles:
    - Gemini: resp.text or resp.parts[0].text
    - OpenAI/Anthropic wrapper: resp.text (via SimpleNamespace)
    - Raw strings
    - Markdown code fences (```json ... ```)
    - Preamble/postamble text around JSON (common with Anthropic models)
    """
    import json

    json_text = ""

    # 1. Extract raw text from response object
    try:
        if hasattr(resp, 'text'):
            json_text = resp.text.strip()
        elif hasattr(resp, 'parts') and resp.parts:
            json_text = resp.parts[0].text.strip()
        elif isinstance(resp, str):
            json_text = resp.strip()
        else:
            return None, f"LLM response format unexpected: {type(resp)}"

    except ValueError as e:
        return None, f"Response blocked or empty (Safety Filter): {e}"
    except Exception as e:
        return None, f"Error extracting text from response: {e}"

    if not json_text:
        return None, "Empty response from LLM"

    # 2. Strip Markdown code fences
    if json_text.startswith("```json"):
        json_text = json_text[len("```json"):].strip()
    elif json_text.startswith("```"):
        json_text = json_text[len("```"):].strip()

    if json_text.endswith("```"):
        json_text = json_text[:-len("```")].strip()

    # 3. Try direct parse first (fast path — works for Gemini and clean responses)
    try:
        return json.loads(json_text), None
    except json.JSONDecodeError:
        pass  # Fall through to extraction logic

    # 4. Extract JSON object from surrounding text (handles Anthropic preamble)
    #    Find the outermost { ... } by brace matching
    first_brace = json_text.find('{')
    if first_brace == -1:
        return None, (
            f"No JSON object found in response. "
            f"First 300 chars: {json_text[:300]}"
        )

    # Match braces to find the complete JSON object
    depth = 0
    in_string = False
    escape_next = False
    last_brace = -1

    for i in range(first_brace, len(json_text)):
        ch = json_text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                last_brace = i
                break

    if last_brace == -1:
        return None, (
            f"Unbalanced braces in response. "
            f"First 300 chars: {json_text[:300]}"
        )

    extracted = json_text[first_brace:last_brace + 1]

    try:
        return json.loads(extracted), None
    except json.JSONDecodeError:
        pass

    # Attempt to fix broken Unicode escapes (e.g. \u00B instead of µ)
    # by replacing malformed \uXXX sequences with the Unicode replacement char.
    import re
    sanitized = re.sub(
        r'\\u([0-9a-fA-F]{1,3})(?![0-9a-fA-F])',
        lambda m: chr(int(m.group(1), 16)) if len(m.group(1)) >= 2 else '�',
        extracted,
    )
    try:
        return json.loads(sanitized), None
    except json.JSONDecodeError as e:
        return None, (
            f"Failed to decode JSON: {e}. "
            f"Extracted text (first 500 chars): {extracted[:500]}"
        )


def retrieve_context(kb: Any, query: str, top_k: int = 10, dedupe: bool = True) -> str:
    """Retrieve the top-k chunks for ``query`` from a ``KnowledgeBase`` and
    format them into a prompt-ready context block.

    The generic RAG retrieval step. Returns an empty string when the KB is
    empty or unbuilt.
    """
    if not (kb is not None and kb.index and kb.index.ntotal > 0):
        return ""

    chunks = kb.retrieve(query, top_k=top_k)
    if dedupe:
        chunks = list({c['text']: c for c in chunks}.values())

    return "\n\n---\n\n".join(
        f"Source: {Path(c['metadata'].get('source', 'N/A')).name}\n"
        f"Type: {c['metadata'].get('content_type')}\n\n{c['text']}"
        for c in chunks
    )


def run_rag(query: str,
            instructions: str,
            kb: Any,
            model: Any,
            generation_config: Any,
            *,
            images: Optional[List[Any]] = None,
            image_descriptions: Optional[List[str]] = None,
            external_context: Optional[str] = None,
            additional_context: Optional[str] = None,
            primary_data_str: Optional[str] = None,
            skill_context: Optional[str] = None,
            fallback_instructions: Optional[str] = None,
            task_name: str = "RAG",
            return_context: bool = False,
            mode_key: Optional[str] = None) -> Any:
    """Generic RAG generation loop.

    Retrieves context for ``query`` from ``kb``, builds a multimodal prompt
    (``instructions`` + query + optional primary data / images / extra
    context + retrieved context + skill context), generates JSON, and — if the
    model reports an "Insufficient" context error and ``fallback_instructions``
    are supplied — retries once in fallback mode.

    Args:
        query: The objective / question driving retrieval and generation.
        instructions: System/task instructions placed first in the prompt.
        kb: A ``KnowledgeBase`` (may be empty/unbuilt).
        model: An LLM with a ``generate_content`` method.
        generation_config: Passed straight to ``model.generate_content``.
        images: Optional list of image file paths or pre-loaded PIL images.
        image_descriptions: Optional structured descriptions for the images.
        external_context: Optional external literature block.
        additional_context: Optional free-text extra context.
        primary_data_str: Optional primary dataset summary.
        skill_context: Optional domain-skill context block.
        fallback_instructions: Optional instructions used to retry once when
            strict generation reports insufficient context.
        task_name: Label used in progress logging.
        return_context: When True, return ``(result, retrieved_context_str)``
            so callers can reuse the exact grounding evidence the generation
            saw (e.g. a downstream critic). Default False preserves the
            original ``result``-only return for all existing callers.
        mode_key: When set, stamp the returned dict with
            ``result[mode_key] = "fallback" | "strict"`` so the caller can
            tell which instruction tier actually produced the plan. Off by
            default — existing callers' results are untouched.

    Returns:
        The parsed JSON dict (or ``{"error": ...}`` on failure). When
        ``return_context`` is True, a ``(result, retrieved_context_str)`` tuple.
    """
    # --- 1. Retrieve context ---
    print(f"\n--- Retrieving Context for {task_name} ---")
    rag_str = retrieve_context(kb, query, top_k=10)

    if not rag_str and not primary_data_str and not external_context:
        retrieved_context_str = "No specific documents found in Knowledge Base."
    else:
        retrieved_context_str = ""
        if external_context:
            retrieved_context_str += f"## 🌍 External Scientific Literature\n{external_context}\n\n"
        if rag_str:
            retrieved_context_str += f"## 📂 Retrieved Local Documents\n{rag_str}"

    def _ret(payload):
        # Honor the opt-in context return without disturbing the original
        # result-only contract for callers that don't request it.
        return (payload, retrieved_context_str) if return_context else payload

    # --- 2. Build multimodal prompt ---
    loaded_images = []
    if images and PIL_Image:
        for img in images:
            if isinstance(img, str):
                try:
                    loaded_images.append(PIL_Image.open(img))
                except Exception as e:
                    print(f"  - ⚠️ Could not load image {img}: {e}")
            else:
                loaded_images.append(img)  # assume already a PIL image

    img_desc_str = json.dumps(image_descriptions, indent=2) if image_descriptions else ""

    prompt_parts = [instructions, f"## User Objective:\n{query}"]

    if primary_data_str:
        prompt_parts.append(f"\n## 📊 Primary Experimental Data:\n{primary_data_str}")

    if loaded_images:
        prompt_parts.append("\n## Provided Images: (See attached)")
        prompt_parts.extend(loaded_images)
        if img_desc_str:
            prompt_parts.append(f"\n## Image Descriptions:\n{img_desc_str}")

    if additional_context:
        prompt_parts.append(f"\n## Additional Context:\n{additional_context}")

    prompt_parts.append(f"\n## Retrieved Context:\n{retrieved_context_str}")

    if skill_context:
        prompt_parts.append(skill_context)

    # --- 3. Generation & fallback ---
    print(f"--- Generating {task_name} ---")
    try:
        # Attempt 1: strict RAG generation
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        result, error_msg = parse_json_from_response(response)

        if error_msg:
            return _ret({"error": f"JSON Parsing Error: {error_msg}"})

        # Check for insufficient-context signal
        needs_fallback = bool(
            result.get("error") and "Insufficient" in str(result.get("error"))
        )

        used_fallback = False
        if needs_fallback:
            print(f"    - ⚠️ Strict generation failed: {result.get('error')}")
            if not fallback_instructions:
                return _ret(result)  # no fallback available

            print("    - 🔄 Entering Fallback Mode (General Knowledge)...")
            prompt_parts[0] = fallback_instructions
            fallback_response = model.generate_content(
                prompt_parts, generation_config=generation_config
            )
            result, error_msg_fb = parse_json_from_response(fallback_response)
            if error_msg_fb:
                return _ret({"error": f"Fallback JSON Parsing Error: {error_msg_fb}"})
            print("    - ✅ Fallback generation successful.")
            used_fallback = True

        if mode_key and isinstance(result, dict):
            result[mode_key] = "fallback" if used_fallback else "strict"
        return _ret(result)

    except Exception as e:
        logging.error(f"Error in run_rag: {e}")
        return _ret({"error": str(e)})
