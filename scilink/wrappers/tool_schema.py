"""Portable tool-parameter schemas (#606).

Tool declarations are authored once, in lenient JSON Schema, and sent to
whichever provider the session uses. Providers disagree on what they accept:

- Gemini validates every declaration up front and rejects ``items`` on a
  non-array schema, so a multi-type ``"type": ["string", "array", "object"]``
  with one shared ``items`` (lenient, accepted by Anthropic and OpenAI) fails
  the WHOLE request once LiteLLM has expanded the type list into ``any_of``
  and copied ``items`` onto every variant. LiteLLM also drops ``oneOf`` to an
  empty schema for Gemini, which silently un-types the parameter.
- OpenAI's reasoning models refuse function tools on chat completions unless
  ``reasoning_effort`` is ``"none"`` (handled in the LiteLLM wrapper).

:func:`normalize_schema` rewrites any schema into the common subset every
provider accepts: a multi-type ``type`` becomes an explicit ``anyOf`` whose
variants carry only the keywords that apply to their own type (``items`` on
arrays, ``properties`` on objects, ...); ``oneOf`` becomes ``anyOf``; nested
schemas are normalized recursively; everything else is left alone. It is
applied centrally, in the provider wrappers, so every agent and mode is
covered without touching the tool authors' declarations.
:func:`gemini_schema_problems` is the matching rules check that a test can
run over every registered tool so a non-portable schema fails in CI.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

# Keywords that only make sense on one JSON type; a multi-type schema's
# shared keywords are handed to the variant they belong to.
_TYPE_KEYS: Dict[str, tuple] = {
    "array": ("items", "minItems", "maxItems", "uniqueItems", "prefixItems", "contains"),
    "object": ("properties", "required", "additionalProperties", "minProperties",
               "maxProperties", "patternProperties", "propertyOrdering"),
    "string": ("minLength", "maxLength", "pattern", "format"),
    "number": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
    "integer": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"),
}
_ALL_TYPE_KEYS = {k for keys in _TYPE_KEYS.values() for k in keys}
# Keywords that stay on the parent of an anyOf (they describe the parameter,
# not one of its shapes).
_SHARED_KEYS = ("description", "title", "default", "nullable", "deprecated")
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions")


def normalize_schema(schema: Any) -> Any:
    """A provider-portable copy of ``schema`` (never mutates the input)."""
    return _normalize(copy.deepcopy(schema))


def _normalize(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize(v) for v in node]
    if not isinstance(node, dict):
        return node
    # oneOf is not in Gemini's schema subset (LiteLLM drops it to {}); anyOf
    # is the portable spelling and is equivalent for tool arguments.
    if "oneOf" in node:
        variants = node.pop("oneOf")
        node["anyOf"] = list(node.get("anyOf", [])) + list(variants)
    t = node.get("type")
    if isinstance(t, list):
        node = _expand_type_list(node, [x for x in t])
    # Recurse into every nested schema position.
    for key in _SUBSCHEMA_MAPS:
        if isinstance(node.get(key), dict):
            node[key] = {k: _normalize(v) for k, v in node[key].items()}
    for key in _SUBSCHEMA_LISTS:
        if isinstance(node.get(key), list):
            node[key] = [_normalize(v) for v in node[key]]
    for key in ("items", "additionalProperties", "not", "contains"):
        if isinstance(node.get(key), dict):
            node[key] = _normalize(node[key])
    return node


def _expand_type_list(node: Dict[str, Any], types: List[Any]) -> Dict[str, Any]:
    """``{"type": [A, B], <keywords>}`` → ``{"anyOf": [{A + A's keywords},
    {B + B's keywords}], <shared keywords>}``. A single-element list is just
    unwrapped. ``null`` becomes its own ``{"type": "null"}`` variant (the
    standard spelling; LiteLLM turns it into Gemini's ``nullable``)."""
    types = [x for x in types if isinstance(x, str)]
    if not types:
        node.pop("type", None)
        return node
    if len(types) == 1:
        node["type"] = types[0]
        return node
    variants = []
    enum = node.get("enum")
    for t in types:
        v: Dict[str, Any] = {"type": t}
        for key in _TYPE_KEYS.get(t, ()):
            if key in node:
                v[key] = node[key]
        if enum is not None and t not in ("null", "array", "object"):
            v["enum"] = [e for e in enum if _enum_matches(e, t)] or enum
        variants.append(v)
    out: Dict[str, Any] = {}
    for key in _SHARED_KEYS:
        if key in node:
            out[key] = node[key]
    existing = node.get("anyOf")
    out["anyOf"] = variants + (list(existing) if isinstance(existing, list) else [])
    return out


def _enum_matches(value: Any, t: str) -> bool:
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def normalize_tools(tools: Optional[List[Any]]) -> Optional[List[Any]]:
    """Normalize the ``parameters`` of every OpenAI-style tool declaration
    (``{"type": "function", "function": {"parameters": ...}}``); a Google-style
    ``function_declarations`` list is handled the same way. Unknown shapes
    pass through untouched."""
    if not tools:
        return tools
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        tool = dict(tool)
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
            fn = dict(fn)
            fn["parameters"] = normalize_schema(fn["parameters"])
            tool["function"] = fn
        elif isinstance(tool.get("parameters"), dict):
            tool["parameters"] = normalize_schema(tool["parameters"])
        elif isinstance(tool.get("function_declarations"), list):
            tool["function_declarations"] = normalize_tools(tool["function_declarations"])
        out.append(tool)
    return out


# ── the rules a Gemini function declaration must satisfy ──────────────────
def gemini_schema_problems(schema: Any, path: str = "parameters") -> List[str]:
    """Violations of Gemini's function-declaration schema subset, as
    ``"<path>: <problem>"`` strings (empty = portable). Checks the shapes that
    are known to fail at request time: a ``type`` list, ``oneOf``,
    type-specific keywords on a schema of another type (``items`` on a
    non-array is the #606 failure), and an ``anyOf`` variant without a type."""
    problems: List[str] = []
    if not isinstance(schema, dict):
        return problems
    t = schema.get("type")
    if isinstance(t, list):
        problems.append(f"{path}: 'type' is a list {t} (must be a single type or anyOf)")
    if "oneOf" in schema:
        problems.append(f"{path}: 'oneOf' is not accepted (use anyOf)")
    if isinstance(t, str):
        for owner, keys in _TYPE_KEYS.items():
            if owner == t or (owner == "number" and t == "integer") or (owner == "integer" and t == "number"):
                continue
            for key in keys:
                if key in schema and key not in _TYPE_KEYS.get(t, ()):
                    problems.append(f"{path}: '{key}' on a {t!r} schema (only valid on {owner})")
    for i, v in enumerate(schema.get("anyOf") or []):
        if isinstance(v, dict) and "type" not in v and "anyOf" not in v:
            problems.append(f"{path}.anyOf[{i}]: variant without a type")
        problems += gemini_schema_problems(v, f"{path}.anyOf[{i}]")
    for key in ("allOf", "prefixItems"):
        for i, v in enumerate(schema.get(key) or []):
            problems += gemini_schema_problems(v, f"{path}.{key}[{i}]")
    for key in _SUBSCHEMA_MAPS:
        for name, v in (schema.get(key) or {}).items():
            problems += gemini_schema_problems(v, f"{path}.{key}[{name}]")
    for key in ("items", "additionalProperties", "not", "contains"):
        if isinstance(schema.get(key), dict):
            problems += gemini_schema_problems(schema[key], f"{path}.{key}")
    return problems


def tool_schema_problems(tools: Optional[List[Any]]) -> List[str]:
    """:func:`gemini_schema_problems` over a tool list, prefixed by tool name."""
    out: List[str] = []
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name", "?") if isinstance(fn, dict) else "?"
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if not isinstance(params, dict):
            continue
        if params.get("type") != "object":
            out.append(f"{name}: parameters.type must be 'object'")
        out += [f"{name}: {p}" for p in gemini_schema_problems(params)]
    return out
