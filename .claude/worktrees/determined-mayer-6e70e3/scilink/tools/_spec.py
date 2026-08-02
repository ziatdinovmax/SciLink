"""Structured tool spec for agent prompt injection and function-call schemas.

A ``ToolSpec`` describes *what* a tool is (bare mechanics: name, signature,
parameters, returns). Domain-specific usage recipes (how to combine tools for
a particular analysis) belong in skill files, not here.

Tool modules in ``scilink/tools/`` declare their spec at module level:

    # single-function module
    TOOL_SPEC = ToolSpec(...)

    # multi-function module
    TOOL_SPECS = [ToolSpec(...), ToolSpec(...)]

The walker in ``scilink.tools._registry`` discovers these and filters by the
``agents`` tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Metadata for a single callable tool.

    Field vocabulary mirrors ``AnalysisOrchestratorTools._register_tool`` so
    the same spec can produce both a prompt block (``to_prompt``) and an
    OpenAI function-call schema (``to_openai_schema``).
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    import_line: str = ""
    signature: str = ""
    agents: list[str] = field(default_factory=list)
    when_to_use: str = ""
    returns: str = ""
    example: str = ""

    def to_prompt(self) -> str:
        """Render as a markdown block for injection into planner / code-gen prompts."""
        lines = [f"### `{self.name}`"]
        if self.description:
            lines.append(self.description)
        if self.when_to_use:
            lines.append(f"**When to use:** {self.when_to_use}")
        if self.import_line:
            lines.append(f"**Import:** `{self.import_line}`")
        if self.signature:
            lines.append(f"**Signature:** `{self.signature}`")
        if self.parameters:
            lines.append("**Parameters:**")
            for pname, pinfo in self.parameters.items():
                pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else str(pinfo)
                ptype = pinfo.get("type", "") if isinstance(pinfo, dict) else ""
                req_marker = " *(required)*" if pname in self.required else ""
                type_str = f" ({ptype})" if ptype else ""
                lines.append(f"- `{pname}`{type_str}{req_marker}: {pdesc}")
        if self.returns:
            lines.append(f"**Returns:** {self.returns}")
        if self.example:
            lines.append(f"**Example:**\n```python\n{self.example}\n```")
        return "\n".join(lines)

    def to_openai_schema(self) -> dict:
        """Render as an OpenAI function-call schema (matches ``_register_tool`` output)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }
