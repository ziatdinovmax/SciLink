"""OpenFF backend for the uniform ``build_parameterized_system`` contract.

Every ``force_field`` skill exposes a ``build_parameterized_system`` tool that
returns a plain-dict payload; ``ForceFieldAgent.parameterize`` selects a backend
skill, dispatches to this tool through the registry, and wraps the payload into
an engine-neutral ``ParameterizedSystem``. That keeps the agent free of any
force-field-package or MD-engine name — the backend lives entirely in the skill.

The payload dict is the uniform contract (same keys for every backend):

    {
      "source_format": "interchange" | "amber",
      "backend":       "<provenance string>",
      "n_atoms":       int,
      "total_charge":  float,
      "interchange_path": str,          # source_format == "interchange"
      "amber_files":      [prmtop, inpcrd],  # source_format == "amber"
      "components":       [...],         # optional
      "coordinates_file": str,          # optional
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec
from .build_interchange import build_interchange


def build_parameterized_system(
    *,
    components: Optional[List[Dict[str, Any]]] = None,
    coordinates_file: Optional[str] = None,
    working_dir: str = ".",
    force_field: str = "openff-2.2.0.offxml",
    extra_force_fields: Optional[List[str]] = None,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Parameterize a packed box with SMIRNOFF + NAGL into a serialized
    Interchange, returned as the uniform payload dict. Extra keyword arguments
    (``pdb_file``, ``research_goal``, …) other backends consume are ignored."""
    if not (components and coordinates_file):
        raise ValueError(
            "the OpenFF backend needs `components` + `coordinates_file` "
            "(a packed box of SMILES-defined species)."
        )
    res = build_interchange(
        components, coordinates_file, working_dir=working_dir,
        force_field=force_field, extra_force_fields=extra_force_fields,
    )
    return {
        "source_format": "interchange",
        "backend": force_field,
        "n_atoms": int(res["n_atoms"]),
        "total_charge": float(res["total_charge"]),
        "interchange_path": res["interchange_path"],
        "components": components,
        "coordinates_file": coordinates_file,
    }


TOOL_SPEC = ToolSpec(
    name="build_parameterized_system",
    description=(
        "OpenFF backend: parameterize a packed box of SMILES-defined species "
        "(components + coordinates) into an engine-neutral ParameterizedSystem "
        "payload (a serialized SMIRNOFF Interchange). The uniform backend "
        "contract behind ForceFieldAgent.parameterize."
    ),
    parameters={
        "components": {"type": "list",
                       "description": "[{name, smiles, count}] in coordinate-file order"},
        "coordinates_file": {"type": "string",
                             "description": "packed-box coordinates with cell + PBC"},
        "working_dir": {"type": "string", "description": "where to write the payload"},
        "force_field": {"type": "string", "description": "base SMIRNOFF .offxml (default Sage)"},
        "extra_force_fields": {"type": "list",
                               "description": "additional .offxml (e.g. water/ion model)"},
    },
    required=["components", "coordinates_file"],
    signature=("build_parameterized_system(*, components, coordinates_file, "
               "working_dir='.', force_field='openff-2.2.0.offxml', "
               "extra_force_fields=None) -> dict"),
    import_line=("from scilink.skills.force_field.openff.build_parameterized_system "
                 "import build_parameterized_system"),
    agents=["simulation"],
    returns="uniform ParameterizedSystem payload dict (source_format='interchange')",
)
