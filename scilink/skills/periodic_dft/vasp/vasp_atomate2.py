"""Rule-based VASP input generation via atomate2 / pymatgen.

A deterministic generation backend for the ``vasp`` skill: it produces
INCAR / KPOINTS / POSCAR from Materials-Project-standard input sets
(atomate2's ``VaspInputGenerator``, falling back to ``MPRelaxSet``) rather
than from an LLM. Selected through the pipeline's ``method='atomate2'``
path and discovered via the skill registry as ``generate_inputs_atomate2``.

This is the foundation-agent "optional code helper" pattern: the LLM path
is the agent's baseline generation; named deterministic backends live in
the engine's skill bundle as tools. Requires the ``[sim]`` extras
(pymatgen + atomate2), imported lazily so the bundle stays discoverable
without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..._shared._spec import ToolSpec


def generate_inputs_atomate2(
    structure_file: str,
    request: str = "",
    output_dir: str = ".",
) -> Dict[str, Any]:
    """Generate VASP inputs from Materials-Project input sets.

    Args:
        structure_file: Path to the structure (read via ASE).
        request: Scientific request. Accepted for interface parity with
            the LLM generator; the rule-based set does not consume it.
        output_dir: Directory to write INCAR / KPOINTS / POSCAR into.

    Returns:
        A dict with ``status`` and, on success, an ``input_files`` mapping
        (filename → contents) plus a ``summary``. Returns an error dict
        when the ``[sim]`` extras are missing or generation fails.
    """
    try:
        from ....agents.sim_agents.atomate2_utils import Atomate2Input
    except ImportError as e:
        return {
            "status": "error",
            "message": (
                "method='atomate2' requires the [sim] extras "
                "(pymatgen, atomate2). Install with: pip install 'scilink[sim]'. "
                f"Original error: {e}"
            ),
        }

    try:
        from ase.io import read as ase_read
        structure = ase_read(structure_file)
        Atomate2Input().generate(structure=structure, output_dir=output_dir)
    except Exception as e:
        return {"status": "error", "message": f"atomate2 generation failed: {e}"}

    out = Path(output_dir)
    input_files: Dict[str, str] = {}
    for name in ("INCAR", "KPOINTS", "POSCAR"):
        p = out / name
        if p.exists():
            input_files[name] = p.read_text()

    if "INCAR" not in input_files:
        return {
            "status": "error",
            "message": "atomate2 generation produced no INCAR.",
        }

    return {
        "status": "success",
        "input_files": input_files,
        "summary": "Materials Project standard relaxation set (atomate2/pymatgen).",
    }


TOOL_SPEC = ToolSpec(
    name="generate_inputs_atomate2",
    description=(
        "Deterministic VASP input generation from Materials-Project "
        "standard input sets (atomate2 / pymatgen MPRelaxSet). The "
        "rule-based alternative to LLM generation; returns a normalized "
        "input_files map. Requires the [sim] extras."
    ),
    parameters={
        "structure_file": {
            "type": "string",
            "description": "Path to the structure file (read via ASE).",
        },
        "request": {
            "type": "string",
            "description": "Scientific request (unused by the rule-based set).",
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to write INCAR / KPOINTS / POSCAR into.",
        },
    },
    required=["structure_file"],
    signature="generate_inputs_atomate2(structure_file, request, output_dir) -> dict",
    import_line="from scilink.skills.periodic_dft.vasp.vasp_atomate2 import generate_inputs_atomate2",
    agents=["simulation"],
    returns="dict with status, input_files (filename->contents), summary.",
)
