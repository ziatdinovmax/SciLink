"""
The ParameterizedSystem descriptor — the contract between the agent that
*assigns a force field* (today: ForceFieldAgent) and the per-engine skill that
*writes the engine's input files*.

Architectural rationale (mirrors DeployedPotential in ``_potential.py``):
force-field parameterization and engine-native file writing are separate
problems. A parameterized molecular system — atoms, types, charges, bonds,
angles, the force-field terms — is engine-neutral; turning it into a LAMMPS
data file vs. a GROMACS topology vs. an OpenMM System is the *engine's* job.
So the FF agent's work ends at "here is a parameterized system"; each MD
engine's skill answers "given a parameterized system, here is how *I* write my
inputs" via the conventional ``write_md_inputs`` tool.

Extensibility design — N + M, not N × M:

  ParameterizedSystem is **engine-neutral**. It names no engine and carries no
  engine file format. It holds one portable payload — a serialized OpenFF
  Interchange (the canonical multi-engine parameterized system, which exports
  natively to LAMMPS / GROMACS / OpenMM) or an AmberTools ``(prmtop, inpcrd)``
  pair bridged through ParmEd — plus engine-neutral metadata.

  Adding an MD engine = one engine-skill ``write_md_inputs`` writer (zero FF
  code). Adding a force-field backend = one FF skill that emits this descriptor
  (zero engine code). Neither side enumerates the other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ComponentSpec:
    """One molecular species in the system, defined by its chemistry.

    A packed box is just coordinates; a force field needs to know which atoms
    form which molecule. ``smiles`` is the chemistry definition a SMIRNOFF
    backend parameterizes against; ``count`` is the number of copies, in the
    order they appear in the coordinate file (load-bearing — the parameterized
    topology aligns atom-by-atom with those coordinates).

    Fields
    ------
    name:
        Human label, e.g. ``"water"``, ``"PF6-"``.
    smiles:
        Canonical SMILES for the species.
    count:
        Number of copies in the box.
    charge:
        Formal molecular charge (used for neutrality checks).
    """
    name: str
    smiles: str
    count: int
    charge: float = 0.0


@dataclass
class ParameterizedSystem:
    """A force-field-parameterized molecular system, handed from the FF agent
    to an MD engine's ``write_md_inputs`` tool.

    Engine-neutral by construction — see the module docstring. Carries exactly
    one payload: a serialized OpenFF Interchange (``interchange_path``) when the
    backend is SMIRNOFF/OpenFF, or an AmberTools ``(prmtop, inpcrd)`` pair
    (``amber_files``) bridged through ParmEd. ``source_format`` says which.

    Attributes
    ----------
    backend:
        Force-field family keyword the producing agent used, e.g.
        ``"openff-2.2.0"``, ``"amber-ff19SB+gaff2"``. Provenance only — engine
        writers branch on ``source_format``, not on this.
    source_format:
        ``"interchange"`` or ``"amber"`` — which payload is populated, and how
        an engine writer should load it.
    n_atoms:
        Total atom count, for a cross-check against the coordinate file.
    total_charge:
        Net system charge, for a neutrality check.
    components:
        Per-species chemistry + counts, in coordinate-file order.
    coordinates_file:
        Path to the packed-box coordinates (e.g. ``.extxyz``) the parameters
        correspond to.
    box:
        Orthorhombic cell lengths (Å), or ``None`` for a non-periodic system.
    interchange_path:
        Path to a serialized OpenFF Interchange (JSON), or ``""``.
    amber_files:
        ``(prmtop, inpcrd)`` paths, or ``("", "")``.
    notes:
        Free-text provenance (FF-selection rationale, caveats) — surfaced in
        the run README.
    """
    backend: str
    source_format: str
    n_atoms: int
    total_charge: float
    components: List[ComponentSpec] = field(default_factory=list)
    coordinates_file: str = ""
    box: Optional[List[float]] = None
    interchange_path: str = ""
    amber_files: Tuple[str, str] = ("", "")
    notes: str = ""

    def to_json(self) -> str:
        """Serialize to JSON (survives the process boundary into an engine
        writer's tool call)."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "ParameterizedSystem":
        """Reconstruct from :meth:`to_json` output."""
        data = dict(json.loads(text))
        data["components"] = [ComponentSpec(**c) for c in data.get("components", [])]
        amber = data.get("amber_files") or ("", "")
        data["amber_files"] = tuple(amber)
        return cls(**data)
