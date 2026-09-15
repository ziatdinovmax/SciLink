"""Build an OpenFF Interchange (the engine-neutral parameterized system) from a
packed box of component molecules.

This is the OpenFF force-field backend's load-bearing callable: given the
per-component chemistry (SMILES + counts, in coordinate-file order) and the
packed coordinates, it parameterizes the system with a SMIRNOFF force field and
NAGL charges and serializes an Interchange. An MD engine's ``write_md_inputs``
then exports it natively (``to_lammps`` / ``to_gromacs`` / ``to_openmm``).

NAGL (OpenFF's graph-net AM1-BCC surrogate) supplies partial charges, so this
path needs no AmberTools/OpenEye and stays pip-installable.

Heavy deps (openff-toolkit, openff-interchange, openff-nagl) are imported lazily
and gated behind the ``scilink[ff]`` extra.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..._shared._spec import ToolSpec

_FF_EXTRA_HINT = (
    "OpenFF force-field parameterization requires the OpenFF stack, which is "
    "conda-forge only (not pip):  conda install -c conda-forge openff-toolkit "
    "openff-interchange openff-nagl openff-nagl-models parmed rdkit"
)

# OpenFF's released NAGL AM1-BCC surrogate model — partial charges without QM.
_DEFAULT_NAGL_MODEL = "openff-gnn-am1bcc-1.0.0.pt"

# vdW supplement for metal cations Sage does not cover (auto-included). See the
# file header for provenance; the completeness gate backstops anything not here.
_ION_VDW_OFFXML = os.path.join(os.path.dirname(__file__), "metal_ion_vdw.offxml")


def _read_coordinates(coordinates_file: str):
    """Return (positions_angstrom, box_lengths_angstrom_or_None, n_atoms) from a
    structure file, via ASE (handles extxyz / pdb / ...)."""
    from ase.io import read as ase_read
    atoms = ase_read(coordinates_file)
    pos = atoms.get_positions()  # Angstrom
    box = None
    if bool(atoms.pbc.all()) and atoms.cell.rank == 3:
        box = atoms.cell.lengths().tolist()  # orthorhombic lengths, Angstrom
    return pos, box, len(atoms)


def _assert_fully_parameterized(interchange, n_atoms: int,
                                atom_provenance: List[tuple]) -> None:
    """Raise if any atom lacks a Lennard-Jones (vdW) term.

    A missing vdW term is the silent failure mode of ``create_interchange`` for
    chemistry the force field does not cover (notably divalent metal cations):
    the call succeeds but the atom is a zero-repulsion point charge. We treat an
    atom as parameterized only if it maps to a vdW potential with positive sigma.
    ``atom_provenance`` is the per-atom ``(component_name, element)`` list in
    topology order, used to name offenders.
    """
    vdw = interchange.collections.get("vdW")
    covered = set()
    if vdw is not None:
        for tkey, pkey in vdw.key_map.items():
            sigma = vdw.potentials[pkey].parameters.get("sigma")
            if sigma is not None and float(sigma.m) > 0:
                covered.update(tkey.atom_indices)

    missing = [i for i in range(n_atoms) if i not in covered]
    if not missing:
        return

    # Summarize offenders by (component, element) rather than dumping indices.
    offenders = sorted({atom_provenance[i] for i in missing
                        if i < len(atom_provenance)})
    detail = ", ".join(f"{comp} ({el})" for comp, el in offenders) or "unknown atoms"
    raise ValueError(
        f"build_interchange: {len(missing)} atom(s) received no Lennard-Jones "
        f"parameters and would be unphysical point charges: {detail}. The force "
        "field does not cover this chemistry (commonly a divalent/transition-"
        "metal cation). Add its vdW to the ion supplement "
        "(force_field/openff/metal_ion_vdw.offxml), pass a force field that "
        "covers it via extra_force_fields, or use a backend (e.g. amber) that does."
    )


def build_interchange(components: List[Dict[str, Any]],
                      coordinates_file: str,
                      working_dir: str = ".",
                      force_field: str = "openff-2.2.0.offxml",
                      extra_force_fields: List[str] | None = None,
                      nagl_model: str = _DEFAULT_NAGL_MODEL) -> Dict[str, Any]:
    """Parameterize a packed box into a serialized OpenFF Interchange.

    Parameters
    ----------
    components:
        Per-species ``{"name", "smiles", "count"}`` in the SAME order the species
        appear in ``coordinates_file`` (load-bearing — the topology aligns
        atom-by-atom with the coordinates).
    coordinates_file:
        Packed-box coordinates (e.g. ``.extxyz``) with cell + PBC.
    working_dir:
        Directory to write the serialized Interchange JSON into.
    force_field, extra_force_fields:
        SMIRNOFF force-field file(s). ``force_field`` is the base (Sage); pass
        e.g. a water/ion model via ``extra_force_fields`` when needed.
    nagl_model:
        NAGL model file for partial charges.

    Returns
    -------
    dict with ``interchange_path`` (serialized Interchange JSON), ``n_atoms``,
    ``total_charge``.
    """
    try:
        from openff.toolkit import ForceField, Molecule, Topology
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
        from openff.units import unit
    except ImportError as e:
        raise ImportError(f"{_FF_EXTRA_HINT}\n(original error: {e})") from e

    if not components:
        raise ValueError("build_interchange: no components supplied")

    nagl = NAGLToolkitWrapper()
    charged_unique = []   # one charged Molecule per component (for charge_from_molecules)
    molecules_in_order = []  # count copies per component, in coordinate-file order
    atom_provenance = []  # per-atom (component_name, element) in topology order
    total_charge = 0.0
    for comp in components:
        smiles, count = comp["smiles"], int(comp["count"])
        name = comp.get("name", smiles)
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        if mol.n_atoms == 1:
            # Monatomic ion: the charge IS the formal charge. NAGL (a graph-net
            # AM1-BCC surrogate for organics) cannot run on a lone metal/ion atom
            # and would reject the element outright.
            mol.partial_charges = unit.Quantity(
                [float(mol.total_charge.m)], unit.elementary_charge)
        else:
            mol.assign_partial_charges(nagl_model, toolkit_registry=nagl)
        charged_unique.append(mol)
        total_charge += float(mol.total_charge.m) * count
        elements = [atom.symbol for atom in mol.atoms]
        for _ in range(count):
            molecules_in_order.append(Molecule(mol))
            atom_provenance.extend((name, el) for el in elements)

    topology = Topology.from_molecules(molecules_in_order)

    positions, box, n_coords = _read_coordinates(coordinates_file)
    if topology.n_atoms != n_coords:
        raise ValueError(
            f"build_interchange: topology has {topology.n_atoms} atoms but "
            f"{coordinates_file} has {n_coords}. The components manifest "
            "(SMILES/counts/order) must match the packed coordinates exactly."
        )
    if box is None:
        raise ValueError(
            f"build_interchange: {coordinates_file} has no periodic cell; a "
            "periodic box is required (LAMMPS/GROMACS electrostatics use PME)."
        )
    topology.box_vectors = unit.Quantity(
        [[box[0], 0, 0], [0, box[1], 0], [0, 0, box[2]]], unit.angstrom
    )

    # Base Sage + the bundled metal-ion vdW supplement (adds parameters for
    # cations Sage lacks; matches nothing in an organic-only system) + caller
    # extras (e.g. a water model).
    ff_files = [force_field, _ION_VDW_OFFXML] + list(extra_force_fields or [])
    ff = ForceField(*ff_files)
    interchange = ff.create_interchange(topology, charge_from_molecules=charged_unique)
    interchange.positions = unit.Quantity(positions, unit.angstrom)

    # Completeness gate: create_interchange silently leaves unsupported atoms
    # (e.g. a divalent metal Sage has no vdW for) with NO Lennard-Jones term —
    # a zero-repulsion point charge that collapses into its neighbours. Refuse to
    # emit such a system; name the offending species so the fix is actionable
    # (add its parameters to the ion supplement, or use a backend that covers it).
    _assert_fully_parameterized(interchange, topology.n_atoms, atom_provenance)

    os.makedirs(working_dir, exist_ok=True)
    out = os.path.join(working_dir, "system_interchange.json")
    # OpenFF Interchange (>=0.5, pydantic v2) serializes via model_dump_json;
    # the engine writer reloads with Interchange.model_validate_json. Both run in
    # the same [ff] env, so the round-trip is version-safe.
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(interchange.model_dump_json())

    return {
        "interchange_path": out,
        "n_atoms": int(topology.n_atoms),
        "total_charge": round(total_charge, 4),
    }


TOOL_SPEC = ToolSpec(
    name="build_interchange",
    description=(
        "Parameterize a packed box of component molecules (SMILES + counts, in "
        "coordinate order) with a SMIRNOFF force field and NAGL charges, and "
        "serialize an engine-neutral OpenFF Interchange. The MD engine then "
        "exports it natively via write_md_inputs."
    ),
    parameters={
        "components": {"type": "list",
                       "description": "[{name, smiles, count}] in coordinate-file order"},
        "coordinates_file": {"type": "string",
                             "description": "packed-box coordinates (extxyz/pdb) with cell + PBC"},
        "working_dir": {"type": "string", "description": "where to write the Interchange JSON"},
        "force_field": {"type": "string", "description": "base SMIRNOFF .offxml (default Sage)"},
        "extra_force_fields": {"type": "list",
                               "description": "additional .offxml (e.g. water/ion model)"},
        "nagl_model": {"type": "string",
                       "description": ("NAGL graph-net charge model "
                                       "(default openff-gnn-am1bcc-1.0.0.pt); "
                                       "one released model today")},
    },
    required=["components", "coordinates_file"],
    signature=("build_interchange(components, coordinates_file, working_dir='.', "
               "force_field='openff-2.2.0.offxml', extra_force_fields=None, "
               "nagl_model='openff-gnn-am1bcc-1.0.0.pt') -> dict"),
    import_line="from scilink.skills.force_field.openff.build_interchange import build_interchange",
    agents=["simulation"],
    returns="dict with interchange_path, n_atoms, total_charge",
)
