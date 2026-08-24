"""Write engine-native MD inputs from an engine-neutral ParameterizedSystem.

The per-engine writing is delegated to OpenFF Interchange's own exporters
(``to_lammps`` / ``to_gromacs`` / ``to_openmm``), which are maintained upstream.
So adding an MD engine is "Interchange supports it", not new SciLink code: this
helper dispatches generically by engine name (SciLink's ``software`` keywords —
``lammps`` / ``gromacs`` / ``openmm`` — line up with the Interchange exporter
method names). The only per-engine handling is the file-writing engines vs.
``to_openmm`` (which returns an object, not files).

A ParameterizedSystem produced by the AMBER pipeline (a prmtop/inpcrd pair) is
bridged into an Interchange first, so it uses the same exporters — one writing
mechanism for every backend.

Heavy deps (openff-interchange, openff-toolkit, parmed) are imported lazily
inside the functions and gated behind the ``scilink[ff]`` extra, so importing
this module never requires them.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ._parameterized_system import ParameterizedSystem

# Engines whose Interchange exporter writes input files to disk (vs. to_openmm,
# which returns an in-memory System). Kept tiny and generic — not a per-engine
# code path, just the file-vs-object distinction the exporters themselves have.
_FILE_WRITING_ENGINES = {"lammps", "gromacs"}

_FF_EXTRA_HINT = (
    "OpenFF Interchange is required to write engine inputs from a parameterized "
    "system. The OpenFF stack is conda-forge only (not pip):  conda install -c "
    "conda-forge openff-toolkit openff-interchange openff-nagl openff-nagl-models "
    "parmed rdkit"
)


def _require_interchange():
    """Import openff.interchange.Interchange or raise an actionable error."""
    try:
        from openff.interchange import Interchange  # noqa: F401
        return Interchange
    except ImportError as e:  # pragma: no cover - exercised only without [ff]
        raise ImportError(f"{_FF_EXTRA_HINT}\n(original error: {e})") from e


def _load_interchange(system: ParameterizedSystem):
    """Return an OpenFF Interchange for ``system``, regardless of backend.

    ``source_format == "interchange"`` deserializes the stored Interchange JSON;
    ``"amber"`` bridges the prmtop/inpcrd pair into an Interchange so AMBER-typed
    systems use the same exporters as the SMIRNOFF path.
    """
    Interchange = _require_interchange()

    if system.source_format == "interchange":
        if not system.interchange_path or not os.path.isfile(system.interchange_path):
            raise FileNotFoundError(
                f"ParameterizedSystem.interchange_path missing: "
                f"{system.interchange_path!r}"
            )
        # OpenFF Interchange (>=0.5, pydantic v2): deserialize the JSON the FF
        # skill wrote with model_dump_json (same [ff] env → version-safe).
        with open(system.interchange_path) as fh:
            return Interchange.model_validate_json(fh.read())

    if system.source_format == "amber":
        prmtop, inpcrd = system.amber_files
        if not (prmtop and inpcrd and os.path.isfile(prmtop) and os.path.isfile(inpcrd)):
            raise FileNotFoundError(
                f"ParameterizedSystem.amber_files missing: {system.amber_files!r}"
            )
        # Bridge AMBER → Interchange via OpenMM (ParmEd loads the prmtop and
        # builds an OpenMM system + topology that Interchange ingests). Imported
        # lazily; part of the [ff] extra.
        try:
            import parmed as pmd
        except ImportError as e:
            raise ImportError(f"{_FF_EXTRA_HINT}\n(original error: {e})") from e
        structure = pmd.load_file(prmtop, xyz=inpcrd)
        return Interchange.from_openmm(
            system=structure.createSystem(),
            topology=structure.topology,
            positions=structure.positions,
            box_vectors=getattr(structure, "box_vectors", None),
        )

    raise ValueError(
        f"Unknown ParameterizedSystem.source_format: {system.source_format!r} "
        "(expected 'interchange' or 'amber')"
    )


def write_md_inputs(system: ParameterizedSystem, software: str,
                    working_dir: str) -> Dict[str, Any]:
    """Write engine-native MD inputs from a ParameterizedSystem.

    Parameters
    ----------
    system:
        The engine-neutral parameterized system (FF agent output).
    software:
        MD engine keyword (``"lammps"`` / ``"gromacs"`` / ``"openmm"``) — matches
        the Interchange exporter ``to_<software>``.
    working_dir:
        Directory to write the engine inputs into.

    Returns
    -------
    dict with ``structure_file`` (the engine's topology+coords file, e.g. a
    LAMMPS data file — the deck's ``read_data`` target) and ``force_field_files``
    (a ``{name: path}`` map of any additional style/settings includes; may be
    empty when the exporter folds everything into the structure file).
    """
    os.makedirs(working_dir, exist_ok=True)
    interchange = _load_interchange(system)

    exporter = getattr(interchange, f"to_{software}", None)
    if exporter is None:
        raise ValueError(
            f"OpenFF Interchange has no exporter for engine {software!r} "
            f"(to_{software}). Supported file-writing engines: "
            f"{sorted(_FILE_WRITING_ENGINES)}."
        )
    if software not in _FILE_WRITING_ENGINES:
        raise NotImplementedError(
            f"write_md_inputs writes input files; engine {software!r} "
            f"(Interchange.to_{software}) returns an in-memory object, not files."
        )

    return _export_files(interchange, software, working_dir)


def _export_files(interchange, software: str, working_dir: str) -> Dict[str, Any]:
    """Call the file-writing Interchange exporter for ``software`` and return the
    {structure_file, force_field_files} contract.

    Exporter output shapes differ slightly per engine (LAMMPS writes one data
    file; GROMACS writes .gro + .top), so the file collection is engine-aware
    here — but the *parameterization* is fully upstream/neutral.
    """
    prefix = os.path.join(working_dir, "system")
    if software == "lammps":
        # Write ONLY the typed data file (topology + masses + Pair/Bond/Angle
        # coeffs); the deck that read_data's it — pair_style, kspace, fixes — is
        # the MD skill's job. We deliberately avoid Interchange.to_lammps, which
        # ALSO emits a runnable input script and raises on mixed h-bond
        # constraints (rigid water + flexible solute); to_lammps_datafile is the
        # deterministic data-only export. Name matches the LAMMPS skill's
        # convention (system.data — the deck's default read_data target).
        data_file = os.path.join(working_dir, "system.data")
        interchange.to_lammps_datafile(data_file)
        _orthogonalize_zero_tilt(data_file)
        return {"structure_file": data_file, "force_field_files": {}}
    if software == "gromacs":
        interchange.to_gromacs(prefix)
        return {
            "structure_file": _first_existing(f"{prefix}.gro"),
            "force_field_files": {"topology": _first_existing(f"{prefix}.top")},
        }
    raise ValueError(f"unhandled file-writing engine {software!r}")


def _orthogonalize_zero_tilt(data_file: str) -> None:
    """Drop a zero-tilt ``xy xz yz`` line so LAMMPS treats the box as orthogonal.

    OpenFF Interchange always writes an ``xy xz yz`` tilt line, which makes
    LAMMPS treat the box as triclinic even when the tilt is exactly zero. A
    needlessly triclinic box forces ``kspace_style`` to be (re)declared after
    ``read_data`` — PPPM aborts with "Must redefine kspace_style after changing
    to triclinic box" otherwise — and adds overhead. For the common orthorhombic
    case we strip the line so the deck reads a plain orthogonal box. A nonzero
    tilt (a genuinely triclinic cell) is left untouched.
    """
    try:
        with open(data_file) as fh:
            lines = fh.readlines()
    except OSError:
        return
    out, changed = [], False
    for ln in lines:
        parts = ln.split()
        if len(parts) == 6 and parts[3:] == ["xy", "xz", "yz"]:
            try:
                if all(abs(float(p)) < 1e-8 for p in parts[:3]):
                    changed = True
                    continue  # drop zero-tilt line -> orthogonal box
            except ValueError:
                pass
        out.append(ln)
    if changed:
        with open(data_file, "w", encoding="utf-8") as fh:
            fh.writelines(out)


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Interchange exporter did not produce any of: {paths}"
    )
