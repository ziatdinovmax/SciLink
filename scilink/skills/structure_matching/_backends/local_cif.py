"""Local CIF directory backend — walks a directory of CIF files."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from . import _cif_index
from ._base import QuerySpec, StructureCandidate

try:
    from pymatgen.core import Structure
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    Structure = None  # type: ignore

_logger = logging.getLogger(__name__)


class LocalCIFBackend:
    """Backend that walks a directory of CIF files.

    Root directory comes from the ``root_dir`` constructor arg or the
    ``SCILINK_LOCAL_CIF_DIR`` env var. Subdirectories are walked
    recursively. Files are parsed lazily via pymatgen on each query.
    """

    name = "local"

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        if self.root_dir is None:
            env = os.getenv("SCILINK_LOCAL_CIF_DIR")
            if env:
                self.root_dir = Path(env)

    def is_available(self) -> bool:
        return (
            PYMATGEN_AVAILABLE
            and self.root_dir is not None
            and self.root_dir.is_dir()
        )

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        if not self.is_available():
            return []

        # First try the parquet index (fast filter on chemistry + symmetry +
        # lattice before any pymatgen parses). On small mirrors and when
        # pyarrow is unavailable, this returns None and we fall back to the
        # direct walk-and-parse path.
        index = _cif_index.load_or_build_index(self.root_dir)  # type: ignore[arg-type]
        if index is not None:
            return self._query_via_index(index, spec)
        return self._query_via_walk(spec)

    def _query_via_index(self, index, spec: QuerySpec) -> list[StructureCandidate]:
        paths = _cif_index.filter_index(
            index,
            chemistry=list(spec.chemistry),
            space_group_hints=spec.space_group_hints,
            lattice_param_ranges=spec.lattice_param_ranges,
            top_n=spec.top_n,
        )
        candidates: list[StructureCandidate] = []
        for cif_path_str in paths:
            cand = self._build_candidate_from_cif(Path(cif_path_str), spec)
            if cand is not None:
                candidates.append(cand)
            if len(candidates) >= spec.top_n * 3:
                break
        candidates.sort(key=lambda c: c.id)
        return candidates[:spec.top_n]

    def _query_via_walk(self, spec: QuerySpec) -> list[StructureCandidate]:
        candidates: list[StructureCandidate] = []
        for cif_path in sorted(self.root_dir.rglob("*.cif")):  # type: ignore[union-attr]
            cand = self._build_candidate_from_cif(cif_path, spec)
            if cand is not None:
                candidates.append(cand)
            if len(candidates) >= spec.top_n * 3:
                break
        candidates.sort(key=lambda c: c.id)
        return candidates[:spec.top_n]

    def _build_candidate_from_cif(
        self, cif_path: Path, spec: QuerySpec,
    ) -> Optional[StructureCandidate]:
        """Parse one CIF and return a StructureCandidate if it matches the spec."""
        target_elements = set(spec.chemistry)
        try:
            struct = Structure.from_file(str(cif_path))
        except Exception as e:
            _logger.debug("Skipping %s: parse failed (%s)", cif_path, e)
            return None

        present_elements = {str(el) for el in struct.composition.elements}
        if target_elements:  # empty = cell-only (blind) query: no element filter
            if not target_elements.issubset(present_elements):
                return None
            if target_elements != present_elements:
                return None

        sg_symbol = None
        sg_number = None
        try:
            sg = struct.get_space_group_info()
            sg_symbol, sg_number = sg[0], sg[1]
        except Exception:
            pass

        if spec.space_group_hints and sg_number not in spec.space_group_hints:
            return None

        lattice = struct.lattice
        if spec.lattice_param_ranges and not _lattice_within_ranges(
            lattice, spec.lattice_param_ranges,
        ):
            return None

        if spec.z_range is not None:
            n_sites = struct.num_sites
            if not (spec.z_range[0] <= n_sites <= spec.z_range[1]):
                return None

        density = None
        if spec.density_range is not None:
            density = float(struct.density)
            if not (spec.density_range[0] <= density <= spec.density_range[1]):
                return None

        anon = None
        if spec.anonymous_formula is not None:
            anon = str(struct.composition.anonymized_formula)
            if anon != spec.anonymous_formula:
                return None

        return StructureCandidate(
            id=cif_path.stem,
            source=self.name,
            formula=str(struct.composition.reduced_formula),
            space_group=sg_symbol,
            structure_path=str(cif_path),
            metadata={
                "spacegroup_number": sg_number,
                "lattice_a": lattice.a,
                "lattice_b": lattice.b,
                "lattice_c": lattice.c,
                "nsites": struct.num_sites,
                "density": density if density is not None else float(struct.density),
                "formula_anonymous": anon if anon is not None
                    else str(struct.composition.anonymized_formula),
            },
            rank_score=1.0,
        )


def _lattice_within_ranges(lattice, ranges: dict) -> bool:
    """Check lattice parameters against ``{a: (min, max), b: ..., c: ...}`` ranges.

    Also accepts a ``volume`` key (Å³) — permutation-invariant, so it is the
    robust primary filter when the axis convention of the query cell (e.g. from
    autoindexing) may differ from the database entry's setting."""
    for key in ("a", "b", "c", "alpha", "beta", "gamma", "volume"):
        if key in ranges:
            lo, hi = ranges[key]
            val = getattr(lattice, key)
            if not (lo <= val <= hi):
                return False
    return True
