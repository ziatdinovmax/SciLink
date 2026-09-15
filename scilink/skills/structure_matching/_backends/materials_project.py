"""Materials Project backend — multi-candidate structure queries via mp-api."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ._base import QuerySpec, StructureCandidate

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    MPRester = None  # type: ignore

_logger = logging.getLogger(__name__)


def _normalize_spacegroup(sg: Any) -> Optional[str]:
    """Coerce MP's symmetry payload (object or dict) to a Hermann-Mauguin symbol."""
    if sg is None:
        return None
    if isinstance(sg, str):
        return sg
    symbol = getattr(sg, "symbol", None) or getattr(sg, "spacegroup_symbol", None)
    if symbol:
        return str(symbol)
    if isinstance(sg, dict):
        return sg.get("symbol") or sg.get("spacegroup_symbol")
    return None


def _rank_score_from_e_hull(e_hull: Optional[float]) -> float:
    """Map energy-above-hull (eV/atom) to a 0..1 rank score; smaller is better."""
    if e_hull is None:
        return 0.0
    return max(0.0, 1.0 - e_hull / 0.5)


class MaterialsProjectBackend:
    """Backend that queries the Materials Project for candidate structures.

    Auth via ``scilink.auth.APIKeyManager``; falls back to the ``MP_API_KEY``
    env var. Results are cached per query for the lifetime of the instance.
    """

    name = "mp"

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            try:
                from ....auth import get_api_key
                api_key = get_api_key("materials_project")
            except Exception:
                api_key = os.getenv("MP_API_KEY") or os.getenv("MATERIALS_PROJECT_API_KEY")
        self.api_key = api_key
        self._cache: dict[tuple, list[StructureCandidate]] = {}

    def is_available(self) -> bool:
        return MP_API_AVAILABLE and bool(self.api_key)

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        if not self.is_available():
            return []
        if not spec.chemistry:
            # Cell-only (blind) queries are not supported here: mp-api searches
            # are chemsys-keyed. COD / local-CIF handle the cell-only path.
            _logger.debug("MP backend skipping cell-only query (needs chemistry)")
            return []

        chemsys = "-".join(sorted(spec.chemistry))
        cache_key = (
            chemsys,
            tuple(spec.space_group_hints) if spec.space_group_hints else None,
            spec.max_e_above_hull,
            spec.top_n,
            spec.z_range,
            spec.density_range,
            spec.anonymous_formula,
        )
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        fields = [
            "material_id", "formula_pretty", "symmetry",
            "energy_above_hull", "structure", "nsites", "density",
            "formula_anonymous",
        ]
        kwargs: dict[str, Any] = {"chemsys": chemsys, "fields": fields}
        if spec.max_e_above_hull is not None:
            kwargs["energy_above_hull"] = (0.0, float(spec.max_e_above_hull))
        if spec.z_range is not None:
            kwargs["num_sites"] = (int(spec.z_range[0]), int(spec.z_range[1]))
        if spec.density_range is not None:
            kwargs["density"] = (float(spec.density_range[0]), float(spec.density_range[1]))
        # anonymous_formula is post-filtered client-side because the mp-api
        # kwarg name has varied across versions ('formula' template vs
        # 'formula_anonymous'); client-side is robust to either.

        try:
            with MPRester(self.api_key) as mpr:
                raw = mpr.materials.summary.search(**kwargs)
        except Exception as e:
            _logger.warning("Materials Project query failed for %s: %s", chemsys, e)
            return []

        candidates: list[StructureCandidate] = []
        for r in raw:
            sg_symbol = _normalize_spacegroup(getattr(r, "symmetry", None))
            sg_number = None
            sym = getattr(r, "symmetry", None)
            if sym is not None:
                sg_number = getattr(sym, "number", None) or (
                    sym.get("number") if isinstance(sym, dict) else None
                )

            if spec.space_group_hints and sg_number not in spec.space_group_hints:
                continue

            anon = getattr(r, "formula_anonymous", None)
            if spec.anonymous_formula is not None:
                if anon is None or str(anon) != spec.anonymous_formula:
                    continue

            e_hull = getattr(r, "energy_above_hull", None)
            candidates.append(StructureCandidate(
                id=str(getattr(r, "material_id", "")),
                source=self.name,
                formula=str(getattr(r, "formula_pretty", "")),
                space_group=sg_symbol,
                metadata={
                    "energy_above_hull": e_hull,
                    "spacegroup_number": sg_number,
                    "nsites": getattr(r, "nsites", None),
                    "density": getattr(r, "density", None),
                    "formula_anonymous": anon,
                    "_structure": getattr(r, "structure", None),
                },
                rank_score=_rank_score_from_e_hull(e_hull),
            ))

        candidates.sort(key=lambda c: -c.rank_score)
        candidates = candidates[:spec.top_n]
        self._cache[cache_key] = list(candidates)
        return candidates
