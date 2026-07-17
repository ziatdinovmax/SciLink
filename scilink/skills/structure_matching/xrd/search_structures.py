"""``search_structures`` tool — multi-backend structure-database query.

Dispatches a :class:`QuerySpec` across the available backends, dedupes
across sources, ranks, materializes CIF files into ``output_dir``, and
returns a JSON-serializable dict.

The tool is callable from any analysis script (the skill registry exposes
it once ``structure_matching/xrd`` is in ``active_skills``). It also
prints a ``DB_MATCHES_JSON:`` line on stdout so the curve-fitting
stdout-parser (extended in commit 7) can lift the candidate list into
``fit_results['db_matches']`` for downstream synthesis and reporting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .._backends import (
    QuerySpec,
    StructureBackend,
    StructureCandidate,
    get_backend_factory,
    registered_backends,
)
from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# Source rank order for dedup: when the same structure appears in multiple
# backends, prefer the more authoritative source. User-registered backends
# not in this map default to rank 50 (between built-ins and "unknown");
# tweak ``register_source_preference`` to override.
_SOURCE_PREFERENCE: dict[str, int] = {"mp": 0, "cod": 1, "local": 2}


def register_source_preference(name: str, rank: int) -> None:
    """Set the dedup priority of a backend (lower = preferred).

    Mirrors :func:`scilink.skills.structure_matching._backends.register_backend`
    — a user adding ICSD might call
    ``register_source_preference('icsd', 0)`` to prefer it over MP.
    """
    _SOURCE_PREFERENCE[name] = int(rank)


TOOL_SPEC = ToolSpec(
    name="search_structures",
    description=(
        "Query crystal-structure databases for candidate structures matching a "
        "chemistry + symmetry spec. Dispatches across Materials Project, local "
        "CIF, and COD; dedupes across sources; returns a ranked list of "
        "candidates with materialized CIF paths."
    ),
    import_line="from scilink.skills.structure_matching.xrd.search_structures import search_structures",
    signature="search_structures(query: dict, sources: list[str] | None = None, output_dir: str = './candidates') -> dict",
    parameters={
        "query": {
            "type": "dict",
            "description": (
                "Spec: {chemistry: list[str] OR list[list[str]] (required "
                "UNLESS lattice_param_ranges is given), space_group_hints?: "
                "list[int], lattice_param_ranges?: {a: (min,max), ..., "
                "volume?: (min,max) Å³}, max_e_above_hull?: float, top_n?: int "
                "(default 10, per chemistry hypothesis), z_range?: "
                "(int, int), density_range?: (float, float) (g/cm³), "
                "anonymous_formula?: str (e.g. 'AB2').\n\n"
                "CELL-ONLY (blind) query: when the sample chemistry is UNKNOWN, "
                "omit chemistry and pass lattice_param_ranges from "
                "index_pattern's recovered cell — the unit cell becomes the "
                "search key. Prefer including the permutation-invariant "
                "'volume' range (axis conventions can differ between the "
                "indexed cell and a database setting). Served by the COD and "
                "local-CIF backends; Materials Project requires chemistry and "
                "skips cell-only queries.\n\n"
                "Use a single list for one hypothesis: chemistry=['Ti','O']. "
                "Use a list of lists when the chemistry is unknown and you "
                "want to test multiple candidates in one call: "
                "chemistry=[['Si'], ['C'], ['Ge']]. Each sublist becomes "
                "its own database query; results are merged + deduped + "
                "ranked across all hypotheses. This is the canonical way "
                "to handle the post-fit (no-hint) path — one search_structures "
                "call covering several element hypotheses, never multiple "
                "calls from inside a loop.\n\n"
                "z_range filters by number of sites per unit cell (MP's "
                "`nsites`, pymatgen's `Structure.num_sites`). density_range "
                "filters by bulk density. anonymous_formula filters by the "
                "stoichiometry template (matches pymatgen's "
                "`Composition.anonymized_formula`). The three are all "
                "optional and ignored by backends that don't carry the data."
            ),
        },
        "sources": {
            "type": "list",
            "description": (
                "Backend names in priority order: 'mp', 'local', 'cod'. None "
                "auto-detects whichever backends are available in the current "
                "environment (MP_API_KEY set → 'mp'; SCILINK_LOCAL_CIF_DIR "
                "exists → 'local')."
            ),
        },
        "output_dir": {
            "type": "str",
            "description": (
                "Directory where materialized CIF files are written "
                "(created if absent). Default './candidates'."
            ),
        },
    },
    required=["query"],
    returns=(
        "dict with 'candidates' (list of {id, source, formula, space_group, "
        "structure_path, rank_score, metadata}), 'sources_queried' (list of "
        "backend names), 'warnings' (list of human-readable strings)."
    ),
    when_to_use=(
        "Whenever a candidate-structure list is needed — pre-fit (chemistry "
        "hypothesized up front) or post-fit (extracted lattice/symmetry "
        "filters the DB). Combine with simulate_xrd_pattern and "
        "score_xrd_match for end-to-end identification."
    ),
)


def search_structures(
    query: dict,
    sources: Optional[list[str]] = None,
    output_dir: str = "./candidates",
) -> dict[str, Any]:
    """Query structure databases. See ``TOOL_SPEC`` for full contract."""
    specs = _coerce_query_specs(query)
    warnings: list[str] = []

    backends = _build_backends(sources, warnings)
    if not backends:
        result = {
            "candidates": [], "sources_queried": [],
            "warnings": warnings or ["No backends available"],
        }
        _emit_db_matches_marker(result)
        return result

    accumulated: list[StructureCandidate] = []
    sources_queried: list[str] = []
    queried_set: set[str] = set()
    for spec in specs:
        for backend in backends:
            try:
                results = backend.query(spec)
            except Exception as e:
                warnings.append(
                    f"{backend.name} backend raised on chemistry={spec.chemistry}: {e}"
                )
                _logger.warning("Backend %s failed: %s", backend.name, e)
                continue
            if backend.name not in queried_set:
                queried_set.add(backend.name)
                sources_queried.append(backend.name)
            accumulated.extend(results)

    deduped = _dedupe(accumulated)
    deduped.sort(key=lambda c: -c.rank_score)
    # Per-chemistry top_n still applies inside backend.query; here we cap the
    # final merged + deduped list at the per-spec top_n * number of hypotheses
    # so a 3-chemistry / top_n=5 call returns at most 15 candidates pre-dedup,
    # whatever dedup produces after.
    global_cap = max(spec.top_n for spec in specs) * max(len(specs), 1)
    deduped = deduped[:global_cap]

    _materialize_cifs(deduped, Path(output_dir), warnings)

    result = {
        "candidates": [_candidate_to_dict(c) for c in deduped],
        "sources_queried": sources_queried,
        "warnings": warnings,
    }
    _emit_db_matches_marker(result)
    return result


# --- Helpers ------------------------------------------------------------------

def _coerce_query_specs(query: dict) -> list[QuerySpec]:
    """Return one QuerySpec per chemistry hypothesis.

    Accepts both legacy single-hypothesis (``chemistry=["Ti", "O"]``) and
    multi-hypothesis (``chemistry=[["Si"], ["C"], ["Ge"]]``) shapes. The
    multi-hypothesis form is what the LLM should use for the no-hint
    workflow path — one call covering several element hypotheses,
    instead of looping over single-hypothesis calls (which historically
    led to consolidation bugs).

    The other QuerySpec fields (space_group_hints, lattice_param_ranges,
    max_e_above_hull, top_n) are shared across hypotheses.
    """
    chemistry = query.get("chemistry")
    if not chemistry:
        # Cell-only (blind) query: no chemistry, the lattice filter is the key.
        # This is the index-first identification path — the unit cell recovered
        # by index_pattern drives the search when the composition is unknown.
        if not query.get("lattice_param_ranges"):
            raise ValueError(
                "query needs chemistry (element symbols, or list-of-lists for "
                "multiple hypotheses) and/or lattice_param_ranges (cell-only "
                "blind query, e.g. from index_pattern's recovered cell)"
            )
        chemistries: list[list[str]] = [[]]
    elif not isinstance(chemistry, list):
        raise ValueError(
            f"query.chemistry must be a list, got {type(chemistry).__name__}"
        )
    # Normalize to list[list[str]].
    elif all(isinstance(c, str) for c in chemistry):
        chemistries = [list(chemistry)]
    elif all(isinstance(c, list) and c and all(isinstance(e, str) for e in c)
              for c in chemistry):
        chemistries = [list(c) for c in chemistry]
    else:
        raise ValueError(
            "query.chemistry must be either a list[str] (single hypothesis, "
            "e.g. ['Ti', 'O']) or a list[list[str]] (multiple hypotheses, "
            "e.g. [['Si'], ['C'], ['Ge']])"
        )

    lattice_ranges = query.get("lattice_param_ranges")
    if lattice_ranges:
        # JSON serialization turns tuples into lists; normalize back.
        lattice_ranges = {k: tuple(v) for k, v in lattice_ranges.items()}

    space_group_hints = query.get("space_group_hints")
    max_e = query.get("max_e_above_hull")
    top_n = int(query.get("top_n", 10))

    z_range = query.get("z_range")
    if z_range is not None:
        z_range = (int(z_range[0]), int(z_range[1]))
    density_range = query.get("density_range")
    if density_range is not None:
        density_range = (float(density_range[0]), float(density_range[1]))
    anonymous_formula = query.get("anonymous_formula")
    if anonymous_formula is not None:
        anonymous_formula = str(anonymous_formula)

    return [
        QuerySpec(
            chemistry=c,
            space_group_hints=space_group_hints,
            lattice_param_ranges=lattice_ranges,
            max_e_above_hull=max_e,
            top_n=top_n,
            z_range=z_range,
            density_range=density_range,
            anonymous_formula=anonymous_formula,
        )
        for c in chemistries
    ]


def _build_backends(
    sources: Optional[list[str]],
    warnings: list[str],
) -> list[StructureBackend]:
    """Construct backend instances for ``sources`` (or auto-detect).

    Backends come from :mod:`scilink.skills.structure_matching._backends`,
    which carries both the built-in factories and any third-party backends
    registered via ``register_backend()`` or the
    ``scilink.structure_backends`` entry-point group.
    """
    if sources is None:
        sources = list(registered_backends().keys())

    backends: list[StructureBackend] = []
    for name in sources:
        factory = get_backend_factory(name)
        if factory is None:
            warnings.append(f"Unknown source '{name}' — skipping")
            continue
        try:
            b = factory()
        except Exception as e:
            warnings.append(f"Failed to construct {name} backend: {e}")
            continue
        if not b.is_available():
            warnings.append(f"Backend '{name}' not available (missing key/dir/dependency)")
            continue
        backends.append(b)
    return backends


def _dedupe(candidates: Iterable[StructureCandidate]) -> list[StructureCandidate]:
    """Collapse duplicates across backends, preferring the more authoritative source."""
    bucket: dict[tuple, StructureCandidate] = {}
    for cand in candidates:
        key = (cand.formula, cand.space_group)
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = cand
            continue
        if _SOURCE_PREFERENCE.get(cand.source, 99) < _SOURCE_PREFERENCE.get(existing.source, 99):
            bucket[key] = cand
    return list(bucket.values())


def _materialize_cifs(
    candidates: list[StructureCandidate],
    output_dir: Path,
    warnings: list[str],
) -> None:
    """Write CIFs to ``output_dir`` for candidates that don't already have a path on disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        if cand.structure_path and Path(cand.structure_path).is_file():
            continue
        struct = cand.metadata.get("_structure")
        if struct is None:
            warnings.append(f"No structure object for {cand.source}:{cand.id} — cannot materialize CIF")
            continue
        path = output_dir / f"{cand.source}_{cand.id}.cif"
        try:
            with open(path, "w") as f:
                f.write(struct.to(fmt="cif"))
            cand.structure_path = str(path)
        except Exception as e:
            warnings.append(f"Failed to write CIF for {cand.source}:{cand.id}: {e}")


def _candidate_to_dict(cand: StructureCandidate) -> dict[str, Any]:
    """JSON-safe dict (drops the in-memory pymatgen Structure)."""
    d = asdict(cand)
    d["metadata"] = {k: v for k, v in d["metadata"].items() if not k.startswith("_")}
    return d


def _emit_db_matches_marker(result: dict[str, Any]) -> None:
    """Emit a ``DB_MATCHES_JSON:`` line on stdout for the framework parser."""
    safe = {
        "candidates": result["candidates"],
        "sources_queried": result["sources_queried"],
        "warnings": result["warnings"],
    }
    try:
        line = "DB_MATCHES_JSON: " + json.dumps(safe, default=str)
    except Exception as e:
        _logger.debug("Skipping DB_MATCHES_JSON emit: %s", e)
        return
    print(line, file=sys.stdout, flush=True)
