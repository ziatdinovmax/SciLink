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
    QuerySkipped,
    QuerySpec,
    StructureBackend,
    StructureCandidate,
    get_backend_factory,
    registered_backends,
)
from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# Source rank order for dedup: when the same structure appears in multiple
# backends, prefer the more authoritative source. A measured cell (COD) comes
# before a DFT-relaxed one (MP, typically 1-3 % too large), which a match then
# has to absorb as a lattice scale (#775); the entry kept carries the other
# source's id and stability. User-registered backends not in this map default
# to rank 99; tweak ``register_source_preference`` to override.
_SOURCE_PREFERENCE: dict[str, int] = {"cod": 0, "mp": 1, "local": 2}


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
        "backend names), 'warnings' (list of human-readable strings), and "
        "'search_record': the query as sent and, per hypothesis and backend, "
        "what happened (ok with n / failed with the error / skipped, matches "
        "dropped, a row cap hit), with 'clean_empty' true only when the "
        "databases answered and hold nothing for this query. A failed or "
        "partial search is never an empty one: do not turn a raised error "
        "into an empty candidate list."
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
    # what this call did, per (hypothesis, backend): the record a reader holds
    # an empty result to — a database that answered with nothing is a finding,
    # one that failed, was skipped, dropped matches or was cut is not
    outcomes: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    backends = _build_backends(sources, warnings, unavailable)
    if not backends:
        result = {
            "candidates": [], "sources_queried": [],
            "warnings": warnings or ["No backends available"],
            "search_record": _search_record(query, specs, unavailable, 0),
        }
        _emit_db_matches_marker(result)
        return result

    accumulated: list[StructureCandidate] = []
    sources_queried: list[str] = []
    queried_set: set[str] = set()
    for spec in specs:
        for backend in backends:
            outcome: dict[str, Any] = {"chemistry": list(spec.chemistry), "backend": backend.name}
            try:
                backend.last_report = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - a backend that takes no attribute reports nothing
                pass
            try:
                results = backend.query(spec)
            except QuerySkipped as e:
                outcomes.append({**outcome, "status": "skipped", "reason": str(e)[:200]})
                continue
            except Exception as e:
                warnings.append(
                    f"{backend.name} backend raised on chemistry={spec.chemistry}: {e}"
                )
                _logger.warning("Backend %s failed: %s", backend.name, e)
                outcomes.append({**outcome, "status": "failed", "error": str(e)[:200]})
                continue
            report = getattr(backend, "last_report", None)
            outcome.update(status="ok", n=len(results))
            if isinstance(report, dict):
                # only the report's own fields: never the dispatcher's status or count
                outcome.update({k: report[k] for k in _REPORT_KEYS if report.get(k)})
            else:
                outcome["report"] = "unknown"
            outcomes.append(outcome)
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
        "search_record": _search_record(query, specs, unavailable + outcomes, len(deduped)),
    }
    _emit_db_matches_marker(result)
    return result


_REPORT_KEYS = ("dropped", "drop_reasons", "truncated", "unparsed")


def _search_record(query: dict, specs: list[QuerySpec], outcomes: list[dict], n: int) -> dict[str, Any]:
    """The call as asked (every hypothesis and filter) and what each backend
    did with it. ``clean_empty`` is true only when at least one backend
    answered every hypothesis, nothing failed, was dropped, truncated or left
    unparsed, and nothing came back: the one empty result that says the
    databases hold no match for this query."""
    filters = {k: v for k, v in query.items() if k not in ("chemistry", "top_n") and v is not None}
    served = {o["backend"] for o in outcomes if o.get("status") == "ok"}
    answered_all = any(
        all(any(o.get("backend") == b and o.get("status") == "ok" and o.get("chemistry") == list(sp.chemistry)
                for o in outcomes) for sp in specs)
        for b in served)
    unclean = any(o.get("status") == "failed" or o.get("dropped") or o.get("truncated")
                  or o.get("unparsed") or o.get("report") == "unknown" for o in outcomes)
    return {
        "query": {"chemistry": [list(sp.chemistry) for sp in specs], "filters": filters,
                  "top_n": specs[0].top_n if specs else None},
        "outcomes": outcomes,
        "n_candidates": n,
        "clean_empty": bool(n == 0 and answered_all and not unclean),
    }


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
    unavailable: Optional[list[dict]] = None,
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
        skipped = unavailable if unavailable is not None else []
        if factory is None:
            warnings.append(f"Unknown source '{name}' — skipping")
            skipped.append({"backend": name, "status": "skipped", "reason": "unknown source"})
            continue
        try:
            b = factory()
        except Exception as e:
            warnings.append(f"Failed to construct {name} backend: {e}")
            skipped.append({"backend": name, "status": "failed", "error": f"construction: {e}"[:200]})
            continue
        if not b.is_available():
            warnings.append(f"Backend '{name}' not available (missing key/dir/dependency)")
            skipped.append({"backend": name, "status": "skipped",
                            "reason": "not available (missing key/dir/dependency)"})
            continue
        backends.append(b)
    return backends


def _dedupe(candidates: Iterable[StructureCandidate]) -> list[StructureCandidate]:
    """Collapse duplicates across backends, preferring the more authoritative
    source. The entry kept records the other sources' ids (``also_in``) and
    keeps the better rank and an energy above hull the other source reported,
    so a measured cell does not lose the stability ranking a computed one had."""
    bucket: dict[tuple, StructureCandidate] = {}
    for cand in candidates:
        key = _dedupe_key(cand)
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = cand
            continue
        keep, drop = ((cand, existing)
                      if _SOURCE_PREFERENCE.get(cand.source, 99) < _SOURCE_PREFERENCE.get(existing.source, 99)
                      else (existing, cand))
        if drop.source != keep.source:
            keep.metadata.setdefault("also_in", []).append({"source": drop.source, "id": drop.id})
        if keep.metadata.get("energy_above_hull") is None and drop.metadata.get("energy_above_hull") is not None:
            keep.metadata["energy_above_hull"] = drop.metadata["energy_above_hull"]
        keep.rank_score = max(keep.rank_score, drop.rank_score)
        bucket[key] = keep
    return list(bucket.values())


def _dedupe_key(cand: StructureCandidate) -> tuple:
    """One key per structure whatever the backend's notation (#775 review):
    COD writes a spaced Hill formula and a spaced symbol with its origin
    setting ('O2 Sn', 'P 42/m n m'), MP ``formula_pretty`` and the compact
    symbol ('SnO2', 'P4_2/mnm'). The reduced formula and the space-group
    NUMBER, falling back to the strings as given."""
    formula = cand.formula
    try:
        from pymatgen.core import Composition
        formula = Composition(str(cand.formula).replace(" ", "")).reduced_formula
    except Exception:  # noqa: BLE001 - an unparseable formula keys as written
        pass
    number = (cand.metadata or {}).get("spacegroup_number")
    try:
        number = int(number)
    except (TypeError, ValueError):
        number = None
    return (formula, number if number and number > 0 else cand.space_group)


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
            with open(path, "w", encoding="utf-8") as f:
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
        "search_record": result.get("search_record"),
    }
    try:
        line = "DB_MATCHES_JSON: " + json.dumps(safe, default=str)
    except Exception as e:
        _logger.debug("Skipping DB_MATCHES_JSON emit: %s", e)
        return
    print(line, file=sys.stdout, flush=True)
