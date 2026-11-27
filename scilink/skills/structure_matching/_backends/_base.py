"""Protocol + dataclasses for structure-database backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# How a backend without its own relevance score (COD, local CIFs) orders the
# distinct phases (one per reduced formula + space group) of a chemistry query:
#   'reported' — most often deposited first (default): a phase reported many
#                times is the one a sample is most likely to contain;
#   'simplest' — fewest atoms in the formula first;
#   'newest'   — most recent deposition first.
RANK_BY_CHOICES = ("reported", "simplest", "newest")
# which database entry represents a phase (COD; see cod._order_group)
REPRESENTATIVE_CHOICES = ("consensus", "newest")


@dataclass
class QuerySpec:
    """Specification for a structure-database query.

    Fields are interpreted independently by each backend; a backend may
    ignore any field it does not support (e.g. ``LocalCIFBackend`` ignores
    ``max_e_above_hull`` since CIFs on disk carry no thermodynamic data).

    Additional filter fields (all optional):

    - ``z_range``: number of sites per unit cell (Materials Project's
      ``nsites``; pymatgen's ``Structure.num_sites`` for local). NOT
      strictly Z = formula-units-per-cell — MP does not expose Z directly
      as a filterable field, so ``nsites`` is the conventional proxy.
    - ``density_range``: bulk density in g/cm³.
    - ``anonymous_formula``: anonymized composition string ("AB2", "ABC3").
      Matches pymatgen's ``Composition.anonymized_formula``.
    - ``rank_by``: ordering of distinct phases for backends with no relevance
      score of their own (see ``RANK_BY_CHOICES``).
    - ``representative``: which entry represents a phase with many entries
      (see ``REPRESENTATIVE_CHOICES``): 'consensus' (default) — experimental,
      ambient, from the densest cluster of cell volumes, newest publication
      within it; 'newest' — the highest database id.
    """

    chemistry: list[str]
    space_group_hints: Optional[list[int]] = None
    lattice_param_ranges: Optional[dict[str, tuple[float, float]]] = None
    max_e_above_hull: Optional[float] = None
    top_n: int = 10
    z_range: Optional[tuple[int, int]] = None
    density_range: Optional[tuple[float, float]] = None
    anonymous_formula: Optional[str] = None
    rank_by: str = "reported"
    representative: str = "consensus"

    def __post_init__(self) -> None:
        # Chemistry-free (cell-only) queries are allowed when a lattice filter is
        # present — the blind-identification path, where the unit cell recovered
        # by autoindexing (index_pattern) is the search key instead of elements.
        # A backend that cannot search by cell alone returns [] for such specs.
        if not self.chemistry and not self.lattice_param_ranges:
            raise ValueError(
                "QuerySpec needs chemistry and/or lattice_param_ranges — pass "
                "element symbols, or a cell-only query (lattice_param_ranges "
                "from index_pattern) when the chemistry is unknown"
            )
        if self.top_n <= 0:
            raise ValueError("QuerySpec.top_n must be positive")
        if self.rank_by not in RANK_BY_CHOICES:
            raise ValueError(
                f"QuerySpec.rank_by must be one of {RANK_BY_CHOICES}; got {self.rank_by!r}")
        if self.representative not in REPRESENTATIVE_CHOICES:
            raise ValueError(
                f"QuerySpec.representative must be one of {REPRESENTATIVE_CHOICES}; "
                f"got {self.representative!r}")
        if self.z_range is not None:
            lo, hi = self.z_range
            if lo < 1 or hi < lo:
                raise ValueError(f"z_range must satisfy 1 ≤ lo ≤ hi; got {self.z_range}")
        if self.density_range is not None:
            lo, hi = self.density_range
            if lo < 0 or hi < lo:
                raise ValueError(
                    f"density_range must satisfy 0 ≤ lo ≤ hi; got {self.density_range}"
                )


@dataclass
class StructureCandidate:
    """A single candidate structure returned by a backend.

    ``structure_path`` is populated by the dispatching tool after CIFs are
    materialized into the session directory — backends construct the
    candidate first and the dispatcher writes the file.
    """

    id: str
    source: str
    formula: str
    space_group: Optional[str] = None
    structure_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0


class QuerySkipped(Exception):
    """Raised by ``query`` when a backend cannot serve this kind of query at all
    (e.g. a chemsys-keyed database asked a cell-only query). The dispatcher
    records the backend as skipped, neither queried nor failed."""


@runtime_checkable
class StructureBackend(Protocol):
    """Protocol every backend must satisfy.

    Backends are stateless once constructed (any caching is internal). The
    dispatching tool calls ``is_available()`` to filter the registered list,
    then ``query()`` on the survivors.
    """

    name: str

    def is_available(self) -> bool:
        """True when this backend can answer queries.

        A backend that needs an API key returns False when the key is
        missing; a backend that needs a local directory returns False when
        the directory does not exist. Must never raise.
        """
        ...

    def query(self, spec: QuerySpec) -> list[StructureCandidate]:
        """Return candidates matching ``spec``.

        Returned candidates are unsorted; ranking and dedup are the
        dispatcher's responsibility. A real error (network failure,
        malformed response) RAISES — the dispatcher records it and continues
        with other backends; an empty list means the database answered and
        holds nothing for this spec. ``QuerySkipped`` says the spec is not
        one this backend serves.

        A backend may also set ``last_report`` for the call: ``dropped`` (how
        many entries that matched could not be loaded) with
        ``drop_reasons``, ``truncated`` (a row cap cut the matches before
        ranking) and ``unparsed`` (files read while searching that did not
        parse). The dispatcher records it beside the call's outcome.
        """
        ...
