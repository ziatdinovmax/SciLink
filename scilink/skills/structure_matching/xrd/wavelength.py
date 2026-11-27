"""``resolve_wavelength`` tool — pick an XRD source wavelength from metadata.

Looks at structured `experiment.*` fields first, then falls back to a
free-text scan for canonical source names. Returns either a named source
string (passable to ``simulate_xrd_pattern`` as ``wavelength``) or a
float wavelength in angstroms.

Eliminates the silent-CuKa-default footgun: when metadata clearly names
a non-Cu source (e.g. MoKa, CoKa), the LLM script should call this
function instead of hard-coding ``wavelength='CuKa'``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Union

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# Anode line names -> Å, the ONE table every XRD tool resolves a named wavelength
# with (refinement engine, search-match, zero calibration, space-group analysis),
# so every name resolve_wavelength can return is accepted everywhere. 'XKa'
# keeps the value the tools have always used; 'XKa1' / 'XKa2' are the lines.
ANODE_ALIASES = {
    "cuka": 1.5406, "cuka1": 1.54056, "cuka2": 1.54439,
    "moka": 0.71073, "moka1": 0.70930, "moka2": 0.71359,
    "coka": 1.78897, "coka1": 1.788965, "coka2": 1.79285,
    "feka": 1.93604, "feka1": 1.936042, "feka2": 1.93998,
    "crka": 2.28970, "crka1": 2.289700, "crka2": 2.293606,
    "agka": 0.55941, "agka1": 0.559421, "agka2": 0.563813,
}


def wavelength_angstrom(wavelength: Any) -> float:
    """A wavelength spec in Å: a number as given, or an anode line name
    ('CuKa', 'MoKa1', 'Co K-a2', ...; case, spaces and hyphens ignored)."""
    if isinstance(wavelength, (int, float)) and not isinstance(wavelength, bool):
        return float(wavelength)
    key = str(wavelength).strip().lower().replace(" ", "").replace("-", "")
    if key in ANODE_ALIASES:
        return ANODE_ALIASES[key]
    raise ValueError(
        f"Unrecognized wavelength {wavelength!r}. Pass a numeric wavelength in Å "
        f"or one of {sorted(ANODE_ALIASES)}."
    )


# Recognized named sources. Keys are the canonical names accepted by
# pymatgen's XRDCalculator; values are aliases that may appear in free text.
_ANODES = {"Cu": "copper", "Mo": "molybdenum", "Co": "cobalt", "Fe": "iron", "Cr": "chromium", "Ag": "silver"}
_KA_SPELLINGS = ("ka", "kα", "k-alpha", "k alpha", "kalpha", "k-α")


def _aliases(symbol: str, element: str, suffix: str) -> tuple[str, ...]:
    out = []
    for name in (symbol.lower(), element):
        for ka in _KA_SPELLINGS:
            for sep in ("", " ", "-"):
                if name == element and sep == "":
                    continue
                out.append(f"{name}{sep}{ka}{suffix}")
                if suffix:
                    out.append(f"{name}{sep}{ka} {suffix}")
    return tuple(dict.fromkeys(out))


# Recognized named sources. Keys are the canonical names accepted by
# pymatgen's XRDCalculator; values are aliases that may appear in free text.
# 'XKa' is the tube's K-alpha radiation (K-alpha1 + K-alpha2 unless the
# metadata says otherwise); 'XKa1' is the K-alpha1 line alone (monochromated).
_NAMED_SOURCES: dict[str, tuple[str, ...]] = {}
for _sym, _el in _ANODES.items():
    _NAMED_SOURCES[f"{_sym}Ka"] = _aliases(_sym, _el, "")
    _NAMED_SOURCES[f"{_sym}Ka1"] = _aliases(_sym, _el, "1")

# Radiation that carries BOTH K-alpha lines, stated outright ...
_DOUBLET_RE = re.compile(
    r"k\s*-?\s*(?:alpha|α|a)\s*-?\s*1\s*(?:\+|,|/|&|and)\s*(?:k\s*-?\s*(?:alpha|α|a)\s*-?\s*)?2"
    r"|k\s*-?\s*(?:alpha|α|a)\s*-?\s*2\s*(?:present|included|retained|not (?:stripped|removed))"
    r"|k\s*-?\s*(?:alpha|α|a)\s*-?\s*1\s*\+\s*k", re.IGNORECASE)
# ... and radiation reduced to K-alpha1 (a primary / Johansson monochromator).
_KA1_ONLY_RE = re.compile(
    r"strictly monochromatic|(?:primary|incident[- ]beam|johansson)[^.;]{0,40}monochromat"
    r"|monochromat\w*[^.;]{0,40}(?:primary|incident[- ]beam|johansson)"
    r"|k\s*-?\s*(?:alpha|α|a)\s*-?\s*1\s*only|k\s*-?\s*(?:alpha|α|a)\s*-?\s*2\s*(?:stripped|removed|eliminated|suppressed)",
    re.IGNORECASE)


TOOL_SPEC = ToolSpec(
    name="resolve_wavelength",
    description=(
        "Pick an XRD source wavelength from experiment metadata. Reads "
        "structured fields (experiment.wavelength, experiment.source, "
        "experiment.x_ray_source) and falls back to a free-text scan "
        "for canonical source names (CuKa, MoKa, CoKa, FeKa, CrKa, "
        "AgKa, CuKa1). Returns the resolved wavelength as a string "
        "(named source) or float (angstroms)."
    ),
    import_line="from scilink.skills.structure_matching.xrd.wavelength import resolve_wavelength",
    signature=(
        "resolve_wavelength(system_info: dict | str | None, "
        "default: str | float = 'CuKa') -> str | float"
    ),
    parameters={
        "system_info": {
            "type": "dict | str | None",
            "description": (
                "System info dict (with 'experiment' subkey) or a free-text "
                "metadata string. Pass None to get the default."
            ),
        },
        "default": {
            "type": "str | float",
            "description": "Fallback when no source can be inferred. Default 'CuKa'.",
        },
    },
    required=["system_info"],
    returns=(
        "str (named source, e.g. 'MoKa') or float (wavelength in angstroms). "
        "Pass directly to simulate_xrd_pattern's wavelength parameter."
    ),
    when_to_use=(
        "Before calling simulate_xrd_pattern, to avoid silently defaulting "
        "to CuKa on a Mo/Co/Cr-source measurement. The wrong wavelength "
        "gives uniformly bad scores for every candidate."
    ),
)


def resolve_wavelength(
    system_info: Union[dict, str, None],
    default: Union[str, float] = "CuKa",
) -> Union[str, float]:
    """Resolve an XRD source wavelength from metadata. See ``TOOL_SPEC``."""
    if system_info is None:
        return default

    text_blobs = _collect_text(system_info) if isinstance(system_info, dict) else [str(system_info)]
    text = " ".join(text_blobs)

    # 1. Structured fields take priority.
    if isinstance(system_info, dict):
        structured = None
        experiment = system_info.get("experiment")
        for block, keys in ((experiment, ("wavelength", "x_ray_wavelength", "source", "x_ray_source")),
                            (system_info, ("wavelength", "x_ray_wavelength"))):
            if not isinstance(block, dict):
                continue
            # the named keys first, then any other '*wavelength*' / 'lambda' key
            # (e.g. 'wavelength_angstrom') — an unread wavelength field must not
            # fall through to the CuKa default
            extra = [k for k in block if isinstance(k, str) and k not in keys
                     and ("wavelength" in k.lower() or k.lower().startswith("lambda"))]
            for key in (*keys, *extra):
                structured = _coerce_field(block.get(key))
                if structured is not None:
                    break
            if structured is not None:
                break
        if structured is not None:
            return _arbitrate_lab_line(structured, text)

    # 2. Free-text scan across all string-valued metadata.
    for blob in text_blobs:
        named = _scan_text_for_named_source(blob)
        if named is not None:
            return _arbitrate_lab_line(named, text)
        angstroms = _scan_text_for_angstroms(blob, need_context=True)
        if angstroms is not None:
            return _arbitrate_lab_line(angstroms, text)

    return default


def _anode_of(value: Union[str, float]) -> Union[str, None]:
    """The anode whose K-alpha line a value names or equals (within 0.002 Å)."""
    if isinstance(value, str):
        m = re.match(r"([A-Z][a-z])Ka1?$", value)
        return m.group(1) if m and m.group(1) in _ANODES else None
    for sym, lines in _ANODE_LINES.items():
        if any(abs(float(value) - lam) < 0.002 for lam in lines):
            return sym
    return None


# K-alpha1, K-alpha2 and the weighted K-alpha (pymatgen's 'XKa') of each anode, in Å
_ANODE_LINES = {"Cu": (1.540593, 1.544414, 1.54184), "Mo": (0.709300, 0.713590, 0.71073),
                "Co": (1.788965, 1.792850, 1.79026), "Fe": (1.936042, 1.939980, 1.93735),
                "Cr": (2.289700, 2.293606, 2.29100), "Ag": (0.559421, 0.563813, 0.56087)}


def _arbitrate_lab_line(value: Union[str, float], text: str) -> Union[str, float]:
    """Decide K-alpha doublet vs K-alpha1 for a lab-anode result from what the
    metadata says about the beam. Refinement tools model 'XKa' as the
    K-alpha1 + K-alpha2 doublet and 'XKa1' / a number as one line.

    - explicit K-alpha1-only wording (strictly monochromatic, primary /
      Johansson monochromator, 'K-alpha1 only', K-alpha2 stripped) -> 'XKa1'
    - explicit doublet wording (K-alpha1 + K-alpha2, K-alpha2 present), or a
      tube K-alpha source named in the text -> 'XKa'
    - a bare number with no such wording, or a non-anode value (synchrotron)
      -> unchanged."""
    sym = _anode_of(value)
    if sym is None:
        return value
    if _affirmed(_KA1_ONLY_RE, text):
        return f"{sym}Ka1"
    if _DOUBLET_RE.search(text):
        return f"{sym}Ka"
    if isinstance(value, str):
        return value
    named = _scan_text_for_named_source(text)
    if named in (f"{sym}Ka", f"{sym}Ka1"):
        return named
    return value


_NEGATION_RE = re.compile(r"\b(?:no|not|without|non|absent|removed)\b[^.;,]{0,25}$", re.IGNORECASE)


def _affirmed(pattern: "re.Pattern", text: str) -> bool:
    """A match not negated just before it ('no incident-beam monochromator',
    'without a primary monochromator' describe the doublet, not K-alpha1)."""
    return any(not _NEGATION_RE.search(text[max(0, m.start() - 40):m.start()])
               for m in pattern.finditer(text))


def _coerce_field(value: Any) -> Union[str, float, None]:
    """Convert a metadata field to a wavelength string or float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if 0.1 < v < 5.0:
            return v
        return None
    if isinstance(value, dict):
        return _coerce_line_dict(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        # Exact named-source match (case insensitive, hyphen / space tolerant).
        named = _match_named_source(v)
        if named is not None:
            return named
        # Numeric string in angstroms.
        try:
            num = float(v)
            if 0.1 < num < 5.0:
                return num
        except ValueError:
            pass
        # Substring scan for the field's own text content.
        named = _scan_text_for_named_source(v)
        if named is not None:
            return named
        angstroms = _scan_text_for_angstroms(v)
        if angstroms is not None:
            return angstroms
    return None


def _coerce_line_dict(value: dict) -> Union[str, float, None]:
    """{'K-alpha1': 1.5406, 'K-alpha2': 1.5444, 'K-alpha2/K-alpha1': 0.5}-style
    fields: the anode's doublet name when K-alpha2 is present with a non-zero
    ratio, the K-alpha1 line otherwise."""
    ka1 = ka2 = ratio = None
    for k, v in value.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        n = re.sub(r"[^a-z0-9/]", "", str(k).lower().replace("α", "a").replace("alpha", "a"))
        if "/" in n or "ratio" in n:
            ratio = float(v)
        elif n.endswith("ka1") or n in ("lam1", "lambda1", "a1"):
            ka1 = float(v)
        elif n.endswith("ka2") or n in ("lam2", "lambda2", "a2"):
            ka2 = float(v)
    if ka1 is None or not 0.1 < ka1 < 5.0:
        return None
    sym = _anode_of(ka1)
    if sym is None:
        return ka1
    if ka2 is not None and (ratio is None or ratio > 0):
        return f"{sym}Ka"
    return f"{sym}Ka1"


def _match_named_source(text: str) -> Union[str, None]:
    """Exact match (after normalization) against canonical source names."""
    norm = _normalize(text)
    for canonical, aliases in _NAMED_SOURCES.items():
        if norm == _normalize(canonical):
            return canonical
        for alias in aliases:
            if norm == _normalize(alias):
                return canonical
    return None


def _scan_text_for_named_source(text: str) -> Union[str, None]:
    """Substring scan for canonical source names. Most-specific match wins."""
    norm = _normalize(text)
    best: tuple[str, int] | None = None  # (canonical, alias_length)
    for canonical, aliases in _NAMED_SOURCES.items():
        for alias in (canonical, *aliases):
            alias_norm = _normalize(alias)
            if alias_norm and alias_norm in norm:
                if best is None or len(alias_norm) > best[1]:
                    best = (canonical, len(alias_norm))
    return best[0] if best else None


_ANGSTROM_RE = re.compile(
    r"(?P<value>\d+\.\d+)\s*(?:[åÅ]|angstrom|A\b)",
    re.IGNORECASE,
)


# In free text a length in Å is a WAVELENGTH only when its clause says so:
# 'λ = 0.4959 Å', 'wavelength 0.7093 Å', 'Cu Kα (1.5406 Å)', 'synchrotron ...'.
# Lattice parameters and d-spacings ('anatase, a = 3.785 Å', 'd(200) = 2.82 Å')
# are lengths in the same range and must not be read as the wavelength.
_RADIATION_CONTEXT = re.compile(
    r"wavelength|λ|lambda|radiation|k\s*-?\s*(?:alpha|α)\b|kα|synchrotron|beamline|"
    r"\b(?:cu|mo|co|fe|cr|ag)\s*-?\s*k", re.IGNORECASE)


def _clause_before(text: str, start: int, span: int = 40) -> str:
    """The part of ``text`` just before ``start`` that belongs to the same clause."""
    return re.split(r"[;,]|\.\s", text[max(0, start - span):start])[-1]


def _scan_text_for_angstroms(text: str, need_context: bool = False) -> Union[float, None]:
    """Pull a 'N.NNN Å' / 'N.NNN angstrom' style number from text. With
    ``need_context`` (free text), only a value whose clause names the radiation."""
    for match in _ANGSTROM_RE.finditer(text):
        try:
            v = float(match.group("value"))
        except ValueError:
            continue
        if not 0.1 < v < 5.0:
            continue
        if need_context and not _RADIATION_CONTEXT.search(_clause_before(text, match.start())):
            continue
        return v
    return None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _collect_text(d: Any, _seen: set | None = None) -> list[str]:
    """Recursively collect all string values from a dict-like metadata blob."""
    if _seen is None:
        _seen = set()
    if id(d) in _seen:
        return []
    _seen.add(id(d))

    blobs: list[str] = []
    if isinstance(d, str):
        blobs.append(d)
    elif isinstance(d, dict):
        for v in d.values():
            blobs.extend(_collect_text(v, _seen))
    elif isinstance(d, (list, tuple)):
        for item in d:
            blobs.extend(_collect_text(item, _seen))
    return blobs


# --- diffractometer geometry -------------------------------------------------
# Flat-plate reflection (Bragg-Brentano) and capillary / transmission
# (Debye-Scherrer) differ in the sample-displacement term and in whether
# crystallites lie preferentially (texture); refinement needs to know which.
_REFLECTION_RE = re.compile(
    r"bragg[- ]?brentano|reflection|flat[- ]?plate|theta\s*/\s*2\s*-?\s*theta|θ\s*/\s*2θ|"
    r"theta[- ]2[- ]?theta|θ-2θ|front[- ]loaded|back[- ]loaded|rear[- ]loaded|"
    r"zero[- ]background (?:holder|plate)|spinning (?:sample|stage)",
    re.IGNORECASE)
_TRANSMISSION_RE = re.compile(
    r"debye[- ]?scherrer|capillary|transmission|kapton|foil|cylindrical sample",
    re.IGNORECASE)


def resolve_geometry(system_info: Union[dict, str, None]) -> Optional[str]:
    """'bragg-brentano', 'debye-scherrer' or None. See ``_SPEC_GEOMETRY``."""
    if system_info is None:
        return None
    if isinstance(system_info, dict):
        # an explicit geometry field (e.g. the sample mode an instrument file
        # records) decides before free text, where stage or holder names such as
        # 'reflection-transmission spinner' can name both
        for block in (system_info, system_info.get("experiment")):
            if not isinstance(block, dict):
                continue
            for key in ("geometry", "diffraction_geometry", "sample_mode"):
                if isinstance(block.get(key), str):
                    g = _geometry_from_text(block[key])
                    if g is not None:
                        return g
    text = " ".join(_collect_text(system_info)) if isinstance(system_info, dict) else str(system_info)
    return _geometry_from_text(text)


def _geometry_from_text(text: str) -> Optional[str]:
    refl, trans = bool(_REFLECTION_RE.search(text)), bool(_TRANSMISSION_RE.search(text))
    if refl and not trans:
        return "bragg-brentano"
    if trans and not refl:
        return "debye-scherrer"
    if trans and refl:
        # 'transmission' can describe a Kapton window on a reflection stage;
        # an explicit geometry name decides
        if re.search(r"debye[- ]?scherrer|capillary", text, re.IGNORECASE):
            return "debye-scherrer"
        if re.search(r"bragg[- ]?brentano", text, re.IGNORECASE):
            return "bragg-brentano"
    return None


_SPEC_GEOMETRY = ToolSpec(
    name="resolve_geometry",
    description=(
        "Read the diffractometer geometry from experiment metadata: "
        "'bragg-brentano' (flat-plate reflection, theta/2theta, spinning or "
        "front/back-loaded holder) or 'debye-scherrer' (capillary, "
        "transmission). None when the metadata does not say."
    ),
    import_line="from scilink.skills.structure_matching.xrd.wavelength import resolve_geometry",
    signature="resolve_geometry(system_info: dict | str | None) -> str | None",
    parameters={
        "system_info": {"type": "dict | str | None",
                        "description": "The record's metadata (searched as text) or a free-text string."},
    },
    required=["system_info"],
    returns="'bragg-brentano', 'debye-scherrer', or None (unknown or contradictory).",
    when_to_use=(
        "Before Rietveld refinement: pass the result as geometry= to "
        "refine_rietveld / refine_rietveld_multiphase / refine_rietveld_series. "
        "It sets the sample-displacement term and whether the automatic "
        "preferred-orientation correction applies (flat-plate reflection only). "
        "None lets the refinement infer it from the wavelength."
    ),
)

TOOL_SPECS = [_SPEC_GEOMETRY]
