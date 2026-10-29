"""
Which element is each atom type of a LAMMPS data file.

A LAMMPS data file says which element an atom type is only through its
``Masses`` section: a mass, often followed by a ``# label``. This module is
the one place that reads it. Every caller (``structure_io``, the LAMMPS
skill's ``parse_data_file`` and ``pair_coeff`` writer, the ASE run-script
generator) gets the same answer and the same reason.

``read_type`` tries the readings of ``_READINGS`` in order. Each is a row:
what it takes from the label, and how close the mass must be.

======== ======================================= ===========================
basis    reading                                 mass check
======== ======================================= ===========================
label    the label is an element symbol ("Na")   within the isotope window
label    "D" / "T" (deuterium / tritium) -> H    within the isotope window
name     a CHARMM ion name ("SOD" -> Na)         within ``tolerance``
name     a full element name ("Carbon")          within ``tolerance``
prefix   the label's first 2, then 1 letters     within ``tolerance``
mass     no label reading: the mass alone        exactly one element within
                                                 ``tolerance``
======== ======================================= ===========================

An isotope-window reading is refused when the mass is another element's
standard mass (within ``tolerance``): "15.999 # N" is a swapped label, not
an isotope of nitrogen, and the later rows are tried instead.

A capitalised label that no reading accepts is a conflict ("CH2" at 14.03, a
coarse-grained "W" bead), and so is a mass with no element, or several, within
``tolerance``. A lowercase label that no reading accepts is a type name
("opls_140") and the mass decides. A conflict is reported, never guessed: a
caller passes ``type_elements`` to say what the file cannot (see
``apply_type_elements``). A new case is a new row plus a test.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ase.data

# How close a mass must be to an element's standard atomic mass for a
# derived reading (name, prefix) or a mass-only reading to count. 0.5 amu
# admits rounded masses ("59.0 # Ni" is 0.31 off) and refuses united-atom
# masses ("CH2" at 14.03 is 2 off carbon). Lower it for a strict check; raise
# it to accept heavier pseudo-atoms (and expect more mass-only ambiguities).
DEFAULT_MASS_TOLERANCE = 0.5

# How far an exact element-symbol label may sit from the standard mass and
# still be that element (an isotope or a repartitioned mass): D at 2.014 and a
# hydrogen-mass-repartitioned H at 3.024 are within it; a Martini "W" bead at
# 72 (tungsten is 183.84) is not. Raise it for heavier repartitioning.
DEFAULT_ISOTOPE_WINDOW = 3.0

# Ion type names whose letters are not the element symbol (CHARMM).
ION_TYPE_NAMES = {
    "SOD": "Na", "POT": "K", "CAL": "Ca", "CES": "Cs", "LIT": "Li",
    "RUB": "Rb", "BAR": "Ba", "ZN2": "Zn",
}

ISOTOPE_ALIASES = {"D": "H", "T": "H"}


def _std_mass(symbol: str) -> float:
    return float(ase.data.atomic_masses[ase.data.atomic_numbers[symbol]])


def _is_symbol(s: str) -> bool:
    # atomic_numbers maps the placeholder "X" to 0: not an element.
    return ase.data.atomic_numbers.get(s, 0) > 0


def _split_label(label: str) -> Tuple[str, str]:
    """``(word, letters)``: the label's first alphanumeric word and its
    leading letters ("Na+" -> ("Na", "Na"), "C12" -> ("C12", "C"))."""
    m = re.match(r"[A-Za-z0-9]+", (label or "").strip())
    if not m or not m.group()[0].isalpha():
        return "", ""
    word = m.group()
    return word, re.match(r"[A-Za-z]+", word).group()


# ── readings: (basis, candidates(word, letters), check) ──────────────────

def _exact_symbol(word, letters):
    return [word] if word == letters and _is_symbol(word) else []


def _isotope_alias(word, letters):
    return [ISOTOPE_ALIASES[word]] if word in ISOTOPE_ALIASES else []


def _ion_name(word, letters):
    return [ION_TYPE_NAMES[word.upper()]] if word.upper() in ION_TYPE_NAMES else []


def _element_name(word, letters):
    return [ase.data.chemical_symbols[z]
            for z, name in enumerate(ase.data.atomic_names)
            if z > 0 and name.lower() == letters.lower()]


def _prefixes(word, letters):
    out = []
    if len(letters) >= 2:
        out.append(letters[0].upper() + letters[1].lower())
    out.append(letters[0].upper())
    return [s for s in out if _is_symbol(s)]


_ISOTOPE, _DERIVED = "isotope", "derived"

_READINGS = (
    ("label",  _exact_symbol,  _ISOTOPE),
    ("label",  _isotope_alias, _ISOTOPE),
    ("name",   _ion_name,      _DERIVED),
    ("name",   _element_name,  _DERIVED),
    ("prefix", _prefixes,      _DERIVED),
)


def elements_near(mass: float, tolerance: float = DEFAULT_MASS_TOLERANCE) -> List[str]:
    """Elements whose standard mass is within ``tolerance`` of ``mass``,
    nearest first."""
    hits = [(abs(float(ase.data.atomic_masses[z]) - mass), ase.data.chemical_symbols[z])
            for z in range(1, len(ase.data.atomic_masses))
            if ase.data.atomic_masses[z] > 0]
    return [s for d, s in sorted(hits) if d <= tolerance]


def read_type(type_id: int, mass: float, label: str = "",
              tolerance: float = DEFAULT_MASS_TOLERANCE,
              isotope_window: float = DEFAULT_ISOTOPE_WINDOW) -> Dict[str, Any]:
    """The element of one atom type, with the basis for it, or a conflict.

    Returns ``{"type", "mass", "label", "element", "basis", "isotope",
    "conflict", "source"}``. ``element`` is None exactly when ``conflict``
    says why; ``basis`` is "label", "name", "prefix" or "mass"; ``isotope``
    is True when a symbol label is accepted beyond ``tolerance`` (D2O's
    "2.014 # H"). ``source`` is "file".
    """
    rec = {"type": int(type_id), "mass": float(mass), "label": label or "",
           "element": None, "basis": None, "isotope": False,
           "conflict": None, "source": "file"}
    word, letters = _split_label(label)
    if word:
        for basis, candidates, check in _READINGS:
            for sym in candidates(word, letters):
                delta = abs(_std_mass(sym) - mass)
                limit = isotope_window if check == _ISOTOPE else tolerance
                if check == _ISOTOPE and delta > tolerance and \
                        elements_near(mass, tolerance):
                    continue        # another element's mass: not an isotope
                if delta <= limit:
                    rec.update(element=sym, basis=basis,
                               isotope=check == _ISOTOPE and delta > tolerance)
                    return rec
        if label.strip()[0].isupper():
            # A capitalised label reads as an element claim ("CH2", "W") but
            # names no element of this mass.
            rec["conflict"] = (f"label '# {label}' names no element of mass "
                               f"{mass:g} (tolerance {tolerance:g} amu)")
            return rec
        # A lowercase label ("opls_140", "type1") is a type name: the mass decides.
    near = elements_near(mass, tolerance)
    if len(near) == 1:
        rec.update(element=near[0], basis="mass")
    elif not near:
        rec["conflict"] = (f"mass {mass:g} is not within {tolerance:g} amu of "
                           f"any element")
    else:
        rec["conflict"] = (f"mass {mass:g} is within {tolerance:g} amu of "
                           f"{' and '.join(near)}; a label must say which")
    return rec


# ── the data file ────────────────────────────────────────────────────────

def parse_masses(lines: Iterable[str]) -> List[Tuple[int, float, str]]:
    """``(type, mass, label)`` for each row of the ``Masses`` section.

    The section ends at the first line that does not start with a type
    number: the next section header, with or without a style comment
    (``Pair Coeffs # lj/cut/coul/long``, ``PairIJ Coeffs``, ``Atoms # full``).
    """
    rows, inside = [], False
    for line in lines:
        stripped = line.strip()
        if not inside:
            inside = stripped.split("#", 1)[0].strip() == "Masses"
            continue
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("#", 1)[0].split()
        if not parts or not parts[0].isdigit():
            break
        try:
            mass = float(parts[1])
        except (IndexError, ValueError):
            continue
        label = stripped.split("#", 1)[1].strip() if "#" in stripped else ""
        rows.append((int(parts[0]), mass, label))
    return rows


_COUNT = re.compile(r"^\s*(\d+)\s+(atoms|bonds|atom types)\s*(#.*)?$")


def header_counts(lines: Iterable[str]) -> Dict[str, int]:
    """``{"atoms", "bonds", "atom types"}`` counts from the header."""
    out = {"atoms": 0, "bonds": 0, "atom types": 0}
    for line in lines:
        s = line.strip()
        if s.split("#", 1)[0].strip() in ("Masses", "Atoms"):
            break
        m = _COUNT.match(s)
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def read_types(lines: Iterable[str],
               tolerance: float = DEFAULT_MASS_TOLERANCE,
               isotope_window: float = DEFAULT_ISOTOPE_WINDOW) -> Dict[int, Dict[str, Any]]:
    """``{type: read_type(...)}`` for every row of the ``Masses`` section."""
    return {t: read_type(t, m, lab, tolerance, isotope_window)
            for t, m, lab in parse_masses(lines)}


def check_type_elements(type_elements: Optional[Dict[Any, str]]) -> Dict[int, str]:
    """``{int type: symbol}`` from a caller's map (JSON gives string keys).
    Raises ValueError for a key that is not a type number or a value that is
    not an element symbol."""
    out = {}
    for key, element in (type_elements or {}).items():
        try:
            t = int(str(key).strip())
        except (TypeError, ValueError):
            raise ValueError(f"type_elements key {key!r} is not an atom type number")
        element = str(element).strip()
        if not _is_symbol(element):
            raise ValueError(f"type_elements[{t}] = {element!r} is not an element symbol")
        out[t] = element
    return out


def apply_type_elements(readings: Dict[int, Dict[str, Any]],
                        type_elements: Optional[Dict[Any, str]],
                        n_types: int = 0) -> Dict[int, Dict[str, Any]]:
    """Apply a caller's ``{type: element}`` on top of the file's readings.

    The caller (the model, a user) decides; this checks and records. Each
    override must name a real element and a declared type (when the file
    says how many there are). It replaces the file's answer on the record,
    ``source`` becomes "override", and the file's own reading is kept under
    ``file_reading``. Types without an override keep the file's reading.
    Raises ValueError for an override that is not an element, not a type, or
    names a type with no Masses entry (the runners need its mass).
    """
    out = {t: dict(r) for t, r in readings.items()}
    for t, element in check_type_elements(type_elements).items():
        if n_types and not 1 <= t <= n_types:
            raise ValueError(f"type_elements names type {t}, but the file "
                             f"declares {n_types} atom types")
        file_rec = out.get(t)
        if file_rec is None:
            raise ValueError(f"type {t} has no Masses entry; add its mass "
                             f"('{t} <mass> # {element}'): LAMMPS and ASE "
                             f"need it")
        out[t] = {"type": t, "mass": file_rec["mass"], "label": file_rec["label"],
                  "element": element, "basis": "override", "isotope": False,
                  "conflict": None, "source": "override",
                  "file_reading": {k: file_rec[k] for k in
                                   ("element", "basis", "isotope", "conflict")}}
    return out
