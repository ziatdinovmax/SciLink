"""VASP pseudopotential (POTCAR) hooks for the engine-neutral contract (#786).

The agent records its choice in ``POTCAR.spec`` (one variant per element).
``prepare_pseudopotentials.sh`` reads that file on the execution host, walks the
POSCAR species line in order, concatenates the listed variant of each species
from ``$VASP_PP_PATH``, and writes ``pseudopotentials.provenance``. Assembly happens
where the library lives, so local, cluster and UI runs share one path, and a
critic fix that edits ``POTCAR.spec`` takes effect on the next run. POTCAR
contents (licensed) never enter an LLM prompt; only names and header metadata do.

Library layouts accepted under ``$VASP_PP_PATH``: ``<lib>/<variant>/POTCAR``
(the directory is itself a functional set) or ``<lib>/potpaw_PBE/<variant>/POTCAR``
(ASE / pymatgen style; ``potpaw_GGA`` for ``GGA = 91``, ``potpaw`` for LDA).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec
from ..._shared.pseudopotentials import element_of

SPEC_FILE = "POTCAR.spec"
PREP_SCRIPT = "prepare_pseudopotentials.sh"
PROVENANCE_FILE = "pseudopotentials.provenance"
LIBRARY_ENV = "VASP_PP_PATH"
MISSING_EXIT_CODE = 3

# Element -> variant names only (no licensed content). Used when pymatgen's
# Materials Project input set is unavailable; pymatgen is the primary source.
_MP_FALLBACK = {
    "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As", "Au": "Au",
    "B": "B", "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi", "Br": "Br", "C": "C",
    "Ca": "Ca_sv", "Cd": "Cd", "Ce": "Ce", "Cl": "Cl", "Co": "Co",
    "Cr": "Cr_pv", "Cs": "Cs_sv", "Cu": "Cu_pv", "Dy": "Dy_3", "Er": "Er_3",
    "Eu": "Eu", "F": "F", "Fe": "Fe_pv", "Ga": "Ga_d", "Gd": "Gd",
    "Ge": "Ge_d", "H": "H", "He": "He", "Hf": "Hf_pv", "Hg": "Hg",
    "Ho": "Ho_3", "I": "I", "In": "In_d", "Ir": "Ir", "K": "K_sv",
    "Kr": "Kr", "La": "La", "Li": "Li_sv", "Lu": "Lu_3", "Mg": "Mg_pv",
    "Mn": "Mn_pv", "Mo": "Mo_pv", "N": "N", "Na": "Na_pv", "Nb": "Nb_pv",
    "Nd": "Nd_3", "Ne": "Ne", "Ni": "Ni_pv", "Np": "Np", "O": "O",
    "Os": "Os_pv", "P": "P", "Pa": "Pa", "Pb": "Pb_d", "Pd": "Pd",
    "Pm": "Pm_3", "Pr": "Pr_3", "Pt": "Pt", "Pu": "Pu", "Rb": "Rb_sv",
    "Re": "Re_pv", "Rh": "Rh_pv", "Ru": "Ru_pv", "S": "S", "Sb": "Sb",
    "Sc": "Sc_sv", "Se": "Se", "Si": "Si", "Sm": "Sm_3", "Sn": "Sn_d",
    "Sr": "Sr_sv", "Ta": "Ta_pv", "Tb": "Tb_3", "Tc": "Tc_pv", "Te": "Te",
    "Th": "Th", "Ti": "Ti_pv", "Tl": "Tl_d", "Tm": "Tm_3", "U": "U",
    "V": "V_pv", "W": "W_pv", "Xe": "Xe", "Y": "Y_sv", "Yb": "Yb_2",
    "Zn": "Zn", "Zr": "Zr_sv",
}

_FUNCTIONAL_SUBDIRS = {"PE": "potpaw_PBE", "91": "potpaw_GGA", "LDA": "potpaw"}


def _mp_table() -> Dict[str, str]:
    """Materials Project's element -> POTCAR variant table, from pymatgen when
    importable (so it tracks MP), else the vendored names above."""
    try:
        from importlib.resources import files

        import yaml
        text = (files("pymatgen.io.vasp") / "MPRelaxSet.yaml").read_text()
        table = (yaml.safe_load(text) or {}).get("POTCAR") or {}
        if table:
            return {str(k): str(v) for k, v in table.items()}
    except Exception:
        pass
    return dict(_MP_FALLBACK)


def default_pseudopotentials(elements: List[str]) -> Dict[str, str]:
    """Recommended variant per element (Materials Project set)."""
    table = _mp_table()
    return {el: table.get(el, el) for el in elements}


def _library_root(library: Optional[str]) -> Optional[Path]:
    root = library or os.environ.get(LIBRARY_ENV)
    return Path(root) if root and Path(root).is_dir() else None


def _set_dirs(root: Path, gga: Optional[str] = None) -> List[Path]:
    """Directories that may hold ``<variant>/POTCAR``, most specific first."""
    sub = _FUNCTIONAL_SUBDIRS.get((gga or "PE").upper(), "potpaw_PBE")
    return [d for d in (root, root / sub) if d.is_dir()]


def _read_header(potcar: Path) -> Dict[str, Any]:
    """TITEL / ZVAL / ENMAX from a POTCAR header (reads only the header)."""
    info: Dict[str, Any] = {}
    try:
        with open(potcar, "r", errors="replace") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                if "TITEL" in line and "titel" not in info:
                    info["titel"] = line.split("=", 1)[1].strip()
                m = re.search(r"ZVAL\s*=\s*([-\d.]+)", line)
                if m and "zval" not in info:
                    info["zval"] = float(m.group(1))
                m = re.search(r"ENMAX\s*=\s*([-\d.]+)", line)
                if m and "enmax" not in info:
                    info["enmax"] = float(m.group(1))
                if len(info) == 3:
                    break
    except OSError:
        pass
    return info


def list_pseudopotentials(
    elements: List[str],
    library: Optional[str] = None,
    gga: Optional[str] = None,
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Variants available locally for each element, with header metadata.

    Returns ``None`` when no library is readable here (e.g. inputs are generated
    on a workstation and run on a cluster); the choice is then checked when the
    prep script runs on the execution host.
    """
    root = _library_root(library)
    if root is None:
        return None
    wanted = set(elements)
    out: Dict[str, List[Dict[str, Any]]] = {el: [] for el in elements}
    seen = set()
    for d in _set_dirs(root, gga):
        for entry in sorted(d.iterdir()):
            el = element_of(entry.name)
            potcar = entry / "POTCAR"
            if el in wanted and entry.name not in seen and potcar.is_file():
                seen.add(entry.name)
                out[el].append({"variant": entry.name, **_read_header(potcar)})
    return out


def pseudopotential_set(input_files: Dict[str, str]) -> Optional[str]:
    """INCAR ``GGA`` tag when it selects a POTCAR set other than PBE (``91`` ->
    potpaw_GGA, ``LDA`` -> potpaw), else None. Mirrors the prep script."""
    m = re.search(r"^\s*GGA\s*=\s*([A-Za-z0-9]+)",
                  (input_files or {}).get("INCAR", ""), re.M | re.I)
    gga = m.group(1).upper() if m else None
    return gga if gga in _FUNCTIONAL_SUBDIRS and gga != "PE" else None


def poscar_species(poscar_text: str) -> List[str]:
    """Species line (line 6) of a VASP5 POSCAR, in order (repeats kept)."""
    lines = (poscar_text or "").splitlines()
    if len(lines) < 6:
        return []
    toks = lines[5].split()
    return toks if toks and all(element_of(t) == t for t in toks) else []


_PREP_TEMPLATE = r"""#!/bin/bash
# Generated by SciLink: assemble POTCAR from {spec} on the execution host.
# {spec} lists one variant per element; for each species on POSCAR line 6, in
# order, the listed variant of that element is used. Edit it to change a choice.
set -u
LIB="${{{env}:-}}"
fail() {{ echo "MISSING PSEUDOPOTENTIAL $*" >&2; exit {code}; }}
[ -n "$LIB" ] || fail "library: {env} is not set"
[ -f {spec} ] || fail "spec: {spec} not found"
GGA=$(grep -iE '^[[:space:]]*GGA[[:space:]]*=' INCAR 2>/dev/null | head -1 \
      | sed -E 's/^[^=]*=[[:space:]]*([A-Za-z0-9]+).*/\1/' | tr '[:lower:]' '[:upper:]')
case "$GGA" in 91) SUB=potpaw_GGA ;; LDA) SUB=potpaw ;; *) SUB=potpaw_PBE ;; esac
[ -f POSCAR ] || fail "structure: POSCAR not found"
read -r -a SPECIES <<< "$(sed -n 6p POSCAR)"
[ ${{#SPECIES[@]}} -gt 0 ] || fail "POSCAR: line 6 is empty (a VASP5 POSCAR with a species line is required)"
for el in "${{SPECIES[@]}}"; do
  [[ "$el" =~ ^[A-Z][a-z]?$ ]] || fail "POSCAR: line 6 is not a species line ('$el'); a VASP5 POSCAR with element symbols is required"
done
: > POTCAR.tmp; : > {prov}.tmp
for el in "${{SPECIES[@]}}"; do
  v=""
  while read -r cand _; do
    [ -z "$cand" ] && continue
    [ "$(echo "$cand" | sed -E 's/^([A-Z][a-z]?).*/\1/')" = "$el" ] && {{ v="$cand"; break; }}
  done < {spec}
  [ -n "$v" ] || fail "$el: no variant listed in {spec}"
  if [ -f "$LIB/$v/POTCAR" ]; then p="$LIB/$v/POTCAR"
  elif [ -f "$LIB/$SUB/$v/POTCAR" ]; then p="$LIB/$SUB/$v/POTCAR"
  else fail "$el:$v in $LIB"; fi
  cat "$p" >> POTCAR.tmp
  enmax=$(grep -m1 'ENMAX' "$p" | sed -E 's/.*ENMAX[[:space:]]*=[[:space:]]*([0-9.]+).*/\1/')
  titel=$(grep -m1 'TITEL' "$p" | sed -E 's/.*=[[:space:]]*//')
  sum=$( (sha256sum "$p" 2>/dev/null || shasum -a 256 "$p" 2>/dev/null) | cut -d' ' -f1)
  echo "$el $v ${{enmax:-?}} ${{sum:-?}} $titel" >> {prov}.tmp
done
mv POTCAR.tmp POTCAR && mv {prov}.tmp {prov}
"""


def render_pseudopotential_prep(
    choice: Dict[str, Any],
    species: List[str],
) -> Dict[str, str]:
    """``POTCAR.spec`` + the prep script for a resolved choice.

    Args:
        choice: ``{element: variant}`` or ``{element: {"variant": ...}}``.
        species: POSCAR species order (repeats kept), from :func:`poscar_species`.
    """
    def variant(el):
        c = choice.get(el, el)
        return c.get("variant") if isinstance(c, dict) else c

    order = list(dict.fromkeys(species)) or list(choice)
    spec = "\n".join(variant(el) for el in order) + "\n"
    script = _PREP_TEMPLATE.format(spec=SPEC_FILE, prov=PROVENANCE_FILE,
                                   env=LIBRARY_ENV, code=MISSING_EXIT_CODE)
    return {SPEC_FILE: spec, PREP_SCRIPT: script}


def pre_run_command() -> str:
    """Shell step the pipeline prefixes to the run command."""
    return f"bash {PREP_SCRIPT}"


def describe_pseudopotentials(run_dir: str) -> Dict[str, Dict[str, Any]]:
    """Provenance written by the prep script: ``{element: {variant, enmax,
    sha256, titel}}`` (empty if the run never got that far)."""
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(run_dir) / PROVENANCE_FILE
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 4:
            continue
        el, v, enmax, sha = parts[:4]
        try:
            enmax_v: Any = float(enmax)
        except ValueError:
            enmax_v = None
        out.setdefault(el, {"variant": v, "enmax": enmax_v, "sha256": sha,
                            "titel": parts[4] if len(parts) > 4 else None})
    return out


def check_encut_floor(incar_text: str, enmax: Dict[str, float],
                      factor: float = 1.3) -> Optional[str]:
    """A finding when ENCUT is below ``factor`` x the largest ENMAX, else None."""
    m = re.search(r"^\s*ENCUT\s*=\s*([\d.]+)", incar_text or "", re.M | re.I)
    vals = [v for v in (enmax or {}).values() if isinstance(v, (int, float))]
    if not m or not vals:
        return None
    encut, floor = float(m.group(1)), factor * max(vals)
    if encut < floor:
        return (f"ENCUT = {encut:g} eV is below {factor} x max ENMAX "
                f"({max(vals):g} eV) = {floor:.0f} eV for the chosen POTCARs")
    return None


TOOL_SPECS = [
    ToolSpec(
        name="default_pseudopotentials",
        description="Recommended POTCAR variant per element (Materials Project set).",
        parameters={"elements": {"type": "list[str]", "description": "Element symbols."}},
        required=["elements"],
        agents=["simulation"],
        returns="{element: variant}",
    ),
    ToolSpec(
        name="list_pseudopotentials",
        description=("POTCAR variants available in $VASP_PP_PATH for each element, "
                     "with TITEL / ZVAL / ENMAX read from the headers only."),
        parameters={
            "elements": {"type": "list[str]", "description": "Element symbols."},
            "library": {"type": "str", "description": "Library root; defaults to $VASP_PP_PATH."},
            "gga": {"type": "str", "description": "INCAR GGA tag selecting potpaw_PBE / potpaw_GGA / potpaw."},
        },
        required=["elements"],
        agents=["simulation"],
        returns="{element: [{variant, titel, zval, enmax}]} or None when no library is readable",
    ),
    ToolSpec(
        name="render_pseudopotential_prep",
        description="POTCAR.spec + prepare_pseudopotentials.sh for a resolved choice.",
        parameters={
            "choice": {"type": "dict", "description": "{element: variant}"},
            "species": {"type": "list[str]", "description": "POSCAR species order."},
        },
        required=["choice", "species"],
        agents=["simulation"],
        returns="{filename: contents}",
    ),
    ToolSpec(
        name="pseudopotential_set",
        description="INCAR GGA tag when it selects a non-PBE POTCAR set, else None.",
        parameters={"input_files": {"type": "dict", "description": "{filename: contents}"}},
        required=["input_files"],
        agents=["simulation"],
        returns="str | None",
    ),
    ToolSpec(
        name="poscar_species",
        description="Species line (line 6) of a VASP5 POSCAR, in order.",
        parameters={"poscar_text": {"type": "str", "description": "POSCAR contents."}},
        required=["poscar_text"],
        agents=["simulation"],
        returns="list[str]",
    ),
    ToolSpec(
        name="pre_run_command",
        description="Shell step to run before the engine command (assembles POTCAR).",
        agents=["simulation"],
        returns="str",
    ),
    ToolSpec(
        name="describe_pseudopotentials",
        description="Pseudopotentials actually used by a run (from pseudopotentials.provenance).",
        parameters={"run_dir": {"type": "str", "description": "Run directory."}},
        required=["run_dir"],
        agents=["simulation"],
        returns="{element: {variant, enmax, sha256, titel}}",
    ),
    ToolSpec(
        name="check_encut_floor",
        description="Finding when ENCUT < 1.3 x max ENMAX of the chosen POTCARs.",
        parameters={
            "incar_text": {"type": "str", "description": "INCAR contents."},
            "enmax": {"type": "dict", "description": "{element: ENMAX eV}"},
        },
        required=["incar_text", "enmax"],
        agents=["simulation"],
        returns="str | None",
    ),
]
