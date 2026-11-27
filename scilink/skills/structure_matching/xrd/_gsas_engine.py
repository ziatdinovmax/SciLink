"""GSAS-II powder-XRD simulation engine (optional backend for ``simulate_xrd``).

Where the default pymatgen engine returns a *kinematic peak list* (positions +
structure-factor intensities, sufficient for peak-based phase identification),
GSAS-II computes a *full continuous profile* with realistic instrument and
sample broadening. That profile is what full-pattern matching and Rietveld-style
work need, so GSAS-II is offered as a drop-in alternative engine.

To keep the ``_ENGINES`` contract a true drop-in, this engine returns the same
core keys as the pymatgen engine (``two_theta``, ``intensities``, ``d_spacings``,
``wavelength``, ``structure_path``) — peak-picked from the profile — and *adds*
the continuous profile (``profile_two_theta`` / ``profile_intensities``) that
GSAS-II uniquely provides. Peak-based consumers are unaffected; profile-based
consumers use the extra keys.

This module is deliberately self-contained (numpy + a lazy GSAS-II import; no
scilink imports) so it can be exercised in a GSAS-II-only environment. pymatgen,
if present, is used only to canonicalize an ambiguous input CIF (see
``_canonicalize_cif``); it is optional and the engine degrades gracefully without
it.

Installation
------------
GSAS-II is an optional dependency built from source, so a Fortran compiler must
be present *at install time*. The verified recipe (conda provides the compiler
toolchain):

    conda create -n <env> python=3.12
    conda activate <env>
    conda install -c conda-forge fortran-compiler meson ninja
    pip install "GSAS-II @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git"

(GSAS-II is not on PyPI and PyPI forbids direct-URL dependencies in package
metadata, so this cannot be a ``scilink[gsas]`` extra.) The install compiles its
Fortran extensions against the environment's numpy. Install scilink/pymatgen
*before* GSAS-II so the compile links against the final numpy ABI. Note: the
compiler activation only happens inside an activated conda env — building via a
bare ``conda run`` (without activation) can fail with "gfortran cannot compile
programs" because ``CONDA_BUILD_SYSROOT`` is unset; use ``conda activate`` (or
source the env's activate scripts) for the build.

The simulation recipe is adapted, with permission, from the collaborator project
PhaseTransitionGUI (``phase_analysis.pipeline.simulation.gsas2powdersim``);
its physical correctness (peak positions, systematic absences, structure-factor
intensities, wavelength scaling) was independently verified before absorption.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Sequence

import numpy as np

# Anode line names (Å): the table shared by every XRD tool ('CuKa', 'MoKa1', ...).
from .wavelength import ANODE_ALIASES as _ANODE_WAVELENGTHS

# GSAS-II instrument-parameter template (constant-wavelength lab diffractometer).
# Only ``Lam`` varies with the requested wavelength; the U/V/W/X/Y/Z/SH-L profile
# terms are GSAS-II's standard reasonable lab defaults (same values the
# collaborator's Cu/Mo .instprm fixtures carried).
_INSTPRM_TEMPLATE = (
    "#GSAS-II instrument parameter file; do not add/delete items!\n"
    "Type:PXC\nBank:1.0\nLam:{lam:.6f}\nPolariz.:0.7\nAzimuth:0.0\nZero:0.0\n"
    "U:2.0\nV:-2.0\nW:5.0\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\n"
)


# Characteristic K-alpha1 / K-alpha2 lines (Å) of the lab anodes. A sealed tube
# or rotating anode without a K-alpha1 monochromator delivers BOTH lines, at the
# fixed intensity ratio I(Ka2)/I(Ka1) = 0.5; refining such a pattern with one
# wavelength leaves every reflection's Ka2 component unmodelled and biases the
# phase scales (IUCr CPD QPA round robin, samples 1a-1h: MAE 1.87 -> 1.53 wt%).
_KALPHA_DOUBLETS = {
    "cu": (1.540593, 1.544414), "mo": (0.709300, 0.713590), "co": (1.788965, 1.792850),
    "fe": (1.936042, 1.939980), "cr": (2.289700, 2.293606), "ag": (0.559421, 0.563813),
}
_KA2_KA1_RATIO = 0.5


def _radiation(wavelength: Any, kalpha2_ratio: Any = None) -> tuple:
    """(lambda1, lambda2 or None, I(L2)/I(L1)) for a refinement's instrument file.

    An anode K-alpha alias ('CuKa', 'Mo Ka', ...) means the tube's K-alpha1 +
    K-alpha2 doublet; a K-alpha1 alias ('CuKa1') or a number (a monochromated
    lab line, a synchrotron wavelength) is a single line. ``resolve_wavelength``
    returns 'CuKa1' when the metadata says K-alpha2 is absent (K-alpha1 /
    primary monochromator).

    ``kalpha2_ratio`` overrides that: 0 forces a single line, a positive value
    the doublet at that I(Ka2)/I(Ka1) (needs a lab-anode wavelength)."""
    l1, l2, ratio = _radiation_default(wavelength)
    if kalpha2_ratio is None:
        return l1, l2, ratio
    r = float(kalpha2_ratio)
    if r < 0:
        raise ValueError(f"kalpha2_ratio must be >= 0, got {kalpha2_ratio!r}")
    if r == 0:
        return l1, None, 0.0
    if l2 is None:
        sym = next((k for k, (a, _) in _KALPHA_DOUBLETS.items() if abs(a - l1) < 0.002), None)
        if sym is None:
            raise ValueError(f"kalpha2_ratio={r} needs a lab-anode wavelength (CuKa, MoKa, ...); "
                             f"{wavelength!r} is not one")
        l1, l2 = _KALPHA_DOUBLETS[sym]
    return l1, l2, r


def _radiation_default(wavelength: Any) -> tuple:
    if isinstance(wavelength, str):
        key = wavelength.strip().lower().replace(" ", "").replace("-", "").replace("α", "a")
        if key.endswith("ka") and key[:-2] in _KALPHA_DOUBLETS:
            l1, l2 = _KALPHA_DOUBLETS[key[:-2]]
            return l1, l2, _KA2_KA1_RATIO
        if key.endswith("ka1") and key[:-3] in _KALPHA_DOUBLETS:
            return _KALPHA_DOUBLETS[key[:-3]][0], None, 0.0
    return _resolve_wavelength(wavelength), None, 0.0


# Starting instrument profiles. One lab template for every instrument started
# synchrotron data with lab peak widths (about twice the observed width at
# lambda = 0.5 A); microstrain then refines NEGATIVE to narrow the peaks, and the
# fractions of overlapping phases are biased. 'synchrotron' is
# GSAS-II's own default for a high-resolution beamline (APS 11-BM; its
# defaultIparms), with the near-complete linear polarization of a synchrotron
# beam. The refinement adapts either start (instrument_profile stage).
INSTRUMENT_PRESETS = {
    "lab": {"U": 2.0, "V": -2.0, "W": 5.0, "X": 0.0, "Y": 0.0, "SH/L": 0.002, "Polariz.": 0.7},
    "synchrotron": {"U": 1.163, "V": -0.126, "W": 0.063, "X": 0.0, "Y": 0.0, "SH/L": 0.002,
                    "Polariz.": 0.99},
}
GEOMETRIES = {"bragg-brentano": "Bragg-Brentano", "debye-scherrer": "Debye-Scherrer"}
_GEOMETRY_ALIASES = {
    "bragg-brentano": "bragg-brentano", "bb": "bragg-brentano", "reflection": "bragg-brentano",
    "flat-plate": "bragg-brentano", "flat plate": "bragg-brentano", "theta-2theta": "bragg-brentano",
    "debye-scherrer": "debye-scherrer", "ds": "debye-scherrer", "transmission": "debye-scherrer",
    "capillary": "debye-scherrer", "cylinder": "debye-scherrer",
}


def _is_lab_line(wavelength: Any) -> bool:
    """True for a lab-anode K-alpha wavelength (alias, or a number within
    0.003 A of an anode's K-alpha1 / K-alpha2 / weighted mean)."""
    if isinstance(wavelength, str):
        key = wavelength.strip().lower().replace(" ", "").replace("-", "").replace("α", "a")
        if re.match(r"^(cu|mo|co|fe|cr|ag)ka[12]?$", key):
            return True
    try:
        lam = float(_resolve_wavelength(wavelength))
    except Exception:  # noqa: BLE001
        return False
    lines = [v for pair in _KALPHA_DOUBLETS.values() for v in pair] + list(_ANODE_WAVELENGTHS.values())
    return any(abs(lam - v) <= 0.003 for v in lines)


def _resolve_instrument_preset(wavelength: Any, preset: Any = None) -> str:
    """'lab' or 'synchrotron'. None/'auto': 'lab' for a lab-anode line,
    'synchrotron' for any other wavelength."""
    if preset is None or str(preset).lower() == "auto":
        return "lab" if _is_lab_line(wavelength) else "synchrotron"
    key = str(preset).strip().lower()
    if key not in INSTRUMENT_PRESETS:
        raise ValueError(f"instrument_preset must be 'auto' or one of {sorted(INSTRUMENT_PRESETS)}, "
                         f"got {preset!r}")
    return key


def _resolve_geometry(wavelength: Any, geometry: Any = None) -> str:
    """'bragg-brentano' or 'debye-scherrer'. None: Bragg-Brentano (flat-plate
    reflection, the lab standard) for a lab-anode line, Debye-Scherrer
    (capillary transmission) otherwise. Pass the geometry from the metadata
    (``resolve_geometry``) when it is known."""
    if geometry is None or str(geometry).lower() == "auto":
        return "bragg-brentano" if _is_lab_line(wavelength) else "debye-scherrer"
    key = _GEOMETRY_ALIASES.get(str(geometry).strip().lower())
    if key is None:
        raise ValueError(f"geometry must be 'bragg-brentano' or 'debye-scherrer' (or 'auto'), "
                         f"got {geometry!r}")
    return key


def _instprm(wavelength: Any, kalpha2_ratio: Any = None, geometry: Any = None,
             instrument_preset: Any = None) -> str:
    """GSAS-II instrument-parameter text for a refinement: the doublet keys
    (Lam1/Lam2/I(L2)/I(L1)) for tube K-alpha radiation, Lam otherwise; the
    starting profile and polarization of ``instrument_preset`` and, when
    ``geometry`` is given, the diffractometer geometry (which decides the
    sample-displacement parameter GSAS-II offers)."""
    l1, l2, ratio = _radiation(wavelength, kalpha2_ratio)
    pr = INSTRUMENT_PRESETS[_resolve_instrument_preset(wavelength, instrument_preset)]
    lam = (f"Lam:{l1:.6f}\n" if l2 is None else
           f"Lam1:{l1:.6f}\nLam2:{l2:.6f}\nI(L2)/I(L1):{ratio}\n")
    text = ("#GSAS-II instrument parameter file; do not add/delete items!\n"
            f"Type:PXC\nBank:1.0\n{lam}Polariz.:{pr['Polariz.']}\nAzimuth:0.0\nZero:0.0\n"
            f"U:{pr['U']}\nV:{pr['V']}\nW:{pr['W']}\nX:{pr['X']}\nY:{pr['Y']}\nZ:0.0\n"
            f"SH/L:{pr['SH/L']}\n")
    if geometry is not None:
        text += f"Diff-type:{GEOMETRIES[_resolve_geometry(wavelength, geometry)]}\n"
    return text


# ------------------------------------------------- refinement-stage helpers
def _negative_mustrain(phases, hist) -> list:
    """Phases whose refined isotropic microstrain is negative. Microstrain is a
    physical broadening and cannot be negative: a negative value means the
    starting instrument profile is broader than the data and the strain term
    narrows the peaks to compensate. It is reported, not undone: holding the
    term at a positive value made the fit worse on lab data whose peaks are
    only slightly narrower than the lab start (the instrument stage could then
    not absorb the difference). It is the signal to start from a narrower
    profile (instrument_preset='synchrotron', or instrument_terms with 'U')."""
    out = []
    for ph in phases:
        try:
            if float(ph.getHAPvalues(hist)["Mustrain"][1][0]) < 0:
                out.append(ph.name)
        except Exception:  # noqa: BLE001 - a phase without the term is skipped
            continue
    return out


def _sample_displacement_stage(gpx, hist, geometry, rwp_fn, corr_fn, trace) -> dict:
    """Refine the flat-plate sample displacement (Bragg-Brentano 'Shift', um):
    a sample surface off the goniometer axis shifts peaks by ~cos(theta), which
    a constant zero offset cannot follow. Kept only when the fit improves."""
    if geometry != "bragg-brentano":
        return {}
    sp = hist.data["Sample Parameters"]
    if "Shift" not in sp:
        return {}
    before = list(sp["Shift"])
    r0 = rwp_fn()
    try:
        hist.set_refinements({"Sample Parameters": ["Shift"]})
        gpx.do_refinements([{}])
        r1 = rwp_fn()
    except Exception as exc:  # noqa: BLE001
        trace.append({"stage": "sample_displacement", "error": str(exc)[:200]})
        r1 = float("nan")
    if np.isfinite(r1) and (not np.isfinite(r0) or r1 <= r0):
        trace.append({"stage": "sample_displacement", "Rwp": r1, "profile_corr": corr_fn(),
                      "shift_um": float(sp["Shift"][0])})
        return {"shift_um": round(float(sp["Shift"][0]), 2), "refined": True}
    try:
        hist.clear_refinements({"Sample Parameters": ["Shift"]})
        sp["Shift"] = before
        gpx.do_refinements([{}])
    except Exception:  # noqa: BLE001
        pass
    trace.append({"stage": "sample_displacement", "rejected": True, "Rwp": rwp_fn()})
    return {"refined": False}


# Preferred orientation (March-Dollase). Platy or needle crystallites packed in
# a flat-plate holder lie preferentially; the intensities of reflections along
# the preferred axis are scaled up (MD < 1, plates) or down (MD > 1). A platy
# phase such as gypsum can show its (0k0) reflections several times their
# calculated height; uncorrected, that one family dominates the misfit and
# biases the fractions. 'auto' tries, per
# phase above PO_MIN_FRACTION, the three axes and the normals of the phase's
# two strongest reflections, and keeps the best only when it lowers Rwp by at
# least PO_MIN_GAIN (relative) with MD inside PO_MD_RANGE — a correction that
# does not earn its parameter is removed.
PO_MIN_GAIN = 0.03
PO_MD_RANGE = (0.4, 2.5)
PO_MIN_FRACTION = 0.05


def _po_axes_for(ph, hist, n_strong=2) -> list:
    axes = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    try:
        refl = np.asarray(hist.data["Reflection Lists"][ph.name]["RefList"], dtype=float)
        inten = refl[:, 3] * refl[:, 9]           # multiplicity x Fc^2
        for i in np.argsort(inten)[::-1]:
            h = tuple(int(round(v)) for v in refl[i, :3])
            g = int(np.gcd.reduce([abs(v) for v in h if v])) if any(h) else 1
            h = tuple(v // g for v in h)
            if h[0] < 0 or (h[0] == 0 and (h[1] < 0 or (h[1] == 0 and h[2] < 0))):
                h = tuple(-v for v in h)
            if any(h) and h not in axes:
                axes.append(h)
                n_strong -= 1
            if n_strong <= 0:
                break
    except Exception:  # noqa: BLE001 - fall back to the three axes
        pass
    return axes


def _snapshot(phases, hist) -> dict:
    """Deep copy of everything a refinement cycle can move (per-phase HAP
    values and cells; histogram sample, background and instrument
    parameters) so a trial that diverges can be undone exactly."""
    import copy
    return {"hap": {ph.name: copy.deepcopy(ph.data["Histograms"][hist.name]) for ph in phases},
            "cell": {ph.name: copy.deepcopy(ph.data["General"]["Cell"]) for ph in phases},
            "hist": {k: copy.deepcopy(hist.data[k])
                     for k in ("Sample Parameters", "Background", "Instrument Parameters")}}


def _restore(snap, phases, hist):
    import copy
    for ph in phases:
        ph.data["Histograms"][hist.name] = copy.deepcopy(snap["hap"][ph.name])
        ph.data["General"]["Cell"] = copy.deepcopy(snap["cell"][ph.name])
    for k, v in snap["hist"].items():
        hist.data[k] = copy.deepcopy(v)


def _set_po(ph, hist, axis, refine, md=1.0):
    ph.getHAPvalues(hist)["Pref.Ori."] = ["MD", float(md), bool(refine),
                                          [int(a) for a in axis], 0, {}, [""], 0.1]


def _preferred_orientation_stage(gpx, phases, hist, spec, rwp_fn, trace, fractions=None) -> dict:
    """``spec``: 'auto', or {phase index | phase name: [h, k, l]} for a fixed
    axis (refined MD, kept unconditionally). Returns {phase_name: record}."""
    out = {}
    if not spec:
        return out
    if isinstance(spec, dict):
        for k, axis in spec.items():
            ph = (phases[int(k)] if str(k).lstrip("-").isdigit() and int(k) < len(phases)
                  else next((p for p in phases if p.name == k or p.name.endswith(str(k))), None))
            if ph is None:
                trace.append({"stage": "preferred_orientation", "error": f"no phase {k!r}"})
                continue
            _set_po(ph, hist, axis, True)
        try:
            gpx.do_refinements([{}])
        except Exception as exc:  # noqa: BLE001
            trace.append({"stage": "preferred_orientation", "error": str(exc)[:200]})
            return out
        for ph in phases:
            po = ph.getHAPvalues(hist)["Pref.Ori."]
            if po[2]:
                out[ph.name] = {"axis": list(po[3]), "MD": round(float(po[1]), 4), "mode": "given"}
        trace.append({"stage": "preferred_orientation", "Rwp": rwp_fn(), "phases": out})
        return out
    if str(spec).lower() != "auto":
        raise ValueError(f"preferred_orientation must be None, 'auto' or {{phase: [h, k, l]}}, got {spec!r}")
    order = sorted(range(len(phases)), key=lambda i: -(fractions or {}).get(phases[i].name, 1.0))
    for i in order:
        ph = phases[i]
        if fractions and fractions.get(ph.name, 1.0) < PO_MIN_FRACTION:
            continue
        base = rwp_fn()
        snap = _snapshot(phases, hist)
        tried, best = [], None
        for axis in _po_axes_for(ph, hist):
            try:
                _set_po(ph, hist, axis, True)
                gpx.do_refinements([{}])
                r, md = rwp_fn(), float(ph.getHAPvalues(hist)["Pref.Ori."][1])
            except Exception as exc:  # noqa: BLE001
                r, md = float("nan"), float("nan")
                tried.append({"axis": list(axis), "error": str(exc)[:120]})
            else:
                tried.append({"axis": list(axis), "MD": round(md, 4), "Rwp": round(r, 3)})
            # every trial starts from the same state: a diverged one (Rwp 100,
            # intensities ~1e22 observed) must not seed the next
            _restore(snap, phases, hist)
            ok = (np.isfinite(r) and r <= base * (1.0 - PO_MIN_GAIN)
                  and PO_MD_RANGE[0] <= md <= PO_MD_RANGE[1])
            if ok and (best is None or r < best[0]):
                best = (r, axis, md)
        if best is not None:
            _set_po(ph, hist, best[1], True, best[2])
            try:
                gpx.do_refinements([{}])
                if not (np.isfinite(rwp_fn()) and rwp_fn() <= base):
                    best = None
            except Exception as exc:  # noqa: BLE001
                trace.append({"stage": f"preferred_orientation:{ph.name}", "error": str(exc)[:200]})
                best = None
        if best is None:
            _restore(snap, phases, hist)
            try:
                gpx.do_refinements([{}])
            except Exception as exc:  # noqa: BLE001
                trace.append({"stage": f"preferred_orientation:{ph.name}", "error": str(exc)[:200]})
                continue
        rec = {"tried": tried, "Rwp_before": round(base, 3), "Rwp_after": round(rwp_fn(), 3)}
        if best is not None:
            po = ph.getHAPvalues(hist)["Pref.Ori."]
            rec.update(axis=list(po[3]), MD=round(float(po[1]), 4), mode="auto")
        else:
            rec.update(axis=None, MD=None, mode="auto: none kept")
        out[ph.name] = rec
        trace.append({"stage": f"preferred_orientation:{ph.name}", "Rwp": rwp_fn(),
                      "kept": best is not None})
    return out


def _radiation_record(wavelength: Any, kalpha2_ratio: Any = None):
    """The K-alpha doublet a refinement modelled, or None for a single line."""
    l1, l2, ratio = _radiation(wavelength, kalpha2_ratio)
    return None if l2 is None else {"kalpha1": l1, "kalpha2": l2, "I(Ka2)/I(Ka1)": ratio}


def _resolve_wavelength(wavelength: Any) -> float:
    """Resolve a wavelength spec to Å. Accepts a float/int or an anode alias
    ('CuKa', 'MoKa', …) matching pymatgen's XRDCalculator vocabulary."""
    from .wavelength import wavelength_angstrom
    return wavelength_angstrom(wavelength)


def gsas_available() -> bool:
    """True if GSAS-II (GSASIIscriptable) can be imported in this environment."""
    try:
        import GSASII.GSASIIscriptable  # noqa: F401
        return True
    except Exception:
        return False


def _canonicalize_cif(structure_path: str, workdir: str, symprec: float = None) -> str:
    """Re-express a CIF through pymatgen so GSAS-II's structure-factor calculation
    is unambiguous and consistent with the pymatgen engine.

    An origin-choice-ambiguous CIF — only an H-M space-group symbol, no explicit
    ``_symmetry_equiv_pos_as_xyz`` operators (common for terse minimal CIFs) —
    can be resolved to a different origin by GSAS-II than by pymatgen, misplacing
    atoms and producing physically wrong intensities (verified: anatase I4_1/amd
    gave Fcalc(200)=0 from a symbol-only CIF, vs the correct strong (200) once
    re-expressed via pymatgen). Re-writing via pymatgen removes the ambiguity and
    makes the two engines agree on the same input — the drop-in contract.

    ``symprec`` controls how the re-written CIF carries symmetry:

    * ``None`` (default) → expand to **P1** (explicit sites, no space group).
      Correct for *simulation*: symmetry does not change the computed pattern.
    * a float (e.g. 0.1) → re-detect and write **with the space group** preserved.
      Required for *Rietveld refinement*: GSAS-II only adds unit-cell parameters
      to the refinement when the phase carries a space group — a P1 phase silently
      refuses to refine the cell. Both settings equally fix the origin ambiguity
      (the atoms are pymatgen's correct interpretation either way).

    Real database CIFs (COD, Materials Project) carry full symmetry operators and
    are already unambiguous; this only guards the terse-CIF tail. Falls back to
    the original path if pymatgen is unavailable or the CIF cannot be parsed, in
    which case correctness relies on the CIF carrying explicit symmetry."""
    try:
        from pymatgen.core import Structure
        from pymatgen.io.cif import CifWriter
    except Exception:
        return structure_path
    try:
        structure = Structure.from_file(structure_path)
        # Strip oxidation states: a CIF declaring ionic species (Ca2+/F-,
        # ubiquitous in COD entries) sends GSAS-II down a charged-scattering-
        # factor path that produces a catastrophically wrong intensity model
        # (verified: pure-fluorite Rietveld corr 0.73/Rwp 81 with the
        # oxidation loop present vs 0.97/Rwp 34 with it removed — same
        # structure otherwise; in a certified three-phase mixture the charged
        # entries collapsed their phases' weight fractions to ~0). Neutral
        # scattering factors are the standard for routine Rietveld.
        structure.remove_oxidation_states()
        out = os.path.join(workdir, "canonical.cif")
        if symprec is not None:
            symprec = _declared_symprec(structure_path, structure, symprec)
        CifWriter(structure, symprec=symprec).write_file(out)
        return out
    except Exception:
        return structure_path


_SG_NUMBER_RE = re.compile(
    r"^\s*_(?:space_group_IT_number|space_group\.IT_number|symmetry_Int_Tables_number)\s+(\d+)",
    re.MULTILINE | re.IGNORECASE)


def _declared_space_group(structure_path: str):
    """The space-group number a CIF declares, or None."""
    try:
        with open(structure_path, encoding="utf-8", errors="ignore") as fh:
            m = _SG_NUMBER_RE.search(fh.read())
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _declared_symprec(structure_path: str, structure, fallback: float) -> float:
    """The tightest symmetry tolerance at which the parsed structure has the
    space group its CIF declares; ``fallback`` when the CIF declares none or no
    tolerance reproduces it. Re-detecting at a loose tolerance alone can
    promote a slightly distorted structure to a higher symmetry (observed on
    COD entries: a monoclinic C2/m superstructure became cubic Fm-3m, a Cmm2
    phase Cmmm), and the refinement would then hold the wrong symmetry."""
    declared = _declared_space_group(structure_path)
    if not declared or declared == 1:
        return fallback
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception:
        return fallback
    for tol in (1e-4, 1e-3, 1e-2, 3e-2, 0.1):
        try:
            if SpacegroupAnalyzer(structure, symprec=tol).get_space_group_number() == declared:
                return tol
        except Exception:
            continue
    return fallback


_B_TO_U = 1.0 / (8.0 * np.pi ** 2)


def _cif_displacements(structure_path: str) -> dict:
    """Per-element isotropic displacement parameter U (Å^2) from a CIF's atom
    sites: U_iso_or_equiv, else B_iso_or_equiv / 8 pi^2, else U_eq from the
    anisotropic U11/U22/U33 (or B11..). Each element gets the mean over its
    sites weighted by site multiplicity. Empty when the CIF carries none."""
    try:
        from pymatgen.io.cif import CifParser
        blocks = CifParser(structure_path).as_dict()
        blk = next(iter(blocks.values()))
    except Exception:
        return {}

    def num(v):
        try:
            return float(re.sub(r"\(.*\)", "", str(v)))
        except (TypeError, ValueError):
            return None

    def col(*keys):
        for k in keys:
            if k in blk:
                v = blk[k]
                return v if isinstance(v, list) else [v]
        return None

    labels = col("_atom_site_label") or []
    types = col("_atom_site_type_symbol") or labels
    u = col("_atom_site_U_iso_or_equiv")
    b = col("_atom_site_B_iso_or_equiv")
    per_label = {}
    for i, lab in enumerate(labels):
        val = num(u[i]) if u and i < len(u) else None
        if val is None and b and i < len(b) and num(b[i]) is not None:
            val = num(b[i]) * _B_TO_U
        per_label[lab] = val
    alab = col("_atom_site_aniso_label") or []
    for key, scale in (("_atom_site_aniso_U_", 1.0), ("_atom_site_aniso_B_", _B_TO_U)):
        diag = [col(key + d) for d in ("11", "22", "33")]
        if alab and all(diag):
            for i, lab in enumerate(alab):
                vals = [num(c[i]) for c in diag if i < len(c)]
                if per_label.get(lab) is None and len(vals) == 3 and None not in vals:
                    per_label[lab] = scale * sum(vals) / 3.0

    try:  # multiplicity weights: how many sites of the full cell carry each label
        from pymatgen.core import Structure
        counts = {}
        for site in Structure.from_file(structure_path):
            lab = getattr(site, "label", None)
            if lab is not None:
                counts[lab] = counts.get(lab, 0) + 1
    except Exception:
        counts = {}
    acc = {}
    for i, lab in enumerate(labels):
        val = per_label.get(lab)
        if val is None or not 0.0 < val < 0.25:
            continue
        el_m = re.match(r"[A-Z][a-z]?", str(types[i] if i < len(types) else lab))
        if not el_m:
            continue
        w = counts.get(lab, 1)
        s_, n_ = acc.get(el_m.group(0), (0.0, 0))
        acc[el_m.group(0)] = (s_ + w * val, n_ + w)
    return {el: s_ / n_ for el, (s_, n_) in acc.items() if n_}


# Plausible isotropic U (Å^2) for a non-hydrogen atom of an ordered inorganic
# solid at room temperature. A CIF value outside it is far more often a
# transcription or refinement artefact (B entered as U, an unconstrained powder
# refinement) than physics, and it biases quantification: a calcite entry with
# U = 0.034-0.045 overstated calcite by ~6 wt% at short wavelength. Such values
# fall back to the default; 'cif_raw' keeps them.
CIF_U_RANGE = (0.0005, 0.025)


def _apply_displacements(phase, source_path: str, displacement, replaced: dict = None) -> dict:
    """Set the GSAS-II phase's atomic displacement parameters per
    ``displacement``: 'cif' — each element's U from the source CIF where it
    gives one inside ``CIF_U_RANGE`` (hydrogen: any), GSAS-II's default
    0.01 Å^2 otherwise; 'cif_raw' — the CIF values unbounded; 'default' —
    GSAS-II's default for every atom; a number — that U for every atom.
    Returns {element: U} actually set (empty: nothing changed); CIF values
    rejected by the range are added to ``replaced`` when given."""
    if displacement is None or (isinstance(displacement, str) and displacement.lower() == "default"):
        return {}
    if isinstance(displacement, str):
        mode = displacement.lower()
        if mode not in ("cif", "cif_raw"):
            raise ValueError(f"displacement must be 'cif', 'cif_raw', 'default' or a number (Å^2), "
                             f"got {displacement!r}")
        values = _cif_displacements(source_path)
        if mode == "cif":
            lo, hi = CIF_U_RANGE
            bad = {el: u for el, u in values.items() if el not in ("H", "D") and not lo <= u <= hi}
            values = {el: u for el, u in values.items() if el not in bad}
            if replaced is not None:
                replaced.update({el: round(float(u), 5) for el, u in bad.items()})
    else:
        u = float(displacement)
        if not 0.0 < u < 0.25:
            raise ValueError(f"displacement U must be in (0, 0.25) Å^2, got {u}")
        values = None
    cx, ct, cs, cia = phase.data["General"]["AtomPtrs"]
    used = {}
    for atom in phase.data["Atoms"]:
        el = re.sub(r"[^A-Za-z]", "", str(atom[ct]))[:2]
        el = el[0].upper() + el[1:].lower() if el else el
        val = float(displacement) if values is None else values.get(el)
        if val is None:
            continue
        atom[cia] = "I"
        atom[cia + 1] = float(val)
        used[el] = round(float(val), 5)
    return used


def _load_g2sc():
    """Lazily import GSASIIscriptable, raising an actionable error if absent."""
    try:
        from GSASII import GSASIIscriptable as G2sc
        return G2sc
    except Exception as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "The 'gsas' XRD engine requires GSAS-II (GSASIIscriptable), which is "
            "not importable in this environment. Install it with: pip install "
            "\"GSAS-II @ git+https://github.com/AdvancedPhotonSource/GSAS-II.git\" "
            "(builds from source; needs a Fortran compiler, e.g. `conda install "
            "-c conda-forge fortran-compiler meson ninja`)."
        ) from exc


def simulate_gsas(
    structure_path: str,
    wavelength: Any = "CuKa",
    two_theta_range: tuple = (10.0, 90.0),
    two_theta_step: float = 0.02,
    crystallite_um: float = 10.0,
    peak_rel_height: float = 0.01,
) -> dict[str, Any]:
    """Simulate a powder-XRD profile from a CIF with GSAS-II.

    Returns the pymatgen-engine contract keys (peak-picked from the profile) plus
    the continuous profile. See module docstring for the design rationale.

    Parameters
    ----------
    structure_path : str
        Path to a CIF file.
    wavelength : float | str
        Wavelength in Å, or an anode alias ('CuKa', 'MoKa', …).
    two_theta_range : (float, float)
        2θ window (degrees).
    two_theta_step : float
        Profile step (degrees). Sets the number of simulated profile points.
    crystallite_um : float
        Isotropic crystallite size (μm) for the sample size-broadening profile.
        LOWER it (e.g. 0.05) to broaden peaks toward a nanocrystalline sample;
        RAISE it for sharper lines. GSAS-II's collaborator default was 10 μm.
    peak_rel_height : float
        Peak-pick threshold as a fraction of the max profile intensity (default
        0.01). LOWER to report weaker reflections in the peak list; RAISE to keep
        only strong peaks. Only affects the discrete peak list, not the profile.
    """
    G2sc = _load_g2sc()
    lam = _resolve_wavelength(wavelength)
    tmin, tmax = float(min(two_theta_range)), float(max(two_theta_range))
    if tmax <= tmin:
        raise ValueError(f"two_theta_range must be increasing, got {two_theta_range}.")
    npts = max(int(round((tmax - tmin) / float(two_theta_step))) + 1, 2)

    # Work inside a scratch dir: GSAS-II writes a .gpx project + temp files.
    with tempfile.TemporaryDirectory() as tmpd:
        instprm = os.path.join(tmpd, "auto.instprm")
        with open(instprm, "w", encoding="utf-8") as fh:
            fh.write(_INSTPRM_TEMPLATE.format(lam=lam))

        # Normalize the CIF to explicit P1 (if pymatgen is available) so an
        # origin-choice-ambiguous symbol-only CIF cannot be mis-resolved by
        # GSAS-II relative to the pymatgen engine — see _canonicalize_cif.
        cif_for_gsas = _canonicalize_cif(structure_path, tmpd)

        gpx = G2sc.G2Project(newgpx=os.path.join(tmpd, "sim.gpx"))
        phase = gpx.add_phase(phasefile=cif_for_gsas, phasename="phase")
        hist = gpx.add_simulated_powder_histogram(
            histname="sim",
            iparams=instprm,
            Tmin=tmin,
            Tmax=tmax,
            Tstep=(tmax - tmin) / (npts - 1),
            phases=[phase],
        )
        gpx.link_histogram_phase(hist, phase)
        phase.setSampleProfile(hist, "size", "isotropic", crystallite_um)
        gpx.do_refinements([{}])  # evaluate Ycalc (no parameters varied)

        x = np.asarray(hist.getdata("X"), dtype=float)
        ycalc = np.asarray(hist.getdata("Ycalc"), dtype=float)

    two_theta, intensities, d_spacings = _peak_pick(x, ycalc, lam, peak_rel_height)

    return {
        "two_theta": [float(v) for v in two_theta],
        "intensities": [float(v) for v in intensities],
        "hkls": [],  # GSAS profile engine does not emit per-peak hkl; peaks are picked
        "d_spacings": [float(v) for v in d_spacings],
        "wavelength": float(lam),
        "structure_path": structure_path,
        # GSAS-II's unique contribution: the full instrument/sample-broadened profile.
        "profile_two_theta": [float(v) for v in x],
        "profile_intensities": [float(v) for v in ycalc],
        "engine_note": (
            "GSAS-II full-profile simulation; 'two_theta'/'intensities' are peak-"
            "picked from the profile, 'profile_*' hold the continuous pattern."
        ),
    }


def _peak_pick(x: np.ndarray, y: np.ndarray, lam: float, rel_height: float = 0.01):
    """Pick peaks from the continuous profile and convert to d-spacings (Bragg).

    Returns (two_theta_peaks, intensities_0_100, d_spacings). Falls back to a
    plain local-maximum scan if scipy is unavailable.
    """
    y = np.asarray(y, dtype=float)
    ymax = float(np.max(y)) if y.size else 0.0
    if ymax <= 0.0:
        return np.array([]), np.array([]), np.array([])
    thresh = rel_height * ymax
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(y, height=thresh)
    except Exception:  # pragma: no cover - scipy is a GSAS-II dep in practice
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thresh))[0] + 1

    tt = x[idx]
    inten = 100.0 * y[idx] / ymax
    theta = np.radians(tt / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = lam / (2.0 * np.sin(theta))
    return tt, inten, d


# --- Rietveld refinement (tier-3) --------------------------------------------
#
# Full-pattern least-squares refinement of a candidate structure against a
# measured pattern, on the same GSAS-II wrapper as the simulation engine. The
# staged protocol below is the load-bearing part: refine the cheap, well-
# determined parameters first (background, scale) and the geometry (2theta zero)
# BEFORE the unit cell — refining the cell against a pattern with an uncorrected
# 2theta offset diverges (the cell chases the offset into invalid geometry). Peak
# widths for a real specimen are dominated by sample broadening, so isotropic
# Mustrain/Size (crystallite size + microstrain) are refined, not the instrument
# Caglioti U/V/W (which are a calibration property, not a per-sample unknown).
#
# Metric note: for a pattern in ARBITRARY intensity units (no counting
# statistics — the common case for archived/RRUFF-style data), the weighted Rwp
# is inflated because the per-point weights (1/esd^2) are only nominal. The
# profile-fit correlation (Yobs vs Ycalc) is the robust fit indicator there — the
# same "score where there is signal" philosophy as peak_region_r2. Both are
# returned; read the correlation when counting statistics are absent.

def rietveld_refine(
    structure_path: str,
    two_theta,
    intensity,
    wavelength: Any = "CuKa",
    two_theta_range: tuple = None,
    refine_cell: bool = True,
    refine_profile: bool = True,
    refine_atoms: bool = False,
    n_background_terms: int = 6,
    kalpha2_ratio: float = None,
    displacement: Any = "cif",
    geometry: Any = None,
    instrument_preset: Any = None,
    sample_displacement: Any = None,
    preferred_orientation: Any = None,
) -> dict[str, Any]:
    """Rietveld-refine ``structure_path`` against a measured (two_theta, intensity)
    pattern with GSAS-II. Returns refined lattice + esd, fit metrics (Rwp and the
    robust profile-fit correlation), the refined profile (obs/calc/background/
    residual), refined crystallite size + microstrain, and the per-stage
    convergence trace. See the module comment for the protocol rationale.

    Parameters
    ----------
    structure_path : str
        CIF of the candidate phase (canonicalized to P1 like the sim engine).
    two_theta, intensity : sequences
        The measured pattern.
    wavelength : float | str
        Wavelength (Å) or anode alias ('CuKa', ...).
    two_theta_range : (float, float) | None
        Optional (min, max) crop of the measured pattern before refining.
    refine_cell : bool
        Refine the unit-cell parameters (default True). Turn OFF to hold the cell
        fixed (e.g. a trusted reference cell, or when the data is too sparse).
    refine_profile : bool
        Refine isotropic sample broadening — microstrain (default True). This is
        what matches the peak widths; turn OFF only if the instrument profile
        already matches the data. (Only microstrain is refined, not a separate
        crystallite size: the two are correlated broadening terms and refining
        both destabilizes sparse/high-symmetry patterns.)
    refine_atoms : bool
        Also refine atomic coordinates + isotropic ADPs (default False). RISKY on
        noisy/low-resolution data — it can diverge or overfit; enable only for
        high-quality patterns where the cell/profile fit is already good.
    n_background_terms : int
        Chebyshev background terms (default 6). RAISE (10-14) for a strongly
        curved/humped background; LOWER for a flat one.
    geometry, instrument_preset, sample_displacement, preferred_orientation
        As in ``rietveld_refine_multiphase``; ``preferred_orientation`` defaults
        to None here (one phase: texture and the structure model trade off).
    """
    import os
    import tempfile

    G2sc = _load_g2sc()
    lam = _resolve_wavelength(wavelength)
    geom = _resolve_geometry(wavelength, geometry)
    preset = _resolve_instrument_preset(wavelength, instrument_preset)
    # opt-in: with no internal standard the displacement correlates with the
    # cells (on a single phase it absorbed a uniform 0.5 % cell error)
    do_shift = bool(sample_displacement) and geom == "bragg-brentano"
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if two_theta_range is not None:
        lo, hi = float(min(two_theta_range)), float(max(two_theta_range))
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
    if x.size < 20:
        raise ValueError("Rietveld refinement needs a denser pattern (>=20 points in range).")

    with tempfile.TemporaryDirectory() as td:
        # symprec (not P1): Rietveld needs the space group so GSAS-II adds the
        # unit-cell parameters to the refinement — a P1 phase silently refuses.
        cif = _canonicalize_cif(structure_path, td, symprec=0.1)
        xy = os.path.join(td, "data.xy")
        esd = np.sqrt(np.maximum(y, 1.0))
        np.savetxt(xy, np.column_stack([x, y, esd]), fmt="%.6f")
        instf = os.path.join(td, "auto.instprm")
        with open(instf, "w", encoding="utf-8") as fh:
            fh.write(_instprm(wavelength, kalpha2_ratio, geom, preset))

        gpx = G2sc.G2Project(newgpx=os.path.join(td, "riet.gpx"))
        hist = gpx.add_powder_histogram(xy, iparams=instf, fmthint="xye")
        phase = gpx.add_phase(cif, phasename="phase")
        adp_replaced = {}
        adp_used = _apply_displacements(phase, structure_path, displacement, adp_replaced)
        gpx.link_histogram_phase(hist, phase)
        input_cell = dict(phase.get_cell())  # starting cell, to expose refinement drift

        def _rwp():
            r = hist.residuals
            return float(r.get("Rwp", r.get("wR", float("nan"))))

        def _corr():
            yo = np.asarray(hist.getdata("Yobs"), dtype=float)
            yc = np.asarray(hist.getdata("Ycalc"), dtype=float)
            if yo.size != yc.size or yo.size < 2:
                return float("nan")
            return float(np.corrcoef(yo, yc)[0, 1])

        stages = [("background+scale", {"set": {
            "Background": {"type": "chebyschev", "refine": True, "no. coeffs": int(n_background_terms)},
            "Sample Parameters": ["Scale"]}})]
        stages.append(("zero_offset", {"set": {"Instrument Parameters": ["Zero"]}}))
        if refine_profile:
            # Isotropic microstrain absorbs the sample peak-broadening. Refining
            # BOTH Mustrain and Size is over-parameterized (the two are correlated
            # broadening terms) and destabilizes sparse/high-symmetry patterns —
            # verified: a cubic fluorite pattern collapsed the cell with both, and
            # converged cleanly with mustrain alone. So mustrain only, by default.
            stages.append(("mustrain", {"set": {"Mustrain": {"type": "isotropic", "refine": True}}}))
        if refine_cell:
            stages.append(("cell", {"set": {"Cell": True}}))
        if refine_atoms:
            stages.append(("atoms", {"set": {"Atoms": {"all": "XU"}}}))

        trace = []
        shift_rec, po_rec = {}, {}
        # preferred orientation follows the cell stage, or the last stage before the
        # atoms when the cell is held (it must run whenever it is requested)
        po_anchor = "cell" if refine_cell else [lb for lb, _ in stages if lb != "atoms"][-1]
        for label, st in stages:
            try:
                gpx.do_refinements([st])
                trace.append({"stage": label, "Rwp": _rwp(), "profile_corr": _corr()})
            except Exception as exc:  # a stage that diverges is recorded, not fatal
                trace.append({"stage": label, "error": str(exc)[:200]})
            if label == "zero_offset" and do_shift:
                shift_rec = _sample_displacement_stage(gpx, hist, geom, _rwp, _corr, trace)
            if label == po_anchor and preferred_orientation:
                po_rec = _preferred_orientation_stage(gpx, [phase], hist, preferred_orientation,
                                                      _rwp, trace)

        cell = phase.get_cell()
        cell_esd = {}
        try:
            _, cell_esd = phase.get_cell_and_esd()
        except Exception:
            pass
        # Only microstrain is refined by default (see the mustrain stage note), so
        # crystallite size is not separately determined — report it as None rather
        # than a misleading unrefined placeholder.
        _size_um, microstrain = _extract_hap_broadening(phase, hist)
        size_um = None

        prof = {
            "two_theta": [float(v) for v in hist.getdata("X")],
            "y_obs": [float(v) for v in hist.getdata("Yobs")],
            "y_calc": [float(v) for v in hist.getdata("Ycalc")],
            "y_background": [float(v) for v in hist.getdata("Background")],
        }
        prof["residual"] = [o - c for o, c in zip(prof["y_obs"], prof["y_calc"])]
        res = dict(hist.residuals)

    _cellkeys = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma", "volume")
    lattice = {k: float(cell[k]) for k in _cellkeys if k in cell}
    input_lattice = {k: float(input_cell[k]) for k in _cellkeys if k in input_cell}

    # Convergence: Rietveld is a LOCAL optimizer. A starting cell too far from the
    # truth (or a wrong phase) diverges — the cell runs away and the profile
    # correlation crashes. Compute the final fit correlation from the ACTUAL final
    # profile (robust to a last stage that errored, where the trace lacks a corr —
    # a catastrophically diverged refinement gives a flat Ycalc -> corr ~0, not
    # None). converged iff the final correlation is healthy AND the cell stage did
    # not degrade the fit vs the best prior stage.
    _yo, _yc = np.asarray(prof["y_obs"], float), np.asarray(prof["y_calc"], float)
    if _yo.size > 1 and np.std(_yc) > 0 and np.std(_yo) > 0:
        final_corr = float(np.corrcoef(_yo, _yc)[0, 1])
    else:
        final_corr = 0.0  # degenerate (flat) calc pattern -> total divergence
    if final_corr != final_corr:  # nan guard
        final_corr = 0.0
    _corrs = [t["profile_corr"] for t in trace
              if isinstance(t.get("profile_corr"), float) and t["profile_corr"] == t["profile_corr"]]
    best_pre = max(_corrs[:-1]) if len(_corrs) > 1 else final_corr
    converged = bool(final_corr >= 0.5 and final_corr >= best_pre - 0.1)

    return {
        "lattice": lattice,
        "lattice_esd": {k: float(v) for k, v in (cell_esd or {}).items()
                        if isinstance(v, (int, float))},
        "input_lattice": input_lattice,
        "converged": converged,
        "Rwp": _final_rwp(res, trace),
        "profile_corr": final_corr,
        "gof": _safe_gof(res),
        "crystallite_size_um": size_um,
        "microstrain": microstrain,
        "convergence_trace": trace,
        "profile": prof,
        "wavelength": float(lam),
        "kalpha_doublet": _radiation_record(wavelength, kalpha2_ratio),
        "displacement_U": adp_used,
        "displacement_U_replaced": adp_replaced,
        "geometry": geom,
        "instrument_preset": preset,
        "sample_displacement": shift_rec,
        "preferred_orientation": po_rec,
        "negative_microstrain": microstrain is not None and microstrain < 0,
        "structure_path": structure_path,
        "note": ("For arbitrary-unit data (no counting statistics) read 'profile_corr' "
                 "as the fit quality; 'Rwp' is inflated by nominal weights. If "
                 "'converged' is False the starting cell was too far off (or the "
                 "phase is wrong) and the refined 'lattice' ran away — compare it "
                 "against 'input_lattice' and do not trust it. 'converged' is "
                 "necessary but NOT sufficient: a wrong-but-plausible structure "
                 "(especially low-symmetry / triclinic) can reach a moderate "
                 "profile_corr (~0.85) with a wrong cell and pass — so corroborate "
                 "a low-symmetry result with the identification score and expected "
                 "cell, and treat a modest profile_corr with caution."),
    }


def _final_rwp(res, trace):
    """Rwp of the final state (the residuals after the last refinement), not
    of whichever stage was recorded last."""
    for k in ("Rwp", "wR"):
        v = (res or {}).get(k)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return trace[-1].get("Rwp") if trace else None


def _safe_gof(res):
    """Goodness-of-fit = Rwp / Rexp = wR / wRmin (GSAS residuals dict). None if
    the expected-Rwp term is missing or zero (arbitrary-unit data)."""
    wr, wrmin = res.get("wR"), res.get("wRmin")
    try:
        if wr and wrmin and wrmin > 0:
            return float(wr) / float(wrmin)
    except Exception:
        pass
    return None


def _extract_hap_broadening(phase, hist):
    """Best-effort read of the refined isotropic crystallite size (um) and
    microstrain from the phase's histogram-and-phase (HAP) values. Returns
    (size_um, microstrain), either possibly None if not present."""
    try:
        hap = phase.getHAPvalues(hist)
    except Exception:
        return None, None
    size_um = microstrain = None
    try:
        sz = hap.get("Size")
        if sz and isinstance(sz[1], (list, tuple)):
            size_um = float(sz[1][0])
    except Exception:
        pass
    try:
        mu = hap.get("Mustrain")
        if mu and isinstance(mu[1], (list, tuple)):
            microstrain = float(mu[1][0])
    except Exception:
        pass
    return size_um, microstrain


# --- Le Bail cell validation (structure-free whole-pattern fit) ---------------
#
# The arbiter of a CANDIDATE CELL. A Rietveld/simulation score conflates the
# cell with the structure — a correct cell paired with a wrong same-cell
# structure scores badly and gets discarded. Le Bail fits the whole profile
# with the cell alone (reflection intensities are free parameters), answering
# "does this cell account for every peak?" directly. Physics of the verdicts:
# a wrong cell or SUBCELL leaves observed peaks unaccounted -> poor fit; a
# SUPERCELL generates a superset of reflections and always fits at least as
# well as the true cell -> Le Bail cannot demote it, so combine with the
# smallest-cell-that-fits preference (index_pattern's alias ranking).
# Validated on real data (cassiterite): true cell corr=0.96 (and the cell
# refines to 0.01% with no structure), wrong cell 0.47, half subcell 0.21,
# supercell 0.94.

# Generic space group per GSAS Bravais lattice — carries exactly the lattice
# centering extinctions; atoms are irrelevant in Le Bail so a dummy suffices.
_BRAVAIS_TO_SG = {
    "Cubic-F": "F m -3 m", "Cubic-I": "I m -3 m", "Cubic-P": "P m -3 m",
    "Trigonal-R": "R -3 m", "Trigonal/Hexagonal-P": "P 6/m m m",
    "Tetragonal-I": "I 4/m m m", "Tetragonal-P": "P 4/m m m",
    "Orthorhombic-F": "F m m m", "Orthorhombic-I": "I m m m",
    "Orthorhombic-A": "A m m m", "Orthorhombic-B": "B m m m",
    "Orthorhombic-C": "C m m m",
    "Monoclinic-I": "I 2/m", "Monoclinic-A": "A 2/m", "Monoclinic-C": "C 2/m",
    "Monoclinic-P": "P 2/m", "Triclinic": "P -1",
}

_LEBAIL_CIF = """data_lebail
_cell_length_a {a}
_cell_length_b {b}
_cell_length_c {c}
_cell_angle_alpha {al}
_cell_angle_beta {be}
_cell_angle_gamma {ga}
_symmetry_space_group_name_H-M '{sg}'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.0
"""


def lebail_fit(
    two_theta,
    intensity,
    cell,
    bravais: str = "Cubic-P",
    wavelength: Any = "CuKa",
    two_theta_range: tuple = None,
    refine_cell: bool = True,
    refine_zero: bool = True,
    n_background_terms: int = 6,
    extra_cycles: int = 2,
    observed_peaks: Sequence[float] = None,
    peak_tol_deg: float = 0.15,
    kalpha2_ratio: float = None,
) -> dict[str, Any]:
    """Le Bail whole-pattern fit of a CANDIDATE CELL (no structure needed).

    See the section comment for the physics of the verdicts (subcell/wrong cell
    reject; supercell fits — prefer the smallest cell that fits).

    Parameters
    ----------
    two_theta, intensity : sequences
        The measured pattern (raw, with background).
    cell : sequence | dict
        Candidate cell: (a, b, c, alpha, beta, gamma) or a dict with those keys
        (index_pattern's candidate_cells entries work directly).
    bravais : str
        A GSAS Bravais name from index_pattern ('Tetragonal-P', 'Cubic-F', ...)
        — mapped to a generic space group carrying the lattice extinctions — or
        an explicit H-M space-group symbol (e.g. 'P 42/m n m') when known.
    wavelength : float | str
        Wavelength (Å) or anode alias; must match the measurement.
    two_theta_range : (float, float) | None
        Optional crop of the pattern before fitting.
    refine_cell : bool
        Refine the cell during the fit (default True) — a correct approximate
        cell polishes to the precise one with no structure at all.
    refine_zero : bool
        Refine the 2θ zero offset (default True).
    n_background_terms : int
        Chebyshev background terms (default 6). RAISE for humped backgrounds.
    extra_cycles : int
        Additional intensity-extraction cycles after the staged refinement
        (default 2) — Le Bail intensities converge iteratively.
    observed_peaks : sequence of float | None
        Measured peak positions (2θ°, e.g. ``extract_peaks``' 'positions'). When
        given, each is checked against the reflections the fitted cell generates
        and the ORPHANS are returned as ``unaccounted_peaks`` — direct evidence
        of an impurity phase (a few orphans on a good fit) or a wrong/subcell
        (many orphans).
    peak_tol_deg : float
        Match window for the accounting (default 0.15°). RAISE for broad peaks
        or a residual zero error.
    """
    import os
    import tempfile

    G2sc = _load_g2sc()
    lam = _resolve_wavelength(wavelength)
    if isinstance(cell, dict):
        vals = [cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")]
    else:
        vals = list(cell)
    if len(vals) != 6:
        raise ValueError("cell must supply (a, b, c, alpha, beta, gamma)")
    sg = _BRAVAIS_TO_SG.get(bravais, bravais)  # Bravais name or explicit H-M symbol

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if two_theta_range is not None:
        lo, hi = float(min(two_theta_range)), float(max(two_theta_range))
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
    if x.size < 20:
        raise ValueError("Le Bail fit needs a denser pattern (>=20 points in range).")

    with tempfile.TemporaryDirectory() as td:
        cif = os.path.join(td, "lebail.cif")
        a, b, c_, al, be, ga = (float(v) for v in vals)
        with open(cif, "w", encoding="utf-8") as fh:
            fh.write(_LEBAIL_CIF.format(a=a, b=b, c=c_, al=al, be=be, ga=ga, sg=sg))
        xy = os.path.join(td, "data.xy")
        np.savetxt(xy, np.column_stack([x, y, np.sqrt(np.maximum(y, 1.0))]), fmt="%.6f")
        instf = os.path.join(td, "auto.instprm")
        with open(instf, "w", encoding="utf-8") as fh:
            fh.write(_instprm(wavelength, kalpha2_ratio))

        gpx = G2sc.G2Project(newgpx=os.path.join(td, "lb.gpx"))
        hist = gpx.add_powder_histogram(xy, iparams=instf, fmthint="xye")
        phase = gpx.add_phase(cif, phasename="lebail")
        gpx.link_histogram_phase(hist, phase)
        input_cell = dict(phase.get_cell())

        stages = [("init", {})]  # normal cycle first: builds the reflection list
        stages += [("background", {"set": {"Background": {
            "type": "chebyschev", "refine": True,
            "no. coeffs": int(n_background_terms)}}})]
        if refine_zero:
            stages.append(("zero", {"set": {"Instrument Parameters": ["Zero"]}}))
        stages.append(("mustrain", {"set": {"Mustrain": {"type": "isotropic", "refine": True}}}))
        if refine_cell:
            stages.append(("cell", {"set": {"Cell": True}}))
        stages += [(f"extract_{i}", {}) for i in range(int(extra_cycles))]

        trace = []
        for label, st in stages:
            try:
                gpx.do_refinements([st])
                if label == "init":  # flip to Le Bail once the RefList exists
                    phase.set_refinements({"LeBail": True})
                yo = np.asarray(hist.getdata("Yobs"), dtype=float)
                yc = np.asarray(hist.getdata("Ycalc"), dtype=float)
                corr = (float(np.corrcoef(yo, yc)[0, 1])
                        if yo.size > 1 and np.std(yc) > 0 else 0.0)
                trace.append({"stage": label, "profile_corr": corr})
            except Exception as exc:
                trace.append({"stage": label, "error": str(exc)[:200]})

        yo = np.asarray(hist.getdata("Yobs"), dtype=float)
        yc = np.asarray(hist.getdata("Ycalc"), dtype=float)
        final_corr = (float(np.corrcoef(yo, yc)[0, 1])
                      if yo.size > 1 and np.std(yc) > 0 else 0.0)
        if final_corr != final_corr:
            final_corr = 0.0
        cellout = phase.get_cell()
        prof = {
            "two_theta": [float(v) for v in hist.getdata("X")],
            "y_obs": [float(v) for v in yo],
            "y_calc": [float(v) for v in yc],
        }

        # Peak accounting: which observed peaks does the fitted cell generate a
        # reflection for? The histogram's post-fit reflection list holds the
        # generated positions at the refined cell (incl. the fitted zero).
        unaccounted = accounted = None
        if observed_peaks is not None:
            try:
                reflists = hist.data.get("Reflection Lists", {})
                refpos = []
                for _phname, pdata in reflists.items():
                    refl = pdata.get("RefList") if isinstance(pdata, dict) else pdata
                    for rowr in refl:
                        refpos.append(float(rowr[5]))  # col 5 = 2theta position
                refpos = np.asarray(sorted(refpos), dtype=float)
                unaccounted, accounted = [], []
                for p in observed_peaks:
                    p = float(p)
                    ok = refpos.size and float(np.min(np.abs(refpos - p))) <= float(peak_tol_deg)
                    (accounted if ok else unaccounted).append(round(p, 3))
            except Exception as exc:
                _log_note = f"peak accounting failed: {exc}"  # noqa: F841
                unaccounted = accounted = None

    _keys = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta",
             "angle_gamma", "volume")
    out = {
        "profile_corr": final_corr,
        "cell_fits": bool(final_corr >= 0.8),
        "lattice": {k: float(cellout[k]) for k in _keys if k in cellout},
        "input_lattice": {k: float(input_cell[k]) for k in _keys if k in input_cell},
        "space_group_used": sg,
        "convergence_trace": trace,
        "profile": prof,
        "wavelength": float(lam),
        "kalpha_doublet": _radiation_record(wavelength, kalpha2_ratio),
        "note": (
            "Structure-free whole-pattern fit: profile_corr >= ~0.9 means the "
            "cell accounts for the pattern (cell_fits uses 0.8); <= ~0.6 means a "
            "wrong cell or a SUBCELL (peaks unaccounted). A SUPERCELL always fits "
            "at least as well as the true cell — among cells that fit, prefer "
            "the SMALLEST. When cell_fits and refine_cell=True, 'lattice' is the "
            "precise cell, obtained with no structure."
        ),
    }
    if observed_peaks is not None and unaccounted is not None:
        out["accounted_peaks"] = accounted
        out["unaccounted_peaks"] = unaccounted
        out["note"] += (
            " Peak accounting: a FEW unaccounted_peaks on an otherwise good fit "
            "= impurity/second-phase lines (identify them separately); MANY = "
            "wrong cell or subcell."
        )
    return out
#
# Powder autoindexing recovers candidate UNIT CELLS from peak positions alone —
# the entry point to identification when the sample chemistry is unknown, since
# the structure-database search can then filter by lattice parameters instead of
# elements. Wraps GSAS-II's ITO-style ``GSASIIindex.DoIndexPeaks`` (random-cell
# volume sweep per Bravais lattice, least-squares refinement, M20/X20 de Wolff
# figures of merit).
#
# Practical behavior established empirically:
# * Peak count is load-bearing: ~6 peaks fails outright; >=10 clean peaks indexes
#   a cubic cell exactly. Feed it every confident peak, not just the strongest.
# * Equal-M20 SUPERLATTICE ALIASES are normal (e.g. Si returns a=7.68=5.43*sqrt2
#   alongside a=5.43 at the same M20). The standard resolution — prefer the
#   SMALLEST volume among (near-)equal-M20 solutions — is applied in the sort.
# * Low symmetry is intrinsically unreliable: monoclinic/triclinic searches are
#   slow and prone to false cells with imperfect peak lists. Triclinic is
#   therefore excluded from the default lattice set (opt in explicitly).

# GSAS-II Bravais-lattice order used by DoIndexPeaks' ``bravais`` flag list.
_BRAVAIS_NAMES = [
    "Cubic-F", "Cubic-I", "Cubic-P", "Trigonal-R", "Trigonal/Hexagonal-P",
    "Tetragonal-I", "Tetragonal-P", "Orthorhombic-F", "Orthorhombic-I",
    "Orthorhombic-A", "Orthorhombic-B", "Orthorhombic-C", "Orthorhombic-P",
    "Monoclinic-I", "Monoclinic-A", "Monoclinic-C", "Monoclinic-P", "Triclinic",
]
_SYSTEM_TO_IBRAV = {
    "cubic": [0, 1, 2],
    "trigonal": [3, 4],
    "hexagonal": [4],
    "tetragonal": [5, 6],
    "orthorhombic": [7, 8, 9, 10, 11, 12],
    "monoclinic": [13, 14, 15, 16],
    "triclinic": [17],
}
_IBRAV_TO_SYSTEM = {i: s for s, idxs in _SYSTEM_TO_IBRAV.items() for i in idxs
                    if not (s in ("hexagonal",) )}  # ibrav 4 reports as 'trigonal/hexagonal' below


def index_pattern(
    two_theta_peaks,
    wavelength: Any = "CuKa",
    crystal_systems: list = None,
    zero_offset: float = 0.0,
    max_nc_no: int = 4,
    start_volume: float = 25.0,
    timeout_per_lattice: float = 30.0,
    m20_min: float = 2.0,
    top_n: int = 10,
    stage_stop_m20: float = 10.0,
    seed: int = 0,
    alias_m20_factor: float = 2.0,
    min_cell_volume: float = 15.0,
) -> dict[str, Any]:
    """Autoindex a powder pattern: peak positions -> ranked candidate unit cells.

    See the section comment above for the method and its empirically established
    behavior (peak count, superlattice aliases, low-symmetry caveats).

    Parameters
    ----------
    two_theta_peaks : sequence of float
        Peak positions in 2θ degrees (e.g. ``extract_peaks``' 'positions').
        Feed ALL confident peaks — indexing needs >=10 to be reliable.
    wavelength : float | str
        Wavelength in Å or anode alias ('CuKa', ...). Must match the measurement.
    crystal_systems : list[str] | None
        Which crystal systems to search: subset of ['cubic','trigonal',
        'hexagonal','tetragonal','orthorhombic','monoclinic','triclinic'].
        Default None = STAGED SYMMETRY DESCENT: cubic -> tetragonal/hexagonal/
        trigonal -> orthorhombic -> monoclinic, stopping at the first symmetry
        level that yields a convincing solution (M20 >= stage_stop_m20 with
        X20 = 0). This is the standard defense against the false-low-symmetry
        pathology (a spurious monoclinic cell can index everything with a
        spectacular M20); M20 is only comparable WITHIN a symmetry level.
        Passing an explicit list disables staging and searches exactly those
        systems (triclinic only ever runs when explicitly requested).
    zero_offset : float
        2θ zero correction (degrees) applied by the indexer (default 0).
    max_nc_no : int
        Max ratio of calculated-to-observed reflections (de Wolff Nc/Nobs cap,
        default 4). RAISE to allow larger cells; LOWER to force parsimony.
    start_volume : float
        Starting cell volume (Å^3) for the sweep (default 25). The sweep grows
        volume automatically, so only RAISE this to skip past small cells when
        you already know the cell is large (speeds the search up).
    timeout_per_lattice : float
        Seconds allowed per Bravais lattice before moving on (default 30).
        RAISE for a thorough (esp. monoclinic) search; LOWER for a quick triage.
    m20_min : float
        Minimum de Wolff M20 figure of merit to keep a cell (default 2). RAISE
        (10-20) to keep only convincing solutions.
    top_n : int
        Max candidate cells returned (default 10).
    stage_stop_m20 : float
        Staged mode only: stop descending to lower symmetry once a stage yields
        M20 >= this with X20 = 0 (default 10). RAISE to keep descending unless
        the high-symmetry solution is overwhelming; LOWER to stop earlier.
    seed : int | None
        The cell search starts from random trial cells; the seed makes the
        result reproducible (default 0). CHANGE it to re-roll the stochastic
        search when the candidates look like aliases of each other; None uses
        fresh randomness each call.
    alias_m20_factor : float
        M20 ratio below which two solutions are treated as alias-equivalent and
        the SMALLEST cell wins (default 2, de Wolff practice — a supercell's
        extra parameters inflate its M20 without being more right). RAISE to
        prefer small cells more aggressively; set near 1 to rank strictly by
        M20.
    min_cell_volume : float
        Physical floor (Å^3) below which a solution is discarded as an
        arithmetic subcell artifact (default 15 — a cell below ~15 Å^3 cannot
        contain an atom; the smallest real cells, bcc metals, are ~20 Å^3).
    """
    import contextlib
    import io

    try:
        from GSASII import GSASIIindex as G2indx
    except Exception as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "index_pattern requires GSAS-II (GSASIIindex). Install it with: "
            "pip install \"GSAS-II @ git+https://github.com/AdvancedPhotonSource/"
            "GSAS-II.git\" (see module docstring for the Fortran-compiler recipe)."
        ) from exc

    lam = _resolve_wavelength(wavelength)
    tts = sorted(float(t) for t in two_theta_peaks)
    if len(tts) < 5:
        raise ValueError(
            f"index_pattern needs at least 5 peak positions (got {len(tts)}); "
            "autoindexing is only reliable with >=10."
        )
    warnings = []
    if len(tts) < 10:
        warnings.append(
            f"only {len(tts)} peaks supplied — autoindexing is unreliable below "
            "~10 peaks; treat any solution with suspicion."
        )

    # 9-column Index-Peak-List rows: [2theta, I, use, indexed, h, k, l, d-obs, d-calc],
    # sorted by DESCENDING d (the indexer reads dmax/dmin from the ends).
    peaks = []
    for tt in tts:
        d = lam / (2.0 * np.sin(np.radians(tt / 2.0)))
        peaks.append([tt, 100.0, True, False, 0, 0, 0, float(d), 0.0])
    peaks.sort(key=lambda p: -p[7])

    bad = [s for s in (crystal_systems or []) if s not in _SYSTEM_TO_IBRAV]
    if bad:
        raise ValueError(f"unknown crystal system(s) {bad}; "
                         f"choose from {sorted(_SYSTEM_TO_IBRAV)}")

    # controls layout (GSAS-II 'Unit Cells List'): [zero-flag, zero, Nc/No max, start V]
    controls = [0, float(zero_offset), int(max_nc_no), float(start_volume)]

    import random as _random

    def _run_index(systems):
        bravais = [False] * len(_BRAVAIS_NAMES)
        for s in systems:
            for i in _SYSTEM_TO_IBRAV[s]:
                bravais[i] = True
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # DoIndexPeaks prints its search log
            _, _, found = G2indx.DoIndexPeaks(
                peaks, controls, bravais, None,
                timeout=float(timeout_per_lattice), M20_min=float(m20_min))
        return found

    # GSASIIindex draws its trial cells from the stdlib `random` module; seed it
    # (saving/restoring the global state) so results are reproducible.
    _rng_state = _random.getstate()
    if seed is not None:
        _random.seed(int(seed))
    try:

        if crystal_systems is not None:
            # Explicit systems: single search exactly as requested (no staging).
            cells = _run_index(crystal_systems)
            stages_run = [list(crystal_systems)]
            stopped_at = None
        else:
            # STAGED SYMMETRY DESCENT (default). A low-symmetry search has
            # enough free parameters that a large spurious cell can index every
            # peak with a spectacular M20 — a false monoclinic cell beating the
            # true tetragonal one is a classic autoindexing pathology (observed
            # live: a monoclinic V~560 A^3 cell at M20~97 vs the true V~71.5
            # tetragonal cassiterite). The standard defense (TREOR/DICVOL
            # practice) is to search HIGH-symmetry lattices first and only
            # descend while no convincing solution exists. M20 is only
            # comparable within a symmetry level, never across levels.
            cells = []
            stages_run = []
            stopped_at = None
            for stage in (["cubic"], ["tetragonal", "hexagonal", "trigonal"],
                          ["orthorhombic"], ["monoclinic"]):
                found = _run_index(stage)
                stages_run.append(stage)
                cells.extend(found)
                if any(c[0] >= stage_stop_m20 and int(c[1]) == 0 for c in found):
                    stopped_at = "/".join(stage)
                    break
    finally:
        _random.setstate(_rng_state)

    # Dedupe near-identical cells (same lattice within 0.2%), keeping best M20;
    # then sort by M20 desc with SMALLEST VOLUME first among alias-equivalent
    # M20 (superlattice-alias resolution). Alias-equivalence is a FACTOR bucket
    # (default 2x, de Wolff practice): a supercell fits peaks with smaller
    # residuals (more parameters), inflating its M20 well past the true cell's
    # (observed live: a sqrt5 x sqrt5 x 2 supercell at M20=26 outranked the true
    # cell at M20=19), so M20 differences below ~2x are NOT decisive between
    # alias cells — the smallest cell wins. A genuinely better solution beats
    # the bucket by exceeding the factor.
    uniq = []
    for c in cells:
        vec = np.array(c[3:10], dtype=float)  # a,b,c,al,be,ga,V
        if vec[6] < float(min_cell_volume) or min(vec[0], vec[1], vec[2]) < 2.0:
            # Physical floors: a unit cell below ~15 A^3 cannot contain an atom
            # (the smallest real cells — bcc metals — are ~20 A^3), and no real
            # crystal has a cell edge under 2 A (shortest known ~2.29 A, Be).
            # Such solutions are arithmetic subcell artifacts of the d-spacing
            # set (GSAS's own halfCell post-step can emit them below its 2.5 A
            # search floor).
            continue
        for u in uniq:
            if u["ibrav"] == int(c[2]) and np.allclose(vec, u["_vec"], rtol=2e-3):
                if c[0] > u["M20"]:
                    u.update(M20=float(c[0]), X20=int(c[1]), _vec=vec)
                break
        else:
            uniq.append({"M20": float(c[0]), "X20": int(c[1]), "ibrav": int(c[2]),
                         "_vec": vec})

    def _m20_bucket(m20):
        import math
        factor = max(float(alias_m20_factor), 1.001)
        return int(math.floor(math.log(max(m20, 1e-9)) / math.log(factor)))

    uniq.sort(key=lambda u: (-_m20_bucket(u["M20"]), u["_vec"][6]))

    out_cells = []
    for u in uniq[:int(top_n)]:
        a, b, c_, al, be, ga, vol = (float(x) for x in u["_vec"])
        ib = u["ibrav"]
        system = ("trigonal/hexagonal" if ib in (3, 4)
                  else _IBRAV_TO_SYSTEM.get(ib, "?"))
        out_cells.append({
            "M20": round(u["M20"], 2), "X20": u["X20"],
            "bravais": _BRAVAIS_NAMES[ib], "crystal_system": system,
            "a": a, "b": b, "c": c_, "alpha": al, "beta": be, "gamma": ga,
            "volume": vol,
        })

    result = {
        "candidate_cells": out_cells,
        "n_peaks_used": len(tts),
        "dmin_angstrom": float(min(p[7] for p in peaks)),
        "wavelength": float(lam),
        "warnings": warnings,
        "note": (
            "Candidates sorted by M20 (de Wolff figure of merit), smallest volume "
            "first among near-equal M20 (a larger equal-M20 cell is usually a "
            "superlattice alias of the smaller one). M20 > ~10 with X20 = 0 is a "
            "convincing solution; M20 near the cutoff is a guess. Indexing "
            "solutions come in FAMILIES related by common factors (x1/2 subcells "
            "from lattice centering, xsqrt2 / xsqrt3 supercells), so the true "
            "cell may be a simple multiple of a candidate — if a database search "
            "around the top cell finds nothing, try the other candidates and "
            "volume-related multiples of them. The downstream simulate+score "
            "loop is the arbiter: a wrong cell yields no scoring match, so "
            "iterate (more peaks, narrowed crystal_systems, zero_offset) rather "
            "than trust M20 alone."
        ),
        "stages_run": ["/".join(s) for s in stages_run],
        "stopped_at": stopped_at,
    }
    if out_cells:
        best = out_cells[0]
        # Ready-made DB filter around the best cell: ±2% per edge covers
        # determination / temperature / composition differences between the
        # sample and databases. The ±6% 'volume' range (edges cubed) is
        # PERMUTATION-INVARIANT — prefer it as the primary cell filter, since a
        # database entry's axis convention may not match the indexed cell's.
        result["lattice_param_ranges"] = {
            k: (round(best[k] * 0.98, 4), round(best[k] * 1.02, 4))
            for k in ("a", "b", "c")
        }
        result["lattice_param_ranges"]["volume"] = (
            round(best["volume"] * 0.94, 2), round(best["volume"] * 1.06, 2))
    return result




def _phase_cell_mass(phase) -> float:
    """Unit-cell mass (sum of atomic masses x multiplicities) for the GSAS
    weight-fraction formula w_i = f_i * M_cell,i / sum_j f_j * M_cell,j."""
    import GSASII.GSASIIElem as G2elem  # same distribution as G2sc
    mass = 0.0
    general = phase.data["General"]
    atoms = phase.data["Atoms"]
    cx, ct, cs, cia = general["AtomPtrs"]
    for a in atoms:
        el = a[ct].strip("+-0123456789")
        mult = float(a[cs + 1])
        occ = float(a[cx + 3])
        info = G2elem.GetAtomInfo(el)
        mass += info["Mass"] * mult * occ
    return mass


def _wt_fractions(phases, hist) -> dict:
    num = {}
    for ph in phases:
        f = float(ph.getHAPvalues(hist)["Scale"][0])
        num[ph.name] = max(f, 0.0) * _phase_cell_mass(ph)
    tot = sum(num.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in num.items()}


# A HAP phase fraction below this share of the largest one is treated as
# collapsed (GSAS-II floors a vanishing fraction at 1e-12, not 0).
_COLLAPSED_FRACTION = 1e-9


def _wt_fraction_esds(gpx, phases, hist) -> tuple:
    """Least-squares esds of the weight fractions w_i = f_i M_i / sum_j f_j M_j,
    propagated from GSAS-II's covariance of the refined HAP phase fractions f
    (the last refinement's covariance; the first phase's f is held fixed and
    absorbed by the histogram scale, so it contributes through the sum only).

    Returns ({phase_name: esd or None}, note). These are statistical
    (counting-noise) esds scaled by GSAS-II's goodness of fit; they do not
    include systematic errors (wrong or missing phases, preferred orientation,
    microabsorption), which usually dominate in quantitative phase analysis."""
    names = [ph.name for ph in phases]
    try:
        cov = gpx["Covariance"]["data"]
        vary = list(cov["varyList"])
        C = np.asarray(cov["covMatrix"], dtype=float)
    except Exception:
        return {n: None for n in names}, "no covariance matrix from the last refinement"
    if C.ndim != 2 or C.shape[0] != len(vary):
        return {n: None for n in names}, "covariance matrix shape does not match the varied parameters"
    hid = hist.id
    raw = [float(ph.getHAPvalues(hist)["Scale"][0]) for ph in phases]
    idx = []
    for ph in phases:
        key = f"{ph.id}:{hid}:Scale"
        idx.append(vary.index(key) if key in vary else None)
    # A phase collapsed when its fraction went to zero: GSAS-II pins it at a
    # tiny floor (1e-12) with zero variance rather than at exactly 0. The
    # linear propagation is meaningless then and would report a spurious
    # +/-0 for every phase.
    top = max([abs(v) for v in raw] + [1e-300])
    collapsed = [n for n, v, j in zip(names, raw, idx)
                 if not v > _COLLAPSED_FRACTION * top
                 or (j is not None and not C[j, j] > 0)]
    if collapsed:
        return {n: None for n in names}, (
            f"no esd: phase fraction(s) refined to zero for {collapsed} (phase collapsed; "
            "the weight fractions are not a valid refinement result)")
    f = np.array(raw)
    M = np.array([_phase_cell_mass(ph) for ph in phases])
    S = float(np.sum(f * M))
    if S <= 0:
        return {n: None for n in names}, "all phase fractions are zero"
    refined = [k for k, j in enumerate(idx) if j is not None]
    if not refined:
        return {n: None for n in names}, "phase fractions were not varied in the last refinement"
    sub = C[np.ix_([idx[k] for k in refined], [idx[k] for k in refined])]
    J = np.zeros((len(phases), len(refined)))
    for i in range(len(phases)):
        for c, k in enumerate(refined):
            J[i, c] = ((M[i] * S if i == k else 0.0) - f[i] * M[i] * M[k]) / S ** 2
    var = np.einsum("ij,jk,ik->i", J, sub, J)
    esd = {n: (float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else None)
           for n, v in zip(names, var)}
    return esd, ("statistical esd from the refinement covariance (scaled by GOF); "
                 "excludes systematic errors, which usually dominate")


# Instrument-profile terms refined: V, W (Gaussian) and X (Lorentzian, 1/cos)
# — the ones isotropic microstrain cannot mimic. U (Gaussian tan^2) and Y
# (Lorentzian tan) are degenerate with microstrain broadening; refining them
# too drove U and Y negative on certified round-robin data (unphysical widths)
# for no gain in accuracy.
_PROFILE_KEYS = ("V", "W", "X")


def _instrument_profile_stage(gpx, hist, two_theta, corr_fn, rwp_fn, trace,
                              terms=_PROFILE_KEYS) -> bool:
    """Refine the instrument profile (Caglioti V, W; Lorentzian X — see
    ``_PROFILE_KEYS``) and keep it only if it helps: the whole-profile
    correlation must not drop and the widths must stay physical (Gaussian
    FWHM^2 = U tan^2(th) + V tan(th) + W > 0 over the measured range, X >= 0). Otherwise the template values are
    restored and the refinement re-converged without them. Records the outcome
    in ``trace``; returns whether the refined profile was kept."""
    terms = [str(t).upper() for t in terms]
    bad = [t for t in terms if t not in ("U", "V", "W", "X", "Y")]
    if bad or not terms:
        trace.append({"stage": "instrument_profile", "error": f"unknown instrument_terms {bad or terms}"})
        return False
    path = lambda k: ["Instrument Parameters", 0, k]  # noqa: E731
    try:
        before = {k: list(hist.getHistEntryValue(path(k))) for k in terms}
    except Exception as exc:
        trace.append({"stage": "instrument_profile", "error": str(exc)[:200]})
        return False
    c0, c1, kept = corr_fn(), float("nan"), False
    try:
        hist.set_refinements({"Instrument Parameters": list(terms)})
        gpx.do_refinements([{}])
        c1 = corr_fn()
        v = {k: float(hist.getHistEntryValue(path(k))[1]) for k in ("U", "V", "W", "X", "Y")}
        t = np.tan(np.radians(np.asarray(two_theta, dtype=float) / 2.0))
        fwhm2 = v["U"] * t ** 2 + v["V"] * t + v["W"]
        physical = bool(np.all(fwhm2 > 0) and v["X"] >= 0 and v["Y"] >= 0)
        kept = bool(np.isfinite(c1) and physical and (not np.isfinite(c0) or c1 >= c0 - 1e-3))
    except Exception as exc:
        trace.append({"stage": "instrument_profile", "error": str(exc)[:200]})
    if kept:
        trace.append({"stage": "instrument_profile", "terms": list(terms), "Rwp": rwp_fn(), "profile_corr": c1})
        return True
    try:
        hist.clear_refinements({"Instrument Parameters": list(terms)})
        for k in terms:
            hist.setHistEntryValue(path(k), before[k])
        gpx.do_refinements([{}])
    except Exception:
        pass
    trace.append({"stage": "instrument_profile", "terms": list(terms), "rejected": True, "profile_corr_tried": c1,
                  "Rwp": rwp_fn(), "profile_corr": corr_fn()})
    return False


def rietveld_refine_multiphase(
    structure_paths,
    two_theta,
    intensity,
    wavelength="CuKa",
    two_theta_range=None,
    refine_cell=True,
    refine_profile=True,
    n_background_terms=6,
    initial_fractions=None,
    refine_instrument=True,
    instrument_terms=None,
    kalpha2_ratio=None,
    displacement="cif",
    geometry=None,
    instrument_preset=None,
    sample_displacement=None,
    preferred_orientation="auto",
    _displacement_sources=None,
):
    """Multi-phase Rietveld: refine ALL phases against one measured pattern.

    Staged protocol mirroring ``rietveld_refine`` (background+scale -> zero ->
    phase fractions -> mustrain -> instrument profile -> cells), with per-phase HAP scale factors
    refined for every phase after the first (the first is absorbed by the
    histogram scale) and weight fractions computed from the GSAS convention
    w_i = f_i * M_cell,i / sum. Validated round-trip: fractions recovered to
    ~1e-3 and cells to <1e-4 A on a synthetic two-phase pattern.

    ``initial_fractions`` — optional starting HAP phase fractions aligned with
    ``structure_paths`` (the series warm start); default all 1.0.

    ``refine_instrument`` — refine the instrument profile (V, W, X) after
    microstrain; the template's generic values cannot match a given
    diffractometer's peak shape, and a mis-shaped profile moves intensity
    between overlapping phases. Guarded: kept only when the fit does not get
    worse and the widths stay physical (see ``_instrument_profile_stage``).
    ``instrument_terms`` — which of U, V, W, X, Y (default ``_PROFILE_KEYS``).

    ``kalpha2_ratio`` — override the K-alpha2 modelling (see ``_radiation``).

    ``displacement`` — atomic displacement parameters (see
    ``_apply_displacements``): 'cif' (default) takes each element's U from the
    structure file within ``CIF_U_RANGE``, 'cif_raw' unbounded, 'default'
    leaves GSAS-II's 0.01 Å^2, a number sets that U.

    ``geometry`` — 'bragg-brentano' (flat-plate reflection) or
    'debye-scherrer' (capillary / transmission); None infers Bragg-Brentano
    for a lab-anode wavelength and Debye-Scherrer otherwise.
    ``instrument_preset`` — starting profile: 'lab', 'synchrotron' or None
    (by wavelength; see ``INSTRUMENT_PRESETS``).
    ``sample_displacement`` — refine the Bragg-Brentano sample displacement
    (default off; kept only if the fit improves).
    ``preferred_orientation`` — March-Dollase: 'auto' (default; applied for
    Bragg-Brentano only, see ``PO_MIN_GAIN``), None, or {phase: [h, k, l]}.

    ``_displacement_sources`` — internal: the original CIFs to read U from when
    ``structure_paths`` are rewritten copies (the series warm start)."""
    import os
    import tempfile

    G2sc = _load_g2sc()
    lam = _resolve_wavelength(wavelength)
    geom = _resolve_geometry(wavelength, geometry)
    preset = _resolve_instrument_preset(wavelength, instrument_preset)
    # opt-in: with no internal standard the displacement correlates with the
    # cells (on a single phase it absorbed a uniform 0.5 % cell error)
    do_shift = bool(sample_displacement) and geom == "bragg-brentano"
    po_spec = preferred_orientation
    if isinstance(po_spec, str) and po_spec.lower() == "auto" and geom != "bragg-brentano":
        po_spec = None   # capillary / transmission: texture is averaged out
    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if two_theta_range is not None:
        lo, hi = float(min(two_theta_range)), float(max(two_theta_range))
        m = (x >= lo) & (x <= hi)
        x, y = x[m], y[m]
    if x.size < 20:
        raise ValueError("Rietveld refinement needs a denser pattern (>=20 points in range).")
    paths = list(structure_paths)
    if len(paths) < 2:
        raise ValueError("rietveld_refine_multiphase needs >=2 structures; "
                         "use refine_rietveld for a single phase.")

    with tempfile.TemporaryDirectory() as td:
        xy = os.path.join(td, "data.xy")
        np.savetxt(xy, np.column_stack([x, y, np.sqrt(np.maximum(y, 1.0))]),
                   fmt="%.6f")
        instf = os.path.join(td, "auto.instprm")
        with open(instf, "w", encoding="utf-8") as fh:
            fh.write(_instprm(wavelength, kalpha2_ratio, geom, preset))

        gpx = G2sc.G2Project(newgpx=os.path.join(td, "riet.gpx"))
        hist = gpx.add_powder_histogram(xy, iparams=instf, fmthint="xye")
        phases = []
        adp_used = {}
        adp_replaced = {}
        input_cells = {}
        for i, p in enumerate(paths):
            # one canonicalization dir per phase: _canonicalize_cif writes a
            # fixed 'canonical.cif' name, which would collide across phases
            pd = os.path.join(td, f"ph{i}")
            os.makedirs(pd, exist_ok=True)
            cif = _canonicalize_cif(p, pd, symprec=0.1)
            name = f"phase{i}_{os.path.splitext(os.path.basename(p))[0][:24]}"
            ph = gpx.add_phase(cif, phasename=name)
            _rep = {}
            adp_used[name] = _apply_displacements(
                ph, (_displacement_sources or paths)[i], displacement, _rep)
            if _rep:
                adp_replaced[name] = _rep
            gpx.link_histogram_phase(hist, ph)
            if initial_fractions is not None:
                ph.getHAPvalues(hist)["Scale"][0] = max(
                    float(initial_fractions[i]), 1e-4)
            phases.append(ph)
            input_cells[name] = dict(ph.get_cell())

        def _rwp():
            r = hist.residuals
            return float(r.get("Rwp", r.get("wR", float("nan"))))

        def _corr():
            yo = np.asarray(hist.getdata("Yobs"), dtype=float)
            yc = np.asarray(hist.getdata("Ycalc"), dtype=float)
            if yo.size != yc.size or yo.size < 2:
                return float("nan")
            return float(np.corrcoef(yo, yc)[0, 1])

        trace = []

        def _stage(label, refdict=None, hap=None, hap_phases=None):
            try:
                if hap:
                    for ph in (hap_phases or phases):
                        ph.set_HAP_refinements(hap, histograms=[hist])
                gpx.do_refinements([refdict or {}])
                trace.append({"stage": label, "Rwp": _rwp(), "profile_corr": _corr()})
            except Exception as exc:
                trace.append({"stage": label, "error": str(exc)[:200]})

        _stage("background+scale", {"set": {
            "Background": {"type": "chebyschev", "refine": True,
                           "no. coeffs": int(n_background_terms)},
            "Sample Parameters": ["Scale"]}})
        _stage("zero_offset", {"set": {"Instrument Parameters": ["Zero"]}})
        shift_rec = (_sample_displacement_stage(gpx, hist, geom, _rwp, _corr, trace)
                     if do_shift else {})
        # phase fractions for all but the first phase (first absorbed by the
        # histogram scale — refining all N is rank-deficient)
        _stage("phase_fractions", hap={"Scale": True}, hap_phases=phases[1:])
        if refine_profile:
            _stage("mustrain", {"set": {"Mustrain": {"type": "isotropic", "refine": True}}})
        if refine_instrument:
            _instrument_profile_stage(gpx, hist, x, _corr, _rwp, trace,
                                      terms=instrument_terms or _PROFILE_KEYS)
        if refine_cell:
            # cells ONE PHASE AT A TIME, dominant first: a joint cells stage
            # lets a weak phase's ill-determined cell run away on the other
            # phases' residuals (observed: a 10 wt% cubic phase collapsed to
            # a = 0.008 Å while the global fit stayed at corr 0.993 — its
            # intensity is too small for the runaway to hurt the residual).
            # With the dominant cells already settled, each weaker phase's
            # cell refines alone against what is left.
            order = sorted(phases,
                           key=lambda ph: -float(ph.getHAPvalues(hist)["Scale"][0]
                                                 * _phase_cell_mass(ph)))
            for ph in order:
                for other in phases:
                    other.set_refinements({"Cell": other is ph})
                _stage(f"cell:{ph.name}")
            for other in phases:
                other.set_refinements({"Cell": False})

        po_rec = {}
        if po_spec:
            po_rec = _preferred_orientation_stage(gpx, phases, hist, po_spec, _rwp, trace,
                                                  fractions=_wt_fractions(phases, hist))
        neg_mu = _negative_mustrain(phases, hist)

        wt = _wt_fractions(phases, hist)
        wt_esd, wt_esd_note = _wt_fraction_esds(gpx, phases, hist)
        _ck = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta",
               "angle_gamma", "volume")
        per_phase = []
        for ph, p in zip(phases, paths):
            cell = ph.get_cell()
            cell_esd = {}
            try:
                _, cell_esd = ph.get_cell_and_esd()
            except Exception:
                pass
            _size, mustrain = _extract_hap_broadening(ph, hist)
            refined = {k: float(cell[k]) for k in _ck if k in cell}
            inp = {k: float(input_cells[ph.name][k])
                   for k in _ck if k in input_cells[ph.name]}
            # per-phase runaway guard: thermal/compositional drift is well
            # under 10% per axis; beyond that the cell diverged (weak phases
            # can collapse without denting the GLOBAL fit metrics). Report
            # the input cell instead of garbage, and flag it.
            runaway = any(
                abs(refined[k] - inp[k]) > 0.10 * max(inp[k], 1e-9)
                for k in ("length_a", "length_b", "length_c")
                if k in refined and k in inp)
            per_phase.append({
                "structure_path": p,
                "phase_name": ph.name,
                "weight_fraction": round(float(wt.get(ph.name, 0.0)), 4),
                "weight_fraction_esd": (round(wt_esd[ph.name], 4)
                                        if wt_esd.get(ph.name) is not None else None),
                # raw HAP fraction (NOT weight fraction) — the series warm
                # start feeds this back in as the next frame's initial value
                "hap_fraction": float(ph.getHAPvalues(hist)["Scale"][0]),
                "lattice": inp if runaway else refined,
                "cell_runaway": bool(runaway),
                "input_lattice": inp,
                "lattice_esd": {} if runaway else
                               {k: float(v) for k, v in (cell_esd or {}).items()
                                if isinstance(v, (int, float))},
                "microstrain": mustrain,
                "displacement_U": adp_used.get(ph.name, {}),
                "displacement_U_replaced": adp_replaced.get(ph.name, {}),
                "preferred_orientation": po_rec.get(ph.name),
            })

        prof = {
            "two_theta": [float(v) for v in hist.getdata("X")],
            "y_obs": [float(v) for v in hist.getdata("Yobs")],
            "y_calc": [float(v) for v in hist.getdata("Ycalc")],
            "y_background": [float(v) for v in hist.getdata("Background")],
        }
        prof["residual"] = [o - c for o, c in zip(prof["y_obs"], prof["y_calc"])]
        res = dict(hist.residuals)

    _yo, _yc = np.asarray(prof["y_obs"], float), np.asarray(prof["y_calc"], float)
    if _yo.size > 1 and np.std(_yc) > 0 and np.std(_yo) > 0:
        final_corr = float(np.corrcoef(_yo, _yc)[0, 1])
    else:
        final_corr = 0.0
    if final_corr != final_corr:
        final_corr = 0.0
    _corrs = [t["profile_corr"] for t in trace
              if isinstance(t.get("profile_corr"), float)
              and t["profile_corr"] == t["profile_corr"]]
    best_pre = max(_corrs[:-1]) if len(_corrs) > 1 else final_corr
    converged = bool(final_corr >= 0.5 and final_corr >= best_pre - 0.1)

    return {
        "phases": per_phase,
        "weight_fractions": {p["phase_name"]: p["weight_fraction"] for p in per_phase},
        "weight_fraction_esds": {p["phase_name"]: p["weight_fraction_esd"] for p in per_phase},
        "weight_fraction_esd_note": wt_esd_note,
        "converged": converged,
        "Rwp": _final_rwp(res, trace),
        "profile_corr": final_corr,
        "gof": _safe_gof(res),
        "convergence_trace": trace,
        "profile": prof,
        "wavelength": float(lam),
        "kalpha_doublet": _radiation_record(wavelength, kalpha2_ratio),
        "geometry": geom,
        "instrument_preset": preset,
        "sample_displacement": shift_rec,
        "preferred_orientation": po_rec,
        "negative_microstrain": neg_mu,
        "note": ("Weight fractions follow the GSAS convention w_i = f_i*M_cell,i/sum "
                 "— quantitative ONLY when all crystalline phases are included and "
                 "microabsorption is negligible; an amorphous component is invisible "
                 "to Rietveld. Read 'profile_corr' as fit quality for arbitrary-unit "
                 "data. If 'converged' is False do not trust fractions or cells — "
                 "compare each phase's 'lattice' vs 'input_lattice' for runaways."),
    }


def _cif_with_lattice(structure_path: str, lattice: dict, out_path: str) -> str:
    """Rewrite a CIF with an updated unit cell (fractional coordinates kept) —
    the lattice warm start for sequential frame-to-frame refinement. Falls back
    to the original CIF when pymatgen is unavailable or parsing fails."""
    try:
        from pymatgen.core import Lattice, Structure
        s = Structure.from_file(structure_path)
        s.lattice = Lattice.from_parameters(
            lattice["length_a"], lattice["length_b"], lattice["length_c"],
            lattice["angle_alpha"], lattice["angle_beta"], lattice["angle_gamma"])
        s.to(filename=out_path, fmt="cif")
        return out_path
    except Exception:
        return structure_path


def rietveld_refine_series(
    structure_paths,
    frames,
    wavelength="CuKa",
    two_theta_range=None,
    refine_cell=True,
    refine_profile=True,
    n_background_terms=6,
    warm_start=True,
    refine_instrument=True,
    instrument_terms=None,
    kalpha2_ratio=None,
    displacement="cif",
    geometry=None,
    instrument_preset=None,
    sample_displacement=None,
    preferred_orientation=None,
):
    """Sequential multi-phase Rietveld across an in-situ series: a per-frame
    loop over the validated single-histogram staged protocol
    (``rietveld_refine_multiphase``), each frame warm-started from its
    predecessor's refined phase fractions and lattices (fractions via the
    starting HAP values, lattices via a rewritten starting CIF).

    Why not GSAS-II's native sequential mode: it refines every flagged
    parameter simultaneously with no staging, which proved numerically
    fragile in scriptable use — strain terms diverged (Dij -> -1e6, invalid
    metric tensor) or whole phase blocks were dropped as singular (the cubic
    HStrain 'eA' term is degenerate by construction and poisons the block).
    The staged per-frame loop is the same code path as the validated
    single-pattern refinement, each frame independently staged and
    independently convergence-checked — robust, and trivially resumable.

    ``frames``: list of {"two_theta": [...], "intensity": [...], "label": ...}
    in series order. ``warm_start=False`` restarts every frame from the input
    CIFs (use for non-contiguous frames where carrying state would mislead).
    ``preferred_orientation`` defaults to None here: the 'auto' axis trials
    multiply the per-frame cost; pass {phase: [h, k, l]} (an axis found on one
    frame with refine_rietveld_multiphase) to correct texture across frames."""
    paths = [str(p) for p in structure_paths]
    if not frames:
        raise ValueError("frames is empty")
    if not paths:
        raise ValueError("structure_paths is empty")

    out_frames = []
    prev_fracs = None
    prev_cells = None
    # stable names from the ORIGINAL paths — warm-start frames run on
    # rewritten temp CIFs whose basenames would otherwise leak into the keys
    phase_names = [
        f"phase{i}_{os.path.splitext(os.path.basename(p))[0][:24]}"
        for i, p in enumerate(paths)]
    with tempfile.TemporaryDirectory() as td:
        for k, fr in enumerate(frames):
            paths_k = paths
            if warm_start and prev_cells is not None:
                paths_k = []
                for i, p in enumerate(paths):
                    cell = prev_cells[i]
                    if cell:
                        wp = os.path.join(td, f"warm_{k}_{i}.cif")
                        paths_k.append(_cif_with_lattice(p, cell, wp))
                    else:
                        paths_k.append(p)
            try:
                r = rietveld_refine_multiphase(
                    paths_k, fr["two_theta"], fr["intensity"],
                    wavelength=wavelength, two_theta_range=two_theta_range,
                    refine_cell=refine_cell, refine_profile=refine_profile,
                    n_background_terms=n_background_terms,
                    initial_fractions=prev_fracs if warm_start else None,
                    refine_instrument=refine_instrument,
                    instrument_terms=instrument_terms, kalpha2_ratio=kalpha2_ratio,
                    displacement=displacement, geometry=geometry,
                    instrument_preset=instrument_preset,
                    sample_displacement=sample_displacement,
                    preferred_orientation=preferred_orientation,
                    _displacement_sources=paths)
            except Exception as exc:
                out_frames.append({"frame": k, "label": fr.get("label", k),
                                   "error": str(exc)[:200]})
                continue
            out_frames.append({
                "frame": k,
                "label": fr.get("label", k),
                # phases follow structure_paths order — remap onto stable names
                "weight_fractions": {phase_names[i]: p["weight_fraction"]
                                     for i, p in enumerate(r["phases"])},
                "weight_fraction_esds": {phase_names[i]: p.get("weight_fraction_esd")
                                         for i, p in enumerate(r["phases"])},
                "converged": r["converged"],
                "profile_corr": r["profile_corr"],
                "Rwp": r["Rwp"],
                "cells": {phase_names[i]: p["lattice"]
                          for i, p in enumerate(r["phases"])},
                "cell_runaways": [phase_names[i] for i, p in enumerate(r["phases"])
                                  if p.get("cell_runaway")],
            })
            if r["converged"]:
                # carry refined state forward only from a converged frame — a
                # diverged frame's runaway cell would poison the rest
                prev_fracs = [p["hap_fraction"] for p in r["phases"]]
                prev_cells = [p["lattice"] for p in r["phases"]]

    return {
        "frames": out_frames,
        "phase_names": phase_names,
        "n_frames": len(frames),
        "wavelength": float(_resolve_wavelength(wavelength)),
        "note": ("Per-frame weight fractions (GSAS convention) and per-frame "
                 "refined cells; frames warm-start from the last CONVERGED "
                 "predecessor. Same quantification caveats as "
                 "rietveld_refine_multiphase (all crystalline phases included, "
                 "amorphous invisible, microabsorption). A frame with "
                 "converged=False (or an 'error' entry) should not be trusted "
                 "— check whether the phase set stops describing the pattern "
                 "there (transient phase; cross-check track_phase_series "
                 "residual alerts)."),
    }
