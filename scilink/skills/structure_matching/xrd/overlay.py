"""Draw an identification match overlay where the score was computed (#775).

Every scorer registers the simulated pattern before it scores it — a lattice
scale applied through Bragg's law, a 2θ scale, a zero shift — and reports the
fitted terms as ``registration``. An overlay drawn from the raw simulated
pattern then sits beside the data peaks by exactly the registration the score
already absorbed: the figure contradicts the number. ``register_overlay``
applies the same transform to the full simulated pattern and names it in the
legend; ``plot_match_overlay`` draws the data, the registered overlay(s) and
their difference with labelled axes.

A fitted lattice scale far from 1 is not silently accepted:
``lattice_scale_warnings`` says what it can mean, and ``register_overlay``
prints it as a ``TOOL_WARNINGS_JSON:`` marker for the plotted match, which the
curve agent lifts into the run's caveats.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

#: Lattice-scale bands beyond which a match is a caveat: an experimental
#: reference cell at the measurement's conditions differs from the sample by
#: thermal expansion, strain or composition — rarely 1 % (an inorganic solid
#: expands ~1e-5 /K); a computed (DFT-relaxed) cell is typically 1–3 % off.
SCALE_BAND = {"experimental": 0.01, "computed": 0.03, "unknown": 0.01}
#: Database sources whose cells are measured, and those whose cells are computed.
EXPERIMENTAL_SOURCES = frozenset({"cod", "local", "icsd"})
COMPUTED_SOURCES = frozenset({"mp", "oqmd", "aflow"})

WARNINGS_MARKER = "TOOL_WARNINGS_JSON:"


def registration_record(*, lattice_scale: float = 1.0, two_theta_scale: float = 1.0,
                        zero_shift: float = 0.0, reference_cell: str = "unknown") -> dict:
    """The transform a scorer applied to the simulated positions before
    scoring: ``2θ' = two_theta_scale · bragg(2θ, lattice_scale) + zero_shift``."""
    return {"lattice_scale": float(lattice_scale), "two_theta_scale": float(two_theta_scale),
            "zero_shift": float(zero_shift), "reference_cell": reference_cell_kind(reference_cell)}


def reference_cell_kind(value: Any) -> str:
    """``'experimental'`` / ``'computed'`` / ``'unknown'`` from a stated kind
    or a database source name (``'cod'``, ``'mp'``, ...)."""
    v = str(value or "").strip().lower()
    if v in SCALE_BAND:
        return v
    if v in EXPERIMENTAL_SOURCES:
        return "experimental"
    if v in COMPUTED_SOURCES:
        return "computed"
    return "unknown"


def apply_registration(positions: Sequence[float], registration: Optional[dict]) -> np.ndarray:
    """Move simulated 2θ positions (degrees) by a scorer's registration."""
    pos = np.asarray(positions, dtype=float)
    reg = registration or {}
    a = float(reg.get("lattice_scale", 1.0) or 1.0)
    if a != 1.0:
        s = np.clip(a * np.sin(np.radians(pos / 2.0)), -1.0, 1.0)
        pos = 2.0 * np.degrees(np.arcsin(s))
    return float(reg.get("two_theta_scale", 1.0) or 1.0) * pos + float(reg.get("zero_shift", 0.0) or 0.0)


def lattice_scale_warnings(registration: Optional[dict], formula: str = "") -> list[str]:
    """A caveat when the fitted scale is beyond the band for the reference
    cell's kind, else ``[]``. The scale is the lattice scale where one was
    fitted, else the 2θ scale (a linear stand-in for it)."""
    reg = registration or {}
    kind = reference_cell_kind(reg.get("reference_cell"))
    a = float(reg.get("lattice_scale", 1.0) or 1.0)
    which = "lattice scale"
    if a == 1.0 and float(reg.get("two_theta_scale", 1.0) or 1.0) != 1.0:
        a, which = float(reg["two_theta_scale"]), "2θ scale"
    band = SCALE_BAND[kind]
    if abs(a - 1.0) <= band:
        return []
    who = f"{formula} " if formula else ""
    pct = 100.0 * (a - 1.0)
    if kind == "computed":
        why = (f"beyond the {100 * band:.0f} % a computed (DFT-relaxed) reference cell typically "
               "differs by: a different phase, a strained or substituted lattice, or a wrong reference")
    else:
        why = (f"beyond {100 * band:.0f} % against "
               + ("an experimental reference cell" if kind == "experimental" else "a reference cell of unknown origin")
               + ": a computed (DFT) reference, a different or substituted phase, or a strained lattice — "
               "thermal expansion of an inorganic solid is ~0.001 % per K"
               + ("" if kind == "experimental" else "; a computed cell is typically 1–3 % off"))
    return [f"The {who}match needed a fitted {which} of ×{a:.3f} ({pct:+.1f} %), {why}."]


def register_overlay(simulated: Any, match_result: dict, *, phase: Any = None, formula: str = "",
                     reference_cell: Optional[str] = None, emit_warnings: bool = True) -> dict:
    """The simulated pattern moved by the registration the scorer fitted, with
    its legend label. See ``TOOL_SPEC`` for the contract."""
    if isinstance(simulated, dict):
        two_theta = simulated.get("two_theta") or simulated.get("sim_two_theta") or []
        intensities = (simulated.get("intensities") or simulated.get("intensity")
                       or simulated.get("sim_intensity") or [])
    else:
        two_theta, intensities = simulated, None
    reg = dict(_registration_of(match_result, phase) or registration_record())
    if reference_cell is not None:
        reg["reference_cell"] = reference_cell_kind(reference_cell)
    formula = formula or _formula_of(match_result, phase)
    moved = apply_registration(two_theta, reg)
    terms = []
    if reg["zero_shift"]:
        terms.append(f"zero shift {reg['zero_shift']:+.2f}°")
    if reg["lattice_scale"] != 1.0:
        terms.append(f"lattice scale ×{reg['lattice_scale']:.3f}")
    if reg["two_theta_scale"] != 1.0:
        terms.append(f"2θ scale ×{reg['two_theta_scale']:.3f}")
    label = f"Simulated {formula or 'reference'} (match overlay" + ("; " + ", ".join(terms) if terms else "") + ")"
    # worded exactly as the scorer worded it (the formula only where the
    # scorer knew it, a multiphase phase), so the run's caveats hold it once
    # when the script also copies the scorer's warnings
    warnings = lattice_scale_warnings(reg, _formula_of(match_result, phase))
    if warnings and emit_warnings:
        # the plotted match's caveat reaches the run's result (the curve agent
        # lifts this marker), whatever the script itself prints
        print(WARNINGS_MARKER + json.dumps(warnings), flush=True)
    out = {"two_theta": [float(v) for v in moved], "label": label, "registration": reg, "warnings": warnings}
    if intensities is not None:
        out["intensities"] = [float(v) for v in intensities]
    return out


def plot_match_overlay(ax, exp_two_theta: Sequence[float], exp_intensity: Sequence[float],
                       overlays: Sequence[dict], *, fwhm: Any = "auto", background: Any = "auto",
                       difference_ax=None, log_scale: Optional[bool] = None,
                       title: str = "Data and match overlay") -> dict:
    """Draw the data and REGISTERED overlays (``register_overlay`` results) on
    ``ax``, with labelled axes and ticks; see ``TOOL_SPEC_PLOT``."""
    from .score_match_fast import _broaden_peaks, _estimate_exp_fwhm
    x = np.asarray(exp_two_theta, dtype=float)
    y = np.asarray(exp_intensity, dtype=float)
    step = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 0.02
    w = _estimate_exp_fwhm(y, step) if fwhm == "auto" else float(fwhm)
    base = _background(y, background)
    ymax = float(np.nanmax(y - base)) if y.size else 1.0
    ax.plot(x, y, color="black", lw=1.0, label="Data")
    total = np.zeros_like(x)
    curves = []
    for ov in overlays:
        if not ov.get("intensities"):
            raise ValueError("an overlay needs its intensities: pass the simulate_xrd_pattern dict to "
                             "register_overlay")
        prof = _broaden_peaks(x, np.asarray(ov["two_theta"], float), np.asarray(ov["intensities"], float), w)
        if prof.max() > 0:
            prof = prof * (ymax / prof.max()) * float(ov.get("weight", 1.0))
        total += prof
        curves.append(prof)
        ax.plot(x, base + prof, lw=1.0, label=ov.get("label") or "Simulated (match overlay)")
    if len(curves) > 1:
        ax.plot(x, base + total, lw=1.0, ls="--", color="gray", label="Sum overlay")
    if log_scale is None:
        pos = y[y > 0]
        log_scale = bool(pos.size and np.log10(pos.max() / max(np.percentile(pos, 5), 1e-12)) > 1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelbottom=True)
    if difference_ax is not None:
        difference_ax.plot(x, y - base - total, color="tab:gray", lw=0.8, label="Data − overlay")
        difference_ax.set_xlabel("2θ (°)")
        difference_ax.set_ylabel("Data − overlay")
        difference_ax.tick_params(axis="x", labelbottom=True)
        difference_ax.legend(fontsize=8)
    return {"fwhm_used": w, "log_scale": log_scale, "overlay_sum": total.tolist(), "background": base.tolist()}


def _background(y: np.ndarray, background: Any) -> np.ndarray:
    """The baseline the overlay is drawn on: an array as given, none for
    ``None`` (background-subtracted data), else a smoothed running low
    percentile of the data — the floor between peaks."""
    if background is None:
        return np.zeros_like(y)
    if not isinstance(background, str):
        b = np.asarray(background, dtype=float)
        if b.shape != y.shape:
            raise ValueError("background must have the data's length")
        return b
    from scipy.ndimage import percentile_filter, uniform_filter1d
    size = max(25, int(0.05 * y.size)) | 1
    return uniform_filter1d(percentile_filter(y, 10, size=size, mode="nearest"), size, mode="nearest")


def _registration_of(result: dict, phase: Any) -> Optional[dict]:
    if not isinstance(result, dict):
        return None
    phases = result.get("active_phases")
    if phases is not None and phase is not None:
        for i, p in enumerate(phases):
            if phase in (i, p.get("id"), p.get("formula")):
                return p.get("registration")
        raise ValueError(f"phase {phase!r} is not among the match's active phases")
    if phases and phase is None and len(phases) == 1:
        return phases[0].get("registration")
    return result.get("registration")


def _formula_of(result: dict, phase: Any) -> str:
    phases = (result or {}).get("active_phases") or []
    if phase is None and len(phases) == 1:
        return str(phases[0].get("formula") or "")
    for i, p in enumerate(phases):
        if phase in (i, p.get("id"), p.get("formula")):
            return str(p.get("formula") or "")
    return ""


TOOL_SPEC = ToolSpec(
    name="register_overlay",
    description=(
        "Move a simulated pattern by the registration a match scorer fitted "
        "(zero shift, lattice scale, 2θ scale) so the overlay is drawn where the "
        "figure of merit was computed, with a legend label stating the registration. "
        "A fitted scale beyond the band for the reference cell's kind is returned as "
        "'warnings' and reported as a caveat on the run."
    ),
    import_line="from scilink.skills.structure_matching.xrd.overlay import register_overlay",
    signature=("register_overlay(simulated, match_result, phase=None, formula='', "
                   "reference_cell=None)"),
    parameters={
        "simulated": {"type": "dict | list",
                      "description": "The simulate_xrd_pattern dict (two_theta + intensities), or a list "
                                     "of 2θ positions. Pass the FULL simulated pattern, not the scorer's "
                                     "strongest-peaks subset."},
        "match_result": {"type": "dict",
                         "description": "The result of score_xrd_match_robust, score_xrd_match_multiphase or "
                                        "score_xrd_match_fast for THIS simulated pattern (its 'registration')."},
        "phase": {"type": "int | str",
                  "description": "Multiphase only: the active phase's index, id or formula."},
        "formula": {"type": "str", "description": "Formula for the legend label (multiphase: read from the phase)."},
        "reference_cell": {"type": "str",
                           "description": "'experimental' | 'computed' or the candidate's database source "
                                          "('cod', 'mp', 'local', ...); overrides what the scorer was told. "
                                          "Sets the scale band for the caveat: 1 % experimental, 3 % computed."},
    },
    required=["simulated", "match_result"],
    returns=("dict with 'two_theta' (registered), 'intensities' (unchanged), 'label' "
             "(e.g. 'Simulated TiO2 (match overlay; zero shift -0.05°, lattice scale ×1.022)'), "
             "'registration' and 'warnings'."),
    when_to_use=("Always, for every identification overlay drawn: the raw simulated pattern sits beside "
                 "the data by exactly the registration the score absorbed."),
)

TOOL_SPEC_PLOT = ToolSpec(
    name="plot_match_overlay",
    description=(
        "Draw the data and one or more REGISTERED overlays (register_overlay results), broadened to the "
        "data's peak width, with labelled axes and ticks, a legend naming each overlay with its "
        "registration, a sum curve only for two or more phases, and an optional 'Data − overlay' panel."
    ),
    import_line="from scilink.skills.structure_matching.xrd.overlay import plot_match_overlay",
    signature=("plot_match_overlay(ax, exp_two_theta, exp_intensity, [overlay], fwhm='auto', "
                   "difference_ax=None)"),
    parameters={
        "ax": {"type": "matplotlib Axes", "description": "Axes for the data and overlay."},
        "exp_two_theta": {"type": "list", "description": "Experimental 2θ (degrees)."},
        "exp_intensity": {"type": "list", "description": "Experimental intensity (background-subtracted or raw)."},
        "overlays": {"type": "list[dict]",
                     "description": "register_overlay results; a 'weight' key scales a phase's overlay "
                                    "(e.g. its fraction) for a mixture."},
        "fwhm": {"type": "float | 'auto'",
                 "description": "Broadening (degrees). 'auto' uses the data's narrowest resolved peak "
                                "width; RAISE only if the overlay looks spikier than the data, LOWER if it "
                                "smears resolved peaks together."},
        "background": {"type": "'auto' | None | list",
                       "description": "Baseline the overlay is drawn on: 'auto' (default) a smoothed running "
                                      "low percentile of the data; None for background-subtracted data; or "
                                      "the script's own fitted background array."},
        "difference_ax": {"type": "matplotlib Axes",
                          "description": "Optional second Axes for 'Data − overlay'."},
        "log_scale": {"type": "bool",
                      "description": "Default chooses by dynamic range (log beyond ~1.5 decades); keep "
                                     "one choice across a series' frames."},
    },
    required=["ax", "exp_two_theta", "exp_intensity", "overlays"],
    returns="dict with 'fwhm_used', 'log_scale', 'overlay_sum', 'background'.",
    when_to_use="For the identification figure, instead of drawing overlay sticks by hand.",
)

TOOL_SPECS = [TOOL_SPEC_PLOT]
