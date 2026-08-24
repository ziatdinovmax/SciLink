"""``scherrer`` tool — crystallite size from a single XRD peak's FWHM.

Implements the Scherrer equation

    D = K · λ / (β · cos θ)

with optional instrumental-broadening subtraction in quadrature
(``β² = FWHM² − FWHM_instrumental²``). Pure-function helper used both
inside this skill's scripts and as a registered tool callable by the
LLM.
"""

from __future__ import annotations

import math
from typing import Any

from ..._shared._spec import ToolSpec


TOOL_SPEC = ToolSpec(
    name="scherrer",
    description=(
        "Crystallite (coherent domain) size from a single XRD peak's "
        "FWHM via the Scherrer equation. Subtracts the instrumental "
        "FWHM in quadrature when provided. Returns size in nm plus the "
        "deconvolved sample FWHM in radians."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.scherrer import scherrer",
    signature=(
        "scherrer(fwhm_deg: float, two_theta_deg: float, "
        "wavelength_angstrom: float, K: float = 0.9, "
        "instrumental_fwhm_deg: float = 0.0) -> dict"
    ),
    parameters={
        "fwhm_deg": {
            "type": "float",
            "description": "Total measured FWHM of the peak (degrees).",
        },
        "two_theta_deg": {
            "type": "float",
            "description": "Peak position in 2θ (degrees).",
        },
        "wavelength_angstrom": {
            "type": "float",
            "description": "X-ray wavelength in angstroms (e.g. 1.5406 for CuKa1).",
        },
        "K": {
            "type": "float",
            "description": "Scherrer shape constant. Default 0.9 (spherical, FWHM-based). 0.94 for cubic.",
        },
        "instrumental_fwhm_deg": {
            "type": "float",
            "description": "Instrumental FWHM from a standard (LaB6, Si). Subtracted in quadrature. Default 0.0 (size will be a lower bound).",
        },
    },
    required=["fwhm_deg", "two_theta_deg", "wavelength_angstrom"],
    returns=(
        "dict with 'size_nm' (float, NaN if peak is resolution-limited), "
        "'beta_sample_rad' (deconvolved sample FWHM in radians), "
        "'resolution_limited' (bool — True when instrumental FWHM ≥ measured)."
    ),
    when_to_use=(
        "After fit_profile gives a peak FWHM, to estimate average "
        "crystallite size from that single peak."
    ),
)


def scherrer(
    fwhm_deg: float,
    two_theta_deg: float,
    wavelength_angstrom: float,
    K: float = 0.9,
    instrumental_fwhm_deg: float = 0.0,
) -> dict[str, Any]:
    """Crystallite size in nm from a single peak's FWHM."""
    if fwhm_deg <= 0:
        raise ValueError(f"fwhm_deg must be positive; got {fwhm_deg}")
    if wavelength_angstrom <= 0:
        raise ValueError(f"wavelength_angstrom must be positive; got {wavelength_angstrom}")
    if not (0 < two_theta_deg < 180):
        raise ValueError(f"two_theta_deg must be in (0, 180); got {two_theta_deg}")
    if instrumental_fwhm_deg < 0:
        raise ValueError(f"instrumental_fwhm_deg must be ≥ 0; got {instrumental_fwhm_deg}")

    beta_total_sq = math.radians(fwhm_deg) ** 2
    beta_inst_sq = math.radians(instrumental_fwhm_deg) ** 2
    beta_sample_sq = beta_total_sq - beta_inst_sq

    if beta_sample_sq <= 0:
        return {
            "size_nm": float("nan"),
            "beta_sample_rad": 0.0,
            "resolution_limited": True,
        }

    beta_sample = math.sqrt(beta_sample_sq)
    theta_rad = math.radians(two_theta_deg / 2.0)
    wavelength_nm = wavelength_angstrom / 10.0
    size_nm = K * wavelength_nm / (beta_sample * math.cos(theta_rad))

    return {
        "size_nm": float(size_nm),
        "beta_sample_rad": float(beta_sample),
        "resolution_limited": False,
    }
