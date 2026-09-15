"""``williamson_hall`` tool — size + microstrain from a W-H plot.

Linear regression of  β·cos θ  vs  sin θ  across multiple peaks:

    β · cos θ = K · λ / D + 4 · ε · sin θ

The intercept gives the size term (K·λ/D); the slope gives 4ε. Each
peak's FWHM is first deconvolved from the instrumental FWHM in
quadrature.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..._shared._spec import ToolSpec


TOOL_SPEC = ToolSpec(
    name="williamson_hall",
    description=(
        "Williamson-Hall analysis: fits β·cosθ vs sinθ across multiple "
        "peaks to separate crystallite size (intercept) and microstrain "
        "(slope). Each peak's FWHM is first deconvolved from the "
        "instrumental FWHM in quadrature. Requires ≥ 3 peaks; 5+ "
        "spread across 2θ is the standard practice."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.williamson_hall import williamson_hall",
    signature=(
        "williamson_hall(peaks: list[dict], wavelength_angstrom: float, "
        "K: float = 0.9, instrumental_fwhm_deg: float = 0.0) -> dict"
    ),
    parameters={
        "peaks": {
            "type": "list[dict]",
            "description": "List of {'two_theta': deg, 'fwhm': deg} dicts; at least 3 peaks required.",
        },
        "wavelength_angstrom": {
            "type": "float",
            "description": "X-ray wavelength in angstroms.",
        },
        "K": {
            "type": "float",
            "description": "Scherrer constant. Default 0.9.",
        },
        "instrumental_fwhm_deg": {
            "type": "float",
            "description": "Instrumental FWHM from a standard. Subtracted in quadrature per peak. Default 0.0.",
        },
    },
    required=["peaks", "wavelength_angstrom"],
    returns=(
        "dict with 'size_nm' (from intercept), 'strain' (from slope; "
        "dimensionless), 'r_squared' (of the linear regression), "
        "'slope', 'intercept', 'n_peaks_used' (peaks that survived "
        "instrumental deconvolution)."
    ),
    when_to_use=(
        "After fit_profile gives FWHMs for ≥ 5 peaks spanning a useful "
        "2θ range, to separate size and strain contributions to peak "
        "broadening."
    ),
)


def williamson_hall(
    peaks: Sequence[dict],
    wavelength_angstrom: float,
    K: float = 0.9,
    instrumental_fwhm_deg: float = 0.0,
) -> dict[str, Any]:
    """W-H linear fit. See ``TOOL_SPEC`` for full contract."""
    if wavelength_angstrom <= 0:
        raise ValueError(f"wavelength_angstrom must be positive; got {wavelength_angstrom}")
    if instrumental_fwhm_deg < 0:
        raise ValueError(f"instrumental_fwhm_deg must be ≥ 0; got {instrumental_fwhm_deg}")
    if len(peaks) < 3:
        raise ValueError(f"Williamson-Hall requires at least 3 peaks; got {len(peaks)}")

    beta_inst_rad = math.radians(instrumental_fwhm_deg)

    xs: list[float] = []  # sin θ
    ys: list[float] = []  # β·cos θ (sample, after instrumental deconvolution)
    used = 0
    for p in peaks:
        try:
            two_theta = float(p["two_theta"])
            fwhm = float(p["fwhm"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"peak entry malformed: {p!r} ({e})") from e
        if fwhm <= 0 or not (0 < two_theta < 180):
            continue
        beta_total_rad = math.radians(fwhm)
        beta_sample_sq = beta_total_rad ** 2 - beta_inst_rad ** 2
        if beta_sample_sq <= 0:
            continue  # resolution-limited; skip from regression
        beta_sample = math.sqrt(beta_sample_sq)
        theta_rad = math.radians(two_theta / 2.0)
        xs.append(math.sin(theta_rad))
        ys.append(beta_sample * math.cos(theta_rad))
        used += 1

    if used < 3:
        return {
            "size_nm": float("nan"),
            "strain": float("nan"),
            "r_squared": 0.0,
            "slope": float("nan"),
            "intercept": float("nan"),
            "n_peaks_used": used,
            "note": "fewer than 3 peaks survived instrumental deconvolution",
        }

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    wavelength_nm = wavelength_angstrom / 10.0
    if intercept > 0:
        size_nm = K * wavelength_nm / float(intercept)
    else:
        size_nm = float("nan")
    strain = float(slope) / 4.0

    return {
        "size_nm": float(size_nm),
        "strain": float(strain),
        "r_squared": float(r_squared),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_peaks_used": used,
    }
