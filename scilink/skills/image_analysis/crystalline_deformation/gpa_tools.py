"""Burgers vector tools (line integral of the displacement gradient
tensor beta around a dislocation core).

Reference: Cloete, Tarleton & Hofmann, Proc. R. Soc. A 478 (2022) 20210909.
Ported from MATLAB (https://github.com/JacquesCloete/Burgers-Vector-Calculator).

Strain/rotation inputs come from the shared GPA tool
``scilink.skills._shared.strain.gpa_strain_map`` (exx, eyy, exy, wxy) —
this module deliberately ships no GPA implementation of its own.
"""

import numpy as np
from scipy.integrate import simpson


# ═══════════════════════════════════════════════════════════════════
# Burgers vector computation
# ═══════════════════════════════════════════════════════════════════

def _strains_to_beta_2d(e_xx, e_yy, e_xy, w_z):
    """Convert 2D strain + rotation to displacement gradient beta."""
    Ny, Nx = e_xx.shape
    beta = np.zeros((Ny, Nx, 2, 2), dtype=np.float64)
    beta[:, :, 0, 0] = e_xx
    beta[:, :, 0, 1] = e_xy - w_z
    beta[:, :, 1, 0] = e_xy + w_z
    beta[:, :, 1, 1] = e_yy
    return beta


def _line_integral(beta, interval):
    """Burgers vector via rectangular line integral (Simpson's rule).

    Closed circulation requires the horizontal-edge pair to enter as
    (top − bottom) alongside (right − left). Verified against an analytic
    edge-dislocation field with known |b| — see
    tests/test_crystalline_deformation_gpa.py; flipping either pair of
    signs breaks |b| in a Poisson-ratio-dependent (non-calibratable) way.
    """
    Ny, Nx = beta.shape[:2]
    xv = np.arange(Nx) * interval
    yv = np.arange(Ny) * interval
    b = np.zeros(2)
    for i in range(2):
        b[i] = (+simpson(y=beta[0, :, i, 0], x=xv)
                - simpson(y=beta[:, 0, i, 1], x=yv)
                - simpson(y=beta[-1, :, i, 0], x=xv)
                + simpson(y=beta[:, -1, i, 1], x=yv))
    return b


def _line_integral_repeated(beta, interval, n_shrink=3):
    """Averaged Burgers vector from concentric shrinking circuits."""
    Ny, Nx = beta.shape[:2]
    results = []
    for m in range(n_shrink + 1):
        if 2 * m >= Ny - 2 or 2 * m >= Nx - 2:
            break
        sub = beta[m:Ny - m, m:Nx - m, :, :] if m > 0 else beta
        results.append(_line_integral(sub, interval))
    if not results:
        return np.zeros(2), np.empty((0, 2))
    b_all = np.array(results)
    mags = np.linalg.norm(b_all, axis=1)
    if len(mags) > 2:
        med = np.median(mags)
        inlier = np.abs(mags - med) < 1.5 * np.std(mags)
        b_mean = np.mean(b_all[inlier], axis=0) if inlier.any() else np.mean(b_all, axis=0)
    else:
        b_mean = np.mean(b_all, axis=0)
    return b_mean, b_all


def _compute_burgers_field(beta, interval, half_width=3, cutoff=None):
    """Sliding-window Burgers vector at every pixel."""
    Ny, Nx = beta.shape[:2]
    b_field = np.zeros((Ny, Nx, 2))
    hw = half_width
    for j in range(hw, Ny - hw):
        for i in range(hw, Nx - hw):
            sub = beta[j - hw:j + hw + 1, i - hw:i + hw + 1, :, :]
            b_field[j, i] = _line_integral(sub, interval)
    b_mag = np.linalg.norm(b_field, axis=2)
    if cutoff is not None:
        mask = b_mag < cutoff
        b_field[mask] = 0.0
        b_mag[mask] = 0.0
    return b_field, b_mag


def _find_cores(b_mag, threshold_factor=3.0, min_distance=5):
    """Locate dislocation cores as peaks in |b| field."""
    from skimage.feature import peak_local_max
    nz = b_mag[b_mag > 0]
    if len(nz) == 0:
        return np.empty((0, 2), dtype=int)
    threshold = threshold_factor * np.median(nz)
    return peak_local_max(b_mag, min_distance=min_distance,
                          threshold_abs=threshold)


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def burgers_from_gpa(e_xx, e_yy, e_xy, w_z, pixel_size,
                     dislocation_positions=None, circuit_half_width=10,
                     n_shrink=3, compute_field=False, field_half_width=3,
                     field_cutoff=None):
    """Compute Burgers vectors from GPA strain/rotation maps.

    Parameters
    ----------
    e_xx, e_yy : 2-D ndarray
        Normal strain maps from GPA.
    e_xy : 2-D ndarray
        Shear strain — pass ``exy`` from
        ``scilink.skills._shared.strain.gpa_strain_map``.
    w_z : 2-D ndarray
        Rigid rotation omega_z = 0.5*(duy/dx - dux/dy) — pass
        ``gpa_strain_map``'s ``wxy`` WITH THE + SIGN (note: the legacy
        e_th convention had the opposite sign).
    pixel_size : float
        Physical pixel size in consistent units (nm or m).
    dislocation_positions : list of (row, col) or None
        Known dislocation core positions (pixel coords).  For each,
        a Burgers circuit is placed around the core.
    circuit_half_width : int
        Half-width of circuit in pixels for per-dislocation mode.
    n_shrink : int
        Number of concentric shrinking circuits for averaging.
    compute_field : bool
        If True, compute full-field Burgers vector map.
    field_half_width : int
        Half-width for sliding circuit in field mode.
    field_cutoff : float or None
        Magnitude cutoff for noise filtering in field mode.

    Returns
    -------
    dict with keys:
        'beta' : (Ny, Nx, 2, 2) displacement gradient tensor
        'burgers_vectors' : list of dicts (if dislocation_positions given)
            each with 'position', 'b_vector', 'b_magnitude', 'b_direction'
        'b_field' : (Ny, Nx, 2) vector field (if compute_field)
        'b_mag' : (Ny, Nx) magnitude field (if compute_field)
        'cores' : (N, 2) auto-detected core positions (if compute_field)
    """
    beta = _strains_to_beta_2d(e_xx, e_yy, e_xy, w_z)
    interval = pixel_size
    result = {'beta': beta}

    if dislocation_positions is not None:
        Ny, Nx = e_xx.shape
        bvecs = []
        hw = circuit_half_width
        for pos in dislocation_positions:
            r, c = int(pos[0]), int(pos[1])
            r0, r1 = max(0, r - hw), min(Ny, r + hw + 1)
            c0, c1 = max(0, c - hw), min(Nx, c + hw + 1)
            sub = beta[r0:r1, c0:c1, :, :]
            if sub.shape[0] < 3 or sub.shape[1] < 3:
                continue
            b_mean, _ = _line_integral_repeated(sub, interval, n_shrink)
            mag = np.linalg.norm(b_mean)
            bvecs.append({
                'position': (r, c),
                'b_vector': b_mean,
                'b_magnitude': mag,
                'b_direction': b_mean / mag if mag > 0 else np.zeros(2),
            })
        result['burgers_vectors'] = bvecs

    if compute_field or dislocation_positions is None:
        b_field, b_mag = _compute_burgers_field(
            beta, interval, half_width=field_half_width, cutoff=field_cutoff)
        result['b_field'] = b_field
        result['b_mag'] = b_mag
        result['cores'] = _find_cores(b_mag)

    return result


# ── Tool registry specs: auto-discovered by scilink.skills._shared._registry
#    when the crystalline_deformation bundle is active ───────────────────────
from scilink.skills._shared._spec import ToolSpec

TOOL_SPECS = [
    ToolSpec(
        name="burgers_from_gpa",
        description=(
            "Burgers vectors from GPA strain/rotation maps via line-integral "
            "circuits: per-dislocation (given core positions) or full-field "
            "auto-detection of dislocation cores with a Burgers vector field. "
            "Feed strain/rotation from the shared gpa_strain_map tool "
            "(e_xy=exy, w_z=+wxy) and cores from PTM/CoS analysis when available."
        ),
        import_line="from scilink.skills.image_analysis.crystalline_deformation.gpa_tools import burgers_from_gpa",
        signature=(
            "burgers_from_gpa(e_xx, e_yy, e_xy, w_z, pixel_size, "
            "dislocation_positions=None, circuit_half_width=10, n_shrink=3, "
            "compute_field=False, field_half_width=3, field_cutoff=None)"
        ),
        returns="dict: burgers_vectors (per-dislocation) or cores/b_field/b_mag (full-field)",
    ),
]
