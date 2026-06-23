"""Atom finding utilities for atomic-resolution STEM images.

Classical tools extracted from atomap (Nord et al., 2017), reimplemented
with numpy/scipy/scikit-image only, plus a DCNN wrapper around AtomAI.

  - detect_atoms: classical peak detection + 2D Gaussian refinement
  - detect_atoms_dcnn: AtomNet3 deep-CNN ensemble detection (requires atomai)
  - refine_positions: fit 2D Gaussians at known positions (any source)
  - find_zone_axes: lattice translation vector detection
  - find_missing_atoms: predict fractional-site positions
  - subtract_atoms: remove fitted Gaussians to reveal weaker sublattices
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max


# ---------------------------------------------------------------------------
# 2D Gaussian model (matches atomap/external/gaussian2d.py convention)
# ---------------------------------------------------------------------------

def _gaussian2d(coords, x0, y0, A, sigma_x, sigma_y, theta, offset):
    """Normalised rotated 2D Gaussian.

    Parameters match atomap: ``A`` is integrated volume, peak height is
    ``A / (2π·σx·σy)``.
    """
    x, y = coords
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cos2 = cos_t ** 2
    sin2 = sin_t ** 2
    sin_2t = math.sin(2 * theta)

    sx2 = sigma_x ** 2
    sy2 = sigma_y ** 2

    a = cos2 / (2 * sx2) + sin2 / (2 * sy2)
    b = -sin_2t / (4 * sx2) + sin_2t / (4 * sy2)
    c = sin2 / (2 * sx2) + cos2 / (2 * sy2)

    dx = x - x0
    dy = y - y0

    norm = A / (2 * math.pi * sigma_x * sigma_y)
    return (norm * np.exp(-(a * dx**2 + 2 * b * dx * dy + c * dy**2))
            + offset).ravel()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_atoms(
    image: np.ndarray,
    separation: int,
    threshold_rel: float = 0.02,
    refine: bool = True,
    percent_to_nn: float = 0.40,
    subtract_background: bool = False,
    normalize_intensity: bool = True,
) -> dict:
    """Detect atom column positions with optional 2D Gaussian refinement.

    Follows atomap's ``get_atom_positions`` + ``fit_atom_positions_gaussian``
    pipeline.  Peak detection runs on the (optionally preprocessed) image;
    background subtraction and normalization default to atomap's conventions.

    Args:
        image: 2D grayscale array (HAADF: bright atoms on dark background).
        separation: Minimum atom spacing in **pixels**.
        threshold_rel: Relative peak threshold for ``peak_local_max``.
        refine: Fit a 2D Gaussian per peak for sub-pixel precision.
        percent_to_nn: Gaussian mask radius as fraction of NN distance.
        subtract_background: Gaussian-blur background subtraction before
            peak finding (atomap default: False).
        normalize_intensity: Normalize to 0-1 before peak finding
            (atomap default via ``normalize_signal``).

    Returns:
        dict with keys ``positions`` (N,2 as x,y where x=col y=row),
        ``sigma_x``, ``sigma_y``, ``amplitude``, ``rotation`` (all N,).
    """
    img = image.astype(np.float64)

    # Optional preprocessing (matches atomap's get_atom_positions flags)
    if subtract_background:
        bg = gaussian_filter(img, sigma=max(separation * 2, 30))
        img = np.clip(img - bg, 0, None)
    if normalize_intensity:
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)

    # Peak detection — returns (row, col)
    peaks_rc = peak_local_max(
        image=img,
        min_distance=int(separation),
        threshold_rel=threshold_rel,
    )

    if len(peaks_rc) == 0:
        return _empty_result()

    # Flip to (x, y) = (col, row) to match atomap convention
    positions = np.fliplr(peaks_rc).astype(np.float64)

    # Remove duplicate detections that are too close (atomap default)
    positions = _remove_too_close_atoms(positions, int(separation) / 2)
    if len(positions) == 0:
        return _empty_result()

    if not refine or len(positions) < 2:
        return {
            "positions": positions,
            "sigma_x": None,
            "sigma_y": None,
            "amplitude": None,
            "rotation": None,
        }

    # Use the original (un-preprocessed) image for Gaussian fitting
    fit_img = image.astype(np.float64)

    # --- 2D Gaussian refinement (matches atomap's iterative strategy) ---
    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=2)
    nn_dist = nn_dists[:, 1]  # distance to closest neighbor

    H, W = fit_img.shape
    sigma_x_arr = np.full(len(positions), separation / 4.0)
    sigma_y_arr = np.full(len(positions), separation / 4.0)
    amplitude_arr = np.zeros(len(positions))
    rotation_arr = np.zeros(len(positions))

    for i, (x0, y0) in enumerate(positions):
        mask_r = max(int(nn_dist[i] * percent_to_nn), 2)

        fitted = False
        ptn = percent_to_nn
        for attempt in range(10):
            r = max(int(nn_dist[i] * ptn), 2)

            # Crop bounds (row = y, col = x)
            r0 = max(0, int(y0) - r)
            r1 = min(H, int(y0) + r + 1)
            c0 = max(0, int(x0) - r)
            c1 = min(W, int(x0) + r + 1)

            crop = fit_img[r0:r1, c0:c1].copy()
            if crop.size < 7:
                ptn *= 0.95
                continue

            # Circular mask to exclude neighbouring atoms (atomap convention)
            cy_local = int(y0) - r0
            cx_local = int(x0) - c0
            yy_m, xx_m = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
            circ_mask = ((xx_m - cx_local) ** 2 + (yy_m - cy_local) ** 2) <= r ** 2
            crop = crop * circ_mask

            # Per-iteration background subtraction (atomap: lowest 3%)
            mask_pixels = crop[circ_mask]
            if mask_pixels.size > 0:
                n_low = max(int(0.03 * mask_pixels.size), 1)
                bg_val = float(np.median(np.sort(mask_pixels.ravel())[:n_low]))
                crop = crop - bg_val
                crop[crop < 0] = 0.0

            # Build coordinate grids (x = col index, y = row index)
            yy, xx = np.mgrid[r0:r1, c0:c1]
            coords = (xx.ravel().astype(np.float64),
                      yy.ravel().astype(np.float64))

            # Robust amplitude init (atomap: median of upper 3%)
            n_high = max(int(0.03 * mask_pixels.size), 1)
            peak_val = float(np.median(np.sort(mask_pixels.ravel())[-n_high:]))
            s0 = separation / 4.0

            p0 = [x0, y0, peak_val * 2 * math.pi * s0 * s0,
                  s0, s0, 0.01, 0.0]
            bounds_lo = [c0, r0, 0, 0.5, 0.5, -math.pi, -np.inf]
            bounds_hi = [c1, r1, np.inf, r * 2, r * 2, math.pi, np.inf]

            try:
                popt, _ = curve_fit(
                    _gaussian2d, coords, crop.ravel().astype(np.float64),
                    p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=2000,
                )
                fx, fy, fA, fsx, fsy, ftheta, foff = popt

                # Validation (matches atomap criteria)
                if abs(fx - x0) > r or abs(fy - y0) > r:
                    raise ValueError("center outside mask")
                if fA < 0:
                    raise ValueError("negative amplitude")
                ratio = max(fsx, fsy) / max(min(fsx, fsy), 0.1)
                if ratio > 4:
                    raise ValueError("sigma ratio > 4")

                positions[i] = [fx, fy]
                sigma_x_arr[i] = fsx
                sigma_y_arr[i] = fsy
                amplitude_arr[i] = fA
                rotation_arr[i] = ftheta % math.pi
                fitted = True
                break
            except Exception:
                ptn *= 0.95

        if not fitted:
            # Fallback: center of mass
            crop = fit_img[max(0, int(y0) - mask_r):min(H, int(y0) + mask_r + 1),
                           max(0, int(x0) - mask_r):min(W, int(x0) + mask_r + 1)]
            if crop.sum() > 0:
                yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
                total = crop.sum()
                cx = (xx * crop).sum() / total + max(0, int(x0) - mask_r)
                cy = (yy * crop).sum() / total + max(0, int(y0) - mask_r)
                positions[i] = [cx, cy]
            amplitude_arr[i] = fit_img[int(round(y0)), int(round(x0))]

    return {
        "positions": positions,
        "sigma_x": sigma_x_arr,
        "sigma_y": sigma_y_arr,
        "amplitude": amplitude_arr,
        "rotation": rotation_arr,
    }


def _remove_too_close_atoms(positions, tolerance, max_iter=20):
    """Remove atoms closer than *tolerance* px, keeping the brighter one.

    Ported from atomap's ``_remove_too_close_atoms``.
    """
    if len(positions) < 2:
        return positions
    # Use descending index as proxy for intensity (peak_local_max returns
    # brightest first).
    intensities = np.arange(len(positions))[::-1]
    for _ in range(max_iter):
        tree = cKDTree(positions)
        pairs = tree.query_pairs(tolerance)
        if not pairs:
            break
        pairs_ar = np.array(list(pairs))
        pair_int = intensities[pairs_ar]
        min_col = np.argmin(pair_int, axis=1)
        min_idx = pairs_ar[np.arange(len(min_col)), min_col]
        max_idx = pairs_ar[np.arange(len(min_col)), 1 - min_col]
        keep_mask = ~np.isin(max_idx, min_idx)
        remove = np.unique(min_idx[keep_mask])
        if len(remove) == 0:
            break
        positions = np.delete(positions, remove, axis=0)
        intensities = np.delete(intensities, remove, axis=0)
    return positions


def find_zone_axes(
    positions: np.ndarray,
    n_neighbors: int = 9,
    distance_tolerance: Optional[float] = None,
) -> list:
    """Detect lattice translation vectors from atom positions.

    Args:
        positions: (N, 2) array of atom positions (x, y).
        n_neighbors: Number of nearest neighbors to examine per atom.
        distance_tolerance: Clustering tolerance in pixels.
            Default: median NN distance / 3.

    Returns:
        List of (dx, dy) tuples — unique lattice vectors, shortest first.
    """
    if len(positions) < 3:
        return []

    tree = cKDTree(positions)
    k = min(n_neighbors + 1, len(positions))
    dists, indices = tree.query(positions, k=k)

    # Median nearest-neighbor distance
    nn1 = dists[:, 1]
    med_nn = float(np.median(nn1))

    if distance_tolerance is None:
        distance_tolerance = med_nn / 3.0

    # Collect all displacement vectors (neighbor - atom)
    all_vectors = []
    for i in range(len(positions)):
        for j in range(1, k):
            d = positions[indices[i, j]] - positions[i]
            all_vectors.append(d)
    all_vectors = np.array(all_vectors)

    # Cluster via 2D histogram
    max_range = med_nn * (n_neighbors ** 0.5) * 1.5
    bin_size = distance_tolerance
    n_bins = max(int(2 * max_range / bin_size), 10)
    bins = np.linspace(-max_range, max_range, n_bins + 1)

    hist, xedges, yedges = np.histogram2d(
        all_vectors[:, 0], all_vectors[:, 1], bins=[bins, bins],
    )

    # Label connected regions above threshold
    threshold_count = max(len(positions) * 0.15, 3)
    labeled, n_clusters = label(hist >= threshold_count)

    candidates = []
    for c in range(1, n_clusters + 1):
        mask = labeled == c
        ys, xs = np.nonzero(mask)
        # Weighted centroid of the cluster
        weights = hist[mask]
        cx = np.average((xedges[xs] + xedges[xs + 1]) / 2, weights=weights)
        cy = np.average((yedges[ys] + yedges[ys + 1]) / 2, weights=weights)
        length = math.hypot(cx, cy)
        if length > med_nn * 0.3:  # skip near-zero vectors
            candidates.append((cx, cy, length))

    # Sort by length
    candidates.sort(key=lambda v: v[2])

    # Remove parallel/antiparallel duplicates and integer multiples
    unique = []
    for cx, cy, length in candidates:
        is_dup = False
        for ux, uy, _ in unique:
            for n in range(-4, 6):
                if n == 0:
                    continue
                ref_x, ref_y = ux * n, uy * n
                if math.hypot(cx - ref_x, cy - ref_y) < distance_tolerance:
                    is_dup = True
                    break
            if is_dup:
                break
        if not is_dup:
            # Canonical direction: prefer positive first nonzero component
            if cx < -1e-6 or (abs(cx) < 1e-6 and cy < -1e-6):
                cx, cy = -cx, -cy
            unique.append((round(cx, 2), round(cy, 2),
                           math.hypot(cx, cy)))

    return [(vx, vy) for vx, vy, _ in unique]


def find_missing_atoms(
    positions: np.ndarray,
    zone_vector: tuple,
    fraction: float = 0.5,
    min_distance: float = 3.0,
) -> np.ndarray:
    """Predict atom positions at fractional lattice sites.

    Args:
        positions: (N, 2) array of detected atoms (x, y).
        zone_vector: (dx, dy) lattice vector from :func:`find_zone_axes`.
        fraction: Fractional position along the vector (0.5 = midpoint).
        min_distance: Minimum distance from existing atoms to keep.

    Returns:
        (M, 2) array of predicted new positions.
    """
    if len(positions) < 2:
        return np.empty((0, 2))

    zv = np.array(zone_vector, dtype=np.float64)
    zv_len = np.linalg.norm(zv)
    if zv_len < 1e-6:
        return np.empty((0, 2))

    tree = cKDTree(positions)
    tolerance = zv_len * 0.5  # neighbor must be within 50% of expected

    new_positions = []
    for p in positions:
        expected = p + zv
        dist, idx = tree.query(expected)
        if dist < tolerance:
            neighbor = positions[idx]
            interp = p * (1 - fraction) + neighbor * fraction
            new_positions.append(interp)

    if not new_positions:
        return np.empty((0, 2))

    new_arr = np.array(new_positions)

    # Deduplicate within new positions
    if len(new_arr) > 1:
        new_tree = cKDTree(new_arr)
        pairs = new_tree.query_pairs(min_distance * 0.5)
        to_remove = set()
        for i, j in pairs:
            to_remove.add(max(i, j))
        if to_remove:
            keep = [i for i in range(len(new_arr)) if i not in to_remove]
            new_arr = new_arr[keep]

    # Remove positions too close to existing atoms
    if len(new_arr) > 0:
        dists, _ = tree.query(new_arr)
        new_arr = new_arr[dists >= min_distance]

    return new_arr


def subtract_atoms(
    image: np.ndarray,
    positions: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    amplitude: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Subtract fitted 2D Gaussians from image.

    Args:
        image: 2D array.
        positions: (N, 2) atom positions (x, y) from :func:`detect_atoms`.
        sigma_x, sigma_y, amplitude: Per-atom Gaussian parameters.
        rotation: Per-atom rotation (radians). Default 0 for all.

    Returns:
        Residual image (clipped ≥ 0).
    """
    img = image.astype(np.float64)
    model = np.zeros_like(img)
    H, W = img.shape

    if rotation is None:
        rotation = np.zeros(len(positions))

    X, Y = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))

    for i, (x0, y0) in enumerate(positions):
        sx = max(float(sigma_x[i]), 0.5)
        sy = max(float(sigma_y[i]), 0.5)
        A = float(amplitude[i])
        theta = float(rotation[i])

        r = int(5 * max(sx, sy))
        r0 = max(0, int(y0) - r)
        r1 = min(H, int(y0) + r + 1)
        c0 = max(0, int(x0) - r)
        c1 = min(W, int(x0) + r + 1)

        if r1 <= r0 or c1 <= c0:
            continue

        xc = X[r0:r1, c0:c1]
        yc = Y[r0:r1, c0:c1]

        # Evaluate normalised rotated 2D Gaussian (no offset)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        sx2 = sx ** 2
        sy2 = sy ** 2
        a = cos_t**2 / (2*sx2) + sin_t**2 / (2*sy2)
        b = -math.sin(2*theta) / (4*sx2) + math.sin(2*theta) / (4*sy2)
        c = sin_t**2 / (2*sx2) + cos_t**2 / (2*sy2)

        dx = xc - x0
        dy = yc - y0
        norm = A / (2 * math.pi * sx * sy)
        model[r0:r1, c0:c1] += norm * np.exp(
            -(a * dx**2 + 2 * b * dx * dy + c * dy**2)
        )

    return np.clip(img - model, 0, None)


def refine_positions(
    image: np.ndarray,
    positions: np.ndarray,
    percent_to_nn: float = 0.40,
) -> dict:
    """Fit 2D Gaussians at known atom positions to get sub-pixel coordinates
    and per-atom sigma, amplitude, and rotation values.

    Use this to refine positions from any source (e.g. ``detect_atoms_dcnn``)
    so that ``subtract_atoms`` has the Gaussian parameters it needs.

    Args:
        image: 2D grayscale array.
        positions: (N, 2) array of atom positions as (x, y) where x=col, y=row.
        percent_to_nn: Gaussian mask radius as fraction of NN distance.

    Returns:
        dict with keys ``positions`` (N,2 refined x,y), ``sigma_x``,
        ``sigma_y``, ``amplitude``, ``rotation`` (all N arrays).
    """
    positions = np.array(positions, dtype=np.float64).copy()
    if len(positions) < 2:
        return {
            "positions": positions,
            "sigma_x": None,
            "sigma_y": None,
            "amplitude": None,
            "rotation": None,
        }

    img = image.astype(np.float64)
    H, W = img.shape

    tree = cKDTree(positions)
    nn_dists, _ = tree.query(positions, k=2)
    nn_dist = nn_dists[:, 1]
    med_sep = float(np.median(nn_dist))

    sigma_x_arr = np.full(len(positions), med_sep / 4.0)
    sigma_y_arr = np.full(len(positions), med_sep / 4.0)
    amplitude_arr = np.zeros(len(positions))
    rotation_arr = np.zeros(len(positions))

    for i, (x0, y0) in enumerate(positions):
        mask_r = max(int(nn_dist[i] * percent_to_nn), 2)
        fitted = False
        ptn = percent_to_nn
        for attempt in range(10):
            r = max(int(nn_dist[i] * ptn), 2)
            r0 = max(0, int(y0) - r)
            r1 = min(H, int(y0) + r + 1)
            c0 = max(0, int(x0) - r)
            c1 = min(W, int(x0) + r + 1)

            crop = img[r0:r1, c0:c1].copy()
            if crop.size < 7:
                ptn *= 0.95
                continue

            cy_local = int(y0) - r0
            cx_local = int(x0) - c0
            yy_m, xx_m = np.ogrid[0:crop.shape[0], 0:crop.shape[1]]
            circ_mask = ((xx_m - cx_local) ** 2 + (yy_m - cy_local) ** 2) <= r ** 2
            crop = crop * circ_mask

            mask_pixels = crop[circ_mask]
            if mask_pixels.size > 0:
                n_low = max(int(0.03 * mask_pixels.size), 1)
                bg_val = float(np.median(np.sort(mask_pixels.ravel())[:n_low]))
                crop = crop - bg_val
                crop[crop < 0] = 0.0

            yy, xx = np.mgrid[r0:r1, c0:c1]
            coords = (xx.ravel().astype(np.float64),
                      yy.ravel().astype(np.float64))

            n_high = max(int(0.03 * mask_pixels.size), 1)
            peak_val = float(np.median(np.sort(mask_pixels.ravel())[-n_high:]))
            s0 = med_sep / 4.0

            p0 = [x0, y0, peak_val * 2 * math.pi * s0 * s0,
                  s0, s0, 0.01, 0.0]
            bounds_lo = [c0, r0, 0, 0.5, 0.5, -math.pi, -np.inf]
            bounds_hi = [c1, r1, np.inf, r * 2, r * 2, math.pi, np.inf]

            try:
                popt, _ = curve_fit(
                    _gaussian2d, coords, crop.ravel().astype(np.float64),
                    p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=2000,
                )
                fx, fy, fA, fsx, fsy, ftheta, foff = popt
                if abs(fx - x0) > r or abs(fy - y0) > r:
                    raise ValueError("center outside mask")
                if fA < 0:
                    raise ValueError("negative amplitude")
                if max(fsx, fsy) / max(min(fsx, fsy), 0.1) > 4:
                    raise ValueError("sigma ratio > 4")

                positions[i] = [fx, fy]
                sigma_x_arr[i] = fsx
                sigma_y_arr[i] = fsy
                amplitude_arr[i] = fA
                rotation_arr[i] = ftheta % math.pi
                fitted = True
                break
            except Exception:
                ptn *= 0.95

        if not fitted:
            crop = img[max(0, int(y0) - mask_r):min(H, int(y0) + mask_r + 1),
                       max(0, int(x0) - mask_r):min(W, int(x0) + mask_r + 1)]
            if crop.sum() > 0:
                yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
                total = crop.sum()
                cx = (xx * crop).sum() / total + max(0, int(x0) - mask_r)
                cy = (yy * crop).sum() / total + max(0, int(y0) - mask_r)
                positions[i] = [cx, cy]
            amplitude_arr[i] = img[int(round(y0)), int(round(x0))]

    return {
        "positions": positions,
        "sigma_x": sigma_x_arr,
        "sigma_y": sigma_y_arr,
        "amplitude": amplitude_arr,
        "rotation": rotation_arr,
    }


# ---------------------------------------------------------------------------
# DCNN-based detection
# ---------------------------------------------------------------------------

_cached_model_dir: Optional[str] = None


def detect_atoms_dcnn(
    image: np.ndarray,
    fov_nm: float,
    model_dir: Optional[str] = None,
    target_pixel_size: float = 0.25,
    threshold: float = 0.8,
    refine: bool = True,
) -> dict:
    """Detect atom columns using the AtomNet3 DCNN ensemble.

    Wraps the existing AtomAI-based pipeline (rescale, ensemble predict,
    coordinate transform) into the same return format as :func:`detect_atoms`
    so downstream tools work unchanged.

    Args:
        image: 2D grayscale array.
        fov_nm: Field of view in **nanometers** (from metadata or calibration).
        model_dir: Path to directory containing ``atomnet3*.tar`` model files.
            If *None*, models are auto-discovered or downloaded on first call.
        target_pixel_size: Target pixel size in **ÅNGSTRÖMS (Å), NOT nm** — the
            only parameter on this tool in Å; ``fov_nm`` here and
            ``pixel_size_nm`` on the other atomic_stem tools are in nm. Default
            0.25 Å (= 0.025 nm/px). The image is resampled to this size before
            inference: ``resampled_px = fov_nm * 10 / target_pixel_size``. Keep
            it within ~1–2× of the image's native pixel size in Å
            (= ``fov_nm * 10 / image_width_px``); a typical good range is
            0.15–0.30 Å. WARNING: passing an nm value here (e.g. 0.02 instead of
            0.20) is ~10× too small, over-resamples to a tiny grid, and collapses
            detection to ~0 columns — if detection collapses, suspect a unit
            error first. Lower (finer) to recover a missed dim sublattice; raise
            (coarser) if columns split/duplicate.
        threshold: Detection confidence threshold (0-1). Default 0.8.
        refine: Sub-pixel Gaussian refinement on detected peaks.

    Returns:
        dict with keys ``positions`` (N,2 as x,y in original image pixels),
        ``sigma_x``, ``sigma_y``, ``amplitude``, ``rotation`` (all *None*),
        and ``heatmap`` (2D probability map in original image space).
    """
    import logging

    try:
        from .atomic_stem import rescale_for_model, predict_with_ensemble
        from scilink.skills._shared.atomistic_model_manager import get_or_download_atomistic_model
    except ImportError as exc:
        raise ImportError(
            "detect_atoms_dcnn requires atomai and opencv-python. "
            "Install with: pip install atomai opencv-python"
        ) from exc

    logger = logging.getLogger(__name__)

    # --- Resolve model directory ---
    global _cached_model_dir
    if model_dir is not None:
        resolved_dir = model_dir
    elif _cached_model_dir is not None:
        resolved_dir = _cached_model_dir
    else:
        resolved_dir = get_or_download_atomistic_model({}, logger)
        if resolved_dir is None:
            raise RuntimeError(
                "Could not locate or download AtomNet3 models. "
                "Pass model_dir explicitly or ensure internet access."
            )
        _cached_model_dir = resolved_dir

    # --- Rescale image to model's expected pixel size ---
    rescaled_image, scale_factor, _ = rescale_for_model(
        image, fov_nm, target_pixel_size_A=target_pixel_size
    )
    # Per-axis scale factors for non-square images
    target_px = rescaled_image.shape[0]
    sf_row = target_px / image.shape[0]
    sf_col = target_px / image.shape[1]

    # --- Run DCNN ensemble ---
    heatmap_rescaled, coords = predict_with_ensemble(
        dir_path=resolved_dir,
        image=rescaled_image,
        logger=logger,
        thresh=threshold,
        refine=refine,
    )

    # Resize heatmap back to original image dimensions
    import cv2
    heatmap = cv2.resize(
        heatmap_rescaled, (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    if coords is None or len(coords) == 0:
        result = _empty_result()
        result["heatmap"] = heatmap
        return result

    # --- Transform coordinates to original image space ---
    # predict_with_ensemble returns (row, col, class) in rescaled space
    rows = coords[:, 0] / sf_row
    cols = coords[:, 1] / sf_col
    # Flip to (x, y) = (col, row) to match detect_atoms convention
    positions = np.column_stack([cols, rows])

    return {
        "positions": positions,
        "sigma_x": None,
        "sigma_y": None,
        "amplitude": None,
        "rotation": None,
        "heatmap": heatmap,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result():
    return {
        "positions": np.empty((0, 2)),
        "sigma_x": None,
        "sigma_y": None,
        "amplitude": None,
        "rotation": None,
    }


def local_env_gmm(
    image,
    positions,
    n_components=4,
    window_size=32,
    covariance="diag",
    random_state=1,
):
    """GMM clustering on local image patches around detected atomic positions.

    Crops a ``(window_size, window_size)`` window around each position,
    flattens each crop into a feature vector, and clusters with a Gaussian
    Mixture Model. Captures the full local neighborhood (neighboring column
    arrangement, not just peak intensity) — the right tool for sublattice
    separation in complex structures where intensity alone is ambiguous.

    Args:
        image: 2D grayscale array.
        positions: (N, 2) array of atom positions as (x, y) — the format
            returned by ``detect_atoms`` / ``detect_atoms_dcnn``.
        n_components: number of GMM clusters (default 4).
        window_size: side length of the local patch in pixels (default 32).
            Should be roughly the lattice parameter so each patch contains
            the central atom plus its immediate neighbors.
        covariance: GMM covariance type — ``'diag'`` (default), ``'full'``,
            ``'tied'``, or ``'spherical'``. ``'diag'`` is the safe default
            for high-dimensional patch features (full covariance overfits
            when ``window_size`` × ``window_size`` is large relative to
            the atom count).
        random_state: RNG seed passed through to the underlying GMM
            (default 1). Fixing this makes cluster assignments reproducible
            across runs on the same input.

    Returns:
        dict with:
        - ``centroids``: (n_components, window_size, window_size) array,
          one average local-environment image per cluster.
        - ``positions``: (M, 2) ``(x, y)`` positions of the atoms that were
          successfully classified (atoms whose patches fell outside the
          image border are dropped by ``imlocal``).
        - ``classes``: (M,) cluster id (0-indexed) per surviving atom.
    """
    try:
        import atomai as aoi
    except ImportError as exc:
        raise ImportError(
            "local_env_gmm requires atomai. Install with: pip install atomai"
        ) from exc

    pos_xy = np.asarray(positions, dtype=float)
    if pos_xy.ndim != 2 or pos_xy.shape[1] not in (2, 3):
        raise ValueError(
            f"positions must be (N,2) or (N,3); got shape {pos_xy.shape}"
        )
    if pos_xy.shape[1] == 3:
        pos_xy = pos_xy[:, :2]

    # ``imlocal`` expects (row, col, class) per atom; swap to match.
    coords_rcc = np.column_stack([
        pos_xy[:, 1],
        pos_xy[:, 0],
        np.zeros(len(pos_xy)),
    ])

    img = np.asarray(image)
    if img.ndim == 2:
        expdata = img[None, ..., None]
    elif img.ndim == 3 and img.shape[-1] == 1:
        expdata = img[None, ...]
    else:
        raise ValueError(
            f"image must be 2D (or 2D + trailing single channel); "
            f"got shape {img.shape}"
        )

    imstack = aoi.stat.imlocal(
        expdata, {0: coords_rcc}, window_size=int(window_size)
    )
    centroids, _, coords_class = imstack.gmm(
        int(n_components),
        covariance=covariance,
        random_state=int(random_state),
    )

    # Strip any trailing channel dim from centroids: atomai returns
    # (k, win, win, 1) for single-channel input.
    centroids = np.asarray(centroids)
    if centroids.ndim == 4 and centroids.shape[-1] == 1:
        centroids = centroids[..., 0]

    if coords_class is None or len(coords_class) == 0:
        return {
            "centroids": centroids,
            "positions": np.empty((0, 2)),
            "classes": np.empty(0, dtype=int),
        }

    # Convert imlocal's 1-indexed classes to 0-indexed and re-emit
    # positions in (x, y) order to match the rest of this module.
    coords_class = np.asarray(coords_class, dtype=float).copy()
    coords_class[:, 2] = coords_class[:, 2] - 1
    out_positions = np.column_stack([coords_class[:, 1], coords_class[:, 0]])
    return {
        "centroids": centroids,
        "positions": out_positions,
        "classes": coords_class[:, 2].astype(int),
    }


def _sublattice_vacancies(cp, med_nn, margin, shape, enclosure,
                          image=None, occupied_intensity=None):
    """Interior sites of one sublattice with no detected column (vacancies).

    Data-driven and lattice-agnostic (no global indexing, so no drift over
    large or strained fields): for every atom, reflect each of its nearest
    neighbours to the opposite side (``2p - q``) — that opposite site is where
    the lattice "expects" another atom. A reflected site that is (a) empty,
    (b) interior, and (c) enclosed by at least ``enclosure`` detected columns
    within ~1 NN is a vacancy candidate. Works for square, hexagonal, and
    oblique sublattices alike.

    Critical disambiguation (when ``image``/``occupied_intensity`` are given):
    a true vacancy has *no atom*, so the image there sits near the
    inter-column background; a mere *detection miss* still has a real column
    (full intensity) at the site. Candidates whose image intensity is a sizable
    fraction of a real column are rejected as missed detections, not vacancies
    — so vacancy counts do not inflate when detection under-finds a dim
    sublattice.
    """
    H, W = shape
    cp = np.asarray(cp, float)
    if len(cp) < 12:
        return []
    tree = cKDTree(cp)
    # Primitive vectors of THIS sublattice, from the most frequent short
    # neighbour displacements (a high-recurrence displacement is a true lattice
    # translation). This carries the correct, possibly anisotropic per-direction
    # lengths — square, rectangular (a≠b), hexagonal and oblique alike — and is
    # more reliable for a single sublattice than generic zone-axis detection.
    kk = min(7, len(cp))
    dists0, idxs0 = tree.query(cp, k=kk)
    short = float(np.median(dists0[:, 1]))
    binsz = 0.3 * short
    counts = {}
    for i in range(len(cp)):
        for j in range(1, kk):
            d = cp[idxs0[i, j]] - cp[i]
            key = (int(round(d[0] / binsz)), int(round(d[1] / binsz)))
            if key == (0, 0):
                continue
            counts.setdefault(key, []).append(d)
    # keep displacements recurring across a meaningful fraction of atoms
    vecs = sorted((np.mean(v, 0) for k, v in counts.items() if len(v) > 0.2 * len(cp)),
                  key=lambda u: np.hypot(*u))
    if len(vecs) < 2:
        return []
    v1 = vecs[0]
    v2 = None
    for v in vecs[1:]:
        if abs(v1[0] * v[1] - v1[1] * v[0]) > 0.2 * np.linalg.norm(v1) * np.linalg.norm(v):
            v2 = v
            break
    if v2 is None:
        return []
    Lmin = min(np.linalg.norm(v1), np.linalg.norm(v2))
    Lmax = max(np.linalg.norm(v1), np.linalg.norm(v2))
    empty_tol = 0.35 * Lmin
    # Every lattice site sits at p ± v1 or p ± v2 from a neighbour, so these
    # offsets generate all candidate sites (including around a vacancy).
    cand = []
    for p in cp:
        for off in (v1, -v1, v2, -v2):
            s = p + off
            if tree.query(s)[0] > empty_tol:  # genuinely empty
                cand.append(s)
    if not cand:
        return []
    cand = np.array(cand)
    bond_max = 1.2 * Lmax
    # Merge votes for the same empty site, then keep only interior, enclosed ones.
    ctree = cKDTree(cand)
    used = np.zeros(len(cand), bool)
    vac = []
    for i in range(len(cand)):
        if used[i]:
            continue
        grp = ctree.query_ball_point(cand[i], empty_tol)
        used[list(grp)] = True
        s = cand[list(grp)].mean(0)
        if not (margin < s[0] < W - margin and margin < s[1] < H - margin):
            continue
        if tree.query(s)[0] <= empty_tol:  # an atom is actually there
            continue
        if len(tree.query_ball_point(s, bond_max)) < enclosure:  # not enclosed
            continue
        if image is not None and occupied_intensity:
            # reject missed detections: a real (undetected) column still has
            # near-full intensity here; a true vacancy sits near background.
            xi, yi = int(round(s[0])), int(round(s[1]))
            rr = max(1, int(round(0.15 * Lmin)))
            patch = image[max(0, yi - rr):yi + rr + 1, max(0, xi - rr):xi + rr + 1]
            if patch.size and float(patch.max()) > 0.6 * occupied_intensity:
                continue
        vac.append((float(s[0]), float(s[1])))
    return vac


def type_sublattice_defects(
    image,
    positions,
    n_sublattices=2,
    window_size=None,
    pixel_size_nm=None,
    dopant_sigma=3.5,
    edge_margin_px=None,
    vacancy_enclosure=3,
    random_state=1,
):
    """Per-sublattice point-defect typing on a multi-sublattice lattice.

    Encapsulates the chain that an LLM tends to get wrong when it improvises:
    (1) separate sublattices by **local environment** (``local_env_gmm``) —
    NOT by raw column intensity, because a substitutional dopant is itself an
    anomalous-intensity column, so an intensity split misfiles it into the
    wrong sublattice and hides it; (2) per sublattice, flag **vacancies**
    (interior ideal-lattice sites with no detected column) and **dopants**
    (amplitude outliers *within* that sublattice). Border columns are excluded
    (unreliable fits). Run only after BOTH sublattices are resolved by
    detection (confirm via NN at the inter-sublattice spacing).

    Args:
        image: 2D grayscale array — pass it RAW (un-normalized); intensity
            carries the Z-contrast that distinguishes species and dopants.
        positions: (N, 2) detected column positions (x, y).
        n_sublattices: number of sublattices to separate (default 2).
        window_size: local-environment patch side in px (default ~1.5× median
            NN ≈ one lattice parameter). Larger captures more neighbourhood.
        pixel_size_nm: nm/px; if given, defect coordinates are also in nm.
        dopant_sigma: robust (MAD) z-score threshold for an intensity outlier
            within a sublattice. LOWER (e.g. 2.5) to catch subtler
            substitutions, RAISE (e.g. 5) if noise produces false dopants.
        edge_margin_px: exclude defects within this many px of the border
            (default 1.5× median NN). RAISE if border fits are unreliable,
            LOWER to include near-edge defects.
        vacancy_enclosure: how many of a candidate empty site's 4 lattice
            neighbours must be occupied for it to count (default 3 of 4).
            RAISE to 4 for only fully-enclosed vacancies (fewer false
            positives at domain edges), LOWER to 2 near boundaries.
        random_state: seed for the GMM separation.

    Returns:
        dict with 'figure_bytes' (PNG: sublattices colour-coded + vacancies +
        dopants), 'metrics' (per-sublattice counts, separation_score,
        vacancies, dopants — each with x/y[/x_nm/y_nm], sublattice, and for
        dopants a signed z and a lighter/heavier-substitution label), and
        'flags'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    img = np.asarray(image, float)
    if img.ndim != 2:
        img = img[..., 0] if (img.ndim == 3 and img.shape[-1] <= 4) else img.reshape(img.shape[-2:])
    H, W = img.shape
    pos = np.asarray(positions, float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"positions must be (N,2); got {pos.shape}")
    pos = pos[:, :2]
    flags = []
    if len(pos) < 20:
        return {"figure_bytes": None,
                "metrics": {"error": "too few columns for sublattice typing"},
                "flags": ["insufficient_columns"]}

    med_nn = float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]))
    win = int(window_size) if window_size else max(8, int(round(1.5 * med_nn)))
    margin = float(edge_margin_px) if edge_margin_px is not None else 1.5 * med_nn

    # (1) separation by LOCAL ENVIRONMENT (not raw intensity)
    sep = local_env_gmm(img, pos, n_components=int(n_sublattices),
                        window_size=win, random_state=int(random_state))
    spos, scls = sep["positions"], sep["classes"]
    if len(spos) < 12:
        return {"figure_bytes": None,
                "metrics": {"error": "sublattice separation produced too few classified columns"},
                "flags": ["separation_failed"]}

    r = max(1, int(round(med_nn * 0.2)))
    def _peak(p):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].max())
    inten = np.array([_peak(p) for p in spos])

    present = [c for c in range(int(n_sublattices)) if np.any(scls == c)]
    means = {c: float(inten[scls == c].mean()) for c in present}
    order = sorted(present, key=lambda c: -means[c])  # brightest first
    pooled = np.mean([inten[scls == c].std() for c in present]) + 1e-9
    sep_score = (max(means.values()) - min(means.values())) / pooled if len(present) > 1 else 0.0
    if sep_score < 2.0:
        flags.append("low_separation_score")

    def _interior(p):
        return margin < p[0] < W - margin and margin < p[1] < H - margin

    sublattices, all_vac, all_dop = [], [], []
    for rank, c in enumerate(order):
        cp, ci = spos[scls == c], inten[scls == c]
        # Gradient-robust intensity: ratio of each column to the median of its
        # OWN-sublattice neighbours, so a slow illumination/thickness gradient
        # across the field does not masquerade as dopants. A dopant is then a
        # column anomalous vs its LOCAL same-species surroundings.
        if len(cp) >= 8:
            kk = min(13, len(cp))
            nb_idx = cKDTree(cp).query(cp, k=kk)[1]
            local_ref = np.array([np.median(ci[nb_idx[t, 1:]]) for t in range(len(cp))])
            metric = ci / (local_ref + 1e-9)
        else:
            metric = ci
        med = np.median(metric); mad = np.median(np.abs(metric - med)) * 1.4826 + 1e-9
        z = (metric - med) / mad
        dops = []
        _sel = np.abs(z) > dopant_sigma
        for p, zz, mv in zip(cp[_sel], z[_sel], metric[_sel]):
            if not _interior(p):
                continue
            # On a very UNIFORM sublattice the MAD is tiny, so a real, moderately
            # off-intensity column (e.g. a substitution 40% dimmer/brighter than its
            # neighbours) scores an absurd |z| (~25) and gets wrongly dismissed
            # downstream as "physically impossible". Report a sanity-CAPPED z for the
            # outlier flag, plus the raw intensity_ratio (this column / median of its
            # same-sublattice neighbours) as the PHYSICAL descriptor — a ratio well
            # above background (~vacancy) but below 1 is a genuine lighter substitution.
            d = {"x": float(p[0]), "y": float(p[1]),
                 "z": float(np.clip(zz, -10.0, 10.0)),
                 "intensity_ratio": round(float(mv), 3),
                 "kind": "lighter-substitution" if zz < 0 else "heavier-substitution",
                 "sublattice": rank}
            if pixel_size_nm:
                d["x_nm"], d["y_nm"] = float(p[0] * pixel_size_nm), float(p[1] * pixel_size_nm)
            dops.append(d)
        vacs = []
        for vx, vy in _sublattice_vacancies(cp, med_nn, margin, (H, W),
                                            int(vacancy_enclosure),
                                            image=img, occupied_intensity=float(np.median(ci))):
            v = {"x": vx, "y": vy, "sublattice": rank}
            if pixel_size_nm:
                v["x_nm"], v["y_nm"] = float(vx * pixel_size_nm), float(vy * pixel_size_nm)
            vacs.append(v)
        all_dop += dops; all_vac += vacs
        sublattices.append({"rank": rank, "n_columns": int(len(cp)),
                            "mean_intensity": means[c],
                            "n_vacancies": len(vacs), "n_dopants": len(dops)})

    # Superstructure / ordered-phase guard. This tool assumes a COMPLETE
    # sublattice with SPARSE, random point defects. When the flagged defect
    # fraction is large, the columns are more likely an ordered phase (ordered
    # vacancies/dopants, charge ordering) than a population of independent
    # point defects — that regime is a superstructure to MAP, not point defects
    # to count (use fourier_reflection_map on the satellite reflection).
    defect_frac = (len(all_vac) + len(all_dop)) / max(1, len(spos))
    if defect_frac > 0.15:
        flags.append("high_defect_fraction_possible_superstructure")
    # A large sublattice-population imbalance is the other superstructure tell
    # (e.g. ordered vacancies that leave a self-consistent sparser lattice the
    # geometry alone cannot see as vacancies): flag it for a reflection check.
    if len(sublattices) >= 2:
        cnts = sorted(s["n_columns"] for s in sublattices)
        if cnts[0] > 0 and cnts[-1] / cnts[0] > 1.5:
            flags.append("sublattice_population_imbalance_check_superstructure")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for a in ax:
        a.imshow(img, cmap="gray"); a.axis("off")
    ax[0].set_title("Sublattices (by local environment)")
    for rank, c in enumerate(order):
        cp = spos[scls == c]
        ax[0].scatter(cp[:, 0], cp[:, 1], s=8, color=colors[rank],
                      label=f"sublattice {rank} (I~{means[c]:.2f})")
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax[1].set_title(f"Defects: {len(all_vac)} vacancies (□), {len(all_dop)} dopants (○)")
    for v in all_vac:
        ax[1].scatter(v["x"], v["y"], s=90, facecolors="none", edgecolors="cyan", marker="s", linewidths=1.6)
    for d in all_dop:
        col = "lime" if d["z"] < 0 else "magenta"
        ax[1].scatter(d["x"], d["y"], s=90, facecolors="none", edgecolors=col, marker="o", linewidths=1.6)
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)

    return {
        "figure_bytes": buf.getvalue(),
        "metrics": {
            "n_columns_classified": int(len(spos)),
            "separation_score": float(sep_score),
            "sublattices": sublattices,
            "n_vacancies": len(all_vac),
            "n_dopants": len(all_dop),
            "vacancies": all_vac,
            "dopants": all_dop,
            "median_nn_px": med_nn,
        },
        "flags": flags,
    }


def map_polarization(image, positions, displaced="auto", n_cage=4,
                     pixel_size_nm=None, edge_margin_px=None,
                     max_offset_frac=0.45, magnitude_outlier_k=6.0,
                     max_magnitude_nm=0.12, domain_coherence_floor=0.4,
                     max_wall_fraction=0.15, local_coherence_floor=0.82,
                     random_state=1):
    """Per-unit-cell ferroelectric polarization / cation off-centering map.

    For a two-sublattice (ABO3-type / interpenetrating) lattice where BOTH
    sublattices are already resolved by detection, this measures the polar
    distortion: the displacement of the "displaced" cation sublattice from the
    centrosymmetric position defined by the surrounding "reference cage"
    sublattice. That displacement field is the ferroelectric order parameter —
    its direction maps domains, its discontinuities map domain walls.

    Method (lattice-agnostic, no material constants): refine positions to
    sub-pixel (polarization IS a sub-pixel displacement), split the columns
    into two sublattices by intensity (2-component GMM — robust, not a fixed
    threshold), pick the displaced sublattice, and for each displaced atom take
    the centroid of its ``n_cage`` nearest reference-sublattice neighbours as
    the centrosymmetric reference. ``P = reference_centroid - displaced_atom``.

    Args:
        image: 2D grayscale HAADF array (RAW — intensity separates the species).
        positions: (N, 2) detected columns (x, y) with BOTH sublattices
            resolved (confirm NN sits at the inter-sublattice spacing first).
        displaced: which sublattice is the off-centering one — ``'auto'``
            (default = the dimmer sublattice, the usual lighter B-cation
            convention), ``'dim'``, or ``'bright'`` (when the off-centering
            cation is the heavier/brighter one). Only flips the reference
            choice / overall sign; the domain structure is unchanged.
        n_cage: number of reference neighbours forming the centrosymmetric
            cage (default 4 for a perovskite [100] projection; raise to match
            the projected coordination of other structures/zone axes).
        pixel_size_nm: nm/px; if given, magnitude is also reported in nm.
        edge_margin_px: drop cells within this of the border (default 1.5×NN).
        max_offset_frac: a displaced cation is accepted only if it sits within
            this fraction of the reference-cage NN spacing from the cage centre
            (default 0.45). This is the robustness knob against extra/spurious
            columns (over-detection or a real third faint population): RAISE
            toward ~0.6 if genuine large polar displacements are being dropped,
            LOWER toward ~0.3 to be stricter when extra columns inflate the
            field. The scale is the *reference* sublattice NN, so it is not
            corrupted by extra dim columns.
        random_state: GMM seed.

    Returns: dict with 'figure_bytes' (PNG: polarization quiver + direction-
        domain map + magnitude map), and 'metrics' (n_cells, median magnitude
        px/nm, per-cell 'polarization' vectors, 'xy', 'magnitude', 'angle',
        'direction_coherence' ~1 for a smooth/domain-structured field and ~0 for
        salt-and-pepper noise — a magnitude-independent way to tell a real field
        from a degenerate one — but it assumes domains LARGER than the NN scale:
        a finely-twinned/antiferroelectric ordering on the ~1-2-cell scale reads
        as LOW coherence too, so low coherence means "random noise OR domains
        finer than ~3 cells" — for the latter check fourier_reflection_map for a
        superstructure satellite before concluding noise), and 'flags' (incl.
        low_direction_coherence_noise_OR_finer_than_NN_domains).
        The DIRECTION panel of the figure is a SMOOTHED domain map (locally
        averaged unit vectors; opacity = local coherence) so coherent domains/walls
        show as solid colour while noise washes out — do not judge coherence from a
        raw per-cell scatter. NOTE: the
        sign follows the displacement→reference convention above (polarization
        is often reported opposite to the cation displacement; flip via
        ``displaced`` if matching a specific reference). LIMITS: direction/domains
        are the robust output; magnitude is trustworthy only on CLEAN detection —
        over-detection inflates it (~20% on real data) and extreme over-detection
        returns ``{'error': 'no valid cells'}`` (fix ``target_pixel_size`` first
        when magnitude matters), and magnitude is also imprecise for very small
        |P| near the position-fit noise floor. Assumes a displacive perovskite-
        type structure (off-centering cation ~at the cage centre; the two cation
        sublattices are the intensity extremes — brightest cage, dimmest
        off-centering; a middle population is ignored and flagged); exotic
        orderings may violate this.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from sklearn.mixture import GaussianMixture

    img = np.asarray(image, float)
    if img.ndim != 2:
        img = img[..., 0] if (img.ndim == 3 and img.shape[-1] <= 4) else img.reshape(img.shape[-2:])
    H, W = img.shape
    pos = np.asarray(positions, float)[:, :2]
    if len(pos) < 20:
        return {"figure_bytes": None, "metrics": {"error": "too few columns"}, "flags": ["insufficient_columns"]}
    flags = []
    try:
        pos = np.asarray(refine_positions(img, pos)["positions"])[:, :2]  # sub-pixel for accurate |P|
    except Exception:
        flags.append("refine_failed_using_raw_positions")

    nn = float(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]))
    margin = float(edge_margin_px) if edge_margin_px is not None else 1.5 * nn
    rr = max(1, int(round(0.2 * nn)))
    I = np.array([img[max(0, int(round(p[1])) - rr):int(round(p[1])) + rr + 1,
                      max(0, int(round(p[0])) - rr):int(round(p[0])) + rr + 1].max() for p in pos])
    # Intensity-cluster the columns. Pick the number of populations by BIC over
    # {2, 3}: real data often has a THIRD faint population (extra/O columns, or
    # over-detection) beyond the two cation sublattices — forcing k=2 would lump
    # it into the reference and corrupt the cage. With k=3 we use only the
    # cleanest extremes (brightest = reference cage, dimmest = off-centering
    # cation) and simply ignore the middle population.
    Iv = I.reshape(-1, 1)
    cand = []
    for k in (2, 3):
        if len(pos) > k * 5:
            g = GaussianMixture(k, random_state=int(random_state)).fit(Iv)
            cand.append((g.bic(Iv), g))
    gm = min(cand, key=lambda t: t[0])[1]
    means = gm.means_.ravel(); lab = gm.predict(Iv); order = np.argsort(means)
    bright_lab, dim_lab = int(order[-1]), int(order[0])
    sep = abs(means[bright_lab] - means[dim_lab]) / (np.sqrt(gm.covariances_.ravel()).mean() + 1e-9)
    if sep < 1.5:
        flags.append("weak_sublattice_intensity_separation")
    if gm.n_components == 3:
        flags.append("third_intensity_population_present_using_extremes")
    if displaced == "bright":
        disp_lab, ref_lab = bright_lab, dim_lab
    else:  # 'auto' / 'dim' -> the dimmer (lighter) cation off-centers
        disp_lab, ref_lab = dim_lab, bright_lab
    disp = pos[lab == disp_lab]; ref = pos[lab == ref_lab]   # middle cluster (if any) ignored
    if len(disp) < 10 or len(ref) < 10:
        return {"figure_bytes": None, "metrics": {"error": "sublattice split failed",
                "n_displaced": int(len(disp)), "n_reference": int(len(ref))}, "flags": flags + ["split_failed"]}

    # Pair each off-centering cation to its reference cage and measure the offset.
    # Robustness to extra / spurious columns (over-detection, or a real third faint
    # population the 2-way split lumps in): anchor everything on the REFERENCE-cage
    # NN spacing (the cleanest scale, unaffected by extra dim columns), and accept a
    # displaced candidate only when it sits ~at the centrosymmetric centre of its
    # cage — a genuine off-centering cation is within max_offset_frac of a cage
    # centre, whereas an extra/spurious column elsewhere is not. This keeps real
    # signal (we do NOT delete detections) while rejecting non-cage columns.
    def _field(disp_s, ref_s):
        nnr = float(np.median(cKDTree(ref_s).query(ref_s, k=2)[0][:, 1]))
        tr = cKDTree(ref_s); Pl = []; XYl = []
        for d in disp_s:
            dd, ii = tr.query(d, k=int(n_cage))
            if dd.max() > 1.3 * nnr * np.sqrt(2):   # cage neighbours must be local to the ref lattice
                continue
            cen = ref_s[ii].mean(0)
            if np.linalg.norm(cen - d) > max_offset_frac * nnr:  # must sit ~at the cage centre
                continue
            if not (margin < d[0] < W - margin and margin < d[1] < H - margin):
                continue
            Pl.append(cen - d); XYl.append(d)
        return np.array(Pl), np.array(XYl), nnr

    # Spatial coherence (the noise guard): mean direction correlation to each strong
    # cell's nearest neighbours — ~1 for a smooth/domain-structured field, ~0 for
    # salt-and-pepper noise. Lets a verifier distinguish a real (even weak)
    # polarization field from a degenerate one independent of how the per-cell
    # figure renders, and drives the geometric-fallback decision below.
    def _coherence(Pl, XYl):
        if len(Pl) < 5:
            return float("nan")
        m = np.linalg.norm(Pl, axis=1); u = Pl / (m[:, None] + 1e-9)
        st = m > np.median(m) * 0.5
        if st.sum() <= 20:
            return float("nan")
        sx = XYl[st]; su = u[st]; kk = min(7, len(sx))
        jj = cKDTree(sx).query(sx, k=kk)[1]
        return float(np.median([np.mean(su[i] @ su[jj[i, 1:]].T) for i in range(len(sx))]))

    P, XY, nn_ref = _field(disp, ref)
    coherence = _coherence(P, XY)

    # Outcome-based geometric-separation fallback for weak intensity contrast
    # (near-identical-Z cations / weak HAADF contrast where the intensity GMM split
    # is unreliable): if the intensity-split field is incoherent or empty, retry
    # with a GEOMETRIC split by local environment (local_env_gmm) — the two
    # interpenetrating sublattices have distinct neighbour arrangements in
    # projection even when equally bright — and keep whichever split yields the more
    # coherent field. Outcome-based (not a brittle intensity-separation threshold)
    # because GMM separation on near-degenerate intensities is itself noise-
    # dependent. The reference/displaced choice is then arbitrary, but the field is
    # invariant to it up to a global sign (a convention anyway).
    if not (coherence == coherence) or coherence < 0.5 or len(P) < 5:
        try:
            win = max(8, int(round(1.5 * nn)))
            lg = local_env_gmm(img, pos, n_components=2, window_size=win,
                               random_state=int(random_state))
            gpos = np.asarray(lg["positions"], float)[:, :2]
            gcls = np.asarray(lg["classes"])
            dg = gpos[gcls == 0]; rg = gpos[gcls == 1]
            if len(dg) >= 10 and len(rg) >= 10:
                Pg, XYg, nng = _field(dg, rg)
                cg = _coherence(Pg, XYg)
                if (cg == cg) and len(Pg) >= 5 and (not (coherence == coherence) or cg > coherence):
                    P, XY, nn_ref, coherence = Pg, XYg, nng, cg
                    flags.append("geometric_separation_used_intensity_ambiguous")
        except Exception:
            flags.append("geometric_separation_attempt_failed")

    if len(P) < 5:
        return {"figure_bytes": None, "metrics": {"error": "no valid cells", "n_cells": int(len(P))}, "flags": flags}
    # Robust per-cell |P| outlier guard: a cell whose offset sits many MADs above
    # the field median is a position-fit failure / mis-paired A-B column, not a
    # real polar distortion — it shows up as an unphysical |P| (e.g. >>80 pm) that
    # corrupts the quiver, the |P| colorbar, and the domain field. Drop these cells
    # (median/MAD are robust, so a genuinely uniform large field is NOT clipped).
    _m0 = np.linalg.norm(P, axis=1)
    _med0 = float(np.median(_m0)); _mad0 = float(np.median(np.abs(_m0 - _med0))) + 1e-9
    _drop = _m0 > _med0 + magnitude_outlier_k * 1.4826 * _mad0           # relative (MAD) outliers
    # Absolute physical ceiling: the MAD guard is magnitude-relative, so on a
    # STRONG field (large median -> large MAD) a >physical tail survives. A
    # ferroelectric cation off-centering above ~max_magnitude_nm (default 120 pm,
    # well over the 40-80 pm common range) is almost surely a sublattice-pairing /
    # fit failure, not a real displacement. Needs pixel_size_nm to convert.
    n_over_ceiling = 0
    if pixel_size_nm and max_magnitude_nm:
        _over = _m0 > (float(max_magnitude_nm) / float(pixel_size_nm))
        n_over_ceiling = int(_over.sum())
        _drop = _drop | _over
    n_dropped = int(_drop.sum())
    if n_dropped and (len(P) - n_dropped) >= 5:
        P, XY = P[~_drop], XY[~_drop]
        flags.append(f"clipped_{n_dropped}_unphysical_magnitude_cells_likely_misfit")
    # Surface the ceiling SPECIFICALLY only when a non-trivial FRACTION is over it
    # (a systematic pairing/scale error worth the caller's attention) — a 1% tail is
    # already covered by the generic clip flag above, so don't cry wolf on it.
    if n_over_ceiling >= max(5, int(0.03 * len(_m0))):
        flags.append(f"{n_over_ceiling}_cells_above_physical_ceiling_{int(round(max_magnitude_nm * 1000))}pm_check_pairing_or_scale")
    mag = np.linalg.norm(P, axis=1); ang = (np.degrees(np.arctan2(P[:, 1], P[:, 0]))) % 360
    unit = P / (mag[:, None] + 1e-9)
    strong = mag > np.median(mag) * 0.5
    if coherence == coherence and coherence < 0.5:
        flags.append("low_direction_coherence_noise_OR_finer_than_NN_domains")

    # --- smoothed / coherence-gated direction-domain map ----------------------
    # Locally average the unit vectors on a coarse grid: the resultant angle is
    # the local domain direction (hue) and the resultant length is the LOCAL
    # coherence (used as opacity) — so coherent domains/walls show as solid
    # colour while incoherent (noise) regions wash out to the grey image beneath.
    import matplotlib.cm as cm
    G = 64
    gx = np.linspace(0, W, G); gy = np.linspace(0, H, G)
    GX, GY = np.meshgrid(gx, gy)
    gpts = np.column_stack([GX.ravel(), GY.ravel()])
    rad = 2.5 * nn_ref
    if strong.sum() >= 5:
        st = cKDTree(XY[strong]); su = unit[strong]
        nbr = st.query_ball_point(gpts, rad)
    else:
        nbr = [[] for _ in gpts]
    hue = np.zeros(len(gpts)); coh_local = np.zeros(len(gpts))
    for i, nb in enumerate(nbr):
        if len(nb) >= 3:
            v = su[nb].mean(0)
            hue[i] = (np.degrees(np.arctan2(v[1], v[0])) % 360) / 360.0
            coh_local[i] = min(1.0, np.hypot(v[0], v[1]))
    rgba = cm.hsv(hue.reshape(G, G)); rgba[..., 3] = coh_local.reshape(G, G)

    # Near-square 2x2 layout (renders large in the HTML report's grid cell — a
    # wide 1x3 strip collapses to an unreadable sliver there). Cells: quiver,
    # direction-domain map, |P| magnitude, and a direction colour-wheel key.
    fig, axes = plt.subplots(2, 2, figsize=(13, 12.5))
    ax = axes.ravel()
    # Show the dark HAADF image only FAINTLY as context, so the coloured
    # overlays (arrows / magnitude scatter) stay clearly visible on top of it.
    for a in (ax[0], ax[1], ax[2]):
        a.imshow(img, cmap="gray", alpha=0.5); a.axis("off"); a.set_aspect("equal")
    # Panel 0: arrows coloured by DIRECTION (hsv = always saturated, so they are
    # visible regardless of |P|, and the hue matches the direction-domain map in
    # panel 1); length ∝ |P|; thin dark edge for contrast on any background.
    if strong.sum() > 0:
        ang = (np.degrees(np.arctan2(P[strong, 1], P[strong, 0])) % 360) / 360.0
        ax[0].quiver(XY[strong, 0], XY[strong, 1], P[strong, 0], P[strong, 1], ang,
                     cmap="hsv", clim=(0.0, 1.0), scale=max(1e-6, np.median(mag)) * 40,
                     width=0.005, headwidth=4.0, edgecolor="k", linewidth=0.3)
    ax[0].set_title("polarization vectors (colour = direction, |P|>0.5·median)")
    ax[1].imshow(rgba, extent=[0, W, H, 0])
    ax[1].set_title(f"DIRECTION domains (smoothed; coherence={coherence:.2f})")
    s2 = ax[2].scatter(XY[:, 0], XY[:, 1], c=mag * (pixel_size_nm or 1), cmap="plasma", s=14,
                       edgecolors="white", linewidths=0.3,
                       vmax=np.percentile(mag, 98) * (pixel_size_nm or 1))
    ax[2].set_title("|P| magnitude" + (" (nm)" if pixel_size_nm else " (px)")); plt.colorbar(s2, ax=ax[2], fraction=0.046)
    # Panel 3: direction colour-wheel — the hue legend for panels 0 & 1 (which
    # had no colorbar). hue at angle theta == arrow/domain hue for direction theta.
    ax[3].remove()
    axk = fig.add_subplot(2, 2, 4, projection="polar")
    _tt = np.linspace(0, 2 * np.pi, 361); _rr = np.linspace(0.55, 1.0, 24)
    _TT, _RR = np.meshgrid(_tt, _rr)
    axk.pcolormesh(_TT, _RR, np.degrees(_TT), cmap="hsv", vmin=0, vmax=360, shading="auto")
    axk.set_yticks([]); axk.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axk.set_xticklabels(["0°", "90°", "180°", "270°"])
    axk.set_title("colour = polarization direction", fontsize=10, pad=14)
    buf = BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)

    # --- domain segmentation + wall localization (returned as DATA) -----------
    # The smoothed direction-domain map above is the gold-standard domain view;
    # return its structure as data too, so a caller reads domains/walls from the
    # tool instead of re-deriving them (e.g. quantizing per-cell angles into
    # fixed cardinal bins, which fragments the field). Domain directions are the
    # PEAKS of the smoothed-direction histogram (adaptive — NOT fixed 0/90/180/270
    # bins), so non-cardinal ferroelectric domains are captured.
    domains = []; n_domains = 0; domain_labels = None; wall_fraction = float("nan")
    local_coherence = float("nan")
    try:
        smv = np.full((len(XY), 2), np.nan)
        if strong.sum() >= 5:
            stree = cKDTree(XY[strong]); su_s = unit[strong]
            for i, nb in enumerate(stree.query_ball_point(XY, 2.5 * nn_ref)):
                if len(nb) >= 3:
                    smv[i] = su_s[nb].mean(0)            # per-cell smoothed direction
        valid_c = ~np.isnan(smv[:, 0]) & (np.hypot(smv[:, 0], smv[:, 1]) > 0.2)
        local_coherence = (float(np.median(np.hypot(smv[valid_c, 0], smv[valid_c, 1])))
                           if valid_c.sum() else float("nan"))
        if valid_c.sum() >= 10:
            th = np.degrees(np.arctan2(smv[valid_c, 1], smv[valid_c, 0])) % 360
            hist, edges = np.histogram(th, bins=36, range=(0, 360))
            hsm = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(3) / 3, "same")[3:-3]
            centers = (edges[:-1] + edges[1:]) / 2
            peaks = [k for k in range(36)
                     if hsm[k] >= 0.25 * hsm.max()
                     and hsm[k] >= hsm[(k - 1) % 36] and hsm[k] >= hsm[(k + 1) % 36]]
            dirs = []
            for k in sorted(peaks, key=lambda k: -hsm[k]):       # merge peaks <30 deg apart
                a = float(centers[k])
                if all(min(abs(a - d), 360 - abs(a - d)) > 30 for d in dirs):
                    dirs.append(a)
            dirs = dirs[:6] or [float(np.median(th))]
            lab = np.full(len(XY), -1, int); idxv = np.where(valid_c)[0]
            da = np.array(dirs); diff = np.abs(th[:, None] - da[None, :])
            lab[idxv] = np.argmin(np.minimum(diff, 360 - diff), axis=1)
            domain_labels = lab
            for di, d in enumerate(dirs):
                m = lab == di
                if m.sum() >= 5:
                    domains.append({"label": int(di), "n_cells": int(m.sum()),
                                    "mean_angle_deg": round(d, 1),
                                    "fraction": round(float(m.sum()) / max(1, valid_c.sum()), 3)})
            n_domains = len(domains)
            if n_domains >= 2:                               # wall cells = label change among spatial NN
                labv = lab[idxv]; kk = min(7, len(idxv))
                nn_i = cKDTree(XY[idxv]).query(XY[idxv], k=kk)[1]
                wall = np.array([np.any(labv[nn_i[i, 1:]] != labv[i]) for i in range(len(idxv))])
                wall_fraction = round(float(wall.mean()), 3)
            # Noise / speckle guard: a genuine multi-domain partition has CONTIGUOUS
            # domains separated by a thin wall (low wall_fraction); a noise-dominated
            # weak field (vectors at/below the position-fit noise floor) fragments
            # into spatially-interspersed speckle (high wall_fraction) or has globally
            # incoherent directions. Either way the multi-domain count is spurious —
            # collapse to ONE 'disordered' field rather than reporting fake domains.
            if n_domains >= 2 and (
                (coherence == coherence and coherence < domain_coherence_floor)      # globally incoherent => pure noise
                or (wall_fraction == wall_fraction and wall_fraction > max_wall_fraction
                    and local_coherence == local_coherence and local_coherence < local_coherence_floor)):
                # Collapse a fragmented partition (high wall_fraction) ONLY when it
                # is ALSO locally incoherent (low within-neighbourhood agreement) =
                # noise. A GENUINE multi-domain mosaic is fragmented but locally
                # COHERENT, so it survives regardless of domain count/size or crop
                # size — local_coherence is count/size-invariant, wall_fraction is
                # not. (wall_fraction alone would over-collapse real fine mosaics.)
                flags.append("field_noise_dominated_domains_unreliable")
                domains = [{"label": 0, "n_cells": int(valid_c.sum()), "mean_angle_deg": None,
                            "fraction": 1.0, "disordered": True}]
                n_domains = 1
                domain_labels = np.zeros(len(XY), int)
                wall_fraction = float("nan")
    except Exception:
        flags.append("domain_segmentation_failed")

    # Net (mean) displacement vs the per-cell magnitude: a registration / scan-
    # offset indicator. net_to_local_ratio « 1 → genuine multi-domain field (the
    # net cancels); » 1 → a uniform offset dominates (registration/scan drift),
    # so the absolute magnitude/direction is suspect even if the field looks
    # coherent. Lets a caller judge the offset instead of hedging about it.
    meanP = P.mean(0)
    net_to_local_ratio = float(np.hypot(meanP[0], meanP[1]) / (np.median(mag) + 1e-9))
    if net_to_local_ratio > 1.5:
        flags.append("offset_dominated_field_check_registration")

    return {
        "figure_bytes": buf.getvalue(),
        "metrics": {
            "n_cells": int(len(P)),
            "median_magnitude_px": float(np.median(mag)),
            "median_magnitude_nm": float(np.median(mag) * pixel_size_nm) if pixel_size_nm else None,
            "direction_coherence": coherence,   # ~1 coherent/domains, ~0 noise
            "local_coherence": local_coherence, # median resultant of the smoothed local direction; < domain_coherence_floor => noise-dominated, domain count unreliable
            "n_domains": int(n_domains),
            "domains": domains,                 # adaptive (peak-clustered) directions, NOT cardinal bins
            "wall_fraction": wall_fraction,
            "net_polarization_px": [float(meanP[0]), float(meanP[1])],
            "net_polarization_nm": ([float(meanP[0] * pixel_size_nm), float(meanP[1] * pixel_size_nm)]
                                    if pixel_size_nm else None),
            "net_to_local_ratio": net_to_local_ratio,
            "displaced_sublattice": displaced,
            "polarization": P.tolist(), "xy": XY.tolist(),
            "magnitude": mag.tolist(), "angle": ang.tolist(),
            "domain_label": domain_labels.tolist() if domain_labels is not None else None,
        },
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

from scilink.skills._shared._spec import ToolSpec

TOOL_SPECS = [
    ToolSpec(
        name="detect_atoms",
        description=(
            "Classical atom-column detection: peak detection plus optional 2D Gaussian "
            "refinement. Returns sub-pixel positions and per-atom Gaussian parameters."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import detect_atoms",
        signature=(
            "detect_atoms(image, separation, threshold_rel=0.02, refine=True, "
            "percent_to_nn=0.40, subtract_background=False, normalize_intensity=True) -> dict"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "Atomic-resolution STEM/HAADF images where atoms appear as bright peaks on a "
            "darker background and the approximate atom separation in pixels is known "
            "(from metadata or by measurement). A good general-purpose baseline across "
            "materials — no training-data dependence."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale array."},
            "separation": {
                "type": "int",
                "description": "Minimum atom spacing in pixels.",
            },
            "threshold_rel": {
                "type": "float",
                "description": "Relative peak threshold (default 0.02).",
            },
            "refine": {
                "type": "bool",
                "description": "Fit a 2D Gaussian per peak for sub-pixel precision (default True).",
            },
            "percent_to_nn": {
                "type": "float",
                "description": "Gaussian mask radius as fraction of nearest-neighbor distance (default 0.40).",
            },
            "subtract_background": {
                "type": "bool",
                "description": "Gaussian-blur background subtraction before peak finding (default False).",
            },
            "normalize_intensity": {
                "type": "bool",
                "description": "Normalize to 0-1 before peak finding (default True).",
            },
        },
        required=["image", "separation"],
        returns=(
            "dict with 'positions' (N,2 array of (x,y) sub-pixel coordinates), "
            "'sigma_x', 'sigma_y', 'amplitude', 'rotation' (each length-N arrays, or "
            "None when refine=False)."
        ),
    ),
    ToolSpec(
        name="detect_atoms_dcnn",
        description=(
            "AtomNet3 deep-CNN ensemble detection. Produces atom positions and a "
            "probability heatmap."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import detect_atoms_dcnn",
        signature=(
            "detect_atoms_dcnn(image, fov_nm, model_dir=None, "
            "target_pixel_size=0.25, threshold=0.8, refine=True) -> dict"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "Relatively clean atomic-resolution images. Known to work well on "
            "transition-metal oxides — simple perovskites, layered perovskites, and "
            "cuprate superconductors (e.g. YBCO, BSCCO) — and on graphene. Preferred "
            "over classical peak finding for these material systems. Requires the "
            "field of view in nanometers (from metadata/calibration). Requires the "
            "atomai package; models auto-download on first call."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale array."},
            "fov_nm": {
                "type": "float",
                "description": "Field of view in nanometers (from metadata or calibration).",
            },
            "model_dir": {
                "type": "str | None",
                "description": "Path to directory with atomnet3*.tar files. None auto-discovers/downloads.",
            },
            "target_pixel_size": {
                "type": "float",
                "description": "Target pixel size in ÅNGSTRÖMS (Å), NOT nm — the only Å param here (fov_nm and pixel_size_nm elsewhere are in nm); default 0.25 Å = 0.025 nm/px. Image is resampled to it before inference (resampled_px = fov_nm*10/target_pixel_size); keep within ~1–2× of native (= fov_nm*10/image_width_px), typically 0.15–0.30 Å. WARNING: passing an nm value (e.g. 0.02 not 0.20) is ~10× too small and COLLAPSES detection to ~0 columns — suspect a unit error first if detection collapses. LOWER (finer) to recover a missed dim sublattice; RAISE (coarser) if columns split/duplicate.",
            },
            "threshold": {
                "type": "float",
                "description": "Detection confidence 0-1 (default 0.8).",
            },
            "refine": {
                "type": "bool",
                "description": "Sub-pixel refinement on detected peaks (default True).",
            },
        },
        required=["image", "fov_nm"],
        returns=(
            "dict with 'positions' (N,2 array (x,y) in original image pixels), "
            "'heatmap' (2D probability map in original image space), and "
            "'sigma_x', 'sigma_y', 'amplitude', 'rotation' (None — call refine_positions "
            "to obtain Gaussian parameters)."
        ),
    ),
    ToolSpec(
        name="refine_positions",
        description=(
            "Fit 2D Gaussians at known atom positions to obtain sub-pixel coordinates "
            "and per-atom sigma, amplitude, and rotation values."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import refine_positions",
        signature="refine_positions(image, positions, percent_to_nn=0.40) -> dict",
        agents=["image_analysis"],
        when_to_use=(
            "After detect_atoms_dcnn (or any source that returns positions without "
            "Gaussian parameters), when downstream tools like subtract_atoms need "
            "sigma / amplitude per atom."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale array."},
            "positions": {
                "type": "ndarray",
                "description": "(N, 2) array of (x, y) atom positions.",
            },
            "percent_to_nn": {
                "type": "float",
                "description": "Gaussian mask radius as fraction of nearest-neighbor distance (default 0.40).",
            },
        },
        required=["image", "positions"],
        returns=(
            "dict with 'positions' (refined N,2 (x,y)), 'sigma_x', 'sigma_y', "
            "'amplitude', 'rotation' (each length-N arrays)."
        ),
    ),
    ToolSpec(
        name="find_zone_axes",
        description=(
            "Detect lattice translation vectors from a set of atom positions. Returns "
            "the unique shortest lattice vectors."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import find_zone_axes",
        signature="find_zone_axes(positions, n_neighbors=9, distance_tolerance=None) -> list",
        agents=["image_analysis"],
        when_to_use=(
            "Once atom positions are known (detect_atoms or detect_atoms_dcnn), to "
            "recover lattice periodicity and pass a zone vector to find_missing_atoms."
        ),
        parameters={
            "positions": {
                "type": "ndarray",
                "description": "(N, 2) array of atom positions (x, y).",
            },
            "n_neighbors": {
                "type": "int",
                "description": "Neighbors per atom to examine (default 9).",
            },
            "distance_tolerance": {
                "type": "float | None",
                "description": "Clustering tolerance in pixels. Default: median NN distance / 3.",
            },
        },
        required=["positions"],
        returns="List of (dx, dy) tuples — unique lattice vectors, shortest first.",
    ),
    ToolSpec(
        name="find_missing_atoms",
        description=(
            "Predict atom positions at fractional lattice sites along a zone vector "
            "(e.g. midpoints for a second sublattice)."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import find_missing_atoms",
        signature="find_missing_atoms(positions, zone_vector, fraction=0.5, min_distance=3.0) -> ndarray",
        agents=["image_analysis"],
        when_to_use=(
            "Multi-sublattice materials where a second (weaker) sublattice sits at "
            "fractional positions between detected atoms. Pair with subtract_atoms to "
            "reveal the weaker sublattice, then re-detect."
        ),
        parameters={
            "positions": {
                "type": "ndarray",
                "description": "(N, 2) array of detected atoms (x, y).",
            },
            "zone_vector": {
                "type": "tuple",
                "description": "(dx, dy) lattice vector from find_zone_axes.",
            },
            "fraction": {
                "type": "float",
                "description": "Fractional position along the vector (0.5 = midpoint).",
            },
            "min_distance": {
                "type": "float",
                "description": "Minimum distance from existing atoms (pixels, default 3.0).",
            },
        },
        required=["positions", "zone_vector"],
        returns="(M, 2) ndarray of predicted positions (x, y).",
    ),
    ToolSpec(
        name="subtract_atoms",
        description=(
            "Subtract fitted 2D Gaussians from an image to produce a residual that "
            "reveals weaker sublattices or features."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import subtract_atoms",
        signature=(
            "subtract_atoms(image, positions, sigma_x, sigma_y, amplitude, "
            "rotation=None) -> ndarray"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "Revealing weaker sublattices in multi-sublattice materials after the "
            "primary sublattice is fit with detect_atoms (or detect_atoms_dcnn + "
            "refine_positions). Feed the residual back into a detector for the second "
            "sublattice."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D array."},
            "positions": {
                "type": "ndarray",
                "description": "(N, 2) atom positions (x, y).",
            },
            "sigma_x": {"type": "ndarray", "description": "Per-atom sigma_x (length N)."},
            "sigma_y": {"type": "ndarray", "description": "Per-atom sigma_y (length N)."},
            "amplitude": {
                "type": "ndarray",
                "description": "Per-atom Gaussian amplitude (length N).",
            },
            "rotation": {
                "type": "ndarray | None",
                "description": "Per-atom rotation in radians. Default 0 for all.",
            },
        },
        required=["image", "positions", "sigma_x", "sigma_y", "amplitude"],
        returns="2D residual image (clipped to >= 0).",
    ),
    ToolSpec(
        name="local_env_gmm",
        description=(
            "GMM clustering on local image patches around detected atomic positions. "
            "Captures the full local neighborhood (neighboring column arrangement, not "
            "just peak intensity) for sublattice separation in complex structures where "
            "intensity alone is ambiguous."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import local_env_gmm",
        signature=(
            "local_env_gmm(image, positions, n_components=4, window_size=32, "
            "covariance='diag', random_state=1) -> dict"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "Sublattice separation in multi-component or complex layered structures "
            "(perovskites, layered oxides, cuprate superconductors) where columns of "
            "different species share similar intensities and intensity-based clustering "
            "alone is insufficient. Run after detection (detect_atoms or "
            "detect_atoms_dcnn). Choose ``window_size`` ≈ lattice parameter so each "
            "patch contains the central atom plus its immediate neighbors. Requires "
            "the atomai package."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale array."},
            "positions": {
                "type": "ndarray",
                "description": "(N, 2) atom positions as (x, y) from any detector.",
            },
            "n_components": {
                "type": "int",
                "description": "Number of GMM clusters (default 4).",
            },
            "window_size": {
                "type": "int",
                "description": (
                    "Side length of the local patch in pixels (default 32). "
                    "Set to ~lattice parameter."
                ),
            },
            "covariance": {
                "type": "str",
                "description": (
                    "GMM covariance type: 'diag' (default — safe for "
                    "high-dim patch features), 'full', 'tied', or "
                    "'spherical'."
                ),
            },
            "random_state": {
                "type": "int",
                "description": (
                    "RNG seed (default 1). Fixed seed makes cluster "
                    "assignments reproducible across runs on the same "
                    "input."
                ),
            },
        },
        required=["image", "positions"],
        returns=(
            "dict with 'centroids' ((n_components, window_size, window_size) array, "
            "average local-environment image per cluster), 'positions' ((M, 2) (x, y) "
            "of atoms successfully classified — may be smaller than N if patches were "
            "cropped at the image border), and 'classes' ((M,) 0-indexed cluster id)."
        ),
    ),
    ToolSpec(
        name="type_sublattice_defects",
        description=(
            "Per-sublattice point-defect typing for a multi-sublattice atomic lattice. "
            "Separates sublattices by LOCAL ENVIRONMENT (not raw intensity, which would "
            "hide a substitutional dopant), then flags vacancies (interior ideal-lattice "
            "sites with no detected column) and dopants (intensity outliers within a "
            "sublattice). Returns a figure and per-sublattice defect lists."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import type_sublattice_defects",
        signature=(
            "type_sublattice_defects(image, positions, n_sublattices=2, window_size=None, "
            "pixel_size_nm=None, dopant_sigma=3.5, edge_margin_px=None, "
            "vacancy_enclosure=3, random_state=1) -> dict"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "After BOTH sublattices of a multi-sublattice lattice are resolved by "
            "detection (perovskite ABO3, dichalcogenide MoS2-type, any structure with a "
            "basis) and the objective is per-sublattice point defects. Confirm both "
            "sublattices are resolved first (NN at the inter-sublattice spacing, not the "
            "unit-cell repeat). Do NOT separate sublattices by raw column intensity "
            "yourself — that misfiles a dim dopant into the dim sublattice and hides it; "
            "this tool separates by local environment. Lattice-agnostic (square, "
            "hexagonal, oblique). Assumes bright-column (HAADF) intensity convention. "
            "Requires the atomai package (used for the local-environment separation). "
            "LIMITS (read the 'flags'): (1) needs clean detection — over/under-"
            "detection fabricates or hides defects, so fix target_pixel_size first "
            "(NN at the physical column spacing). (2) Dopant typing and superstructure "
            "flagging are robust across lattice types; vacancy recall is high on "
            "hexagonal/rectangular lattices and for vacancy clusters, but PARTIAL on "
            "tightly-interpenetrating square lattices where the defective sublattice "
            "sits at dim interstitial sites enclosed by bright neighbours — cross-check "
            "vacancies with fft_defect_map (reciprocal-space, geometry-independent) and "
            "consider lowering vacancy_enclosure. (3) Assumes a COMPLETE sublattice with "
            "SPARSE random defects: an ORDERED/abundant defect arrangement is a "
            "superstructure, not point defects — when 'high_defect_fraction_possible_"
            "superstructure' or 'sublattice_population_imbalance_check_superstructure' is "
            "flagged, map the satellite reflection with fourier_reflection_map instead of "
            "counting sites. (4) HAADF bright-column intensity convention."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale array. Pass RAW (un-normalized) — intensity carries the Z-contrast that distinguishes species and dopants."},
            "positions": {"type": "ndarray", "description": "(N, 2) detected column positions (x, y) from detect_atoms / detect_atoms_dcnn, with BOTH sublattices resolved."},
            "n_sublattices": {"type": "int", "description": "Number of sublattices / distinct column species to separate (default 2; set to 3+ for more complex bases)."},
            "window_size": {"type": "int", "description": "Local-environment patch side in px for separation (default ~1.5x median NN, i.e. ~one lattice parameter). Larger captures more neighbourhood context."},
            "pixel_size_nm": {"type": "float", "description": "nm/px; if given, defect coordinates are also reported in nm."},
            "dopant_sigma": {"type": "float", "description": "Robust (MAD) z-score threshold for an intensity outlier within a sublattice. LOWER (e.g. 2.5) to catch subtler substitutions; RAISE (e.g. 5) if noise produces false dopants."},
            "edge_margin_px": {"type": "float", "description": "Exclude defects within this many px of the border (default 1.5x median NN). RAISE if border fits are unreliable; LOWER to include near-edge defects."},
            "vacancy_enclosure": {"type": "int", "description": "How many of a candidate empty site's 4 lattice neighbours must be occupied to count as a vacancy (default 3 of 4). RAISE to 4 for only fully-enclosed vacancies (fewer false positives at domain/grain edges); LOWER to 2 to catch vacancies near boundaries."},
            "random_state": {"type": "int", "description": "Seed for the GMM separation (default 1; fixed for reproducibility)."},
        },
        required=["image", "positions"],
        returns=(
            "dict with 'figure_bytes' (PNG: sublattices colour-coded + vacancies (squares) "
            "+ dopants (circles)), 'metrics' (separation_score, per-sublattice "
            "n_columns/mean_intensity/n_vacancies/n_dopants, and 'vacancies'/'dopants' "
            "lists each with x/y[/x_nm/y_nm], sublattice index, and for dopants a signed z "
            "(robust MAD z, CAPPED at +-10 so a real outlier on a very uniform sublattice "
            "is not reported as an absurd ~25-sigma 'impossible' value), an intensity_ratio "
            "(this column / its same-sublattice neighbour median — the PHYSICAL descriptor: "
            "<1 dimmer=lighter substitution, >1 brighter=heavier; a ratio near background "
            "would be a vacancy not a dopant), and a lighter/heavier-substitution label), and "
            "'flags' (e.g. low_separation_score when the sublattices are not cleanly "
            "distinguished). To catch subtler substitutions, LOWER dopant_sigma; judge "
            "candidates by intensity_ratio, not the (capped) z."
        ),
    ),
    ToolSpec(
        name="map_polarization",
        description=(
            "Per-unit-cell ferroelectric polarization / cation off-centering map for a "
            "two-sublattice (ABO3-type / interpenetrating) lattice. Measures the displacement "
            "of the displaced cation sublattice from the centrosymmetric centroid of its "
            "reference-cage neighbours — the ferroelectric order parameter whose direction maps "
            "domains and whose discontinuities map domain walls. Returns figure + per-cell vectors."
        ),
        import_line="from scilink.skills.image_analysis.atomic_stem.atom_finding import map_polarization",
        signature=(
            "map_polarization(image, positions, displaced='auto', n_cage=4, "
            "pixel_size_nm=None, edge_margin_px=None, max_offset_frac=0.45, "
            "magnitude_outlier_k=6.0, max_magnitude_nm=0.12, domain_coherence_floor=0.4, "
            "max_wall_fraction=0.15, local_coherence_floor=0.82, random_state=1) -> dict"
        ),
        agents=["image_analysis"],
        when_to_use=(
            "Ferroic/ferroelectric distortion mapping on a perovskite or other interpenetrating "
            "two-sublattice lattice, AFTER detection has resolved BOTH sublattices (the cage "
            "cations AND the off-centering cations) — confirm both are resolved by an NN at the "
            "inter-sublattice spacing, NOT the intra-sublattice repeat. Lattice-agnostic and "
            "free of material constants. Assumes HAADF bright-column intensity. Reports the polar "
            "displacement field; for tetragonality/shear, characterise the reference sublattice "
            "with gpa_strain. "
            "LIMITS: (1) DIRECTION/domains are the robust output; MAGNITUDE is trustworthy only "
            "on CLEAN detection (NN at the inter-sublattice spacing, low nn_cv) — over-detected "
            "input inflates magnitude (~20% on real over-detection), and EXTREME over-detection "
            "returns {'error':'no valid cells'} (an honest failure, not a result), so fix "
            "target_pixel_size first when magnitude matters. (2) magnitude is also imprecise for "
            "very small displacements near the position-fit noise floor. (3) assumes a DISPLACIVE "
            "perovskite-type structure: the off-centering cation sits ~at the reference-cage "
            "centre and the two cation sublattices are the INTENSITY EXTREMES (brightest = cage "
            "reference, dimmest = off-centering cation); a middle intensity population is ignored "
            "(reported via the third_intensity_population_present_using_extremes flag); exotic "
            "orderings may violate these assumptions. (4) when the two sublattices are NOT "
            "intensity-distinguishable (near-identical-Z cations, weak HAADF contrast), the tool "
            "AUTOMATICALLY falls back to GEOMETRIC separation by local environment "
            "(local_env_gmm) — flagged geometric_separation_used_intensity_ambiguous; the "
            "reference/sign is then arbitrary (direction/magnitude unchanged up to a global sign), "
            "so do NOT pre-decline a weak-contrast ferroelectric as single-sublattice."
        ),
        parameters={
            "image": {"type": "ndarray", "description": "2D grayscale HAADF (RAW — intensity separates the two species)."},
            "positions": {"type": "ndarray", "description": "(N,2) detected columns (x,y) with BOTH sublattices resolved."},
            "displaced": {"type": "str", "description": "Which sublattice off-centers: 'auto' (default, the dimmer/lighter cation — standard convention), 'dim', or 'bright' (when the off-centering cation is the heavier/brighter one). Only flips the reference/sign; domains unchanged."},
            "n_cage": {"type": "int", "description": "Number of reference neighbours forming the centrosymmetric cage (default 4 = perovskite [100]; raise to match the projected coordination of another structure/zone axis)."},
            "pixel_size_nm": {"type": "float", "description": "nm/px; if given, magnitude is also reported in nm."},
            "edge_margin_px": {"type": "float", "description": "Exclude cells within this many px of the border (default 1.5x NN)."},
            "max_offset_frac": {"type": "float", "description": "Robustness to extra/spurious columns: a displaced cation is accepted only within this fraction of the reference-cage NN spacing from the cage centre (default 0.45). RAISE (~0.6) if large genuine displacements are dropped; LOWER (~0.3) if extra columns inflate the field. Scale is the reference sublattice NN, robust to over-detection."},
            "magnitude_outlier_k": {"type": "float", "description": "Robust per-cell |P| outlier clip, in MADs above the field median (default 6.0): drops position-fit / A-B-pairing failures (flag clipped_N_unphysical_magnitude_cells). RELATIVE (median/MAD), so a uniformly large real field is NOT clipped, but on a strong field its high MAD lets a >physical tail through — that tail is caught by max_magnitude_nm. LOWER (~4) if stray huge vectors persist; RAISE (~8) if genuine large displacements get clipped."},
            "max_magnitude_nm": {"type": "float", "description": "Absolute physical ceiling on per-cell |P| in nm (default 0.12 = 120 pm; needs pixel_size_nm; None disables). Complements the relative MAD clip: a displacement above this is almost surely a sublattice-pairing/scale error, not real ferroelectric off-centering (40-80 pm common, ~100 pm extreme). Cells above it are dropped and counted in the N_cells_above_physical_ceiling flag (the flag fires even when too many to drop = whole-field scale error). RAISE (~0.2) for exotic large-displacement materials; LOWER (~0.1) to be stricter."},
            "domain_coherence_floor": {"type": "float", "description": "Global-coherence floor for the noise guard (default 0.4): if direction_coherence is below this the field is treated as pure noise and a multi-domain partition is collapsed to ONE 'disordered' domain (flag field_noise_dominated_domains_unreliable). RAISE (~0.5) to be stricter; LOWER (~0.3) if real low-contrast domains get suppressed."},
            "max_wall_fraction": {"type": "float", "description": "Fragmentation half of the noise guard (default 0.15): a fragmented partition (wall_fraction above this) is collapsed to one 'disordered' domain ONLY when it is ALSO locally incoherent (see local_coherence_floor). A genuine multi-domain field has contiguous domains separated by a THIN wall (~0.05); noise speckle is ~0.25+. RAISE (~0.25) if real finely-twinned mosaics are collapsed; LOWER (~0.1) to reject more weak-field speckle. NOTE wall_fraction grows with domain count/fineness and with smaller crops, which is why it is gated by local_coherence (count/size-invariant) rather than used alone."},
            "local_coherence_floor": {"type": "float", "description": "Coherence half of the noise guard (default 0.82): the median within-neighbourhood direction agreement. A fragmented partition collapses only if local_coherence is BELOW this (= noise). A real multi-domain mosaic stays locally coherent (~0.95+) regardless of how many/small its domains are, so it is preserved. RAISE (~0.88) to treat more borderline fields as noise; LOWER (~0.75) if real low-contrast/imperfect domains are wrongly collapsed."},
            "random_state": {"type": "int", "description": "Seed for the intensity-split GMM (default 1)."},
        },
        required=["image", "positions"],
        returns=(
            "dict with 'figure_bytes' (PNG: polarization quiver + SMOOTHED direction-domain map "
            "+ magnitude map). 'metrics': n_cells, median_magnitude_px/nm, direction_coherence "
            "(~1=coherent/domains ~0=noise OR domains finer than ~3 cells [check "
            "fourier_reflection_map for a satellite first]); local_coherence (median local "
            "smoothed-direction resultant; below domain_coherence_floor => noise-dominated, the "
            "field is reported as a single 'disordered' domain via "
            "field_noise_dominated_domains_unreliable — do NOT read a domain count or a clean "
            "single domain off such a field); the DOMAIN SEGMENTATION as data — "
            "n_domains, domains (list of {label, n_cells, mean_angle_deg, fraction}; directions are "
            "PEAK-CLUSTERED, adaptive — NOT fixed 0/90/180/270 bins), wall_fraction, per-cell "
            "domain_label; net_polarization_px/nm and net_to_local_ratio (mean/median|P|: «1 = "
            "genuine multi-domain field [net cancels]; »1 = a uniform offset / scan-registration "
            "drift dominates, so absolute magnitude is suspect — flag "
            "offset_dominated_field_check_registration); per-cell polarization/xy/magnitude/angle "
            "lists. 'flags' (e.g. weak_sublattice_intensity_separation). Sign follows the "
            "displaced->reference convention; flip via 'displaced' to match a specific reference."
        ),
    ),
]

