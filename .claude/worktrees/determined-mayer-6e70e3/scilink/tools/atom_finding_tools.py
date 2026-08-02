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
        target_pixel_size: Target pixel size in **Angstroms** for the model.
            Default 0.25.  This value is approximate and may need tuning
            for different materials.
        threshold: Detection confidence threshold (0-1). Default 0.8.
        refine: Sub-pixel Gaussian refinement on detected peaks.

    Returns:
        dict with keys ``positions`` (N,2 as x,y in original image pixels),
        ``sigma_x``, ``sigma_y``, ``amplitude``, ``rotation`` (all *None*),
        and ``heatmap`` (2D probability map in original image space).
    """
    import logging

    try:
        from ..tools.atomistic_tools import rescale_for_model, predict_with_ensemble
        from ..tools.atomistic_model_manager import get_or_download_atomistic_model
    except ImportError as exc:
        raise ImportError(
            "detect_atoms_dcnn requires atomai and opencv-python. "
            "Install with: pip install atomai opencv-python"
        ) from exc

    logger = logging.getLogger("scilink.tools.atom_finding_tools")

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


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

from ._spec import ToolSpec

TOOL_SPECS = [
    ToolSpec(
        name="detect_atoms",
        description=(
            "Classical atom-column detection: peak detection plus optional 2D Gaussian "
            "refinement. Returns sub-pixel positions and per-atom Gaussian parameters."
        ),
        import_line="from scilink.tools.atom_finding_tools import detect_atoms",
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
        import_line="from scilink.tools.atom_finding_tools import detect_atoms_dcnn",
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
                "description": "Target pixel size in Angstroms (default 0.25).",
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
        import_line="from scilink.tools.atom_finding_tools import refine_positions",
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
        import_line="from scilink.tools.atom_finding_tools import find_zone_axes",
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
        import_line="from scilink.tools.atom_finding_tools import find_missing_atoms",
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
        import_line="from scilink.tools.atom_finding_tools import subtract_atoms",
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
        import_line="from scilink.tools.atom_finding_tools import local_env_gmm",
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
]

