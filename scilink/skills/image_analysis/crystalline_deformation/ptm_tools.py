"""2D Polyhedral Template Matching and Center of Symmetry tools.

PTM algorithm based on Britton & Medlin, Materials Characterization 213
(2024) 114017 (https://github.com/sandialabs/2D-PTM). Classifies each
atom's local neighbor environment against ideal crystal structure
templates using Kabsch-aligned RMSD with optimal point correspondence.

Center of Symmetry uses a robust anti-parallel partner search rather
than fixed i/i+3 pairing, following Mukhopadhyay's implementation.

Both tools take only (x, y) atom positions as input.
"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull


# ── Templates ──────────────────────────────────────────────
# Ideal 7-point motifs (centroid + 6 neighbors) for each structure.
# Defined analytically from lattice parameters following Britton & Medlin.

def _fcc_110_template(a=1.0):
    """FCC viewed along [110]. Regular hexagonal projected pattern."""
    s6 = np.sqrt(6)
    s3 = np.sqrt(3)
    return np.array([
        [0, 0],
        [a * s6 / 4, 0],
        [a / s6, a / s3],
        [-a * s6 / 12, a / s3],
        [-a * s6 / 4, 0],
        [-a / s6, -a / s3],
        [a * s6 / 12, -a / s3],
    ])


def _bcc_111_template(a_fcc=1.0):
    """BCC viewed along [111]. Lattice parameter scaled to FCC close-packed."""
    a = a_fcc * np.sqrt(6) / 3
    s6 = np.sqrt(6)
    s2 = np.sqrt(2)
    return np.array([
        [0, 0],
        [a * s6 / 3, 0],
        [a * s6 / 6, a * s2 / 2],
        [-a * s6 / 6, a * s2 / 2],
        [-a * s6 / 3, 0],
        [-a * s6 / 6, -a * s2 / 2],
        [a * s6 / 6, -a * s2 / 2],
    ])


def _hcp_2110_template(a_fcc=1.0):
    """HCP viewed along [2-1-10]. Lattice parameter scaled to FCC."""
    a = a_fcc / np.sqrt(2)
    s3 = np.sqrt(3)
    s6 = np.sqrt(6)
    return np.array([
        [0, 0],
        [a * s3 / 2, 0],
        [a * s3 / 6, a * s6 / 3],
        [-a * s3 / 3, a * s6 / 3],
        [-a * s3 / 2, 0],
        [-a * s3 / 3, -a * s6 / 3],
        [a * s3 / 6, -a * s6 / 3],
    ])


DEFAULT_TEMPLATES = {
    "FCC": _fcc_110_template,
    "BCC": _bcc_111_template,
    "HCP": _hcp_2110_template,
}

STRUCTURE_CODES = {"FCC": 1, "BCC": 2, "HCP": 3}


# ── Kabsch alignment ──────────────────────────────────────

def _kabsch_2d(P, Q):
    """Kabsch algorithm for 2D point sets. Returns rotation matrix."""
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    return R


def _scaled_rmsd(v, w):
    """Compute scaled RMSD between experimental points v and template w.

    Following Britton & Medlin RMSD.m:
    - s = sqrt(sum||w-wbar||² / sum||v-vbar||²) scales v to template size
    - R = Kabsch rotation aligning template w onto experimental v
    - residual = s*(v_i - vbar) - (R @ (w_i - wbar))

    Returns (rmsd, scale_factor, rotation_matrix).
    """
    N = len(v)
    v_bar = v.mean(axis=0)
    w_bar = w.mean(axis=0)

    v_c = v - v_bar
    w_c = w - w_bar

    sv = np.sum(np.linalg.norm(v_c, axis=1) ** 2)
    sw = np.sum(np.linalg.norm(w_c, axis=1) ** 2)
    s = np.sqrt(sw / sv) if sv > 1e-15 else 1.0

    # Kabsch: find R that rotates w_c onto s*v_c  →  H = P^T @ Q where P=w_c, Q=s*v_c
    H = w_c.T @ (s * v_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])
    R = Vt.T @ S @ U.T

    # Residual: s*v_centered - R @ w_centered
    rotated_w = (R @ w_c.T).T
    residuals = s * v_c - rotated_w
    rmsd = np.sqrt(np.sum(residuals ** 2) / N)
    return rmsd, s, R


# ── Neighbor finding ──────────────────────────────────────

def _find_neighbors_delaunay(positions):
    """Build neighbor lists from Delaunay triangulation.

    Returns dict mapping atom index to list of neighbor indices.
    """
    tri = Delaunay(positions)
    neighbors = {i: set() for i in range(len(positions))}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                neighbors[simplex[i]].add(simplex[j])
                neighbors[simplex[j]].add(simplex[i])
    return neighbors


def _get_6nn_sorted(center, all_neighbor_positions):
    """Select 6 closest neighbors and sort angularly around center.

    If Delaunay gives more than 6 neighbors, keep only the 6 nearest.
    Returns (7, 2) array: center + 6 sorted neighbors, or None if < 6.
    """
    dists = np.linalg.norm(all_neighbor_positions - center, axis=1)
    if len(dists) < 6:
        return None
    closest_idx = np.argsort(dists)[:6]
    nn6 = all_neighbor_positions[closest_idx]
    vecs = nn6 - center
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    order = np.argsort(angles)
    return np.vstack([center.reshape(1, 2), nn6[order]])


# ── Per-atom PTM classification ───────────────────────────

def _classify_atom(sorted_points, templates, threshold):
    """Classify a single atom by testing all templates with circular permutations.

    Following Britton & Medlin: try all 6 cyclic permutations of the
    neighbor ordering for each template, pick the best Kabsch-aligned RMSD.

    Args:
        sorted_points: (7, 2) array — center + 6 angularly-sorted neighbors.
        templates: dict of {name: (7, 2) template array}.
        threshold: max RMSD for valid classification.

    Returns:
        (label, rmsd, angle_deg, scale_factor)
    """
    best_rmsd = np.inf
    best_label = "unidentified"
    best_angle = np.nan
    best_scale = np.nan

    # Center at origin
    center = sorted_points[0]
    pts = sorted_points - center
    neighbors = pts[1:]  # (6, 2)
    N = len(neighbors)

    for name, tmpl in templates.items():
        tmpl_c = tmpl - tmpl[0]

        for shift in range(N):
            perm = np.roll(neighbors, shift, axis=0)
            v = np.vstack([pts[0:1], perm])

            rmsd, s, R = _scaled_rmsd(v, tmpl_c)

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                if rmsd < threshold:
                    best_label = name
                    best_scale = s
                    # Extract rotation angle
                    cos_val = np.clip(R[0, 0], -1.0, 1.0)
                    angle = np.degrees(np.arctan2(R[1, 0], cos_val))
                    if angle < 0:
                        angle += 180.0
                    best_angle = angle

    if best_label == "unidentified":
        best_rmsd = np.nan
        best_angle = np.nan
        best_scale = np.nan

    return best_label, best_rmsd, best_angle, best_scale


# ── Center of Symmetry ────────────────────────────────────

def _compute_cos_single(center, neighbor_positions):
    """Compute Center of Symmetry for one atom.

    Uses robust anti-parallel partner search: for each bond vector j,
    find D[j] = min_i(||bond_i + bond_j||). Then:
        M = sum(D² / 2) / (2 * sum(||bond||²))

    M = 0 is perfectly centrosymmetric. M > 0 indicates broken symmetry.
    """
    bonds = neighbor_positions - center
    n = len(bonds)
    if n < 2:
        return np.nan

    bond_norms_sq = np.sum(bonds ** 2, axis=1)
    denom = 2.0 * np.sum(bond_norms_sq)
    if denom < 1e-15:
        return np.nan

    D_sq_sum = 0.0
    for j in range(n):
        pair_norms = np.sqrt(np.sum((bonds + bonds[j]) ** 2, axis=1))
        D_sq_sum += np.min(pair_norms) ** 2

    M = (D_sq_sum / 2.0) / denom
    return M


# ── Public API ────────────────────────────────────────────

def ptm_classify(x, y, threshold=0.05, structures=None, edge_cutout_nn=3):
    """Run 2D Polyhedral Template Matching on detected atom positions.

    Args:
        x: 1D array of x-coordinates (N atoms).
        y: 1D array of y-coordinates (N atoms).
        threshold: RMSD threshold for classification (default 0.05,
            following Britton & Medlin). Atoms with best RMSD above
            this are labeled 'unidentified'.
        structures: list of structure names to test, e.g. ["FCC", "HCP"].
            Default: all three (FCC, BCC, HCP).
        edge_cutout_nn: exclude atoms within this many NN distances of
            image edges from classification (set label to 'edge').
            Default 3.

    Returns:
        dict with keys:
            labels: list of str — per-atom classification
            rmsd: (N,) float array — best RMSD (NaN for unidentified/edge)
            angle: (N,) float array — rotation angle in degrees
            scale: (N,) float array — scaling factor
            cos: (N,) float array — Center of Symmetry metric
            structure_code: (N,) int array — 1=FCC, 2=BCC, 3=HCP, 0=other, -1=edge
            positions: (N, 2) array — input positions
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    positions = np.column_stack([x, y])
    N = len(positions)

    if structures is None:
        structures = ["FCC", "BCC", "HCP"]

    # Build templates
    templates = {}
    for name in structures:
        if name in DEFAULT_TEMPLATES:
            templates[name] = DEFAULT_TEMPLATES[name]()

    # Neighbor lists via Delaunay
    neighbors = _find_neighbors_delaunay(positions)

    # Compute mean NN distance for edge exclusion
    nn_dists = []
    for i in range(N):
        if neighbors[i]:
            dists = np.linalg.norm(positions[list(neighbors[i])] - positions[i], axis=1)
            nn_dists.append(np.min(dists))
    mean_nn = np.median(nn_dists) if nn_dists else 0.0

    # Edge mask
    margin = edge_cutout_nn * mean_nn
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    edge_mask = (
        (x < xmin + margin) | (x > xmax - margin) |
        (y < ymin + margin) | (y > ymax - margin)
    )

    # Initialize output arrays
    labels = ["unidentified"] * N
    rmsd_arr = np.full(N, np.nan)
    angle_arr = np.full(N, np.nan)
    scale_arr = np.full(N, np.nan)
    cos_arr = np.full(N, np.nan)
    code_arr = np.zeros(N, dtype=int)

    for i in range(N):
        nn_idx = list(neighbors[i])
        nn_pos = positions[nn_idx]

        # CoS is computed for all atoms with enough neighbors (uses all Delaunay NN)
        if len(nn_idx) >= 4:
            # Use 6 closest for CoS
            dists = np.linalg.norm(nn_pos - positions[i], axis=1)
            n_use = min(6, len(nn_idx))
            closest = np.argsort(dists)[:n_use]
            cos_arr[i] = _compute_cos_single(positions[i], nn_pos[closest])

        # Edge atoms: mark but still compute CoS
        if edge_mask[i]:
            labels[i] = "edge"
            code_arr[i] = -1
            continue

        # Need at least 6 neighbors for PTM
        if len(nn_idx) < 6:
            continue

        sorted_pts = _get_6nn_sorted(positions[i], nn_pos)
        if sorted_pts is None:
            continue

        label, rmsd, angle, scale = _classify_atom(sorted_pts, templates, threshold)
        labels[i] = label
        rmsd_arr[i] = rmsd
        angle_arr[i] = angle
        scale_arr[i] = scale
        code_arr[i] = STRUCTURE_CODES.get(label, 0)

    # Clamp CoS artifacts
    cos_arr = np.where(cos_arr > 0.05, np.nan, cos_arr)
    cos_arr = np.where(cos_arr < 0, 0.0, cos_arr)

    return {
        "labels": labels,
        "rmsd": rmsd_arr,
        "angle": angle_arr,
        "scale": scale_arr,
        "cos": cos_arr,
        "structure_code": code_arr,
        "positions": positions,
    }


def ptm_classify_sweep(
    x, y, thresholds=None, structures=None, edge_cutout_nn=3,
    target_unidentified=0.05,
):
    """Sweep RMSD thresholds to find the best classification cutoff.

    Runs Kabsch alignment once (expensive), then reclassifies at each
    candidate threshold (cheap) by comparing stored per-atom RMSD values.

    Args:
        x, y: 1D arrays of atom coordinates.
        thresholds: list of RMSD thresholds to test.  Default covers
            the range 0.03–0.20 in 10 steps.
        structures: structure names to test (default: all).
        edge_cutout_nn: edge exclusion margin in NN distances.
        target_unidentified: target fraction of non-edge unidentified
            atoms.  The sweep picks the smallest threshold that puts
            unidentified fraction below this value.

    Returns:
        dict with keys:
            best_threshold: float — recommended RMSD threshold
            best_result: dict — full ptm_classify output at that threshold
            sweep: list of dicts — per-threshold statistics:
                threshold, n_classified, n_unidentified, frac_unidentified,
                per_structure (dict of structure→count)
            raw_rmsd: (N,) float array — best RMSD per atom (NaN for
                edge atoms), independent of threshold
    """
    if thresholds is None:
        thresholds = [0.03, 0.05, 0.06, 0.075, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    thresholds = sorted(thresholds)

    # Run once with permissive threshold to get raw RMSDs
    base = ptm_classify(x, y, threshold=np.inf, structures=structures,
                        edge_cutout_nn=edge_cutout_nn)

    raw_rmsd = base["rmsd"].copy()
    base_labels = base["labels"]
    base_codes = base["structure_code"]
    N = len(base_labels)

    edge_mask = np.array([l == "edge" for l in base_labels])
    n_interior = int(np.sum(~edge_mask))

    sweep_stats = []
    best_threshold = thresholds[-1]
    best_result = None

    for thr in thresholds:
        labels = list(base_labels)
        codes = base_codes.copy()
        rmsd_out = raw_rmsd.copy()

        for i in range(N):
            if edge_mask[i]:
                continue
            if np.isnan(raw_rmsd[i]) or raw_rmsd[i] >= thr:
                labels[i] = "unidentified"
                codes[i] = 0
                rmsd_out[i] = np.nan

        n_unid = sum(1 for i in range(N) if labels[i] == "unidentified")
        n_classified = n_interior - n_unid
        frac_unid = n_unid / n_interior if n_interior > 0 else 0.0

        per_struct = {}
        for label in labels:
            if label not in ("unidentified", "edge"):
                per_struct[label] = per_struct.get(label, 0) + 1

        sweep_stats.append({
            "threshold": thr,
            "n_classified": n_classified,
            "n_unidentified": n_unid,
            "frac_unidentified": round(frac_unid, 4),
            "per_structure": per_struct,
        })

        if frac_unid <= target_unidentified and best_result is None:
            best_threshold = thr
            best_result = {
                "labels": labels,
                "rmsd": rmsd_out,
                "angle": base["angle"].copy(),
                "scale": base["scale"].copy(),
                "cos": base["cos"].copy(),
                "structure_code": codes,
                "positions": base["positions"],
            }

    # If no threshold met target, use the most permissive
    if best_result is None:
        thr = thresholds[-1]
        labels = list(base_labels)
        codes = base_codes.copy()
        rmsd_out = raw_rmsd.copy()
        for i in range(N):
            if edge_mask[i]:
                continue
            if np.isnan(raw_rmsd[i]) or raw_rmsd[i] >= thr:
                labels[i] = "unidentified"
                codes[i] = 0
                rmsd_out[i] = np.nan
        best_result = {
            "labels": labels,
            "rmsd": rmsd_out,
            "angle": base["angle"].copy(),
            "scale": base["scale"].copy(),
            "cos": base["cos"].copy(),
            "structure_code": codes,
            "positions": base["positions"],
        }
        best_threshold = thr

    return {
        "best_threshold": best_threshold,
        "best_result": best_result,
        "sweep": sweep_stats,
        "raw_rmsd": raw_rmsd,
    }


def compute_cos(x, y, n_neighbors=6, edge_cutout_nn=3):
    """Compute Center of Symmetry metric for all atom positions.

    Standalone CoS calculation (no PTM). Useful when template matching
    is not needed but distortion mapping is.

    Args:
        x, y: 1D arrays of atom coordinates.
        n_neighbors: number of neighbors to use (default 6).
        edge_cutout_nn: edge exclusion in NN distances.

    Returns:
        dict with keys:
            cos: (N,) float array — CoS metric per atom
            positions: (N, 2) array
            edge_mask: (N,) bool array
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    positions = np.column_stack([x, y])
    N = len(positions)

    neighbors = _find_neighbors_delaunay(positions)

    # Mean NN distance
    nn_dists = []
    for i in range(N):
        if neighbors[i]:
            dists = np.linalg.norm(positions[list(neighbors[i])] - positions[i], axis=1)
            nn_dists.append(np.min(dists))
    mean_nn = np.median(nn_dists) if nn_dists else 0.0

    margin = edge_cutout_nn * mean_nn
    xmin, ymin = positions.min(axis=0)
    xmax, ymax = positions.max(axis=0)
    edge_mask = (
        (x < xmin + margin) | (x > xmax - margin) |
        (y < ymin + margin) | (y > ymax - margin)
    )

    cos_arr = np.full(N, np.nan)
    for i in range(N):
        nn_idx = list(neighbors[i])
        if len(nn_idx) < 4:
            continue
        # Use closest n_neighbors if more are available
        dists = np.linalg.norm(positions[nn_idx] - positions[i], axis=1)
        closest = np.argsort(dists)[:n_neighbors]
        nn_pos = positions[np.array(nn_idx)[closest]]
        cos_arr[i] = _compute_cos_single(positions[i], nn_pos)

    cos_arr = np.where(cos_arr > 0.05, np.nan, cos_arr)
    cos_arr = np.where(cos_arr < 0, 0.0, cos_arr)

    return {
        "cos": cos_arr,
        "positions": positions,
        "edge_mask": edge_mask,
    }


# ── Tool registry specs: auto-discovered by scilink.skills._shared._registry
#    when the crystalline_deformation bundle is active ───────────────────────
from scilink.skills._shared._spec import ToolSpec

TOOL_SPECS = [
    ToolSpec(
        name="ptm_classify",
        description=(
            "2D Polyhedral Template Matching: classifies each atom column as "
            "FCC, BCC, HCP, or unidentified by Kabsch-aligned RMSD matching of "
            "its 6-NN environment against ideal zone-axis templates, and "
            "computes the per-atom Center-of-Symmetry (CoS) distortion metric. "
            "Primary defect-identification tool for atomic-resolution images."
        ),
        import_line="from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import ptm_classify",
        signature="ptm_classify(x, y, threshold=0.05, structures=None, edge_cutout_nn=3)",
        returns="dict: labels, rmsd, cos, structure_code, rotation, per-structure counts",
    ),
    ToolSpec(
        name="ptm_classify_sweep",
        description=(
            "RMSD-threshold sweep for PTM: runs Kabsch alignment once, then "
            "reclassifies at each candidate threshold to auto-select the cutoff "
            "meeting a target unidentified fraction. Use when default-threshold "
            "ptm_classify leaves >5% of interior atoms unidentified."
        ),
        import_line="from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import ptm_classify_sweep",
        signature="ptm_classify_sweep(x, y, thresholds=None, structures=None, edge_cutout_nn=3, target_unidentified=0.05)",
        returns="dict: best_threshold, best_result, sweep (per-threshold stats)",
    ),
    ToolSpec(
        name="compute_cos",
        description=(
            "Standalone Center-of-Symmetry metric per atom (0=centrosymmetric "
            "bulk; elevated at defects, boundaries, dislocation cores). Use as "
            "a continuous local-distortion / strain-proxy map, valid across "
            "grain boundaries where GPA is not."
        ),
        import_line="from scilink.skills.image_analysis.crystalline_deformation.ptm_tools import compute_cos",
        signature="compute_cos(x, y, n_neighbors=6, edge_cutout_nn=3)",
        returns="dict: cos (per-atom array), neighbor counts",
    ),
]
