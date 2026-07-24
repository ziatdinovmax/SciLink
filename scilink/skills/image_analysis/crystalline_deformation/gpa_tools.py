"""Geometric Phase Analysis (GPA) and Burgers vector tools.

GPA extracts local strain and rotation fields from atomic-resolution
HRTEM/STEM images by analyzing the phase of Bragg reflections in the FFT.

Reference: Hytch, Snoeck & Kilaas, Ultramicroscopy 74 (1998) 131-146.

Burgers vector computation via line integral of displacement gradient
tensor beta around a dislocation core.

Reference: Cloete, Tarleton & Hofmann, Proc. R. Soc. A 478 (2022) 20210909.

GPA code extracted from stemtool (https://github.com/stemtool/stemtools).
Burgers vector code ported from MATLAB
(https://github.com/JacquesCloete/Burgers-Vector-Calculator).
"""

import numpy as np
import skimage.restoration as skr
import scipy.ndimage as scnd
from scipy.integrate import simpson


# ═══════════════════════════════════════════════════════════════════
# GPA — Geometric Phase Analysis
# ═══════════════════════════════════════════════════════════════════

def _make_circle(size_circ, center_x, center_y, radius):
    p, q = int(size_circ[0]), int(size_circ[1])
    yV, xV = np.mgrid[0:p, 0:q]
    return (((yV - center_y)**2 + (xV - center_x)**2)**0.5 < radius).astype(np.float64)


def _image_normalizer(img):
    lo, hi = np.amin(img), np.amax(img)
    return (img - lo) / (hi - lo) if hi != lo else np.zeros_like(img, dtype=np.float64)


def _phase_diff(angle_image):
    """Differentiate a wrapped phase image without wrapping artifacts."""
    z = np.exp(1j * angle_image)
    dx = np.zeros_like(z)
    dx[:, :-1] = np.diff(z, axis=1)
    dy = np.zeros_like(z)
    dy[:-1, :] = np.diff(z, axis=0)
    conj = np.conj(z)
    return np.imag(conj * dx), np.imag(conj * dy)


def _circ_to_G(circ_pos, image):
    return np.divide(np.flip(np.asarray(circ_pos)), np.asarray(image.shape)) - 0.5


def _G_to_circ(g_vec, image):
    circ = np.zeros(2)
    circ[1] = g_vec[0] * image.shape[0] + 0.5 * image.shape[0]
    circ[0] = g_vec[1] * image.shape[1] + 0.5 * image.shape[1]
    return circ


def _phase_matrix(gvec, image, circ_size=0, g_blur=True):
    """Compute real-space phase by masking around a Bragg spot in FFT."""
    imshape = np.asarray(image.shape)
    circ_rad = np.amin(0.01 * imshape) if circ_size == 0 else circ_size
    yy, xx = np.mgrid[0:imshape[0], 0:imshape[1]]
    circ_pos = np.multiply(np.flip(gvec), imshape) + 0.5 * imshape
    circ_mask = _make_circle(imshape, circ_pos[0], circ_pos[1], circ_rad).astype(bool)
    ham = np.sqrt(np.outer(np.hamming(imshape[0]), np.hamming(imshape[1])))
    ft = np.fft.fftshift(np.fft.fft2(image * ham))

    if g_blur:
        sigma2 = np.sum((0.5 * gvec * imshape)**2)
        zz = ((yy[circ_mask] - circ_pos[1])**2 + (xx[circ_mask] - circ_pos[0])**2) / sigma2
        four_mask = np.zeros_like(yy, dtype=np.float64)
        four_mask[circ_mask] = np.exp(-0.5 * zz)
        return np.angle(np.fft.ifft2(four_mask * ft))
    else:
        return np.angle(np.fft.ifft2(circ_mask * ft))


class _GPA:
    """Internal GPA engine."""

    def __init__(self, image, calib, calib_units="nm", ref_iter=20,
                 use_blur=True, max_strain=0.4):
        self.image = np.asarray(image, dtype=np.float64)
        self.calib = calib
        self.calib_units = calib_units
        self.blur = use_blur
        self.ref_iter = int(ref_iter)
        self.imshape = np.asarray(self.image.shape)
        self.inv_calib = 1.0 / (self.calib * self.imshape)
        self.max_strain = max_strain
        self._spots = False
        self._ref = False
        self._refined = False

    def find_spots(self, circ1, circ2, circ_size=15):
        self.circ_1 = self.imshape / 2 + np.asarray(circ1) / self.inv_calib
        self.circ_2 = self.imshape / 2 + np.asarray(circ2) / self.inv_calib
        self.circ_size = circ_size
        self.gvec_1 = _circ_to_G(self.circ_1, self.image)
        self.gvec_2 = _circ_to_G(self.circ_2, self.image)
        self.P1 = _phase_matrix(self.gvec_1, self.image, circ_size, self.blur)
        self.P2 = _phase_matrix(self.gvec_2, self.image, circ_size, self.blur)
        self._spots = True

    def find_spots_auto(self, n_peaks=6, min_distance=20):
        from skimage.feature import peak_local_max
        ham = np.sqrt(np.outer(np.hamming(self.imshape[0]), np.hamming(self.imshape[1])))
        ft = np.fft.fftshift(np.fft.fft2(self.image * ham))
        ps = scnd.gaussian_filter(np.log10(np.abs(ft) + 1e-10), 3)

        center = self.imshape / 2
        yy, xx = np.mgrid[0:self.imshape[0], 0:self.imshape[1]]
        central = ((yy - center[0])**2 + (xx - center[1])**2) < min_distance**2
        ps_m = ps.copy()
        ps_m[central] = 0

        coords = peak_local_max(ps_m, min_distance=min_distance,
                                num_peaks=n_peaks * 2)
        # Keep one of each Friedel pair
        unique = []
        for c in coords:
            mir = 2 * center - c
            dup = any(np.linalg.norm(u - mir) < min_distance for u in unique)
            if not dup:
                unique.append(c)
            if len(unique) >= n_peaks:
                break

        if len(unique) < 2:
            raise ValueError(f"Only {len(unique)} Bragg peaks found, need 2")

        ints = [ps[c[0], c[1]] for c in unique]
        order = np.argsort(ints)[::-1]
        s1, s2 = unique[order[0]], unique[order[1]]
        g1 = (s1 - center) * self.inv_calib
        g2 = (s2 - center) * self.inv_calib
        self.circ_size = max(min_distance // 2, 10)
        self.find_spots(tuple(g1), tuple(g2), circ_size=self.circ_size)
        return tuple(g1), tuple(g2)

    def define_reference_rect(self, row_slice, col_slice):
        if not self._spots:
            raise RuntimeError("Call find_spots() first")
        self.ref_reg = np.zeros(self.image.shape, dtype=bool)
        self.ref_reg[row_slice, col_slice] = True
        self._ref = True

    def refine_phase(self):
        if not self._ref:
            raise RuntimeError("Call define_reference*() first")
        ry = np.arange(-self.imshape[0] / 2, self.imshape[0] / 2, 1)
        rx = np.arange(-self.imshape[1] / 2, self.imshape[1] / 2, 1)
        Rx, Ry = np.meshgrid(rx, ry)

        g1, g2 = self.gvec_1.copy(), self.gvec_2.copy()
        P1, P2 = self.P1.copy(), self.P2.copy()

        # Mask out zero-coordinate pixels to avoid divide-by-zero
        ref_nz = self.ref_reg & (Ry != 0) & (Rx != 0)

        for _ in range(self.ref_iter):
            G1x, G1y = _phase_diff(P1)
            G2x, G2y = _phase_diff(P2)
            g1r = (G1x + G1y) / (2 * np.pi)
            g2r = (G2x + G2y) / (2 * np.pi)
            g1 += np.array([np.median(g1r[ref_nz] / Ry[ref_nz]),
                            np.median(g1r[ref_nz] / Rx[ref_nz])])
            g2 += np.array([np.median(g2r[ref_nz] / Ry[ref_nz]),
                            np.median(g2r[ref_nz] / Rx[ref_nz])])
            P1 = _phase_matrix(g1, self.image, self.circ_size, self.blur)
            P2 = _phase_matrix(g2, self.image, self.circ_size, self.blur)

        self.gvec_1_fin, self.gvec_2_fin = g1, g2
        self.P1_fin, self.P2_fin = P1, P2
        self._refined = True

    def get_strain(self):
        if not self._refined:
            raise RuntimeError("Call refine_phase() first")

        gm = np.zeros((2, 2))
        gm[0, :] = np.flip(self.gvec_1_fin)
        gm[1, :] = np.flip(self.gvec_2_fin)
        self.a_matrix = np.linalg.inv(gm.T)

        P1u = skr.unwrap_phase(self.P1_fin)
        P2u = skr.unwrap_phase(self.P2_fin)

        rolled = np.array([P1u.ravel(), P2u.ravel()])
        u = self.a_matrix @ rolled
        ux = u[0].reshape(P1u.shape)
        uy = u[1].reshape(P1u.shape)

        exx, exy = _phase_diff(ux)
        eyx, eyy = _phase_diff(uy)
        eth = 0.5 * (exy - eyx)
        edg = 0.5 * (exy + eyx)

        ref = self.ref_reg
        for f in (exx, eyy, eth, edg):
            f -= np.median(f[ref])
        if self.max_strain > 0:
            for f in (exx, eyy, eth, edg):
                np.clip(f, -self.max_strain, self.max_strain, out=f)

        self.e_xx, self.e_yy, self.e_th, self.e_dg = exx, eyy, eth, edg
        return exx, eyy, eth, edg

    def get_phase_maps(self):
        if not self._refined:
            raise RuntimeError("Call refine_phase() first")
        return skr.unwrap_phase(self.P1_fin), skr.unwrap_phase(self.P2_fin)


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
    """Burgers vector via rectangular line integral (Simpson's rule)."""
    Ny, Nx = beta.shape[:2]
    xv = np.arange(Nx) * interval
    yv = np.arange(Ny) * interval
    b = np.zeros(2)
    for i in range(2):
        b[i] = (-simpson(y=beta[0, :, i, 0], x=xv)
                - simpson(y=beta[:, 0, i, 1], x=yv)
                + simpson(y=beta[-1, :, i, 0], x=xv)
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

def compute_gpa_strain(image, calib, calib_units="nm",
                       spot1=None, spot2=None,
                       ref_region=None,
                       circ_size=15, ref_iter=20, max_strain=0.4,
                       auto_spots=True, auto_ref_fraction=0.25):
    """One-call GPA strain computation.

    Parameters
    ----------
    image : 2-D ndarray
        Atomic-resolution HRTEM or STEM image.
    calib : float
        Pixel size in physical units.
    calib_units : str
        Unit of calibration (default "nm").
    spot1, spot2 : tuple or None
        Bragg spot positions in inverse-length units.  If None and
        auto_spots is True, spots are detected automatically.
    ref_region : tuple of 2 slices, or None
        Reference region as ``(slice(r0, r1), slice(c0, c1))`` in pixel
        coordinates.  If None, a central patch is used.
    circ_size : float
        Aperture radius in FFT pixels.
    ref_iter : int
        Number of g-vector refinement iterations.
    max_strain : float
        Clamp strain maps to [-max_strain, max_strain].
    auto_spots : bool
        Auto-detect Bragg spots when spot1/spot2 are None.
    auto_ref_fraction : float
        Fraction of image to use as reference when ref_region is None.

    Returns
    -------
    dict with keys:
        'e_xx', 'e_yy', 'e_th', 'e_dg' : 2-D ndarrays (strain maps)
        'P1', 'P2' : 2-D ndarrays (unwrapped phase maps)
        'gpa' : the internal GPA object
    """
    gpa = _GPA(image, calib, calib_units,
               ref_iter=ref_iter, max_strain=max_strain)

    if spot1 is not None and spot2 is not None:
        gpa.find_spots(spot1, spot2, circ_size=circ_size)
    elif auto_spots:
        gpa.find_spots_auto()
    else:
        raise ValueError("Provide spot1/spot2 or set auto_spots=True")

    if ref_region is not None:
        gpa.define_reference_rect(*ref_region)
    else:
        ny, nx = image.shape
        f = auto_ref_fraction
        r0, r1 = int(ny * (0.5 - f / 2)), int(ny * (0.5 + f / 2))
        c0, c1 = int(nx * (0.5 - f / 2)), int(nx * (0.5 + f / 2))
        gpa.define_reference_rect(slice(r0, r1), slice(c0, c1))

    gpa.refine_phase()
    exx, eyy, eth, edg = gpa.get_strain()
    P1, P2 = gpa.get_phase_maps()

    return {
        'e_xx': exx, 'e_yy': eyy, 'e_th': eth, 'e_dg': edg,
        'P1': P1, 'P2': P2,
        'gpa': gpa,
    }


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
        Shear strain — this is 'e_dg' from compute_gpa_strain.
    w_z : 2-D ndarray
        Rotation — this is 'e_th' from compute_gpa_strain.
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
        name="compute_gpa_strain",
        description=(
            "One-call Geometric Phase Analysis: full-field strain tensor maps "
            "(e_xx, e_yy, shear e_dg, rotation e_th) from an atomic-resolution "
            "image via FFT Bragg-spot phase analysis. ONLY valid on "
            "single-grain, single-phase images — see skill guidance for the "
            "validity gate (never on bicrystals/twin boundaries)."
        ),
        import_line="from scilink.skills.image_analysis.crystalline_deformation.gpa_tools import compute_gpa_strain",
        signature=(
            "compute_gpa_strain(image, calib, calib_units='nm', spot1=None, spot2=None, "
            "ref_region=None, circ_size=15, ref_iter=20, max_strain=0.4, "
            "auto_spots=True, auto_ref_fraction=0.25)"
        ),
        returns="dict: e_xx, e_yy, e_th (rotation), e_dg (shear), spots, ref_region",
    ),
    ToolSpec(
        name="burgers_from_gpa",
        description=(
            "Burgers vectors from GPA strain/rotation maps via line-integral "
            "circuits: per-dislocation (given core positions) or full-field "
            "auto-detection of dislocation cores with a Burgers vector field. "
            "Feed cores from PTM/CoS analysis when available."
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
