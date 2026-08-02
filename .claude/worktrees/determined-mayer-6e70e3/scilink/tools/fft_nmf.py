import os
import numpy as np
from scipy import fftpack, ndimage
from sklearn.decomposition import NMF
from skimage.util import view_as_windows
from skimage import io, color
import warnings

# --- Helper Function for Standalone Usage ---
def load_image(file_path):
    """Simple wrapper to load images using skimage."""
    try:
        img = io.imread(file_path)
        # Handle RGBA/RGB
        if img.ndim == 3 and img.shape[2] in [3, 4]:
            img = color.rgb2gray(img[:, :, :3])
        return img
    except Exception as e:
        raise ValueError(f"Could not load image at {file_path}: {e}")

def _pairwise_cos_sim(X):
    """Plain cosine similarity matrix between rows of X. Shape (n, n)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    Xn = X / norms
    return Xn @ Xn.T


def _pairwise_baseline_subtracted_sim(X):
    """Cosine similarity after subtracting the per-pair element-wise minimum.

    For two NMF components that share most of their energy in a common base
    (DC peak, Hamming envelope, lattice peaks present in both) and differ
    only in a localized peak pair, plain cosine similarity is dominated by
    the shared part and reads close to 1 even though the components are
    physically distinct. Subtracting ``np.minimum(A, B)`` element-wise
    leaves only the differing parts, so the similarity reflects how much
    those differing parts overlap rather than how much the shared base
    overlaps.

    Identical rows return 1.0; rows where one is a strict subset of the
    other (one row has differing peaks, the other has none) return 0.0.
    """
    n = X.shape[0]
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            common = np.minimum(X[i], X[j])
            a = X[i] - common
            b = X[j] - common
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 and nb == 0.0:
                val = 1.0
            elif na == 0.0 or nb == 0.0:
                val = 0.0
            else:
                val = float((a @ b) / (na * nb))
            sim[i, j] = val
            sim[j, i] = val
    return sim


class SlidingFFTNMF:
    def __init__(self, window_size_x=None, window_size_y=None,
                 window_step_x=None, window_step_y=None,
                 interpolation_factor=2, zoom_factor=2,
                 hamming_filter=True, components=4,
                 random_state=42, init='random', n_inits=1):
        """
        Sliding Window FFT with NMF unmixing.
        Supports both Single Images (2D) and Time-Series Stacks (3D).
        """
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.window_step_x = window_step_x
        self.window_step_y = window_step_y

        self.interpol_factor = interpolation_factor
        self.zoom_factor = zoom_factor
        self.hamming_filter = hamming_filter
        self.n_components = components

        # NMF init / randomness
        self.random_state = random_state
        self.init = init
        self.n_inits = max(1, int(n_inits))

        # Internal state
        self.hamming_window = None
        self.windows_shape = None # (n_frames, n_windows_y, n_windows_x)
        self.fft_size = None
        
    def _calculate_window_params(self, image_shape):
        """Auto-calculates window parameters based on the spatial dimensions (H, W)."""
        # image_shape is expected to be (Height, Width) here
        height, width = image_shape
        
        # 1. Defaults for Window Size
        if self.window_size_x is None:
            val = max(32, min(128, height // 8))
            self.window_size_x = 2 ** int(np.log2(val))
        if self.window_size_y is None:
            val = max(32, min(128, width // 8))
            self.window_size_y = 2 ** int(np.log2(val))
            
        # 2. Defaults for Step Size (25% overlap is standard)
        if self.window_step_x is None:
            self.window_step_x = max(1, self.window_size_x // 4)
        if self.window_step_y is None:
            self.window_step_y = max(1, self.window_size_y // 4)
            
        # 3. Create Hamming Window
        bw2d = np.outer(np.hamming(self.window_size_x), np.ones(self.window_size_y))
        self.hamming_window = np.sqrt(bw2d * bw2d.T)

    def _extract_windows_from_frame(self, frame):
        """Extracts sliding windows from a SINGLE 2D frame."""
        # Pad if necessary
        pad_h = max(0, self.window_size_x - frame.shape[0])
        pad_w = max(0, self.window_size_y - frame.shape[1])
        if pad_h > 0 or pad_w > 0:
            frame = np.pad(frame, ((0, pad_h), (0, pad_w)), mode='constant')

        window_shape = (self.window_size_x, self.window_size_y)
        step = (self.window_step_x, self.window_step_y)
        
        # Extract windows: Shape becomes (n_win_y, n_win_x, win_h, win_w)
        windows = view_as_windows(frame, window_shape, step=step)
        
        # Return flattened list of windows and the grid shape
        grid_shape = windows.shape[:2]
        windows_flat = windows.reshape(-1, self.window_size_x, self.window_size_y)
        return windows_flat, grid_shape

    def make_windows(self, data):
        """
        Extract windows from 2D (Single) or 3D (Series) data.
        Performs GLOBAL normalization to preserve relative intensity changes.
        """
        data = data.astype(float)
        
        # 1. Global Normalization (Crucial for Time Series)
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        else:
            data = data - d_min # Handle flat images

        # 2. Determine input type
        if data.ndim == 2:
            # Single Image: Add dummy time dimension -> (1, H, W)
            data = data[np.newaxis, :, :]
            self.is_series = False
        elif data.ndim == 3:
            # Time Series: (Time, H, W)
            self.is_series = True
        else:
            raise ValueError(f"Input must be 2D or 3D array. Got {data.ndim}D")

        # 3. Initialize Parameters based on first frame
        self._calculate_window_params(data.shape[1:])
        
        all_windows = []
        
        # 4. Extract windows from every frame
        for t in range(data.shape[0]):
            windows, grid_shape = self._extract_windows_from_frame(data[t])
            all_windows.append(windows)
            
        # Store grid shape for reconstruction: (Time, Grid_Y, Grid_X)
        self.grid_shape = (data.shape[0], grid_shape[0], grid_shape[1])
        
        # Stack all windows into one massive array: (Total_Windows, H, W)
        # Total_Windows = Time * Grid_Y * Grid_X
        return np.vstack(all_windows)

    def process_fft(self, windows):
        """Compute FFT for a batch of windows."""
        n_windows = windows.shape[0]
        fft_results = []
        
        # Pre-calculate zoom indices to avoid doing it inside the loop
        cx, cy = self.window_size_x // 2, self.window_size_y // 2
        zoom_sz = max(1, self.window_size_x // (2 * self.zoom_factor))
        x_sl = slice(max(0, cx - zoom_sz), min(self.window_size_x, cx + zoom_sz))
        y_sl = slice(max(0, cy - zoom_sz), min(self.window_size_y, cy + zoom_sz))

        for i in range(n_windows):
            w = windows[i]
            if self.hamming_filter:
                w = w * self.hamming_window
                
            # FFT
            fft_res = fftpack.fftshift(fftpack.fft2(w))
            fft_mag = np.log1p(np.abs(fft_res)) # Log magnitude
            
            # Zoom
            zoomed = fft_mag[x_sl, y_sl]
            
            # Interpolate (Optional)
            if self.interpol_factor > 1:
                zoomed = ndimage.zoom(zoomed, self.interpol_factor, order=1)
                
            fft_results.append(zoomed)
            
        # Stack and pad if necessary (though shapes should be uniform)
        # We assume uniform shapes for efficiency here
        self.fft_size = fft_results[0].shape
        return np.array(fft_results)

    def run_nmf(self, fft_data):
        """
        Run NMF on the stacked FFT data.
        Returns reshaped components and abundances.

        Sets diagnostic attributes on ``self`` for the wrapper to surface:
        ``reconstruction_err_``, ``relative_residual_``,
        ``component_cosine_similarity_``,
        ``component_baseline_similarity_``,
        ``abundance_cosine_similarity_``.
        """
        # Flatten: (N_Samples, H*W)
        n_samples = fft_data.shape[0]
        flat_data = fft_data.reshape(n_samples, -1)

        # Clean data
        flat_data = np.nan_to_num(flat_data)
        flat_data = np.maximum(0, flat_data)

        # Safety check for components
        n_comps = min(self.n_components, n_samples, flat_data.shape[1])
        if n_comps != self.n_components:
            warnings.warn(f"Reduced components from {self.n_components} to {n_comps} due to data size.")

        # --- Run NMF (with optional best-of-N for random init) ---
        # Deterministic inits (nndsvd*) ignore n_inits — running them
        # multiple times produces identical output. Cap to 1 in that case.
        deterministic_inits = {"nndsvd", "nndsvda", "nndsvdar"}
        effective_n_inits = (
            1 if self.init in deterministic_inits else self.n_inits
        )

        if effective_n_inits == 1:
            nmf = NMF(
                n_components=n_comps,
                init=self.init,
                random_state=self.random_state,
                max_iter=500,
            )
            W = nmf.fit_transform(flat_data)
            H = nmf.components_
        else:
            best_err = np.inf
            best_W = best_H = best_nmf = None
            seed_rng = np.random.default_rng(self.random_state)
            for _ in range(effective_n_inits):
                seed = int(seed_rng.integers(0, 2**31 - 1))
                trial = NMF(
                    n_components=n_comps,
                    init=self.init,
                    random_state=seed,
                    max_iter=500,
                )
                W_trial = trial.fit_transform(flat_data)
                if trial.reconstruction_err_ < best_err:
                    best_err = trial.reconstruction_err_
                    best_W = W_trial
                    best_H = trial.components_
                    best_nmf = trial
            nmf, W, H = best_nmf, best_W, best_H

        # --- Diagnostics: reconstruction quality & component distinctness ---
        self.reconstruction_err_ = float(nmf.reconstruction_err_)
        data_norm = float(np.linalg.norm(flat_data))
        self.relative_residual_ = (
            self.reconstruction_err_ / data_norm if data_norm > 0 else 0.0
        )
        self.component_cosine_similarity_ = _pairwise_cos_sim(H)
        self.component_baseline_similarity_ = _pairwise_baseline_subtracted_sim(H)
        # Abundance similarity uses W (samples × n_comps); we want pairs of
        # components, so transpose to (n_comps × samples) and take cos-sim.
        self.abundance_cosine_similarity_ = _pairwise_cos_sim(W.T)

        # Per-window relative residual: for each window i, how poorly the
        # basis fits it. Reshape back to grid so it's directly viewable.
        # ||X_i - (W·H)_i||_2 / ||X_i||_2 — dimensionless 0..1+, lower
        # is better. Spikes localize regions the chosen basis cannot
        # represent (artifacts, untrained phases, etc.).
        reconstruction = W @ H
        per_sample_err = np.linalg.norm(flat_data - reconstruction, axis=1)
        per_sample_norm = np.linalg.norm(flat_data, axis=1)
        per_sample_norm = np.where(per_sample_norm > 0, per_sample_norm, 1.0)
        per_sample_relative = per_sample_err / per_sample_norm

        t_steps, grid_y, grid_x = self.grid_shape
        residual_map = per_sample_relative.reshape(t_steps, grid_y, grid_x)
        if not self.is_series:
            residual_map = residual_map[0]
        self.residual_map_ = residual_map

        # --- Reshape Results ---

        # 1. Components: (n_comps, fft_h, fft_w)
        components_img = H.reshape(n_comps, self.fft_size[0], self.fft_size[1])

        # 2. Abundances: Reconstruct spatial/temporal structure
        # W is currently (Time * Grid_Y * Grid_X, n_comps)
        # We need to reshape it back to the grid structure
        t_steps, grid_y, grid_x = self.grid_shape

        # Reshape to (Time, Grid_Y, Grid_X, n_comps)
        abundances_grid = W.reshape(t_steps, grid_y, grid_x, n_comps)

        # Transpose to standard format: (Time, n_comps, Grid_Y, Grid_X)
        abundances_final = abundances_grid.transpose(0, 3, 1, 2)

        # If input was single image, squeeze out the time dimension
        if not self.is_series:
            abundances_final = abundances_final[0] # -> (n_comps, Grid_Y, Grid_X)

        return components_img, abundances_final

    def analyze(self, image_input, output_dir=None):
        """
        Main method. Handles file loading, analysis, and saving.
        
        Returns:
            components: (n_comps, h, w)
            abundances: (Time, n_comps, h, w) OR (n_comps, h, w) for single image
        """
        # 1. Load Data
        if isinstance(image_input, str):
            data = load_image(image_input)
            if output_dir is None:
                name = os.path.splitext(os.path.basename(image_input))[0]
                output_dir = f"{name}_results"
        elif isinstance(image_input, np.ndarray):
            data = image_input
            if output_dir is None:
                output_dir = "nmf_results"
        else:
            raise TypeError("Input must be path (str) or array")
            
        # 2. Execute Pipeline
        print(f"Processing data with shape {data.shape}...")
        all_windows = self.make_windows(data)
        
        print(f"Computed {len(all_windows)} total windows. Running FFT...")
        fft_data = self.process_fft(all_windows)
        
        print("Running NMF...")
        components, abundances = self.run_nmf(fft_data)
        
        # 3. Save Results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, "components.npy"), components)
            np.save(os.path.join(output_dir, "abundances.npy"), abundances)
            print(f"Saved results to {output_dir}")
            
        return components, abundances


def run_fft_nmf_analysis(image_array, params=None):
    """Run sliding FFT + NMF decomposition on an image.

    Convenience wrapper around :class:`SlidingFFTNMF` that mirrors the
    ``run_sam_analysis`` API so generated scripts can call it directly.

    Args:
        image_array: 2D numpy array (grayscale image).
        params: Optional dict with:
            - window_size (int): Side length in pixels (default: auto).
            - n_components (int): Number of NMF components (default: 4).
            - step_fraction (float): Window step as fraction of window
              size — 0.25 means 75 % overlap (default: 0.25).

    Returns:
        dict with ``components`` (n, h, w), ``abundances`` (n, gh, gw),
        ``n_components``, ``window_size``, and ``grid_shape``.
    """
    params = params or {}
    window_size = params.get("window_size")
    n_components = params.get("n_components", 4)
    step_fraction = params.get("step_fraction", 0.25)
    random_state = params.get("random_state", 42)
    init = params.get("init", "random")
    n_inits = params.get("n_inits", 1)

    kwargs = {
        "components": n_components,
        "random_state": random_state,
        "init": init,
        "n_inits": n_inits,
    }
    if window_size is not None:
        kwargs["window_size_x"] = window_size
        kwargs["window_size_y"] = window_size
        step = max(1, int(window_size * step_fraction))
        kwargs["window_step_x"] = step
        kwargs["window_step_y"] = step

    analyzer = SlidingFFTNMF(**kwargs)

    # Call pipeline steps directly to avoid stdout prints and
    # unwanted file saves that analyze() does by default.
    data = image_array.astype(float)
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min)
    windows = analyzer.make_windows(data)
    fft_data = analyzer.process_fft(windows)
    components, abundances = analyzer.run_nmf(fft_data)

    return {
        "components": components,
        "abundances": abundances,
        "n_components": int(components.shape[0]),
        "window_size": (analyzer.window_size_x, analyzer.window_size_y),
        "grid_shape": (
            analyzer.grid_shape[1],
            analyzer.grid_shape[2],
        ),
        "reconstruction_err": analyzer.reconstruction_err_,
        "relative_residual": analyzer.relative_residual_,
        "component_cosine_similarity": analyzer.component_cosine_similarity_,
        "component_baseline_similarity": analyzer.component_baseline_similarity_,
        "abundance_cosine_similarity": analyzer.abundance_cosine_similarity_,
        "residual_map": analyzer.residual_map_,
    }


# =============================================================================
# TOOL SPEC
# =============================================================================

from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="run_fft_nmf_analysis",
    description=(
        "Sliding-window FFT + NMF decomposition. Factorizes local frequency patterns "
        "across an image into a small set of basis spectra and their spatial abundance maps."
    ),
    import_line="from scilink.tools.fft_nmf import run_fft_nmf_analysis",
    signature="run_fft_nmf_analysis(image_array, params=None) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "Materials with defects, disorder, multiple phases, or spatially varying "
        "structure — or when the objective involves characterizing disorder, phase "
        "separation, or local symmetry variations. Also useful for any image where "
        "the goal is periodic patterns, symmetries, or electronic/lattice patterns "
        "rather than individual atom positions.\n"
        "\n"
        "With a window size tuned to the spatial scale of the repeating features and a "
        "simple post-processing step (e.g. inspecting components and abundance maps, "
        "thresholding or clustering abundances to localize distinct regions), this is "
        "already a complete Tier 1 pipeline — no extra processing steps are required.\n"
        "\n"
        "**What the method guarantees vs. does not:** FFT-NMF is a data-driven "
        "non-negative decomposition. It reliably produces (a) non-negative spectral "
        "components with low reconstruction error, (b) spatially coherent abundance "
        "maps when the image contains spatial variation, and (c) visually distinct "
        "components when the image contains distinct patterns. It does NOT assign "
        "semantic labels to components (e.g. one component = 'crystalline', another "
        "= 'disordered') — that is for the user to interpret. Plan's quality_criteria "
        "should target coherent, non-noise outputs rather than idealized textbook "
        "patterns or strict semantic separation the method cannot deliver. "
        "When signals co-vary spatially (lattice × LDOS envelope, topography × "
        "composition, etc.), components typically mix these signals rather than "
        "isolating each — interpret components as basis patterns, not "
        "physics-separated modes.\n"
        "\n"
        "**How to use the diagnostic metrics (relative_residual, component_*_similarity, "
        "residual_map) in `quality_criteria`:** these fields are *informational "
        "properties of the data + decomposition*, not knobs the analysis can tune to "
        "a target. If the data has only ~2 effective patterns at the chosen scale, "
        "components will be similar and abundances correlated regardless of how many "
        "components you ask for or how you tune the window — that is a finding about "
        "the data, not an analysis failure. Reference these as informational checks "
        "or in observations / caveats (e.g. *'if baseline_similarity > 0.95 across "
        "all pairs, report that the decomposition has collapsed and reduce "
        "n_components'*), but do NOT write them as hard pass/fail thresholds (e.g. "
        "*'baseline_similarity must be < 0.7'*) in `quality_criteria`. A criterion "
        "the data cannot satisfy turns into an unwinnable retry loop. Hard criteria "
        "should target things analysis parameters can actually change: non-noise "
        "components, spatially coherent abundance maps, no NaN/Inf values, "
        "abundance values in expected ranges."
    ),
    parameters={
        "image_array": {
            "type": "ndarray",
            "description": "2D grayscale numpy array.",
        },
        "params": {
            "type": "dict",
            "description": (
                "Optional. Keys: "
                "window_size (int pixels, default auto — pick based on the spatial scale "
                "of repeating features), "
                "n_components (int, default 4 — number of distinct patterns expected), "
                "step_fraction (float, default 0.25 — window step as a fraction of window "
                "size; 0.25 = 75% overlap), "
                "random_state (int, default 42 — RNG seed for reproducibility), "
                "init (str, default 'random' — NMF initialization. Use 'nndsvd' or "
                "'nndsvda' for deterministic, often more robust starts; ignored by "
                "n_inits since they are deterministic), "
                "n_inits (int, default 1 — when init='random', run that many random "
                "starts and keep the lowest-residual fit; helps escape local optima at "
                "the cost of N× compute)."
            ),
        },
    },
    required=["image_array"],
    returns=(
        "dict with: "
        "'components' (ndarray, shape (n_components, fft_h, fft_w)) — each is a 2D "
        "FFT power spectrum for one dominant frequency pattern; "
        "'abundances' (ndarray, shape (n_components, grid_h, grid_w)) — spatial "
        "maps of where each component is present; "
        "'n_components' (int); "
        "'window_size' (tuple of two ints, (width, height)); "
        "'grid_shape' (tuple of two ints, (grid_h, grid_w)); "
        "'reconstruction_err' (float, Frobenius norm of NMF residual); "
        "'relative_residual' (float, dimensionless 0-1 — reconstruction_err / "
        "||X||_F; lower is better, useful for comparing across n_components); "
        "'component_cosine_similarity' (n_components × n_components ndarray, "
        "plain cosine similarity between component spectra — sensitive to "
        "shared baseline content); "
        "'component_baseline_similarity' (n_components × n_components ndarray, "
        "cosine similarity AFTER subtracting per-pair element-wise minimum — "
        "isolates the differing peaks, so high values here indicate truly "
        "redundant components; > 0.95 means consider reducing n_components); "
        "'abundance_cosine_similarity' (n_components × n_components ndarray, "
        "cosine similarity of abundance maps — distinct components covering "
        "different spatial regions get low similarity); "
        "'residual_map' (ndarray, shape (grid_h, grid_w) for single image or "
        "(time, grid_h, grid_w) for series) — per-window relative residual "
        "||X_i - (W·H)_i||_2 / ||X_i||_2. Spikes localize regions the chosen "
        "basis cannot represent (artifacts, untrained phases, edges); useful "
        "alongside abundance maps for spotting where the decomposition fails."
    ),
    example=(
        "result = run_fft_nmf_analysis(image_array, params={'n_components': 4})\n"
        "np.save('nmf_components.npy', result['components'])\n"
        "np.save('abundance_maps.npy', result['abundances'])\n"
        "print(f\"residual={result['relative_residual']:.3f}\")\n"
        "# Off-diagonal entries of component_baseline_similarity > 0.95 \n"
        "# indicate redundant components — try a lower n_components."
    ),
)
