import warnings
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.mixture import GaussianMixture
import numpy as np

class SpectralUnmixer:
    """
    Applies various decomposition algorithms to hyperspectral data.
    Automatically handles spatially masked (zero-valued) data by processing
    only valid pixels to improve performance and accuracy.
    """
    def __init__(self, method: str = 'nmf', n_components: int = 4, normalize: bool = False, **kwargs):
        self.method = method
        self.n_components = n_components
        self.normalize = normalize
        self.kwargs = kwargs
        
        # Initialize the underlying sklearn model
        if self.method == 'nmf':
            self.model = NMF(n_components=n_components, **self.kwargs)
        elif self.method == 'pca':
            self.model = PCA(n_components=n_components, **self.kwargs)
        elif self.method == 'ica':
            self.model = FastICA(n_components=n_components, whiten='unit-variance', 
                                 max_iter=self.kwargs.get("max_iter", 200))
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=n_components, **self.kwargs)
        else:
            raise ValueError("Method not recognized. Choose from 'nmf', 'pca', 'ica', 'gmm'.")
        
        self.components_ = None
        self.abundance_maps_ = None
        self.image_shape_ = None

    def fit(self, hspy_data: np.ndarray):
        """
        Fits the model to the hyperspectral cube.
        Automatically filters out zero-valued pixels (background) before processing.
        """
        if hspy_data.ndim != 3:
            raise ValueError("Input data must be a 3D hyperspectral cube (h, w, e).")
        
        self.image_shape_ = hspy_data.shape[:2]
        h, w, e = hspy_data.shape
        
        # 1. FLATTEN
        spectra_matrix = hspy_data.reshape((h * w, e))
        
        # 2. FILTER (Internal Masking)
        # Identify pixels that have actual data (sum > small epsilon)
        pixel_sums = np.sum(spectra_matrix, axis=1)
        valid_pixel_mask = pixel_sums > 1e-6 
        
        # Create a subset containing ONLY valid pixels
        spectra_to_fit = spectra_matrix[valid_pixel_mask]
        n_valid = spectra_to_fit.shape[0]

        # Guardrail: Prevent crashing on tiny regions
        if n_valid < self.n_components:
            # We raise a ValueError so the tool wrapper can catch it gracefully
            raise ValueError(f"Too few valid pixels ({n_valid}) for {self.n_components} components.")

        # --- Normalization (Optional) ---
        l1_norms_subset = None
        if self.normalize:
            # Calculate norms only for the valid subset
            l1_norms_subset = np.sum(spectra_to_fit, axis=1, keepdims=True)
            l1_norms_subset[l1_norms_subset == 0] = 1 
            spectra_to_fit = spectra_to_fit / l1_norms_subset

        # --- Pre-check for NMF ---
        if self.method == 'nmf':
            min_val = np.min(spectra_to_fit)
            if min_val < 0:
                warnings.warn(f"NMF requires non-negative data. Shifting data by {-min_val:.2f}.")
                spectra_to_fit = spectra_to_fit - min_val

        # 3. DECOMPOSE (Run algorithms on the subset)
        subset_abundances = None

        if self.method == 'gmm':
            # Special logic for GMM (PCA dimensionality reduction first)
            pca_param = self.kwargs.get('pca_dims', 0.99)
            
            # PCA on subset
            pca_full = PCA()
            pca_full.fit(spectra_to_fit)
            
            if isinstance(pca_param, int):
                n_components_pca = pca_param
            elif isinstance(pca_param, float) and 0 < pca_param < 1:
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                n_components_pca = np.searchsorted(cumulative_variance, pca_param) + 1
            else:
                n_components_pca = min(spectra_to_fit.shape)

            pca_final = PCA(n_components=n_components_pca)
            projected_data = pca_final.fit_transform(spectra_to_fit)
            
            # Fit GMM
            self.model.fit(projected_data)
            labels = self.model.predict(projected_data)
            subset_abundances = self.model.predict_proba(projected_data)
            
            # Reconstruct component centers (mean spectra)
            self.components_ = np.array([
                spectra_to_fit[labels == i].mean(axis=0) 
                for i in range(self.n_components)
            ])
        else:
            # Standard NMF/PCA/ICA workflow
            subset_abundances = self.model.fit_transform(spectra_to_fit)
            self.components_ = self.model.components_

        # Restore intensity magnitude if we normalized
        if self.normalize and l1_norms_subset is not None:
            subset_abundances = subset_abundances * l1_norms_subset

        # 4. REFOLD (Reconstruct the full 2D images)
        # Create a blank (black) canvas of the original size
        full_abundances = np.zeros((h * w, self.n_components))
        
        # Paint the calculated abundances back into the correct pixel locations
        full_abundances[valid_pixel_mask] = subset_abundances
        
        # Reshape to (H, W, n_components) for visualization
        self.abundance_maps_ = full_abundances.reshape((h, w, self.n_components))
        
        return self.components_, self.abundance_maps_
