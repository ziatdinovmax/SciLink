import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize, Normalize
from botorch.models.transforms.input import Warp, ChainedInputTransform
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.generation import MaxPosteriorSampling
from botorch.sampling import SobolQMCNormalSampler

from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan


ALLOWED_KERNELS = {
    "matern_2.5": {"class": MaternKernel, "kwargs": {"nu": 2.5}},
    "matern_1.5": {"class": MaternKernel, "kwargs": {"nu": 1.5}},
    "matern_0.5": {"class": MaternKernel, "kwargs": {"nu": 0.5}},
    "rbf":        {"class": RBFKernel,    "kwargs": {}},
}

ALLOWED_NOISE_PRIORS = {
    "fixed_low":  {"min_noise": 1e-4},
    "learnable":  {"min_noise": 1e-5},
    "high_noise": {"min_noise": 1e-2},
}

ALLOWED_INPUT_TRANSFORMS = {"none", "warp"}


def build_input_transform(key: str, input_dim: int) -> "ChainedInputTransform | Normalize":
    """Construct the GP's input_transform. 'warp' chains Normalize -> Warp to give
    per-axis non-stationary stretching; 'none' is the default Normalize-only path."""
    normalize = Normalize(d=input_dim)
    if key == "warp":
        warp = Warp(d=input_dim, indices=list(range(input_dim)))
        return ChainedInputTransform(normalize=normalize, warp=warp)
    return normalize

def build_covar_module(kernel_key: str, input_dim: int) -> ScaleKernel:
    """Factory for Kernel selection."""
    if kernel_key not in ALLOWED_KERNELS:
        kernel_key = "matern_2.5"
    
    config = ALLOWED_KERNELS[kernel_key]
    base_kernel = config["class"](ard_num_dims=input_dim, **config["kwargs"])
    return ScaleKernel(base_kernel)

def build_likelihood(noise_key: str) -> GaussianLikelihood:
    """Factory for Likelihood/Noise selection."""
    if noise_key not in ALLOWED_NOISE_PRIORS:
        noise_key = "fixed_low"
        
    config = ALLOWED_NOISE_PRIORS[noise_key]
    noise_constraint = GreaterThan(config["min_noise"])
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
    
    # Initialize slightly above min to aid convergence
    likelihood.noise = torch.tensor(config["min_noise"] * 2.0)
    return likelihood


def _acq_contour_levels(acq_map: np.ndarray, n_levels: int = 30) -> np.ndarray:
    """Compute contour levels that focus on the acquisition peaks.

    Near observed points the acquisition value can be extremely low (e.g.,
    log_ei = -27000), creating dark bands that dominate the colormap.
    We clip the lower bound aggressively so the colormap highlights where
    the optimizer actually wants to sample (the bright regions).
    """
    vmax = np.percentile(acq_map, 99)
    # Use the median as the lower bound — this ensures at least half the
    # landscape gets color variation, regardless of extreme lows
    vmin = np.median(acq_map)
    # If median == max (very flat landscape), fall back to wider range
    if vmin >= vmax:
        vmin = np.percentile(acq_map, 10)
    if vmin >= vmax:
        vmin = acq_map.min()
    if vmin == vmax:
        vmax = vmin + 1
    return np.linspace(vmin, vmax, n_levels)


class SingleObjectiveOptimizer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None
        self.y_train = None
        self.bounds = None
        self.input_dim = 0
        self.feature_names = []
        # Persisted after recommend()
        self.acq_func = None
        self.acq_strategy_name = None

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: List[Tuple[float, float]], 
            model_config: Dict[str, str], feature_names: List[str] = None):
        """Fits the SingleTaskGP."""
        self.X_train = torch.tensor(X, dtype=torch.double, device=self.device)
        self.y_train = torch.tensor(y, dtype=torch.double, device=self.device)
        if self.y_train.ndim == 1: self.y_train = self.y_train.unsqueeze(-1)
        
        self.input_dim = self.X_train.shape[-1]
        self.bounds = torch.tensor(bounds, dtype=torch.double, device=self.device).T
        self.feature_names = feature_names or [f"x{i}" for i in range(self.input_dim)]

        kernel_choice = model_config.get("kernel", "matern_2.5")
        noise_choice = model_config.get("noise", "fixed_low")
        input_transform_choice = model_config.get("input_transform", "none")

        covar = build_covar_module(kernel_choice, self.input_dim)
        likelihood = build_likelihood(noise_choice)
        input_transform = build_input_transform(input_transform_choice, self.input_dim)

        self.model = SingleTaskGP(
            self.X_train,
            self.y_train,
            covar_module=covar,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=Standardize(m=1)
        )
        
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        
        # Clear stale acquisition function from previous fit
        self.acq_func = None
        self.acq_strategy_name = None

    def recommend(self, n_candidates: int = 1, strategy: str = 'log_ei', params: Dict[str, float] = None) -> np.ndarray:
        """
        Generates n_candidates. 
        Supports 'thompson' for high-throughput batches, and 'ucb'/'log_ei' for precision.
        """
        if self.model is None: raise RuntimeError("Call fit() first.")
        params = params or {}
        
        self.acq_strategy_name = strategy

        # --- 1. Thompson Sampling (High Throughput / Diversity) ---
        if strategy == 'thompson':
            self.acq_func = None  # No persistent acq object for Thompson
            n_pool = min(10000, max(2000, 100 * n_candidates))
            X_cand = draw_sobol_samples(bounds=self.bounds, n=n_pool, q=1).squeeze(1)
            
            thompson_sampler = MaxPosteriorSampling(model=self.model, replacement=False)
            candidates = thompson_sampler(X_cand, num_samples=n_candidates)
            return candidates.detach().cpu().numpy()

        # --- 2. Acquisition Functions ---
        if strategy == 'ucb':
            beta = params.get('beta', 2.0)
            acq_func = qUpperConfidenceBound(model=self.model, beta=beta)
            
        elif strategy == 'max_variance':
            acq_func = qUpperConfidenceBound(model=self.model, beta=1000.0)
            
        else: # Default: 'log_ei'
            best_f = self.y_train.max()
            if n_candidates > 1:
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
                acq_func = qLogExpectedImprovement(
                    model=self.model, 
                    best_f=best_f,
                    sampler=sampler
                )
            else:
                acq_func = LogExpectedImprovement(model=self.model, best_f=best_f)

        # Persist the acquisition function
        self.acq_func = acq_func

        # --- 3. Optimization (Greedy Batch) ---
        is_large_batch = n_candidates > 10
        use_sequential = n_candidates > 1 and strategy != 'thompson'
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_candidates,
            num_restarts=2 if is_large_batch else 10,
            raw_samples=128 if is_large_batch else 512,
            sequential=use_sequential
        )
        return candidates.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Acquisition function evaluation (for constrained batch planning)
    # ------------------------------------------------------------------ #

    def evaluate_acquisition(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate acquisition landscape at arbitrary points.
        
        For standard acquisition functions (EI, UCB): evaluates acq_func directly.
        For Thompson sampling: draws a single posterior sample and evaluates it.
        For no strategy yet: raises RuntimeError.
        
        Args:
            X: (N, D) array of points to evaluate
            
        Returns:
            (N,) array of acquisition values
        """
        X_t = torch.tensor(X, dtype=torch.double, device=self.device)
        
        if self.acq_func is not None:
            # Standard acquisition function (EI, UCB, max_variance)
            acq_values = np.empty(len(X_t))
            chunk_size = 256
            with torch.no_grad():
                for start in range(0, len(X_t), chunk_size):
                    end = min(start + chunk_size, len(X_t))
                    chunk = X_t[start:end].unsqueeze(1)  # (chunk, 1, d)
                    acq_values[start:end] = self.acq_func(chunk).cpu().numpy()
            return acq_values
        
        elif self.acq_strategy_name == 'thompson':
            # Thompson sampling: a single posterior draw IS the acquisition surface.
            # Must chunk to avoid OOM — GP posterior builds full N×N covariance.
            # We use posterior mean + scaled posterior std with a fixed random draw
            # to create a consistent pseudo-sample across chunks.
            self.model.eval()
            acq_values = np.empty(len(X_t))
            chunk_size = 256
            
            # Draw a single global random seed vector for consistency across chunks
            rng = np.random.RandomState(42)
            
            with torch.no_grad():
                for start in range(0, len(X_t), chunk_size):
                    end = min(start + chunk_size, len(X_t))
                    chunk = X_t[start:end]
                    posterior = self.model.posterior(chunk)
                    mean = posterior.mean.squeeze(-1)       # (chunk_size,)
                    std = posterior.variance.sqrt().squeeze(-1)  # (chunk_size,)
                    # Consistent random perturbation per point
                    z = torch.tensor(
                        rng.randn(end - start), 
                        dtype=torch.double, device=self.device
                    )
                    sample = mean + std * z
                    acq_values[start:end] = sample.cpu().numpy()
            
            return acq_values
        
        else:
            raise RuntimeError(
                "No acquisition function available. "
                "Call recommend() first to establish a strategy."
            )

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return posterior mean and variance at points X.
        
        Args:
            X: (N, D) array of points
            
        Returns:
            Tuple of (mean, variance) arrays, each shape (N,)
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.double, device=self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_t)
            mean = posterior.mean.cpu().numpy().squeeze()
            variance = posterior.variance.cpu().numpy().squeeze()
        return mean, variance

    # ------------------------------------------------------------------ #
    #  Acquisition function evaluation helpers (for plotting)
    # ------------------------------------------------------------------ #

    def _evaluate_acq_on_grid(self, dim1: int, dim2: int,
                               anchor: np.ndarray, resolution: int = 50
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the acquisition function on a 2D grid by sweeping dim1 and dim2,
        holding all other dimensions fixed at *anchor* values.

        Returns:
            (grid_x1, grid_x2, acq_values) — meshgrid arrays and evaluated values.
        """
        b_min = self.bounds[0].cpu().numpy()
        b_max = self.bounds[1].cpu().numpy()

        x1 = np.linspace(b_min[dim1], b_max[dim1], resolution)
        x2 = np.linspace(b_min[dim2], b_max[dim2], resolution)
        g1, g2 = np.meshgrid(x1, x2)

        flat = np.tile(anchor, (resolution * resolution, 1))
        flat[:, dim1] = g1.ravel()
        flat[:, dim2] = g2.ravel()

        acq_values = self.evaluate_acquisition(flat)

        return g1, g2, acq_values.reshape(resolution, resolution)

    def _evaluate_acq_1d(self, dim: int, anchor: np.ndarray,
                          resolution: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the acquisition function along a single dimension,
        holding all other dimensions fixed at *anchor* values.

        Returns:
            (x_values, acq_values)
        """
        b_min = self.bounds[0, dim].item()
        b_max = self.bounds[1, dim].item()

        x_vals = np.linspace(b_min, b_max, resolution)
        X_sweep = np.tile(anchor, (resolution, 1))
        X_sweep[:, dim] = x_vals

        acq_vals = self.evaluate_acquisition(X_sweep)

        return x_vals, acq_vals

    # ------------------------------------------------------------------ #
    #  Public: plot & save acquisition landscape
    # ------------------------------------------------------------------ #

    def plot_acquisition(self, candidate_x: np.ndarray, save_path: str,
                         dims: Optional[List[int]] = None,
                         resolution: int = 50) -> str:
        """
        Plots the acquisition function landscape and saves to disk.

        For 1D inputs: line plot of acquisition value vs the single parameter.
        For 2D inputs: filled-contour heatmap of the full landscape.
        For ND inputs (N>2): 2D heatmap of the top-2 most important dimensions
            (by Sobol sensitivity) plus 1D marginal slices for every dimension.

        Works with all acquisition strategies including Thompson sampling 
        (uses a single posterior sample as the acquisition surface).

        Args:
            candidate_x: Recommended candidate(s) from recommend(). Shape (n_candidates, d).
                          The first candidate is used as the anchor point for slicing.
            save_path: File path to save the plot (.png).
            dims: Optional pair of dimension indices for the 2D heatmap.
                  If None, auto-selects via Sobol sensitivity.
            resolution: Grid resolution per axis.

        Returns:
            save_path on success.

        Raises:
            RuntimeError: If recommend() hasn't been called yet.
        """
        if self.acq_func is None and self.acq_strategy_name != 'thompson':
            raise RuntimeError(
                "No acquisition function available. "
                "Call recommend() first to establish a strategy."
            )

        anchor = candidate_x[0] if candidate_x.ndim == 2 else candidate_x

        if self.input_dim == 1:
            return self._plot_acq_1d(anchor, save_path, resolution)
        elif self.input_dim == 2:
            return self._plot_acq_2d(anchor, candidate_x, save_path, dims=[0, 1],
                                      resolution=resolution)
        else:
            return self._plot_acq_nd(anchor, candidate_x, save_path, dims=dims,
                                      resolution=resolution)

    def _plot_acq_1d(self, anchor: np.ndarray, save_path: str, resolution: int) -> str:
        x_vals, acq_vals = self._evaluate_acq_1d(dim=0, anchor=anchor,
                                                   resolution=resolution * 4)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, acq_vals, 'b-', linewidth=2,
                label=f'Acq ({self.acq_strategy_name})')
        ax.axvline(anchor[0], color='red', linestyle='--', linewidth=1.5,
                    label='Candidate')
        ax.scatter(self.X_train.cpu().numpy().flatten(),
                   np.full(len(self.X_train), acq_vals.min()),
                   marker='|', color='black', s=100, zorder=5, label='Observations')

        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel('Log(EI)' if self.acq_strategy_name == 'log_ei' else 'Acquisition Value')
        ax.set_title(f'Acquisition Function: {self.acq_strategy_name}')
        ax.legend(); ax.grid(True, alpha=0.3)

        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
        return save_path

    def _plot_acq_2d(self, anchor: np.ndarray, candidate_x: np.ndarray,
                      save_path: str, dims: List[int], resolution: int) -> str:
        d1, d2 = dims[0], dims[1]
        g1, g2, acq_map = self._evaluate_acq_on_grid(d1, d2, anchor, resolution)

        fig, ax = plt.subplots(figsize=(10, 8))
        levels = _acq_contour_levels(acq_map, 50)
        contour = ax.contourf(g1, g2, acq_map, levels=levels, cmap='viridis', extend='both')
        fig.colorbar(contour, ax=ax, label='Log(EI)' if self.acq_strategy_name == 'log_ei' else 'Acquisition Value')

        X_np = self.X_train.cpu().numpy()
        ax.scatter(X_np[:, d1], X_np[:, d2], c='white', edgecolors='black',
                   s=40, zorder=5, label='Observations')

        cands = candidate_x if candidate_x.ndim == 2 else candidate_x[np.newaxis, :]
        ax.scatter(cands[:, d1], cands[:, d2], c='red', marker='*',
                   s=200, edgecolors='darkred', zorder=6, label='Candidates')

        ax.set_xlabel(self.feature_names[d1])
        ax.set_ylabel(self.feature_names[d2])
        ax.set_title(f'Acquisition Function: {self.acq_strategy_name}')
        ax.legend(loc='upper right')

        fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
        return save_path

    def _plot_acq_nd(self, anchor: np.ndarray, candidate_x: np.ndarray,
                      save_path: str, dims: Optional[List[int]],
                      resolution: int) -> str:
        """
        For N>2 dimensions:
        - Top panel: 2D heatmap slice through the two most important dimensions.
        - Bottom panels: 1D acquisition slices for each dimension.
        """
        if dims and len(dims) >= 2:
            top_dims = dims[:2]
        else:
            try:
                names, scores = self._compute_sensitivity()
                sorted_idx = np.argsort(scores)[::-1]
                top_dims = [sorted_idx[0], sorted_idx[1]]
            except Exception:
                top_dims = [0, 1]

        import matplotlib.gridspec as gridspec

        n_1d = self.input_dim
        n_1d_rows = max(1, (n_1d + 2) // 3)
        fig = plt.figure(figsize=(14, 5 + 3 * n_1d_rows))
        gs = gridspec.GridSpec(1 + n_1d_rows, 3, figure=fig, hspace=0.45, wspace=0.35)

        # --- 2D heatmap (top, full width) ---
        ax_2d = fig.add_subplot(gs[0, :])
        d1, d2 = top_dims
        g1, g2, acq_map = self._evaluate_acq_on_grid(d1, d2, anchor, resolution)

        levels = _acq_contour_levels(acq_map, 50)
        contour = ax_2d.contourf(g1, g2, acq_map, levels=levels, cmap='viridis', extend='both')
        fig.colorbar(contour, ax=ax_2d, label='Log(EI)' if self.acq_strategy_name == 'log_ei' else 'Acquisition Value', fraction=0.02)

        X_np = self.X_train.cpu().numpy()
        ax_2d.scatter(X_np[:, d1], X_np[:, d2], c='white', edgecolors='black',
                      s=30, zorder=5)
        cands = candidate_x if candidate_x.ndim == 2 else candidate_x[np.newaxis, :]
        ax_2d.scatter(cands[:, d1], cands[:, d2], c='red', marker='*',
                      s=200, edgecolors='darkred', zorder=6, label='Candidates')
        ax_2d.set_xlabel(self.feature_names[d1])
        ax_2d.set_ylabel(self.feature_names[d2])
        ax_2d.set_title(f'Acquisition ({self.acq_strategy_name}) — '
                        f'{self.feature_names[d1]} vs {self.feature_names[d2]}')
        ax_2d.legend(loc='upper right', fontsize=8)

        # --- 1D slices (bottom rows) ---
        for i in range(n_1d):
            row = 1 + i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            x_vals, acq_vals = self._evaluate_acq_1d(dim=i, anchor=anchor,
                                                       resolution=resolution * 4)
            ax.plot(x_vals, acq_vals, 'b-', linewidth=1.5)
            ax.axvline(anchor[i], color='red', linestyle='--', alpha=0.7,
                        label='Candidate')
            ax.scatter(X_np[:, i], np.full(len(X_np), acq_vals.min()),
                       marker='|', color='black', s=60, alpha=0.5)
            ax.set_xlabel(self.feature_names[i])
            ax.set_ylabel('Acq Value')
            ax.set_title(f'Slice: {self.feature_names[i]}', fontsize=10)
            ax.grid(True, alpha=0.2)

        fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        return save_path

    def save_acquisition_data(self, candidate_x: np.ndarray, save_path: str,
                               resolution: int = 50) -> Dict[str, Any]:
        """
        Evaluates and saves the acquisition function landscape to a .npz file
        for later analysis or custom plotting.

        Works with all acquisition strategies including Thompson sampling.

        Saved arrays:
        - 1D sweeps for every dimension (keyed as 'x_dim0', 'acq_dim0', etc.)
        - If input_dim <= 4, pairwise 2D grids ('grid1_0_1', 'grid2_0_1', 'acq_0_1')
        - Metadata: strategy name, feature names, bounds, candidate, training data

        Args:
            candidate_x: Candidate(s) from recommend(). First row used as anchor.
            save_path: Output path (.npz).
            resolution: Grid points per axis.

        Returns:
            Dict with keys saved and the file path.

        Raises:
            RuntimeError: If recommend() hasn't been called yet.
        """
        if self.acq_func is None and self.acq_strategy_name != 'thompson':
            raise RuntimeError(
                "No acquisition function available. "
                "Call recommend() first to establish a strategy."
            )

        anchor = candidate_x[0] if candidate_x.ndim == 2 else candidate_x
        data = {
            "strategy": np.array(self.acq_strategy_name),
            "feature_names": np.array(self.feature_names),
            "bounds_lower": self.bounds[0].cpu().numpy(),
            "bounds_upper": self.bounds[1].cpu().numpy(),
            "candidate": anchor,
            "X_train": self.X_train.cpu().numpy(),
            "y_train": self.y_train.cpu().numpy(),
        }

        # 1D sweeps for every dimension
        for i in range(self.input_dim):
            x_vals, acq_vals = self._evaluate_acq_1d(dim=i, anchor=anchor,
                                                       resolution=resolution * 4)
            data[f"x_dim{i}"] = x_vals
            data[f"acq_dim{i}"] = acq_vals

        # 2D grids for low-dimensional problems
        if self.input_dim <= 4:
            import itertools
            for d1, d2 in itertools.combinations(range(self.input_dim), 2):
                g1, g2, acq_map = self._evaluate_acq_on_grid(d1, d2, anchor, resolution)
                data[f"grid1_{d1}_{d2}"] = g1
                data[f"grid2_{d1}_{d2}"] = g2
                data[f"acq_{d1}_{d2}"] = acq_map

        np.savez_compressed(save_path, **data)

        return {
            "path": save_path,
            "keys": list(data.keys()),
            "strategy": self.acq_strategy_name,
            "n_dims": self.input_dim
        }

    # ------------------------------------------------------------------ #
    #  Existing diagnostics (unchanged)
    # ------------------------------------------------------------------ #

    def _compute_sensitivity(self, n_samples=2048) -> Tuple[List[str], List[float]]:
        """Helper: Total-order Sobol indices (Jansen 1999) for diagnostics.

        ST_i = (1 / 2N) * sum_k (f(A) - f(A_B^{(i)}))^2 / Var(Y), where A_B^{(i)}
        is A with its i-th column replaced by B's i-th column. ST_i captures all
        variance contributions involving X_i — first-order plus every interaction —
        so variables whose influence is purely through interactions don't get
        hidden at zero.

        Note on sampling: A and B must be independent. Splitting a single
        Sobol sequence in half does NOT give independence — the halves share
        the low-discrepancy lattice and column i of the two halves ends up
        near-identical for low dimensions, which collapses ST_i to ~0. We
        instead draw a single 2d-dimensional Sobol sample and split columns.
        """
        if self.input_dim == 1: return [self.feature_names[0]], [1.0]

        # Double-width bounds so the first d columns and last d columns sample
        # the same physical space but use different Sobol dimensions (i.e. are
        # effectively independent).
        double_bounds = torch.cat([self.bounds, self.bounds], dim=1)
        X_sobol = draw_sobol_samples(bounds=double_bounds, n=n_samples, q=1).squeeze(1)
        A, B = X_sobol[:, :self.input_dim], X_sobol[:, self.input_dim:]

        def predict(X):
            with torch.no_grad(): return self.model.posterior(X).mean.flatten()

        f_A, f_B = predict(A), predict(B)
        var_Y = torch.var(torch.cat([f_A, f_B])) + 1e-9

        indices = []
        for i in range(self.input_dim):
            AB_i = A.clone(); AB_i[:, i] = B[:, i]
            numerator = 0.5 * torch.mean((f_A - predict(AB_i)) ** 2)
            indices.append(max(0.0, (numerator / var_Y).item()))
        return self.feature_names, indices

    def generate_diagnostics(self, candidate_x: np.ndarray, history_y: List[float], save_path: str, n_initial: int = 0):
        """Generates 4-Panel Dashboard: LOO-CV Residuals, Trend, Acquisition Slice, Sensitivity."""
        x_plot = candidate_x[0:1]

        y_np = self.y_train.cpu().numpy().flatten()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # --- 1. LOO-CV Residuals (Top Left) ---
        ax_res = axes[0, 0]
        n_pts = len(y_np)
        LOO_MAX = 50  # Only run LOO-CV for small datasets

        if n_pts <= LOO_MAX:
            # Leave-one-out cross-validation
            loo_residuals = np.zeros(n_pts)
            loo_std = np.zeros(n_pts)
            X_all = self.X_train.clone()
            y_all = self.y_train.clone()

            for i in range(n_pts):
                # Remove point i
                mask = torch.ones(n_pts, dtype=torch.bool)
                mask[i] = False
                X_loo = X_all[mask]
                y_loo = y_all[mask]

                # Fit a temporary GP on N-1 points
                try:
                    from botorch.models import SingleTaskGP
                    from botorch.fit import fit_gpytorch_mll
                    from gpytorch.mlls import ExactMarginalLogLikelihood

                    loo_model = SingleTaskGP(X_loo, y_loo)
                    mll = ExactMarginalLogLikelihood(loo_model.likelihood, loo_model)
                    fit_gpytorch_mll(mll)

                    loo_model.eval()
                    with torch.no_grad():
                        pred = loo_model.posterior(X_all[i:i+1], observation_noise=False)
                        loo_residuals[i] = y_np[i] - pred.mean.cpu().numpy().flatten()[0]
                        loo_std[i] = pred.variance.sqrt().cpu().numpy().flatten()[0]
                except Exception:
                    loo_residuals[i] = 0
                    loo_std[i] = 0

            max_resid = np.abs(loo_residuals).max()
            bar_width = 0.8
            ax_res.bar(np.arange(n_pts), loo_residuals, width=bar_width,
                       color='steelblue', alpha=0.7, edgecolor='white')
            ax_res.axhline(0, color='red', linestyle='--', linewidth=1)
            # Show prediction uncertainty from LOO — extend band to cover full bar width
            if loo_std.max() > 0:
                # Create step-wise band that covers each bar fully
                x_band = []
                y_lo = []
                y_hi = []
                for i in range(n_pts):
                    x_band.extend([i - bar_width / 2, i + bar_width / 2])
                    y_lo.extend([-loo_std[i], -loo_std[i]])
                    y_hi.extend([loo_std[i], loo_std[i]])
                ax_res.fill_between(x_band, y_lo, y_hi,
                                    color='red', alpha=0.1, label='LOO prediction uncertainty (1\u03c3)')
            y_extent = max(max_resid, loo_std.max()) * 1.3
            if y_extent == 0:
                y_extent = 0.1
            ax_res.set_ylim(-y_extent, y_extent)
            ax_res.set_title('1. LOO-CV Residuals')
        else:
            # Too many points for LOO — show training residuals instead
            self.model.eval()
            with torch.no_grad():
                posterior = self.model.posterior(self.X_train, observation_noise=False)
                pred_mean = posterior.mean.cpu().numpy().flatten()
            residuals = y_np - pred_mean
            max_resid = np.abs(residuals).max() if len(residuals) > 0 else 0
            ax_res.bar(np.arange(n_pts), residuals, color='steelblue', alpha=0.7, edgecolor='white')
            ax_res.axhline(0, color='red', linestyle='--', linewidth=1)
            y_extent = max_resid * 1.5 if max_resid > 0 else 0.1
            ax_res.set_ylim(-y_extent, y_extent)
            ax_res.set_title(f'1. Training Residuals (n={n_pts}, LOO skipped)')

        ax_res.set_xlabel('Data point index')
        ax_res.set_ylabel('Residual (Observed \u2212 Predicted)')
        if ax_res.get_legend_handles_labels()[1]:
            ax_res.legend(fontsize=8)

        # --- 2. Data Overview / Optimization Trend (Top Right) ---
        ax_trend = axes[0, 1]
        steps = np.arange(1, len(history_y) + 1)
        if n_initial > 0 and n_initial >= len(history_y):
            # No BO-guided points yet — show initial data only
            ax_trend.plot(steps, history_y, 's', color='gray', alpha=0.5, label='Initial data')
            ax_trend.set_title("2. Initial Data Overview")
        elif n_initial > 0 and n_initial < len(history_y):
            # Mix of initial + BO-guided
            ax_trend.plot(steps[:n_initial], history_y[:n_initial], 's', color='gray', alpha=0.4, label='Initial data')
            ax_trend.plot(steps[n_initial:], history_y[n_initial:], 'ko-', alpha=0.5, label='BO-guided')
            ax_trend.axvline(n_initial + 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)
            ax_trend.plot(steps, np.maximum.accumulate(history_y), 'g-', linewidth=2, label='Best Found')
            ax_trend.set_title("2. Optimization Trend")
        else:
            ax_trend.plot(steps, history_y, 'ko-', alpha=0.3)
            ax_trend.plot(steps, np.maximum.accumulate(history_y), 'g-', linewidth=2, label='Best Found')
            ax_trend.set_title("2. Optimization Trend")
        ax_trend.legend(fontsize=8)

        # --- 3. Sensitivity (Bottom Right) ---
        ax_sens = axes[1, 1]
        top_dim_idx = 0
        sorted_idx = np.arange(self.input_dim)  # default: original order
        sensitivity_data = {}
        try:
            names, scores = self._compute_sensitivity()
            sorted_idx = np.argsort(scores)[::-1]
            top_dim_idx = sorted_idx[0]
            sensitivity_data = {names[i]: round(scores[i], 4) for i in sorted_idx}

            y_pos = np.arange(len(names))
            ax_sens.barh(y_pos, [scores[i] for i in sorted_idx], align='center', color='skyblue')
            ax_sens.set_yticks(y_pos)
            ax_sens.set_yticklabels([names[i] for i in sorted_idx])
            ax_sens.invert_yaxis()
            ax_sens.set_xlabel('Total-Order Sobol Index')
            ax_sens.set_title("4. Model Sensitivity")
        except Exception as e:
            ax_sens.text(0.5, 0.5, f"Analysis Error: {str(e)}", ha='center')

        # --- 3. Acquisition Function (Bottom Left) ---
        # Uses evaluate_acquisition() which handles all strategies including Thompson.
        ax_acq = axes[1, 0]
        anchor = x_plot[0]
        X_np = self.X_train.cpu().numpy()

        has_acquisition = (self.acq_func is not None) or (self.acq_strategy_name == 'thompson')

        if has_acquisition:
            if self.input_dim == 1:
                # --- 1D: full acquisition curve ---
                x_vals, acq_vals = self._evaluate_acq_1d(
                    dim=0, anchor=anchor, resolution=200
                )
                ax_acq.plot(x_vals, acq_vals, 'b-', linewidth=2,
                            label=self.acq_strategy_name)
                ax_acq.axvline(anchor[0], color='red', linestyle='--',
                               linewidth=1.5, label='Candidate')
                ax_acq.scatter(X_np.flatten(),
                               np.full(len(X_np), acq_vals.min()),
                               marker='|', color='black', s=80, zorder=5,
                               label='Observations')
                ax_acq.set_xlabel(self.feature_names[0])
                ax_acq.set_ylabel('Acquisition Value')
                ax_acq.set_title(f"3. Acquisition Function ({self.acq_strategy_name})")

            elif self.input_dim == 2:
                # --- 2D: full heatmap ---
                g1, g2, acq_map = self._evaluate_acq_on_grid(
                    0, 1, anchor, resolution=50
                )
                _levels = _acq_contour_levels(acq_map, 30)
                contour = ax_acq.contourf(g1, g2, acq_map, levels=_levels,
                                           cmap='viridis', extend='both')
                _acq_label = 'Log(EI)' if self.acq_strategy_name == 'log_ei' else 'Acq. Value'
                fig.colorbar(contour, ax=ax_acq, fraction=0.046, pad=0.04, label=_acq_label)
                ax_acq.scatter(X_np[:, 0], X_np[:, 1], c='white',
                               edgecolors='black', s=30, zorder=5,
                               label='Observations')
                cands = candidate_x if candidate_x.ndim == 2 else candidate_x[np.newaxis, :]
                ax_acq.scatter(cands[:, 0], cands[:, 1], c='red', marker='*',
                               s=150, edgecolors='darkred', zorder=6,
                               label='Candidates')
                ax_acq.set_xlabel(self.feature_names[0])
                ax_acq.set_ylabel(self.feature_names[1])
                ax_acq.set_title(f"3. Acquisition Function ({self.acq_strategy_name})")

            else:
                # --- d>2: 2D heatmap of top-2 Sobol dimensions ---
                # top_dim_idx is already sorted_idx[0]; get second
                second_dim_idx = sorted_idx[1] if len(sorted_idx) > 1 else (1 if top_dim_idx == 0 else 0)
                g1, g2, acq_map = self._evaluate_acq_on_grid(
                    top_dim_idx, second_dim_idx, anchor, resolution=50
                )
                _levels = _acq_contour_levels(acq_map, 30)
                contour = ax_acq.contourf(g1, g2, acq_map, levels=_levels,
                                           cmap='viridis', extend='both')
                _acq_label = 'Log(EI)' if self.acq_strategy_name == 'log_ei' else 'Acq. Value'
                fig.colorbar(contour, ax=ax_acq, fraction=0.046, pad=0.04, label=_acq_label)
                ax_acq.scatter(X_np[:, top_dim_idx], X_np[:, second_dim_idx],
                               c='white', edgecolors='black', s=30, zorder=5,
                               label='Observations')
                cands = candidate_x if candidate_x.ndim == 2 else candidate_x[np.newaxis, :]
                ax_acq.scatter(cands[:, top_dim_idx], cands[:, second_dim_idx],
                               c='red', marker='*', s=150, edgecolors='darkred',
                               zorder=6, label='Candidates')
                dim1_name = self.feature_names[top_dim_idx]
                dim2_name = self.feature_names[second_dim_idx]
                ax_acq.set_xlabel(dim1_name)
                ax_acq.set_ylabel(dim2_name)
                ax_acq.set_title(f"3. Acquisition ({self.acq_strategy_name}) — "
                                 f"{dim1_name} vs {dim2_name}")

        else:
            # Fallback: no strategy set at all — show GP posterior
            dim_name = self.feature_names[top_dim_idx]
            b_min = self.bounds[0, top_dim_idx].item()
            b_max = self.bounds[1, top_dim_idx].item()
            x_sweep = np.linspace(b_min, b_max, 200)
            X_slice = np.tile(x_plot, (200, 1))
            X_slice[:, top_dim_idx] = x_sweep
            with torch.no_grad():
                post = self.model.posterior(
                    torch.tensor(X_slice, dtype=torch.double, device=self.device)
                )
                mu = post.mean.cpu().numpy().flatten()
                sigma = post.variance.sqrt().cpu().numpy().flatten()
            ax_acq.plot(x_sweep, mu, 'b-', label='GP Mean')
            ax_acq.fill_between(x_sweep, mu - 1.96 * sigma, mu + 1.96 * sigma,
                                alpha=0.2, color='b')
            ax_acq.axvline(anchor[top_dim_idx], color='red', linestyle='--',
                           label='Candidate')
            ax_acq.set_xlabel(dim_name)
            ax_acq.set_ylabel('Predicted Value')
            ax_acq.set_title(f"3. GP Posterior along '{dim_name}' (no acq func)")

        ax_acq.legend()
        ax_acq.grid(True, alpha=0.3)

        fig.tight_layout(); fig.savefig(save_path); plt.close(fig)
        return sensitivity_data


class MultiObjectiveOptimizer:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None
        self.y_train = None
        self.bounds = None
        self.input_dim = 0
        self.output_dim = 0
        self.feature_names = []
        # Persisted after recommend()
        self.acq_func = None
        self.acq_strategy_name = None

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: List[Tuple[float, float]], 
            model_config: Dict[str, str], feature_names: List[str] = None):
        self.X_train = torch.tensor(X, dtype=torch.double, device=self.device)
        self.y_train = torch.tensor(y, dtype=torch.double, device=self.device)
        self.input_dim = self.X_train.shape[-1]
        self.output_dim = self.y_train.shape[-1]
        self.bounds = torch.tensor(bounds, dtype=torch.double, device=self.device).T
        self.feature_names = feature_names or [f"x{i}" for i in range(self.input_dim)]
        # Independent GPs
        models = []
        for i in range(self.output_dim):
            kernel_choice = model_config.get("kernel", "matern_2.5")
            noise_choice = model_config.get("noise", "fixed_low")
            input_transform_choice = model_config.get("input_transform", "none")

            models.append(
                SingleTaskGP(
                    self.X_train, self.y_train[:, i : i + 1],
                    covar_module=build_covar_module(kernel_choice, self.input_dim),
                    likelihood=build_likelihood(noise_choice),
                    input_transform=build_input_transform(input_transform_choice, self.input_dim),
                    outcome_transform=Standardize(m=1)
                )
            )
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        
        # Clear stale acquisition function from previous fit
        self.acq_func = None
        self.acq_strategy_name = None

    def recommend(self, n_candidates: int = 1, strategy: str = 'pareto', params: Dict[str, Any] = None) -> np.ndarray:
        if self.model is None: raise RuntimeError("Call fit() first.")
        params = params or {}
        
        self.acq_strategy_name = strategy

        if strategy == 'weighted':
            weights = params.get('weights', [1.0]*self.output_dim)
            beta = params.get('beta', 0.1)
            weights_t = torch.tensor(weights, device=self.device)
            objective = LinearMCObjective(weights=weights_t)
            acq_func = qUpperConfidenceBound(model=self.model, beta=beta, objective=objective)
            
        elif strategy == 'max_variance':
            weights = params.get('weights', [1.0]*self.output_dim)
            weights_t = torch.tensor(weights, device=self.device)
            objective = LinearMCObjective(weights=weights_t)
            acq_func = qUpperConfidenceBound(model=self.model, beta=1000.0, objective=objective)
            
        else: # Default: 'pareto'
            ref_point = self.y_train.min(dim=0)[0] - 0.1 * torch.abs(self.y_train.min(dim=0)[0])
            acq_func = qLogNoisyExpectedHypervolumeImprovement(
                model=self.model, 
                X_baseline=self.X_train, 
                prune_baseline=True, 
                ref_point=ref_point
            )

        # Persist the acquisition function
        self.acq_func = acq_func

        is_large = n_candidates > 10
        candidates, _ = optimize_acqf(
            acq_function=acq_func, bounds=self.bounds, q=n_candidates,
            num_restarts=2 if is_large else 10, 
            raw_samples=128 if is_large else 256,
            sequential=True
        )
        return candidates.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    #  Acquisition function evaluation (for constrained batch planning)
    # ------------------------------------------------------------------ #

    def evaluate_acquisition(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate MOO acquisition landscape at arbitrary points.
        
        For EHVI/UCB: evaluates the persisted acq_func in chunks.
        
        Args:
            X: (N, D) array of points to evaluate
            
        Returns:
            (N,) array of acquisition values
        """
        if self.acq_func is None:
            raise RuntimeError(
                "No acquisition function available. "
                "Call recommend() first to establish a strategy."
            )
        
        X_t = torch.tensor(X, dtype=torch.double, device=self.device)
        acq_values = np.empty(len(X_t))
        chunk_size = 256
        with torch.no_grad():
            for start in range(0, len(X_t), chunk_size):
                end = min(start + chunk_size, len(X_t))
                chunk = X_t[start:end].unsqueeze(1)  # (chunk, 1, d)
                acq_values[start:end] = self.acq_func(chunk).cpu().numpy()
        return acq_values

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return posterior mean and variance at points X (summed across objectives).
        
        Used as fallback for acquisition landscape when acq_func unavailable.
        Returns summed variance as a scalar proxy for multi-output uncertainty.
        
        Args:
            X: (N, D) array of points
            
        Returns:
            Tuple of (mean_sum, variance_sum) arrays, each shape (N,)
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.double, device=self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_t)
            # Sum across objectives for a scalar proxy
            mean = posterior.mean.sum(dim=-1).cpu().numpy()
            variance = posterior.variance.sum(dim=-1).cpu().numpy()
        return mean, variance

    def generate_diagnostics(self, save_path: str):
        """
        Generates Pairwise Pareto Scatter Plots for ANY N >= 2.
        """
        import itertools
        import math
        
        y_np = self.y_train.cpu().numpy()
        
        is_efficient = np.ones(y_np.shape[0], dtype=bool)
        for i, c in enumerate(y_np):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(y_np[is_efficient] > c, axis=1) | (y_np[is_efficient] == c).all(axis=1)

        pairs = list(itertools.combinations(range(self.output_dim), 2))
        n_plots = len(pairs)
        
        if n_plots == 1:
            rows, cols = 1, 1
            figsize = (8, 6)
        else:
            cols = min(3, n_plots)
            rows = math.ceil(n_plots / cols)
            figsize = (5 * cols, 4 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if n_plots == 1:
            axes_list = [axes]
        else:
            axes_list = axes.flatten()
        
        for idx, (dim_x, dim_y) in enumerate(pairs):
            ax = axes_list[idx]
            ax.scatter(y_np[~is_efficient][:, dim_x], y_np[~is_efficient][:, dim_y], 
                       c='gray', alpha=0.3, label='Dominated' if idx == 0 else "")
            ax.scatter(y_np[is_efficient][:, dim_x], y_np[is_efficient][:, dim_y], 
                       c='red', edgecolors='darkred', s=40, label='Pareto' if idx == 0 else "")
            
            name_x = self.feature_names[dim_x] if self.feature_names and dim_x < len(self.feature_names) else f"Obj {dim_x+1}"
            name_y = self.feature_names[dim_y] if self.feature_names and dim_y < len(self.feature_names) else f"Obj {dim_y+1}"
            
            ax.set_xlabel(name_x); ax.set_ylabel(name_y)
            ax.set_title(f"{name_x} vs {name_y}")
            ax.grid(True, linestyle='--', alpha=0.5)
            if idx == 0: ax.legend()

        for i in range(n_plots, len(axes_list)):
            axes_list[i].axis('off')

        fig.suptitle(f"Multi-Objective Diagnostics ({self.output_dim} Objectives)", fontsize=16)
        fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def get_optimizer(is_moo: bool, device: str = "cpu"):
    return MultiObjectiveOptimizer(device) if is_moo else SingleObjectiveOptimizer(device)