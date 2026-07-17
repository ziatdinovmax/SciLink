"""Peak-region quality metric for XRD profile fits.

A weak or noisy powder pattern can have its Bragg peaks fit correctly yet score
a low whole-pattern R², because the long stretches of background-only channels
between reflections are noise-dominated and inflate the total sum of squares. A
correct fit of a low-SNR pattern is then mis-scored as a failure (see the
RRUFF Hematite reference: global R² ≈ 0.55 on a fit whose peaks track the data,
peak-region R² ≈ 0.89). The gate should measure the fit **where there is
diffracted intensity**.

``peak_region_r2`` is the shared 1D-spectroscopy metric (implemented in
``scilink.skills._shared._quality_metrics`` and also used by the NMR skill). It
computes R² over the union of the points where the *data* rises above the noise
(so a missed real reflection is still penalized — a genuine peak-deficient fit on
a dense low-symmetry pattern is NOT rescued) and the points where the *fitted
model* places intensity. It degenerates to the global R² on a high-SNR pattern
whose signal spans most channels, so it never inflates an already-good fit.

For a background-subtracted (``intensity_corrected``) pattern pass
``baseline=[0]*len`` so the signal/noise split is taken about zero.
"""

from __future__ import annotations

from ..._shared._spec import ToolSpec
from ..._shared._quality_metrics import peak_region_r2  # noqa: F401  (re-exported for the import_line)


TOOL_SPEC = ToolSpec(
    name="peak_region_r2",
    description=(
        "Compute the peak-region R² of an XRD profile fit — R² over the signal "
        "region (channels where the data rises above noise ∪ where the model "
        "places intensity) rather than the whole 2θ range — so a correct fit of "
        "a weak/noisy pattern is not scored as a failure by the noise-dominated "
        "background channels. It does NOT rescue a fit that genuinely misses real "
        "reflections (those channels are above noise and stay in the region), and "
        "it equals the global R² on a high-SNR pattern. Report it alongside the "
        "global R² as the gate metric."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.quality import peak_region_r2",
    signature=(
        "peak_region_r2(x, y, y_fit, baseline=None, k_sigma=3.0, min_run=3, "
        "dilate=5, model_frac=0.02, min_points=15) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "2θ axis (degrees)."},
        "y": {"type": "list[float]", "description": "Intensity the fit was computed against. Pass the SAME array fit_curve was fit to: the background-subtracted intensity_corrected when fitting a corrected pattern, or the raw intensity when the model includes the background."},
        "y_fit": {"type": "list[float]", "description": "Fitted model evaluated on x (fit_curve)."},
        "baseline": {"type": "list[float]", "description": "Baseline on x. For a background-subtracted pattern (y = intensity_corrected) pass [0]*len(x) so the signal/noise split is about zero; for a raw-intensity fit pass the fitted/estimated background so the split is correct."},
        "k_sigma": {"type": "float", "description": "How many noise σ a channel must exceed to count as diffracted signal (default 3). RAISE (4–5) on very noisy patterns so the region excludes noise; LOWER (2) to include weak reflections that barely clear the noise."},
        "min_run": {"type": "int", "description": "Minimum contiguous run length (points) for a data feature to count as a real peak vs an isolated noise spike (default 3). Increase for finely-sampled (0.01° step) scans."},
        "dilate": {"type": "int", "description": "Points to grow the signal mask on each side, to include peak wings/tails (default 5). Increase for broad nanocrystalline peaks so the wings are scored; decrease for very sharp lines."},
        "model_frac": {"type": "float", "description": "Also count channels where the fitted model exceeds this fraction of its own max as signal (default 0.02) — so the metric scores where the model claims reflections even if data there is weak."},
        "min_points": {"type": "int", "description": "If fewer than this many signal points are found, fall back to the global R² (default 15) — guards the metric on a near-empty pattern."},
    },
    required=["x", "y", "y_fit"],
    returns=(
        "dict with 'peak_region_r2', global 'r_squared', 'n_signal_points', and "
        "'fell_back_to_global'. Put 'peak_region_r2' (and 'r_squared') into the "
        "fit_quality block of the fit-results JSON."
    ),
    when_to_use=(
        "As the final quality step of an XRD profile fit — the skill's quality "
        "gate scores by 'peak_region_r2'. A large gap (low global R², high "
        "peak-region R²) means a low-SNR pattern whose peaks are nonetheless well "
        "fit; a low peak-region R² means real reflections are mis- or un-fit."
    ),
)
