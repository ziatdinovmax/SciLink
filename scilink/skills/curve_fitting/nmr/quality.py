"""Peak-region quality metric for 1D NMR fits.

NMR spectra have an enormous digitized width relative to the resonances, so a
correct fit of a narrow peak is dominated, in a whole-window R², by the
noise-filled empty regions — a good fit can score R² ≈ 0.1 (see the low-SNR
²³Na / ⁶⁷Zn references). The gate metric should instead measure how well the
model reproduces the spectrum **where there is signal**.

The metric is implemented once in ``scilink.skills._shared._quality_metrics``
(it is a general 1D-spectroscopy metric, shared with the XRD profile skill);
this module re-exports it and declares the NMR-flavoured ``TOOL_SPEC`` so the
tool stays skill-gated. The ``import_line`` below still resolves
``peak_region_r2`` from here, so callers are unaffected.
"""

from __future__ import annotations

from ..._shared._spec import ToolSpec
from ..._shared._quality_metrics import (  # noqa: F401  (re-exported for the TOOL_SPEC import_line)
    peak_region_r2,
    _robust_noise,
    _contiguous_runs,
)


TOOL_SPEC = ToolSpec(
    name="peak_region_r2",
    description=(
        "Compute the peak-region R² of a 1D NMR fit — R² over the signal "
        "region (data-above-noise ∪ where the model places intensity) rather "
        "than the whole digitized window — so a correct fit of a narrow peak in "
        "a wide, mostly-empty spectrum is not scored as a failure by the "
        "noise-dominated baseline. Report this as the gate metric."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.quality import peak_region_r2",
    signature=(
        "peak_region_r2(x, y, y_fit, baseline=None, k_sigma=3.0, min_run=3, "
        "dilate=5, model_frac=0.02, min_points=15) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "Chemical-shift axis (ppm)."},
        "y": {"type": "list[float]", "description": "Spectrum intensity (the data fit to)."},
        "y_fit": {"type": "list[float]", "description": "Fitted model evaluated on x."},
        "baseline": {"type": "list[float]", "description": "Fitted baseline on x (default: median of y). Pass the actual fitted baseline so the signal/noise split is correct on a sloped or rolling background."},
        "k_sigma": {"type": "float", "description": "How many noise σ a point must exceed to count as signal (default 3). RAISE (4–5) on noisy spectra so the region excludes noise; LOWER (2) to include a weak, broad resonance that barely clears the noise."},
        "min_run": {"type": "int", "description": "Minimum contiguous run length (points) for a data feature to count as a real peak vs an isolated noise spike (default 3). Increase for very finely-sampled spectra."},
        "dilate": {"type": "int", "description": "Points to grow the signal mask on each side, to include peak wings (default 5). Increase for broad lines so the wings are scored; decrease for very sharp lines."},
        "model_frac": {"type": "float", "description": "Also count points where the fitted model exceeds this fraction of its own max as signal (default 0.02) — so the metric scores where the model claims peaks even if data there is weak."},
        "min_points": {"type": "int", "description": "If fewer than this many signal points are found, fall back to the global R² (default 15) — guards the metric when there is essentially no signal."},
        "struct_apex_frac": {"type": "float", "description": "Threshold on apex_resid_frac (largest residual as a fraction of peak height) for the residual_structured flag (default 0.15 = a ≥15% apex miss). The raw value is always returned; raise it to flag only gross misfits, lower it to be stricter."},
        "struct_autocorr": {"type": "float", "description": "Threshold on lag-1 residual autocorrelation for residual_structured (default 0.9) — near 1 means a systematic (non-random) residual rather than noise."},
    },
    required=["x", "y", "y_fit"],
    returns=(
        "dict with 'peak_region_r2', global 'r_squared', 'n_signal_points', "
        "'fell_back_to_global', and residual-structure diagnostics over the "
        "signal region. Judge fit adequacy by the SIGNAL-RELATIVE quantities — "
        "'apex_resid_frac' (largest residual / peak height) and 'resid_rms_frac' "
        "(RMS residual / peak height) — together with 'resid_autocorr_lag1'. "
        "NOTE: 'apex_resid_over_noise' and 'frac_resid_gt_3sigma' are "
        "NOISE-relative and are therefore large even for a PERFECT fit of a "
        "high-SNR peak (e.g. ~0.8 of points exceed 3σ on a clean line) — do not "
        "read them as misfit on their own. 'residual_structured' (boolean) = "
        "systematic (autocorr high) AND severe at the apex (apex_resid_frac "
        "above threshold). Put 'peak_region_r2' and 'r_squared' into the "
        "fit_quality block; if 'residual_structured' is True the fit has the "
        "wrong MODEL where the signal is — escalate (see the skill), don't accept "
        "it on R² alone."
    ),
    when_to_use=(
        "Always, as the final quality step of an NMR fit — the skill's quality "
        "gate scores by 'peak_region_r2'."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
