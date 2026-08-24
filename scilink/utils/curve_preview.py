"""Curve preview helpers for LLM-facing prompts.

Pure rendering utilities — no controller / pipeline state involved. Shared
between the in-pipeline `SeriesScoutController` and the lit-search query
optimizer so both produce visually consistent previews.
"""
from __future__ import annotations

import io

import numpy as np


def extract_xy(curve_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) arrays from common curve-data shapes."""
    if curve_data.ndim == 1:
        return np.arange(len(curve_data)), curve_data
    if curve_data.shape[0] == 2:
        return curve_data[0], curve_data[1]
    if curve_data.shape[1] == 2:
        return curve_data[:, 0], curve_data[:, 1]
    raise ValueError(f"Unexpected curve_data shape: {curve_data.shape}")


def render_curve_overlay(
    scout_curves: list[dict],
    system_info: dict | None = None,
    title_suffix: str = " — Scout Overlay",
) -> bytes:
    """Render an overlay plot of multiple curves and return raw PNG bytes.

    Args:
        scout_curves: list of {"label": str, "curve_data": np.ndarray}.
        system_info: optional metadata dict with xlabel/ylabel/title keys.
        title_suffix: appended to the plot title.

    Returns:
        Raw PNG bytes, suitable for `{mime_type: image/png, data: <bytes>}`
        prompt parts consumed by LLM wrappers.
    """
    import matplotlib.pyplot as plt

    system_info = system_info or {}
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.viridis
    n = len(scout_curves)
    for i, entry in enumerate(scout_curves):
        x, y = extract_xy(entry["curve_data"])
        color = cmap(i / max(n - 1, 1))
        ax.plot(x, y, color=color, linewidth=1.2, label=entry["label"])

    ax.set_xlabel(system_info.get("xlabel", "X"))
    ax.set_ylabel(system_info.get("ylabel", "Y"))
    ax.set_title(system_info.get("title", "Data") + title_suffix)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_curve_single(curve_data: np.ndarray, system_info: dict | None = None) -> bytes:
    """Render a single curve and return raw PNG bytes."""
    import matplotlib.pyplot as plt

    system_info = system_info or {}
    x, y = extract_xy(curve_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color="steelblue", linewidth=1.4)
    ax.set_xlabel(system_info.get("xlabel", "X"))
    ax.set_ylabel(system_info.get("ylabel", "Y"))
    ax.set_title(system_info.get("title", "Data"))
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
