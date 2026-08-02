import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import os

logger = logging.getLogger(__name__)


def load_curve_data(data_path: str) -> np.ndarray:
    """
    Robustly loads curve data (X, Y) from various file formats.
    Handles .npy, .h5/.hdf5 (NeXus), CSV, TSV, and whitespace separation
    automatically.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    # Native numpy format
    if data_path.endswith('.npy'):
        return np.load(data_path)

    # NeXus / HDF5 — pull the signal and (when present) its axis so we
    # can return (X, Y) pairs for callers that expect a 2D layout.
    if data_path.lower().endswith(('.h5', '.hdf5')):
        from .hdf5_utils import load_hdf5_signal
        signal, axes = load_hdf5_signal(data_path, return_axes=True)
        if signal.ndim == 1 and axes and axes[0] is not None and axes[0].size == signal.size:
            return np.column_stack([axes[0], signal])
        return signal

    attempts = [
        dict(),                                # whitespace-delimited, no header
        dict(skiprows=1),                      # whitespace-delimited, skip header
        dict(delimiter=','),                   # CSV, no header
        dict(delimiter=',', skiprows=1),       # CSV, skip header
    ]

    for kw in attempts:
        try:
            data = np.loadtxt(data_path, **kw)
            if data.size > 0:
                return data
        except Exception:
            pass

    # If all fail, raise descriptive error
    raise ValueError(f"Unsupported file format or invalid data structure in {data_path}.")

def plot_curve_to_bytes(curve_data: np.ndarray, system_info: dict, title_suffix: str = "") -> bytes:
    """
    Plots a 1D curve and returns the image as bytes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
    
    plot_title = system_info.get("title") or "Data"
    ax.set_title(plot_title + title_suffix)

    xlabel_text = system_info.get("xlabel") or "X-axis"
    ax.set_xlabel(xlabel_text)

    ylabel_text = system_info.get("ylabel") or "Y-axis"
    ax.set_ylabel(ylabel_text)
    
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)
    return image_bytes