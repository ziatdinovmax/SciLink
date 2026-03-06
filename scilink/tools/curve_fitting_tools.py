import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import os

logger = logging.getLogger(__name__)


def load_curve_data(data_path: str) -> np.ndarray:
    """
    Robustly loads curve data (X, Y) from text files.
    Handles CSV, TSV, and whitespace separation automatically.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

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