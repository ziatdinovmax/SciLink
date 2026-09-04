"""Server-rendered previews: raster thumbnails (incl. NPY/TIFF heatmaps
with a selectable colormap) and tabular extracts for CSV/TSV/XLSX.

Ports the normalization behavior of the Streamlit file viewer
(scilink/ui/components/file_viewer.py: TIFF/NPY normalized to [0,1] before
display; Excel head-only) but renders to PNG bytes / JSON rows so the
browser does the displaying.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

_RASTER_EXTS = {".png", ".jpg", ".jpeg"}
_ARRAY_EXTS = {".npy", ".tif", ".tiff"}
_TABLE_EXTS = {".csv", ".tsv", ".xlsx"}

_ALLOWED_CMAPS = ("gray", "viridis", "magma", "plasma", "inferno")


def _normalize(arr) -> "Any":
    import numpy as np

    a = np.asarray(arr, dtype=float)
    a = np.nan_to_num(a)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return a


def _array_to_png(arr, cmap: str) -> bytes:
    import numpy as np
    from PIL import Image

    a = np.asarray(arr)
    # 3D stacks / cubes: show the middle slice along the smallest axis
    # collapsed — keep it simple: take the first 2D slice.
    while a.ndim > 2:
        a = a[0] if a.shape[0] <= a.shape[-1] else a[..., 0]
    if a.ndim < 2:
        a = a.reshape(1, -1)
    norm = _normalize(a)
    if cmap == "gray":
        img = Image.fromarray((norm * 255).astype("uint8"), mode="L")
    else:
        from matplotlib import colormaps

        rgba = colormaps[cmap](norm)
        img = Image.fromarray((rgba[..., :3] * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _npy_header(path: Path):
    """(shape, dtype_str) from the .npy header — no data load, no unpickle."""
    import numpy as np

    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return shape, str(dtype)


def _line_plot_png(arr, size: int) -> bytes:
    """1D data (or (N,2)/(2,N) pairs) rendered as a plot — a colormapped
    1-pixel ribbon is the wrong presentation for a spectrum.

    Pairs with monotonic x plot as a curve; non-monotonic x means the pairs
    are coordinates (atom positions, particle centers), which get a scatter
    with equal axis scaling instead of a spaghetti line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a = np.asarray(arr, dtype=float)
    scatter = False
    if a.ndim == 2:
        # x/y pairs: columns if (N, 2), rows if (2, N).
        x, y = (a[:, 0], a[:, 1]) if a.shape[1] == 2 else (a[0], a[1])
        # Curve vs coordinates by X-REVERSAL FRACTION, not monotonicity:
        # hysteresis loops and CV sweeps reverse x a handful of times and
        # are still curves, while unsorted coordinates reverse on ~half the
        # steps. Reversals are counted only on x-steps above a jitter
        # threshold so noisy-but-ordered spectra stay curves.
        d = np.diff(x)
        xr = float(x.max() - x.min())
        sig = d[np.abs(d) > 1e-3 * (xr if xr > 0 else 1.0)]
        if sig.size >= 2:
            reversals = int(np.count_nonzero(np.diff(np.sign(sig)) != 0))
            scatter = reversals / (sig.size - 1) > 0.2
    else:
        x, y = np.arange(a.size), a.ravel()
    fig, ax = plt.subplots(figsize=(5, 3), dpi=max(64, size) / 5)
    if scatter:
        ax.scatter(x, y, s=max(1.0, 4000.0 / max(len(x), 1)), lw=0)
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.plot(x, y, lw=1.0)
        ax.margins(x=0.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _is_xy_pairs(shape) -> bool:
    return (len(shape) == 2 and 2 in shape and max(shape) >= 8
            and min(shape) == 2)


def render_thumbnail(path: Path, size: int = 256,
                     cmap: str = "viridis") -> tuple:
    """(PNG bytes, kind) for an image/array file, longest side ≤ ``size``.

    ``kind`` is ``"image"`` (raster), ``"heatmap"`` (2D array, colormap
    applies), or ``"line"`` (1D / x-y data plotted — colormap is
    meaningless there). Unrenderable arrays raise ValueError carrying the
    shape/dtype from the .npy header so the caller can explain instead of
    serving a broken image.
    """
    from PIL import Image

    if cmap not in _ALLOWED_CMAPS:
        cmap = "viridis"
    size = max(32, min(size, 2048))
    ext = path.suffix.lower()
    kind = "image"

    if ext == ".npy":
        import numpy as np

        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            try:
                shape, dtype = _npy_header(path)
                raise ValueError(
                    f"array of shape {tuple(shape)}, dtype {dtype} cannot "
                    "be rendered as an image — download to inspect")
            except ValueError:
                raise
            except Exception:
                raise ValueError("not a readable .npy array — download to inspect")
        if arr.size <= 1:
            val = f" (value: {arr.reshape(-1)[0]})" if arr.size == 1 else ""
            raise ValueError(f"scalar array{val} — nothing to render")
        if arr.ndim == 1 or _is_xy_pairs(arr.shape):
            return _line_plot_png(arr, size), "line"
        kind = "heatmap"
        png = _array_to_png(arr, cmap)
        img = Image.open(io.BytesIO(png))
    elif ext in (".tif", ".tiff"):
        import numpy as np

        img = Image.open(path)
        arr = np.asarray(img)
        if arr.ndim == 2 and arr.dtype != "uint8":
            kind = "heatmap"
            img = Image.open(io.BytesIO(_array_to_png(arr, cmap)))
        else:
            img = img.convert("RGB")
    elif ext in _RASTER_EXTS:
        img = Image.open(path).convert("RGB")
    else:
        raise ValueError(f"No thumbnail renderer for {ext!r}")

    img.thumbnail((size, size))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue(), kind


def can_thumbnail(name: str) -> bool:
    return Path(name).suffix.lower() in (_RASTER_EXTS | _ARRAY_EXTS)


def extract_table(path: Path, limit: int = 500) -> Dict[str, Any]:
    """Head of a tabular file as ``{columns, rows, total_rows, truncated}``."""
    import pandas as pd

    limit = max(1, min(limit, 5000))
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Not a table file: {ext!r}")
    total = len(df)
    head = df.head(limit)
    return {
        "columns": [str(c) for c in head.columns],
        "rows": head.astype(object).where(head.notna(), None).values.tolist(),
        "total_rows": total,
        "truncated": total > limit,
    }
