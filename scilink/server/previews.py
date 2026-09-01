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


def render_thumbnail(path: Path, size: int = 256, cmap: str = "viridis") -> bytes:
    """PNG bytes for an image/array file, longest side ≤ ``size``."""
    from PIL import Image

    if cmap not in _ALLOWED_CMAPS:
        cmap = "viridis"
    size = max(32, min(size, 2048))
    ext = path.suffix.lower()

    if ext == ".npy":
        import numpy as np

        arr = np.load(path, mmap_mode="r")
        png = _array_to_png(arr, cmap)
        img = Image.open(io.BytesIO(png))
    elif ext in (".tif", ".tiff"):
        import numpy as np

        img = Image.open(path)
        arr = np.asarray(img)
        if arr.ndim == 2 and arr.dtype != "uint8":
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
    return out.getvalue()


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
