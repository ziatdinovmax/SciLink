import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import logging

from ._spec import ToolSpec

logger = logging.getLogger(__name__)

MAX_THUMBNAIL_DIM = 1024


def load_image_data(image_path: str) -> np.ndarray:
    """
    Load image data from file (.npy, .png, .tif, .jpg, .bmp).

    For .npy files, returns the raw array (preserving dtype/shape).
    For standard image formats, uses OpenCV with BGR→RGB conversion.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    if ext == ".npy":
        return np.load(image_path)

    # Standard image formats
    import cv2

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR→RGB for color images; leave grayscale and 2-channel as-is
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    # 2-channel images (e.g., AFM amplitude+phase) need no conversion

    return img


def _jsonify_tag(value):
    """Coerce a PIL TIFF tag value into a JSON-friendly form."""
    try:
        from PIL.TiffImagePlugin import IFDRational
    except Exception:
        IFDRational = ()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if IFDRational and isinstance(value, IFDRational):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonify_tag(v) for v in value]
    return value


def _parse_image_description(desc: str):
    """Parse an ImageDescription block into structured metadata when possible.

    Handles the two common scientific conventions: ImageJ's ``key=value`` lines
    and OME-TIFF's embedded XML (kept verbatim, not parsed, to avoid a dependency).
    """
    if not isinstance(desc, str) or not desc.strip():
        return None
    stripped = desc.lstrip()
    if stripped.startswith("<?xml") or stripped.startswith("<OME"):
        return {"format": "ome-xml", "xml": desc}
    if "=" in desc and "\n" in desc:  # ImageJ-style key=value block
        kv = {}
        for line in desc.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()
        if kv:
            return {"format": "imagej", "fields": kv}
    return None


def _pixel_size_from_tags(named: dict):
    """Derive pixel size from TIFF resolution tags (best effort)."""
    xres, yres = named.get("XResolution"), named.get("YResolution")
    if not xres or not yres:
        return None
    unit = {1: "none", 2: "inch", 3: "cm"}.get(named.get("ResolutionUnit"), "unknown")
    try:
        return {"x": 1.0 / float(xres), "y": 1.0 / float(yres),
                "unit": f"per_{unit}" if unit not in ("none", "unknown") else unit}
    except (ZeroDivisionError, TypeError, ValueError):
        return None


_LEN_UNIT_TO_NM = {
    "nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
    "um": 1e3, "µm": 1e3, "micron": 1e3, "microns": 1e3, "micrometer": 1e3,
    "micrometers": 1e3, "mm": 1e6, "m": 1e9,
    "a": 0.1, "angstrom": 0.1, "angstroms": 0.1, "å": 0.1, "pm": 1e-3,
}


def resolve_pixel_size_nm(metadata, image_shape):
    """Resolve nm-per-pixel from metadata + image shape (best effort, never raises).

    The robust, calibration-correct rule that generated analysis code should use
    instead of hand-rolling it. The common bug it prevents: dividing
    ``field_of_view`` by a metadata pixel-COUNT field (``n_cols`` / ``width``)
    that is usually absent — leaving pixel size ``None`` — when the real
    denominator is the image array shape.

    Priority:
      1. ``experimental_details.spatial_info.field_of_view_{x,y}`` divided by the
         image width / height (the standard SEM/TEM/AFM case).
      2. Embedded TIFF ``pixel_size`` (``embedded_file_metadata.pixel_size`` or a
         top-level ``pixel_size``) when its unit is a known length.

    Parameters
    ----------
    metadata : dict | None
        The system_info / metadata dict (may nest the payload under
        ``"system_info"``).
    image_shape : tuple
        numpy ``image.shape`` — ``shape[0]`` = rows (height), ``shape[1]`` =
        columns (width).

    Returns
    -------
    dict | None
        ``{"x": nm_per_px, "y": nm_per_px, "source": str}`` or ``None`` if it
        cannot be resolved. ``x`` is the horizontal (column) pixel size.
    """
    if not isinstance(metadata, dict) or not image_shape or len(image_shape) < 2:
        return None
    n_rows, n_cols = float(image_shape[0]), float(image_shape[1])
    if n_rows < 1 or n_cols < 1:
        return None

    def _dig(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    # unwrap an optional system_info wrapper
    roots = [metadata]
    if isinstance(metadata.get("system_info"), dict):
        roots.append(metadata["system_info"])

    # 1) field of view / image shape
    for root in roots:
        spat = _dig(root, "experimental_details", "spatial_info") or \
            _dig(root, "spatial_info")
        if isinstance(spat, dict):
            fov_x = spat.get("field_of_view_x")
            fov_y = spat.get("field_of_view_y")
            unit = str(spat.get("field_of_view_units", "nm")).strip().lower()
            scale = _LEN_UNIT_TO_NM.get(unit)
            if scale and (fov_x or fov_y):
                px_x = (float(fov_x) * scale / n_cols) if fov_x else None
                px_y = (float(fov_y) * scale / n_rows) if fov_y else None
                px_x = px_x if px_x else px_y
                px_y = px_y if px_y else px_x
                if px_x and px_x > 0:
                    return {"x": px_x, "y": px_y, "source": "field_of_view"}

    # 2) embedded pixel_size with a known length unit
    for root in roots:
        px = _dig(root, "embedded_file_metadata", "pixel_size") or \
            (root.get("pixel_size") if isinstance(root.get("pixel_size"), dict) else None)
        if isinstance(px, dict):
            unit = str(px.get("unit", "")).strip().lower()
            scale = _LEN_UNIT_TO_NM.get(unit)
            if scale and px.get("x"):
                return {"x": float(px["x"]) * scale,
                        "y": float(px.get("y", px["x"])) * scale,
                        "source": "embedded_tags"}
    return None


def extract_image_metadata(image_path: str) -> dict:
    """Best-effort recovery of embedded metadata from an image file.

    ``load_image_data`` reads pixels with OpenCV, which discards TIFF tags /
    ImageDescription — the common carrier of scientific calibration and
    acquisition parameters (pixel size, instrument, ImageJ/OME-XML blocks). This
    recovers them so they can flow into ``system_info``. TIFF-focused today;
    returns ``{}`` for other formats or when nothing readable is present. Never
    raises — embedded metadata is best-effort, not load-critical.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".tif", ".tiff"):
        return {}
    try:
        from PIL import Image
        from PIL.TiffTags import TAGS as TIFF_TAGS
    except Exception:
        return {}
    meta: dict = {}
    try:
        with Image.open(image_path) as im:
            tags = getattr(im, "tag_v2", None)
            if not tags:
                return {}
            named = {TIFF_TAGS.get(tid, str(tid)): _jsonify_tag(val)
                     for tid, val in tags.items()}
            desc = named.get("ImageDescription")
            if desc:
                meta["image_description"] = desc
                parsed = _parse_image_description(desc)
                if parsed:
                    meta["image_description_parsed"] = parsed
            for tag, key in (("Software", "software"), ("DateTime", "datetime"),
                             ("Make", "make"), ("Model", "model")):
                if named.get(tag):
                    meta[key] = named[tag]
            px = _pixel_size_from_tags(named)
            if px:
                meta["pixel_size"] = px
            meta["tiff_tags"] = named
    except Exception:
        return {}
    return meta


def image_to_thumbnail_bytes(
    image: np.ndarray, max_dim: int = MAX_THUMBNAIL_DIM, quality: int = 85
) -> bytes:
    """
    Resize image to fit within max_dim and return JPEG bytes for LLM prompts.

    Handles grayscale, RGB, and float arrays (auto-normalized to 0-255).
    """
    arr = image.copy()

    # For multi-channel images that aren't standard RGB/RGBA (e.g., 2-channel),
    # create a labeled subplot figure so the LLM sees each channel with its index.
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        n_ch = arr.shape[2]
        fig, axes = plt.subplots(1, n_ch, figsize=(3.5 * n_ch, 3.5))
        if n_ch == 1:
            axes = [axes]
        for c in range(n_ch):
            ch = arr[:, :, c].astype(np.float64)
            finite = ch[np.isfinite(ch)]
            mn, mx = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
            if mx - mn > 1e-6:
                ch = np.clip((ch - mn) / (mx - mn), 0.0, 1.0)
            else:
                ch = np.zeros_like(ch)
            axes[c].imshow(np.nan_to_num(ch, nan=0), cmap="gray", aspect="equal")
            axes[c].set_title(f"Channel {c}", fontsize=10)
            axes[c].axis("off")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="jpeg", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Normalize non-uint8 arrays (float / uint16) to uint8 for display.
    # Use a ROBUST PERCENTILE stretch, not global min-max: this thumbnail's only
    # job is visual inspection by the model (quantitative work uses the staged
    # full-resolution data.npy, not this image). Global min-max renders a
    # low-dynamic-range frame — e.g. a uint16 image whose signal spans a tiny
    # fraction of [min, max], or any frame with a few hot/cold outlier pixels —
    # nearly blank, hiding faint features from both the planner and the verifier.
    # Clipping the [0.5, 99.5] percentile tails maximizes visible contrast.
    # uint8 inputs are already display-ready and left untouched.
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        finite = arr[np.isfinite(arr)]
        lo, hi = np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 1.0)
        if hi - lo > 1e-6:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255
        else:
            arr = np.zeros_like(arr)
        arr = np.nan_to_num(arr, nan=0).astype(np.uint8)

    # Determine spatial dims
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w = arr.shape[:2]
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")

    # Resize if needed
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        pil_img = Image.fromarray(arr)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        pil_img = Image.fromarray(arr)

    # Convert to RGB for JPEG if grayscale
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")

    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def create_image_montage(
    images: list[np.ndarray],
    labels: list[str],
    max_cols: int = 4,
    max_dim: int = MAX_THUMBNAIL_DIM,
) -> bytes:
    """
    Create a labeled montage of multiple images and return JPEG bytes.

    Used for series scouting — shows representative images side by side.
    """
    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.asarray(axes).flatten()

    for i, (img, label) in enumerate(zip(images, labels)):
        arr = img
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float64)
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if mx - mn > 1e-6:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            arr = np.nan_to_num(arr, nan=0)

        if arr.ndim == 3 and arr.shape[2] not in (3, 4):
            arr = arr[:, :, 0]  # show first channel for non-RGB multi-channel
        cmap = "gray" if arr.ndim == 2 else None
        axes[i].imshow(arr, cmap=cmap, aspect="equal")
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="jpeg", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


TOOL_SPECS = [
    ToolSpec(
        name="resolve_pixel_size_nm",
        description=(
            "Resolve nm-per-pixel from metadata + image shape. The robust, "
            "calibration-correct way to get pixel size — use this instead of "
            "hand-rolling the arithmetic. It divides field_of_view by the image "
            "array SHAPE; never divide by a metadata pixel-count field like "
            "n_cols/width (usually absent, silently leaving pixel size None). "
            "Never raises."
        ),
        import_line=(
            "from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm"
        ),
        signature="resolve_pixel_size_nm(metadata, image_shape) -> dict | None",
        agents=["image_analysis"],
        when_to_use=(
            "Whenever a step needs pixel size in nm (spacing/lattice measurements, "
            "fourier_reflection_map, converting pixel distances to physical units). "
            "Pass the system_info/metadata dict and image.shape."
        ),
        parameters={
            "metadata": {
                "type": "dict",
                "description": (
                    "The system_info / metadata dict (may nest the payload under "
                    "a 'system_info' key; both are handled)."
                ),
            },
            "image_shape": {
                "type": "tuple",
                "description": (
                    "numpy image.shape — shape[0]=rows (height), shape[1]=cols (width)."
                ),
            },
        },
        required=["metadata", "image_shape"],
        returns=(
            "dict {'x': nm_per_px, 'y': nm_per_px, 'source': str} where 'x' is the "
            "horizontal (column) pixel size, or None if it cannot be resolved. "
            "Read the scalar as px['x'] — there are NO 'pixel_size_nm' / "
            "'pixel_size_nm_x' keys; handle the None case with a guard."
        ),
        example=(
            "px = resolve_pixel_size_nm(metadata, image.shape)\n"
            "if px is None:\n"
            "    raise ValueError('pixel size unavailable; cannot convert to nm')\n"
            "nm_per_px = px['x']           # horizontal nm/pixel\n"
            "distance_nm = distance_px * nm_per_px"
        ),
    ),
]
