import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import logging

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
            mn, mx = np.nanmin(ch), np.nanmax(ch)
            if mx - mn > 1e-6:
                ch = (ch - mn) / (mx - mn)
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

    # Normalize float arrays to uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val - min_val > 1e-6:
            arr = (arr - min_val) / (max_val - min_val) * 255
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
