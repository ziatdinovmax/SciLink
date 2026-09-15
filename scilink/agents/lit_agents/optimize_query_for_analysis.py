"""Multimodal query optimizer for the analyze-mode `search_literature` tool.

Refines a raw orchestrator-LLM query into a focused, experiment-grounded
question by showing the optimizer LLM the same kind of preview the planner
sees: a thumbnail/montage of the actual data plus normalized metadata.

Always best-effort. On any failure (no data loaded, preview rendering error,
LLM call error) the raw query is returned so Edison still gets *something* to
search on. The refinement is expensive only relative to a single LLM call,
which is cheap compared to the Edison job that follows.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# --- Tunables -----------------------------------------------------------------
DEFAULT_SCOUT_N = 5
DEFAULT_MAX_REFINED_CHARS = 600  # keep refined queries focused, not bloated

# --- Prompt -------------------------------------------------------------------
_OPTIMIZER_PROMPT = """\
You refine literature-search queries for an experimental data analysis system.

The orchestrator wrote a draft query. Your job is to rewrite it so it returns
the most relevant prior art for THIS specific dataset — exploiting the visual
preview, data shape, and experimental metadata you can see below.

Rules:
- Phrase the refined query as a single natural-language research question —
  not a comma-separated bag of specifics or a long noun phrase.
- Bake in concrete experimental specifics from the metadata (material,
  technique, instrument, modality) when they sharpen retrieval.
- Mention salient visual features (grain morphology, peak shape, defects,
  texture) only if the preview clearly shows them.
- Do NOT invent material identities or measurement parameters not present
  in the metadata or clearly visible in the preview. Stay grounded.
- Keep the query under {max_chars} characters.
- Output JSON only: {{"refined_query": "..."}}
"""


def _summarize_metadata(metadata: Any) -> str:
    if not metadata:
        return "(no metadata loaded)"
    try:
        return json.dumps(metadata, indent=2, default=str)[:1500]
    except Exception:
        return str(metadata)[:1500]


def _build_image_preview(data_path: str) -> tuple[bytes, str] | None:
    """Single image → JPEG thumbnail bytes. Series (dir) → montage."""
    from ...skills._shared.image_analysis_tools import (
        create_image_montage,
        image_to_thumbnail_bytes,
    )

    p = Path(data_path)

    def _load(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.load(path)
        from PIL import Image
        return np.asarray(Image.open(path))

    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix.lower() in (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
        if not files:
            return None
        if len(files) == 1:
            return image_to_thumbnail_bytes(_load(files[0])), "image/jpeg"
        n = min(DEFAULT_SCOUT_N, len(files))
        # Uniform-stride sample (no LLM-driven adaptive selection — that's
        # the in-pipeline scout's job; here we just need representative coverage)
        idxs = np.linspace(0, len(files) - 1, n, dtype=int).tolist()
        sampled = [files[i] for i in idxs]
        images = [_load(f) for f in sampled]
        labels = [f.name for f in sampled]
        return create_image_montage(images, labels), "image/jpeg"

    # Single file
    arr = _load(p)
    if arr.ndim == 3 and arr.shape[0] not in (3, 4) and arr.shape[-1] not in (3, 4):
        # Image stack stored as a single .npy file (shape: N x H x W)
        n = min(DEFAULT_SCOUT_N, arr.shape[0])
        idxs = np.linspace(0, arr.shape[0] - 1, n, dtype=int).tolist()
        images = [arr[i] for i in idxs]
        labels = [f"frame {i}" for i in idxs]
        return create_image_montage(images, labels), "image/jpeg"
    return image_to_thumbnail_bytes(arr), "image/jpeg"


def _build_curve_preview(data_path: str, metadata: Any) -> tuple[bytes, str] | None:
    """Single curve → plot. Series (dir or 2D stack) → overlay plot."""
    from ...utils.curve_preview import render_curve_overlay, render_curve_single

    system_info = metadata if isinstance(metadata, dict) else {}
    p = Path(data_path)

    def _load(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.load(path)
        return np.loadtxt(path, delimiter=",")

    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix.lower() in (".npy", ".csv", ".txt", ".dat")
        )
        if not files:
            return None
        if len(files) == 1:
            return render_curve_single(_load(files[0]), system_info), "image/png"
        n = min(DEFAULT_SCOUT_N, len(files))
        idxs = np.linspace(0, len(files) - 1, n, dtype=int).tolist()
        scout = [
            {"label": files[i].name, "curve_data": _load(files[i])}
            for i in idxs
        ]
        return render_curve_overlay(scout, system_info), "image/png"

    arr = _load(p)
    # 2-D stack of N spectra: (N, M) where M >> N typically. Heuristic: if
    # the smaller axis looks like spectra count and the larger like channels,
    # treat as series.
    if arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] != 2:
        n_spectra = arr.shape[0]
        n = min(DEFAULT_SCOUT_N, n_spectra)
        idxs = np.linspace(0, n_spectra - 1, n, dtype=int).tolist()
        scout = [{"label": f"spectrum {i}", "curve_data": arr[i]} for i in idxs]
        return render_curve_overlay(scout, system_info), "image/png"
    return render_curve_single(arr, system_info), "image/png"


def _build_preview(
    data_type: str | None,
    data_path: str | None,
    metadata: Any,
) -> tuple[bytes, str] | None:
    """Dispatch by data_type. Return (bytes, mime_type) or None on failure."""
    if not data_path:
        return None
    try:
        if data_type == "image":
            return _build_image_preview(data_path)
        if data_type == "curve":
            return _build_curve_preview(data_path, metadata)
        # Hyperspectral / unknown: skip preview for v1
        logger.info(
            f"optimize_query_for_analysis: no preview builder for "
            f"data_type={data_type!r}; refining with metadata only."
        )
        return None
    except Exception as e:
        logger.warning(f"optimize_query_for_analysis: preview failed: {e}")
        return None


def optimize_query_for_analysis(
    raw_query: str,
    data_type: str | None,
    data_path: str | None,
    metadata: Any,
    model: Any,
    max_chars: int = DEFAULT_MAX_REFINED_CHARS,
) -> str:
    """Refine `raw_query` using the loaded data preview + metadata.

    Args:
        raw_query: the orchestrator LLM's query string.
        data_type: "image" | "curve" | "hyperspectral" | None — from
            `examine_data` (`self.orch.current_data_type`).
        data_path: path to the loaded data (file or directory).
        metadata: normalized metadata dict (`self.orch.current_metadata`).
        model: LLM client with `.generate_content(parts)` (the orchestrator's
            existing wrapper instance).
        max_chars: soft ceiling for the refined query length.

    Returns:
        Refined query string. On any failure, returns `raw_query` unchanged.
    """
    if not raw_query or not raw_query.strip():
        return raw_query

    preview = _build_preview(data_type, data_path, metadata)
    if preview is None and not metadata:
        # Nothing useful to add; don't burn a token on an LLM round-trip.
        return raw_query

    parts: list = [
        _OPTIMIZER_PROMPT.format(max_chars=max_chars),
        f"\n## Draft query\n{raw_query.strip()}",
        f"\n## Data type\n{data_type or 'unknown'}",
        f"\n## Metadata\n{_summarize_metadata(metadata)}",
    ]
    if preview is not None:
        png_bytes, mime = preview
        parts.append("\n## Data preview")
        parts.append({"mime_type": mime, "data": png_bytes})

    try:
        response = model.generate_content(parts)
        text = (response.text or "").strip()
    except Exception as e:
        logger.warning(f"optimize_query_for_analysis: LLM call failed: {e}")
        return raw_query

    # Parse JSON; tolerate raw-text fallback if the model omitted fences.
    refined: str | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            refined = parsed.get("refined_query")
    except Exception:
        # Last-resort: take the response text as-is if it looks like a query
        # (single line, sensible length). Otherwise give up and return raw.
        if text and "\n" not in text and 10 < len(text) < max_chars * 2:
            refined = text

    if not refined or not isinstance(refined, str) or not refined.strip():
        logger.info("optimize_query_for_analysis: no refined query parsed; using raw.")
        return raw_query

    return refined.strip()
