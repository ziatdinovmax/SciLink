"""``reconstruct_offaxis_hologram_stack`` tool — contract-exact off-axis
hologram reconstruction of a temporal ``(frame, y, x)`` stack.

Turns raw interferograms into wrapped relative phase (radians) against a
reference hologram, then gates the result on producer-supplied validation
frames (circular coherence after a global phase offset). Deterministic; the
carrier and Fourier mask come from the acquisition contract when one exists
and are only auto-picked when the caller explicitly asks for it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..._shared._spec import ToolSpec

SUPPORTED_PROFILES = ("faradaiq_mmzi_v109_offaxis_exact", "generic_offaxis")


def _cosine_mask(shape, center, radius, taper):
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    inner = radius * (1.0 - taper)
    mask = np.zeros(shape, dtype=np.float32)
    mask[dist <= inner] = 1.0
    ring = (dist > inner) & (dist < radius)
    mask[ring] = 0.5 * (1.0 + np.cos(np.pi * (dist[ring] - inner) / (radius - inner)))
    return mask


def _area_resize(data: np.ndarray, shape_yx) -> np.ndarray:
    """cv2.INTER_AREA when available (the producer's exact kernel), else an
    equivalent block-mean for integer ratios / bilinear otherwise."""
    if tuple(data.shape) == tuple(shape_yx):
        return data
    try:
        import cv2
        return cv2.resize(data, (int(shape_yx[1]), int(shape_yx[0])), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001 - optional dependency
        from scipy.ndimage import zoom
        fy, fx = shape_yx[0] / data.shape[0], shape_yx[1] / data.shape[1]
        return zoom(data, (fy, fx), order=1).astype(np.float32)


def auto_pick_carrier(hologram: np.ndarray, shape_yx, dc_exclusion_radius: int = 40) -> tuple:
    """Brightest off-axis sideband in the shifted FFT (negative half-plane
    convention: row < centre). Returns ``(row, col)`` in fftshift coordinates."""
    data = np.asarray(hologram, dtype=np.float32)
    data = _area_resize(data - np.percentile(data, 0.5), shape_yx)
    win = np.outer(np.hanning(shape_yx[0]), np.hanning(shape_yx[1])).astype(np.float32)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(data * win)))
    cy, cx = shape_yx[0] // 2, shape_yx[1] // 2
    y, x = np.ogrid[: shape_yx[0], : shape_yx[1]]
    spec[(y - cy) ** 2 + (x - cx) ** 2 <= dc_exclusion_radius ** 2] = 0
    spec[cy:, :] = 0                       # keep one half-plane (conjugate twin)
    r, c = np.unravel_index(int(np.argmax(spec)), spec.shape)
    return float(r), float(c)


class OffAxisReconstructor:
    """Producer-contract reconstruction (FaradaIQ mMZI v109 profile)."""

    def __init__(self, processed_shape_yx, carrier_row_col, mask_radius_px,
                 carrier_subpixel_row_col=None, mask_taper=0.2, offset_percentile=0.5):
        self.shape = (int(processed_shape_yx[0]), int(processed_shape_yx[1]))
        self.carrier = (float(carrier_row_col[0]), float(carrier_row_col[1]))
        self.subpixel = tuple(float(v) for v in (carrier_subpixel_row_col or carrier_row_col))
        self.radius = float(mask_radius_px)
        self.taper = float(mask_taper)
        self.offset_percentile = float(offset_percentile)
        self.window = np.outer(np.hanning(self.shape[0]).astype(np.float32),
                               np.hanning(self.shape[1]).astype(np.float32))
        self.mask = _cosine_mask(self.shape, self.carrier, self.radius, self.taper)

    @classmethod
    def from_contract(cls, contract: dict) -> "OffAxisReconstructor":
        prof = contract.get("reconstruction_profile", {})
        eff = contract.get("effective_reconstruction", {})
        name = prof.get("name", "generic_offaxis")
        if name not in SUPPORTED_PROFILES:
            raise ValueError(f"Unsupported reconstruction profile: {name}")
        if prof.get("apodization_window", "separable_hann") != "separable_hann":
            raise ValueError("Only the separable Hann apodization window is supported")
        return cls(eff["processed_fft_shape_yx"], eff["carrier_fftshift_row_col"],
                   eff["mask_radius_fft_pixels"],
                   eff.get("carrier_subpixel_fftshift_row_col"),
                   prof.get("mask_taper_fraction", 0.2),
                   prof.get("intensity_offset_percentile", 0.5))

    def field(self, image: np.ndarray) -> np.ndarray:
        data = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(data)
        fill = float(np.median(data[finite])) if finite.any() else 0.0
        data = np.where(finite, data, fill)
        data = data - np.percentile(data, self.offset_percentile)
        data = _area_resize(data, self.shape)
        spec = np.fft.fftshift(np.fft.fft2(data * self.window)) * self.mask
        ic = tuple(int(round(v)) for v in self.carrier)
        spec = np.roll(spec, (self.shape[0] // 2 - ic[0], self.shape[1] // 2 - ic[1]), axis=(0, 1))
        fld = np.fft.ifft2(np.fft.ifftshift(spec))
        ry, rx = self.subpixel[0] - ic[0], self.subpixel[1] - ic[1]
        if ry or rx:
            y, x = np.ogrid[: self.shape[0], : self.shape[1]]
            fld *= np.exp(-2j * np.pi * (ry * y / self.shape[0] + rx * x / self.shape[1]))
        return fld

    def sideband_contrast(self, image: np.ndarray) -> float:
        """|FFT| at the carrier divided by the median |FFT| of the processed
        hologram: an internal fringe-quality gate (>= 5 is a clean sideband)."""
        data = np.asarray(image, dtype=np.float32)
        data = data - np.percentile(data, self.offset_percentile)
        data = _area_resize(data, self.shape)
        spec = np.abs(np.fft.fftshift(np.fft.fft2(data * self.window)))
        r, c = (int(round(v)) for v in self.carrier)
        peak = float(spec[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].max())
        return peak / max(float(np.median(spec)), 1e-9)

    def relative_phase(self, hologram: np.ndarray, reference_field: np.ndarray) -> np.ndarray:
        return np.angle(self.field(hologram) * np.conj(reference_field)).astype(np.float32)


def producer_qc(reconstructed: np.ndarray, target: np.ndarray) -> dict:
    """Circular residual metrics after allowing only a global phase offset."""
    valid = np.isfinite(reconstructed) & np.isfinite(target)
    vec = np.mean(np.exp(1j * (target[valid] - reconstructed[valid])))
    offset = float(np.angle(vec))
    resid = np.angle(np.exp(1j * (target - (reconstructed + offset))))
    return {"circular_coherence": float(abs(vec)), "global_phase_offset_rad": offset,
            "circular_mae_rad": float(np.mean(np.abs(resid[valid]))),
            "circular_rmse_rad": float(np.sqrt(np.mean(resid[valid] ** 2)))}


def _h5_json(h5, path):
    v = h5[path][()]
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return json.loads(v) if isinstance(v, str) else v


def _first_3d_dataset(h5) -> str:
    found = []

    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found.append((obj.size, name))
    h5.visititems(visit)
    if not found:
        raise ValueError("No 3-D dataset (frame, y, x) found in the HDF5 file")
    return sorted(found)[-1][1]


def reconstruct_offaxis_hologram_stack(
    measurement_h5: str,
    output_dir: str,
    reference_h5: Optional[str] = None,
    raw_dataset: Optional[str] = None,
    reference_dataset: Optional[str] = None,
    contract: Any = "hdf5",
    expected_contract_sha256: Optional[str] = None,
    frames: Any = "all",
    validation_frames_npy: Optional[str] = None,
    validation_frame_indices: Optional[list] = None,
    qc_coherence_min: float = 0.95,
    carrier: Any = "contract",
    processed_scale: float = 0.35,
    mask_radius_px: float = 63.0,
    mask_taper: float = 0.2,
    output_name: Optional[str] = None,
) -> dict:
    """Reconstruct a temporal off-axis hologram stack to wrapped relative phase.

    Returns a dict with the output paths, per-frame QC metrics and whether
    the producer-target gate passed. Never converts phase to any physical
    quantity. See ``TOOL_SPEC`` for the parameters.
    """
    import h5py
    t0 = time.time()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    meas_path = Path(measurement_h5)
    ref_path = Path(reference_h5) if reference_h5 else meas_path
    with h5py.File(meas_path, "r") as m, h5py.File(ref_path, "r") as r:
        # ---- contract ----
        contract_dict = None
        if contract == "hdf5":
            key = "metadata/mmzi_processing_contracts_json"
            if key in m:
                vals = _h5_json(m, key)
                contract_dict = vals[0] if isinstance(vals, list) else vals
        elif isinstance(contract, dict):
            contract_dict = contract
        elif isinstance(contract, str) and Path(contract).is_file():
            contract_dict = json.loads(Path(contract).read_text())
        if contract_dict is not None and expected_contract_sha256 and \
                contract_dict.get("contract_sha256") != expected_contract_sha256:
            raise ValueError("Reconstruction contract hash does not match expected_contract_sha256")
        raw_name = raw_dataset or _first_3d_dataset(m)
        raw = m[raw_name]
        n_frames, H, W = raw.shape
        if reference_dataset:
            ref_img = r[reference_dataset][()]
        elif ref_path == meas_path:
            ref_img = raw[0][()]
        else:
            ref_img = r[_first_3d_dataset(r)][0][()] if any(
                isinstance(r[k], h5py.Dataset) and r[k].ndim == 3 for k in r) else None
            if ref_img is None:
                two_d = [k for k in r if isinstance(r[k], h5py.Dataset) and r[k].ndim == 2]
                ref_img = r[two_d[0]][()]
        # ---- reconstructor ----
        carrier_source = "contract"; mask_cap = None; mask_used = None
        if contract_dict is not None and carrier == "contract":
            rec = OffAxisReconstructor.from_contract(contract_dict)
        else:
            shape = (int(round(H * processed_scale)), int(round(W * processed_scale)))
            if carrier == "auto" or carrier == "contract":
                cr = auto_pick_carrier(ref_img, shape); carrier_source = "auto_picked_from_reference"
            else:
                cr = (float(carrier[0]), float(carrier[1])); carrier_source = "caller"
            # Generic path: the sideband mask must never reach the DC term. Cap the
            # radius at 60 % of the carrier-to-centre distance (the FaradaIQ default
            # of 63 px assumes that instrument's carrier geometry).
            dc_dist = float(np.hypot(cr[0] - shape[0] / 2, cr[1] - shape[1] / 2))
            mask_cap = 0.6 * dc_dist
            mask_used = min(float(mask_radius_px), mask_cap) if dc_dist > 0 else float(mask_radius_px)
            rec = OffAxisReconstructor(shape, cr, mask_used, None, mask_taper)
        ref_field = rec.field(ref_img)
        ref_contrast = rec.sideband_contrast(ref_img)
        # ---- frames ----
        vidx = [int(v) for v in (validation_frame_indices or [])]
        if frames == "all":
            requested = list(range(n_frames))
        else:
            requested = sorted({int(f) for f in frames} | set(vidx))
        name = output_name or f"{meas_path.stem}_wrapped_phase.npy"
        phase_path = out / name
        full = len(requested) == n_frames
        stack = np.lib.format.open_memmap(phase_path, mode="w+", dtype=np.float32,
                                          shape=(len(requested), rec.shape[0], rec.shape[1]))
        qc_cache = {}
        for i, fi in enumerate(requested):
            ph = rec.relative_phase(raw[fi], ref_field)
            stack[i] = ph
            if fi in vidx:
                qc_cache[fi] = ph
        stack.flush(); del stack
    # ---- QC gate ----
    qc_frames = []
    if validation_frames_npy and vidx:
        targets = np.load(validation_frames_npy, mmap_mode="r")
        for pos, fi in enumerate(vidx):
            met = producer_qc(qc_cache[fi], targets[pos]); met["frame_index"] = fi
            met["passed"] = met["circular_coherence"] >= qc_coherence_min
            qc_frames.append(met)
    qc_passed = all(q["passed"] for q in qc_frames) if qc_frames else None
    receipt = {
        "schema": "scilink_offaxis_reconstruction_receipt_v1",
        "measurement_h5": str(meas_path), "raw_dataset": raw_name,
        "reference_h5": str(ref_path), "reference_dataset": reference_dataset,
        "contract_sha256": (contract_dict or {}).get("contract_sha256"),
        "profile": (contract_dict or {}).get("reconstruction_profile", {}).get("name", "generic_offaxis"),
        "carrier_source": carrier_source, "carrier_fftshift_row_col": list(rec.carrier),
        "reference_sideband_contrast": ref_contrast,
        "mask_radius_capped_to": (mask_used if (mask_cap is not None and mask_used < float(mask_radius_px)) else None),
        "mask_radius_fft_pixels": rec.radius, "processed_shape_yx": list(rec.shape),
        "frames_reconstructed": requested if not full else "all",
        "frame_index_map": None if full else requested,
        "phase_output": str(phase_path),
        "phase_semantics": "wrapped relative phase in radians, angle(current_field * conj(reference_field))",
        "qc_coherence_min": qc_coherence_min, "producer_validation_passed": qc_passed,
        "producer_validation": qc_frames,
        "absolute_quantities_inferred": False, "seconds": round(time.time() - t0, 1),
    }
    rpath = out / f"{Path(name).stem}_receipt.json"
    rpath.write_text(json.dumps(receipt, indent=2))
    return {"status": "success" if qc_passed in (True, None) else "qc_failed",
            "phase_output": str(phase_path), "receipt": str(rpath),
            "producer_validation_passed": qc_passed,
            "min_circular_coherence": (min(q["circular_coherence"] for q in qc_frames) if qc_frames else None),
            "n_frames": len(requested), "processed_shape_yx": list(rec.shape),
            "carrier_source": carrier_source, "reference_sideband_contrast": ref_contrast,
            "producer_validation": qc_frames}


TOOL_SPEC = ToolSpec(
    name="reconstruct_offaxis_hologram_stack",
    description=(
        "Reconstruct a raw off-axis (Mach-Zehnder / digital holographic) interferogram "
        "stack (frame, y, x) into WRAPPED RELATIVE PHASE in radians against a reference "
        "hologram, using the acquisition contract stored in the HDF5 file (carrier, "
        "Fourier mask, processing scale) when present, and gating the result on "
        "producer-supplied validation frames (circular coherence >= threshold after a "
        "global phase offset). Writes a float32 .npy stack plus a receipt JSON."
    ),
    import_line=("from scilink.skills.data_preparation.mmzi_hologram_reconstruction"
                 ".reconstruct import reconstruct_offaxis_hologram_stack"),
    signature=("reconstruct_offaxis_hologram_stack(measurement_h5: str, output_dir: str, "
               "reference_h5: str | None = None, raw_dataset: str | None = None, "
               "reference_dataset: str | None = None, contract: 'hdf5' | dict | path = 'hdf5', "
               "expected_contract_sha256: str | None = None, frames: 'all' | list[int] = 'all', "
               "validation_frames_npy: str | None = None, validation_frame_indices: list[int] | None = None, "
               "qc_coherence_min: float = 0.95, carrier: 'contract' | 'auto' | (row, col) = 'contract', "
               "processed_scale: float = 0.35, mask_radius_px: float = 63.0, mask_taper: float = 0.2, "
               "output_name: str | None = None) -> dict"),
    parameters={
        "measurement_h5": {"type": "str", "description": "HDF5 file holding the raw (frame, y, x) interferogram stack."},
        "output_dir": {"type": "str", "description": "Directory for the phase stack and receipt (outside the source bundle)."},
        "reference_h5": {"type": "str", "description": "HDF5 file holding the reference interferogram; omit when the reference lives in measurement_h5 (or to use frame 0 as reference)."},
        "raw_dataset": {"type": "str", "description": "HDF5 dataset path of the raw stack; default = the largest 3-D dataset. Prefer the path named by the bundle's manifest."},
        "reference_dataset": {"type": "str", "description": "HDF5 dataset path of the 2-D reference interferogram."},
        "contract": {"type": "str | dict", "description": "'hdf5' (default) reads metadata/mmzi_processing_contracts_json from the file; or a dict / JSON path with reconstruction_profile + effective_reconstruction; or None to force the generic path (auto carrier)."},
        "expected_contract_sha256": {"type": "str", "description": "Hash from the bundle manifest; raises on mismatch so a stale contract is never applied silently."},
        "frames": {"type": "str | list[int]", "description": "'all' (default) or a list of frame indices; validation frames are always added."},
        "validation_frames_npy": {"type": "str", "description": ".npy of producer phase target frames (n_targets, y, x) for the QC gate."},
        "validation_frame_indices": {"type": "list[int]", "description": "Frame indices matching validation_frames_npy, in order."},
        "qc_coherence_min": {"type": "float", "description": "QC gate: minimum circular coherence vs each producer target (default 0.95). LOWER only if the producer's own pipeline is known to differ (e.g. different apodization); RAISE for a stricter byte-level reproduction check."},
        "carrier": {"type": "str | tuple", "description": "'contract' (default: exact saved carrier; falls back to 'auto' only when no contract exists), 'auto' (brightest off-axis sideband of the reference, DC excluded) or an explicit (row, col) in fftshift coordinates of the processed frame. Never auto-pick when an authoritative contract carrier exists."},
        "processed_scale": {"type": "float", "description": "Generic path only: resize factor applied before the FFT (default 0.35, the producer's runtime scale). Smaller = faster/coarser."},
        "mask_radius_px": {"type": "float", "description": "Generic path only: sideband mask radius in processed-FFT pixels (default 63, automatically capped at 60 % of the carrier-to-DC distance so the mask never swallows the DC term; the receipt reports mask_radius_capped_to). RAISE to keep finer phase detail (risk: DC/twin leakage); LOWER to suppress noise."},
        "mask_taper": {"type": "float", "description": "Cosine taper fraction of the sideband mask (default 0.2)."},
        "output_name": {"type": "str", "description": "File name of the phase stack (default <measurement stem>_wrapped_phase.npy)."},
    },
    required=["measurement_h5", "output_dir"],
    returns=("dict: status ('success' | 'qc_failed'), phase_output (.npy path, float32 "
             "(n_frames, y, x) wrapped phase in radians), receipt (JSON path), "
             "producer_validation_passed (bool or None when no targets given), "
             "min_circular_coherence, producer_validation (per target frame: frame_index, circular_coherence, "
             "circular_mae_rad, circular_rmse_rad, passed — surface these in qc.metrics), n_frames, processed_shape_yx, carrier_source, "
             "reference_sideband_contrast (|FFT| at the carrier / median |FFT|; the internal "
             "fringe-quality gate when no producer targets exist — >= 5 means a clean sideband)."),
    when_to_use=("First step for any raw temporal hologram / interferogram stack: turn "
                 "interferograms into wrapped phase BEFORE any curve or image analysis. "
                 "Then call derive_phase_products on the output."),
    agents=["data_preparation"],
)
