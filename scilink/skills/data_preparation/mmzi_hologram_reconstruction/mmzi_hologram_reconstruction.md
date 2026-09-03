---
description: 'Off-axis hologram / interferogram (mMZI, digital holographic microscopy) preparation: contract-exact reconstruction of raw (frame, y, x) interferogram stacks into wrapped relative phase with a producer-target QC gate, then piston-immune steady-state phase maps and ROI phase-vs-time traces joined to a per-frame condition timeline.'
technique: [mMZI, "Mach-Zehnder interferometry", "off-axis holography", "digital holographic microscopy", "quantitative phase imaging", interferogram]
---

# Off-axis hologram reconstruction (mMZI / DHM)

## overview

A raw off-axis interferogram is an intensity image carrying the object phase
on a spatial carrier fringe. It is **not** an image of the sample and **not** a
spectral cube: the third axis of a `(frame, y, x)` stack is time. Nothing
downstream (curve or image agents) may see the raw stack; this skill turns it
into two analysis-ready products:

1. **Steady-state phase difference maps** (2-D `.npy`, radians, global piston
   removed) — for image analysis of *where* the phase changed.
2. **ROI phase-vs-time traces** (`.csv`, piston-immune ROI differences and a
   template-amplitude fraction, joined with the per-frame condition timeline)
   — for curve / time-series analysis of *when* and *how much*.

Two helper tools do the deterministic work: `reconstruct_offaxis_hologram_stack`
(contract-exact reconstruction + producer QC gate) and `derive_phase_products`
(piston-immune products + sidecars). The generated script is the glue: read the
bundle's manifest, run the tools per acquisition, and report QC.

## planning

- **Find the authority.** A prepared bundle usually ships a
  `reconstruction_manifest.json` (per-run raw HDF5, reference HDF5, dataset
  paths, contract hash, producer validation frames and their indices, the QC
  threshold) and same-stem JSON sidecars per raw file. Read the manifest
  first; never pick "the largest dataset" when a manifest names the dataset.
- **Reference.** Each run is reconstructed against a reference interferogram
  (a separate file, or a dataset inside the run's own file, possibly shared
  with a sibling run). Use exactly the reference the manifest names.
- **Contract.** The carrier position, Fourier mask radius and processing scale
  are stored inside the HDF5 (`metadata/mmzi_processing_contracts_json`) in
  processed-FFT coordinates. Pass `contract='hdf5'` and the manifest's
  `reconstruction_contract_sha256` as `expected_contract_sha256`; never
  auto-pick a carrier when a contract exists. For a bundle WITHOUT a contract,
  `carrier='auto'` picks the brightest off-axis sideband — say so in the QC notes.
- **QC gate.** Producer validation frames (typically 5 per run) must be
  reproduced with circular coherence ≥ the manifest threshold (0.95). A run
  that fails the gate is reported as failed, not silently dropped.
- **Condition timeline.** Frame-level conditions (e.g. magnet state) come from
  a per-run timing CSV (`*_frame_conditions.csv`); join by frame order. State
  labels that mean "moving" go into `transition_states`.
- **Products per run**, all under the output directory, never inside the
  bundle: the wrapped phase stack (kept for provenance), one steady-state map
  (last condition minus first), one trace CSV, receipts. Group the traces of
  all runs as one series (`group` field) so the curve agent sees them together.
- **Dense-fringe zone.** Where a strong perturbation makes fringes denser than
  the sampling limit (typically a band beside the wall nearest the
  perturbation), the phase is unresolvable and 2π residues leak into the map
  as sharp-edged plateaus. Never guess this zone: pass
  `auto_exclude_dense_fringes=True` so the derive tool MEASURES it (edge zones
  with median |dφ/dx| above the sampling limit) and reports what it excluded;
  data with no such zone (a weak or interior perturbation) excludes nothing.
  When a zone was excluded, say in the QC notes that the true phase change
  there is larger than anything reported.
- **Memory / time.** A run of 120 frames at 2160×4096 uint16 is ~2 GB on disk
  and takes ~30 s to reconstruct; process runs sequentially and rely on the
  memory-mapped outputs the tools write.

## implementation

```python
import os, json, glob
from pathlib import Path
from scilink.skills.data_preparation.mmzi_hologram_reconstruction.reconstruct import reconstruct_offaxis_hologram_stack
from scilink.skills.data_preparation.mmzi_hologram_reconstruction.derive import derive_phase_products

root = Path(_PREP["input"]); out = Path(_PREP["out_dir"])
manifest_path = next(iter(root.rglob("reconstruction_manifest.json")), None)
manifest = json.loads(manifest_path.read_text()) if manifest_path else {}
bundle = manifest_path.parent.parent if manifest_path else root   # manifest lives in <bundle>/mmzi/
thr = float(manifest.get("qc_gate", {}).get("producer_phase_circular_coherence_minimum", 0.95))

products, receipts, metrics, notes, all_ok = [], [], {}, [], True
for rec in manifest.get("records", []):
    wf = rec["workflow"].lower(); run_dir = out / wf
    r = reconstruct_offaxis_hologram_stack(
        measurement_h5=str(bundle / rec["raw_hdf5"]), output_dir=str(run_dir),
        reference_h5=str(bundle / rec["reference_hdf5"]),
        raw_dataset=rec["raw_dataset"], reference_dataset=rec["reference_dataset"],
        contract="hdf5", expected_contract_sha256=rec["reconstruction_contract_sha256"],
        validation_frames_npy=str(bundle / rec["producer_validation_frames"]),
        validation_frame_indices=rec["producer_validation_frame_indices"],
        qc_coherence_min=thr, output_name=f"mmzi_{wf}_wrapped_phase.npy")
    metrics[f"{wf}_reference_sideband_contrast"] = r["reference_sideband_contrast"]
    for q in r["producer_validation"]:                     # per target frame, straight from the tool
        metrics[f"{wf}_coherence_frame{q['frame_index']}"] = q["circular_coherence"]
        if not q["passed"]:
            notes.append(f"{wf}: frame {q['frame_index']} coherence {q['circular_coherence']:.4f} < {thr}")
    receipts.append(r["receipt"])
    if r["producer_validation_passed"] is False:      # None = no targets exist (caveat, not failure)
        all_ok = False; notes.append(f"{wf}: producer QC gate failed"); continue
    if r["producer_validation_passed"] is None:
        notes.append(f"{wf}: no producer targets; carrier_source={r['carrier_source']} (caveat)")
    d = derive_phase_products(
        wrapped_phase_npy=r["phase_output"], output_dir=str(run_dir),
        timeline_csv=str(bundle / rec["frame_condition_timeline"]),
        state_column="magnet_state", time_column="capture_elapsed_s",
        transition_states=["moving_to_cuvette", "retracting"],
        steady_window_frames=25, bin_factor=2, smoothing_sigma=1.0,
        auto_exclude_dense_fringes=True, stem=f"mmzi_{wf}")
    # the tool measures and reports any excluded edge zone (sidecar interpretation_limits);
    # rerun with bin_factor=1 if the discontinuity fraction still trips the gate
    metrics[f"{wf}_map_discontinuity_fraction"] = d["cross_checks"]["steady_map_discontinuity_fraction"]
    products += [
        {"path": d["diff_map"], "kind": "image", "sidecar": d["diff_map_sidecar"],
         "description": f"{wf} steady-state phase difference map (rad)", "group": "phase_maps"},
        {"path": d["roi_curve"], "kind": "curve", "sidecar": d["roi_curve_sidecar"],
         "description": f"{wf} ROI phase differences vs time (rad)", "group": "phase_traces"},
        {"path": d["quicklook"], "kind": "figure", "sidecar": None,
         "description": f"{wf} quicklook", "group": "figures"}]
print("PREP_RESULT_JSON:" + json.dumps({"products": products,
      "qc": {"passed": all_ok, "metrics": metrics, "notes": notes},
      "receipts": receipts, "summary": f"{len(manifest.get('records', []))} runs reconstructed and derived."}))
```

Adapt, don't copy blindly: a bundle with a single raw file and no manifest
uses `contract='hdf5'` with `reference_h5=None` (frame 0 as reference) or
`carrier='auto'`; a bundle without a timing CSV yields a drift-only product.
When the same reference file serves several runs, that is by design (shared
reference at cycle start) — do not "fix" it.

## interpretation

- The products are **relative optical phase in radians**. Never convert to
  concentration, refractive index or thickness unless wavelength, path length
  and the response coefficient are all recorded and validated; the bundle may
  carry a calibration marked temporary — treat it as not authoritative.
- The interferometer's global piston fluctuates between frames (often by more
  than π), so absolute phase per frame is meaningless; every product here is a
  within-frame spatial difference or a circular-mean field over a steady
  window, which are immune to piston.
- A condition change inside a run is confounded with time; the traces carry the
  first and last steady windows so drift can be bracketed, and single-condition
  control runs give the no-effect null.
- Left/right in the maps are image coordinates; the instrument geometry
  (where a perturbation source sits relative to the field of view) is usually
  not recorded.

## validation

- Contract-less data (no embedded contract, no producer targets): the gate
  that exists is internal — `reference_sideband_contrast` returned by the
  reconstruct tool (>= 5 = clean sideband; put it in `qc.metrics`), the
  steady-map discontinuity fraction and the trace-vs-map agreement below. Pass
  on those, and put "no producer targets; carrier auto-picked" in `qc.notes`
  as a caveat. Do NOT fail QC for a check that has no target, and do not
  invent metrics the tools did not return.
- Every reconstructed run WITH producer targets must report `producer_validation_passed: true` with
  circular coherence ≥ the manifest threshold for **every** target frame; a
  failed gate must appear in `qc.notes` and set `qc.passed` false.
- The reconstruction contract hash must match the manifest (the tool raises
  otherwise); the receipt must name the raw/reference files and datasets.
- The manifest's reference for a run may be a dataset stored inside a sibling
  run's raw HDF5 (a new reference is captured at the start of each cycle and
  shared by that cycle's runs). That is the producer's design, not a defect:
  judge references by the manifest's `reference_hdf5` + `reference_dataset`
  pair, never by the file name.
- Dense-fringe edge zones are MEASURED by the derive tool
  (`auto_exclude_dense_fringes=True`), never hand-picked; when the sidecar reports an
  excluded range, `qc.notes` must say "near-wall phase unresolvable; true change larger
  than reported". A hand-picked exclusion, or one that removes most of the frame, is a
  defect (the tool refuses exclusions leaving fewer than half the columns).
- After that, `steady_map_discontinuity_fraction` should be < 0.005 (must be < 0.02);
  larger values mean remaining 2π residues — lower `bin_factor` or widen the exclusion.
- Trace and map must agree: the band-trace step and the map ROI difference
  for the same ROI pair should match within ~15 %; the sidecar's
  `cross_checks` carries both.
- Products must live outside the input bundle and each `.npy`/`.csv` must have
  its same-stem JSON sidecar stating units (radians), semantics and limits.
- The script must not fit models, classify, or interpret; preparation ends at
  QC-passed products.
