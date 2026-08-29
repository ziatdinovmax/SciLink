# Ferroelectric polarization mapping — atomic-resolution STEM demo

A HAADF-STEM image of a ferroelectric perovskite oxide (ABO3), 1024 × 1024 px
over a 16 × 16 nm field of view (15.6 pm/px). Both cation sublattices are
resolved, so the polar distortion — the off-centering of the B-site cation
from the centre of its A-site cage — can be measured cell by cell.

This is the atomic-resolution example for the image-analysis agent. It
exercises the `atomic_stem` skill end to end: deep-learning column detection
(AtomNet3 ensemble), detection QC, per-unit-cell polarization mapping, and
domain / domain-wall segmentation, orchestrated by an agent-written script.

Real experimental data (not simulated).

## Run

Give the agent this objective verbatim — it is what steers the run toward
polarization mapping rather than generic lattice metrology:

> Map the per-unit-cell ferroelectric polarization in this image and identify
> any ferroelectric domains and domain walls.

**Streamlit UI** (recommended):

```bash
scilink ui
```

— upload `image.npy` and `image.json`, then paste the objective above.

**CLI**:

```bash
scilink analyze --data examples/ferroelectric_polarization_demo/image.npy \
                --metadata examples/ferroelectric_polarization_demo/image.json
```

— then type the objective above as your first chat message.

The first run downloads the AtomNet3 detector weights; a full run takes
roughly 5–10 minutes with a frontier model.

## What to expect

The agent should auto-select the `atomic_stem` skill, detect both cation
sublattices, and produce a per-unit-cell polarization map (a quiver of
off-centering vectors with a direction colour wheel), a domain map, and a
domain-wall map, together with the per-cell arrays and a metrics JSON.

Qualitatively, the image contains more than one ferroelectric domain with a
wall between them; a good run finds the domains, localizes the wall, and
reports a smooth, domain-structured field (high direction-coherence score)
rather than salt-and-pepper noise.
