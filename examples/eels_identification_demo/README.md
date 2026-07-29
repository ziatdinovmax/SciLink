# EELS — identification mode demo

A synthetic core-loss EELS spectrum of SrTiO₃ for testing the
curve-fitting agent's **identification mode** (`task_mode="identification"`).
Energy positions follow van Benthem, Elsässer, Rühle, *Ultramicroscopy*
(2003) [doi:10.1016/s0304-3991(03)00112-8] for the Ti L₃ and O K edge
onsets, and Zhang, Visinoiu, Heyroth et al., *Phys. Rev. B* **71**,
064108 (2005) [doi:10.1103/physrevb.71.064108] for the Ti L₃–L₂
separation and the O K low-energy fine-structure splitting.

The metadata sidecar deliberately does not name the material; the
agent should resolve identity from the spectral fingerprint alone.

## Run

**Streamlit UI** (recommended):

```bash
scilink ui
```

— upload `spectrum.npy` and `spectrum.json`, then ask the agent in
chat to run in identification mode.

**CLI**:

```bash
scilink analyze --data examples/eels_identification_demo/spectrum.npy \
                --metadata examples/eels_identification_demo/spectrum.json
```

— and tell the agent in chat to run in identification mode, e.g.:

> "Run the analysis in identification mode and produce a ranked list
> of candidate materials."

## What to expect

The agent should detect two core-loss edges in the 350–862 eV window
(Ti L₂,₃ near 456 eV, O K near 530 eV), fit the near-edge fine
structure, and emit a ranked list of candidate materials in its
`candidate_identifications` output — typically Ti⁴⁺ titanates such as
SrTiO₃, anatase, and rutile, since the spectrum doesn't include Sr or
Ba edges that would uniquely fingerprint the cation.
