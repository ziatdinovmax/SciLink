# Literature notes: K-edge thickness measurement in spectral X-ray transmission

- For a transmission spectrum I(E)/I0(E) through a foil, the element-specific
  measure of areal thickness is the **K-edge jump**: the discontinuity in the
  linear attenuation at the element's K-shell binding energy (Au: 80.725 keV).
- Standard recipe (K-edge subtraction / dual-energy method): compute
  ln(I0/I) just below and just above the edge in narrow windows (e.g. ±2-4 keV),
  take the difference Δμt = μt(above) − μt(below); thickness t = Δμt / Δμ_tab,
  where Δμ_tab is the tabulated jump in the linear attenuation coefficient
  (Au: μ/ρ jumps from ≈ 1.006 to ≈ 4.575 cm²/g at the K edge; ρ = 19.3 g/cm³).
- A full-spectrum fit of ln(I0/I) against tabulated μ(E) is biased by beam
  hardening, detector saturation at low energies, and scatter; the differential
  edge-jump measure cancels smooth backgrounds and is the accepted standard.
- Energies below ~40 keV in 160 kVp bremsstrahlung radiography of high-Z foils
  are dominated by scatter/noise and should be excluded.
