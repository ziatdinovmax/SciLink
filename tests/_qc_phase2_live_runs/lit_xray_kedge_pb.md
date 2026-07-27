# Literature notes: K-edge thickness measurement in spectral X-ray transmission

- For a transmission spectrum I(E)/I0(E) through a foil, the element-specific
  measure of areal thickness is the **K-edge jump**: the discontinuity in the
  linear attenuation at the element's K-shell binding energy (Pb: 88.005 keV).
- Standard recipe (K-edge subtraction / dual-energy method): compute
  ln(I0/I) just below and just above the edge in narrow windows (e.g. +/-2-4 keV),
  take the difference dmu_t = mu_t(above) - mu_t(below); thickness t = dmu_t / dmu_tab,
  where dmu_tab is the tabulated jump in the linear attenuation coefficient
  (obtain the tabulated mass-attenuation jump for Pb from the registered
  attenuation helper; Pb density rho = 11.35 g/cm3).
- A full-spectrum fit of ln(I0/I) against tabulated mu(E) is biased by beam
  hardening, detector saturation at low energies, and scatter; the differential
  edge-jump measure cancels smooth backgrounds and is the accepted standard.
- Energies below ~40 keV in 160 kVp bremsstrahlung radiography of high-Z foils
  are dominated by scatter/noise and should be excluded.
