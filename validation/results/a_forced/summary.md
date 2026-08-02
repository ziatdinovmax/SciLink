# Lattice-constant validation panel — mode: `a_forced`

| system | crystal | exp a (Å) | vasp a (Å) | vasp Δ% | mace a (Å) | mace Δ% | chgnet a (Å) | chgnet Δ% |
|---|---|---|---|---|---|---|---|---|
| si_diamond | silicon, diamond cubic (Fd-3m) | 5.431 | 5.4696 | +0.71 | 5.4553 | +0.45 | 5.4636 | +0.60 |
| cu_fcc | copper, face-centered cubic (Fm-3m) | 3.615 | 3.6283 | +0.37 | 3.6239 | +0.25 | 3.6180 | +0.08 |
| mgo_rocksalt | magnesium oxide, rock-salt (Fm-3m) | 4.212 | — | — | 4.2541 | +1.00 | 4.2577 | +1.09 |
| c_diamond | diamond carbon (Fd-3m) | 3.567 | 3.5738 | +0.19 | 3.5703 | +0.09 | 3.5718 | +0.13 |

_Δ% is deviation from the experimental lattice constant. PBE and PBE-trained MLIPs are expected to land slightly above experiment; ⚠ flags a value outside the [-0.5%, +2.5%] band._
