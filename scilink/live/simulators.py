"""Simulated experiments for trying a live loop without an instrument.

Four :class:`~scilink.live.instruments.Instrument` implementations, each a small
physical model with a scripted timeline — the things a live loop has to cope
with actually happen (a slow physical trend, an abrupt change of regime, a
glitch frame) and the acquisition parameters really matter (noise, broadening),
so a recommendation has a measurable effect. Every frame carries its ground
``truth`` so a result can be checked; nothing in the loop reads it.

They are stand-ins, not digital twins: simple enough to read in a minute, and
replaced by your own ``Instrument`` subclass when the loop meets a real
controller.

    BeamlineXRD    powder diffraction over a temperature ramp, with a phase transition
    InSituRaman    D / G bands during an anneal; fluorescence, cosmic rays
    AFMForceCurve  force–distance curves across two materials
    STMdIdV        dI/dV point spectra of a gap, broadened by the lock-in modulation
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .instruments import Frame, Instrument
from .recommend import InstrumentSchema


def _gauss(x, c, fwhm):
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _lorentz(x, c, fwhm):
    g = fwhm / 2.0
    return g ** 2 / ((x - c) ** 2 + g ** 2)


class _Simulator(Instrument):
    #: What happens when, for people reading a run: [{"frame", "what"}].
    events: List[Dict[str, Any]] = []

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.frame = 0

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed * 100_003 + self.frame)

    def acquire(self, params: Dict[str, Any]) -> Frame:
        params = self.check(params)
        self.frame += 1
        return self._acquire(params, self._rng())

    def _acquire(self, params, rng) -> Frame:          # pragma: no cover - interface
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────
# Beamline XRD
# ──────────────────────────────────────────────────────────────

class BeamlineXRD(_Simulator):
    """Powder diffraction during a temperature ramp (300 K upward, 6 K / frame).

    Phase A's reflections shift to lower 2θ with thermal expansion. Around
    frame 36 (≈ 510 K) A transforms into phase B over about eight frames: A's
    reflections fade, B's grow at different angles — a regime change no recipe
    written for A describes. Counts scale with ``exposure_s`` (Poisson noise).
    """

    name = "beamline_xrd"
    system_info = {"technique": "synchrotron powder X-ray diffraction",
                   "sample": "oxide powder heated in situ",
                   "x_axis": "2theta (degrees)", "y_axis": "intensity (counts)",
                   "wavelength_angstrom": 0.7293}
    schema = InstrumentSchema.from_dict({
        "exposure_s": {"low": 0.1, "high": 20.0, "units": "s",
                       "description": "detector exposure per pattern; counts scale with it"}})
    defaults = {"exposure_s": 2.0}
    outputs = {"main_peak_2theta": "2theta position of the strongest reflection of the majority phase",
               "main_peak_fwhm": "full width at half maximum of that reflection",
               "main_peak_height": "height of that reflection above the background"}
    targets = ["position, width and height of the strongest reflection"]
    events = [{"frame": 1, "what": "ramp starts at 300 K; phase A, reflections shift with thermal expansion"},
              {"frame": 36, "what": "phase A begins transforming into phase B (≈ 510 K)"},
              {"frame": 44, "what": "transformation complete; phase B only"}]

    A = [(14.20, 1.00), (20.15, 0.55), (24.75, 0.35)]      # (2theta at 300 K, relative intensity)
    B = [(15.35, 0.90), (18.90, 0.60), (26.10, 0.40)]
    T0, DT, T_TRANSITION, WIDTH_FRAMES = 300.0, 6.0, 510.0, 8
    ALPHA = 1.6e-5                                          # linear thermal expansion, 1/K

    def temperature(self, frame: int) -> float:
        return self.T0 + self.DT * (frame - 1)

    def fraction_b(self, frame: int) -> float:
        t = (self.temperature(frame) - self.T_TRANSITION) / (self.DT * self.WIDTH_FRAMES / 2.0)
        return float(1.0 / (1.0 + np.exp(-2.2 * t)))

    def _acquire(self, params, rng) -> Frame:
        x = np.linspace(10.0, 30.0, 1600)
        T = self.temperature(self.frame)
        fb = self.fraction_b(self.frame)
        shift = lambda tt: tt - np.degrees(np.tan(np.radians(tt / 2.0)) * 2.0 * self.ALPHA * (T - self.T0))
        rate = 0.0
        for phase, frac in ((self.A, 1.0 - fb), (self.B, fb)):
            for tt, rel in phase:
                rate = rate + 900.0 * frac * rel * _gauss(x, shift(tt), 0.11)
        rate = rate + 60.0 + 25.0 * np.exp(-(x - 10.0) / 9.0)      # background
        counts = rng.poisson(np.clip(rate * params["exposure_s"], 0, None)).astype(float)
        major = self.A if fb < 0.5 else self.B
        return Frame(x=x, y=counts / params["exposure_s"], params=params,
                     x_label="two_theta_deg", y_label="intensity_cps",
                     meta={"temperature_K": T},
                     truth={"temperature_K": T, "fraction_B": fb,
                            "main_peak_2theta": float(shift(major[0][0])),
                            "main_peak_fwhm": 0.11, "majority_phase": "A" if fb < 0.5 else "B"})


# ──────────────────────────────────────────────────────────────
# In-situ Raman
# ──────────────────────────────────────────────────────────────

class InSituRaman(_Simulator):
    """D and G bands of a carbon film during an anneal (60 frames).

    The D/G height ratio falls from 1.20 to 0.45 and the G band stiffens from
    1582 to 1594 cm⁻¹; a fluorescence background grows through the run; frames
    23 and 41 carry a cosmic-ray spike. Signal scales with
    ``laser_power_mw × integration_s × accumulations`` (shot noise), but above
    8 mW the laser heats the film and the G band softens — more power is not
    free.
    """

    name = "insitu_raman"
    system_info = {"technique": "Raman spectroscopy", "sample": "amorphous carbon film, annealed in situ",
                   "excitation_nm": 532, "x_axis": "Raman shift (cm^-1)", "y_axis": "intensity (counts)"}
    schema = InstrumentSchema.from_dict({
        "laser_power_mw": {"low": 0.5, "high": 20.0, "units": "mW",
                           "description": "laser power at the sample; above ~8 mW it heats the film"},
        "integration_s": {"low": 0.2, "high": 30.0, "units": "s", "description": "integration time per accumulation"},
        "accumulations": {"kind": "int", "low": 1, "high": 20, "description": "accumulations averaged"}})
    defaults = {"laser_power_mw": 4.0, "integration_s": 5.0, "accumulations": 2}
    outputs = {"g_position": "position of the G band (the band near 1585 cm^-1)",
               "d_over_g": "ratio of the D band height to the G band height, each above the background"}
    targets = ["G band position", "D/G height ratio"]
    events = [{"frame": 1, "what": "anneal starts: D/G 1.20, G at 1582 cm^-1"},
              {"frame": 23, "what": "cosmic-ray spike"}, {"frame": 41, "what": "cosmic-ray spike"},
              {"frame": 60, "what": "anneal ends: D/G 0.45, G at 1594 cm^-1"}]
    N = 60
    SPIKES = (23, 41)

    def progress(self, frame: int) -> float:
        return float(np.clip((frame - 1) / (self.N - 1), 0.0, 1.0))

    def _acquire(self, params, rng) -> Frame:
        x = np.linspace(1000.0, 1900.0, 900)
        p = self.progress(self.frame)
        ratio = 1.20 - 0.75 * p
        heating = max(0.0, params["laser_power_mw"] - 8.0)
        g_pos = 1582.0 + 12.0 * p - 0.9 * heating
        dose = params["laser_power_mw"] * params["integration_s"]
        g_h = 55.0 * dose
        rate = (g_h * _lorentz(x, g_pos, 62.0) + ratio * g_h * _lorentz(x, 1350.0, 150.0)
                + (0.35 + 0.9 * p) * g_h * (0.55 + 0.45 * (x - 1000.0) / 900.0))   # fluorescence
        n = params["accumulations"]
        counts = rng.poisson(np.clip(rate, 0, None) * n).astype(float) / n
        if self.frame in self.SPIKES:
            counts[int(rng.integers(150, 750))] += 9.0 * g_h
        return Frame(x=x, y=counts, params=params, x_label="raman_shift_cm-1", y_label="counts",
                     meta={"anneal_progress": p},
                     truth={"g_position": float(g_pos), "d_over_g": float(ratio),
                            "cosmic_ray": self.frame in self.SPIKES})


# ──────────────────────────────────────────────────────────────
# AFM force spectroscopy
# ──────────────────────────────────────────────────────────────

class AFMForceCurve(_Simulator):
    """Approach force–distance curves along a line crossing two materials.

    Before contact the force is flat (with a small attractive snap-in dip);
    after the contact point it rises linearly with stiffness ``k``. Frames 1–25
    sit on a soft matrix (k ≈ 0.8 N/m, contact near 40 nm); from frame 26 the
    tip is on a stiff inclusion (k ≈ 3.2 N/m, surface 12 nm higher). Noise falls
    with slower ``approach_speed_nm_s``; ``trigger_force_nN`` sets how far into
    contact the curve goes.
    """

    name = "afm_force_curve"
    system_info = {"technique": "AFM force spectroscopy (approach curve)",
                   "sample": "stiff inclusions in a soft polymer matrix",
                   "x_axis": "z piezo displacement (nm)", "y_axis": "force (nN)"}
    schema = InstrumentSchema.from_dict({
        "trigger_force_nN": {"low": 2.0, "high": 60.0, "units": "nN",
                             "description": "force at which the approach stops"},
        "approach_speed_nm_s": {"low": 20.0, "high": 2000.0, "units": "nm/s",
                                "description": "z approach speed; slower is quieter"}})
    defaults = {"trigger_force_nN": 20.0, "approach_speed_nm_s": 400.0}
    outputs = {"contact_point_nm": "z displacement at which the tip touches the surface (force starts rising)",
               "stiffness_N_per_m": "slope of the force versus displacement in the contact region, in N/m (nN per nm)"}
    targets = ["contact point", "contact stiffness"]
    events = [{"frame": 1, "what": "soft matrix: k ≈ 0.8 N/m, contact ≈ 40 nm"},
              {"frame": 26, "what": "tip moves onto a stiff inclusion: k ≈ 3.2 N/m, contact ≈ 28 nm"}]

    def _acquire(self, params, rng) -> Frame:
        stiff = self.frame >= 26
        k = (3.2 if stiff else 0.8) * (1.0 + 0.03 * np.sin(self.frame / 3.0))
        z0 = (28.0 if stiff else 40.0) + 0.4 * np.sin(self.frame / 5.0)
        z_end = z0 + params["trigger_force_nN"] / k
        z = np.linspace(0.0, z_end, 500)
        f = np.where(z > z0, k * (z - z0), 0.0) - 0.6 * np.exp(-0.5 * ((z - (z0 - 0.8)) / 0.6) ** 2)
        noise = 0.05 + 0.25 * np.sqrt(params["approach_speed_nm_s"] / 400.0)
        f = f + rng.normal(0.0, noise, z.size) + 0.002 * z           # slight drift
        return Frame(x=z, y=f, params=params, x_label="z_nm", y_label="force_nN",
                     meta={"line_position": self.frame},
                     truth={"contact_point_nm": float(z0), "stiffness_N_per_m": float(k),
                            "material": "inclusion" if stiff else "matrix"})


# ──────────────────────────────────────────────────────────────
# STM point spectroscopy
# ──────────────────────────────────────────────────────────────

class STMdIdV(_Simulator):
    """dI/dV point spectra of a superconducting gap along a line (90 frames).

    A Dynes density of states with gap Δ falling smoothly from 1.40 to 1.05 meV
    along the line. The lock-in ``modulation_mv`` does two things at once: the
    noise falls as 1/modulation, and the spectrum is smeared by it — a large
    modulation gives a quiet, wrong gap. ``averages`` buys noise honestly. At
    frame 31 the tip changes: a strong asymmetric background appears.
    """

    name = "stm_didv"
    system_info = {"technique": "scanning tunnelling spectroscopy (dI/dV point spectra, lock-in)",
                   "sample": ("superconducting film, 1.5 K; point spectra taken one after another "
                              "along a line across the film, so the gap may change from frame to frame"),
                   "x_axis": "sample bias (mV)", "y_axis": "dI/dV (normalised to the normal state)",
                   # What an STM user would write down: the model the gap is read from.
                   "model": ("Dynes (lifetime-broadened BCS) density of states, "
                             "N(E) = |Re[(E - i*Gamma) / sqrt((E - i*Gamma)**2 - Delta**2)]| "
                             "(complex square root; symmetric in E), convolved with a Gaussian "
                             "for the lock-in modulation and temperature")}
    schema = InstrumentSchema.from_dict({
        "modulation_mv": {"low": 0.02, "high": 1.0, "units": "mV rms",
                          "description": "lock-in bias modulation; lowers noise but smears the spectrum"},
        "averages": {"kind": "int", "low": 1, "high": 64, "description": "sweeps averaged"}})
    defaults = {"modulation_mv": 0.10, "averages": 4}
    outputs = {"gap_mev": ("superconducting gap Delta in meV: the gap parameter of the fitted Dynes density of "
                           "states — NOT the coherence-peak position, which broadening pushes outward"),
               "coherence_peak_height": "height of the positive-bias coherence peak above the normal-state level of 1"}
    targets = ["superconducting gap", "coherence peak height"]
    events = [{"frame": 1, "what": "gap 1.40 meV, falling smoothly along the line"},
              {"frame": 31, "what": "tip change: a strong asymmetric background appears"},
              {"frame": 90, "what": "end of line: gap 1.05 meV"}]
    N = 90

    def gap(self, frame: int) -> float:
        return float(1.40 - 0.35 * np.clip((frame - 1) / (self.N - 1), 0, 1))

    def _acquire(self, params, rng) -> Frame:
        v = np.linspace(-5.0, 5.0, 501)
        delta, gamma = self.gap(self.frame), 0.06
        e = v + 1j * gamma
        dos = np.abs(np.real(e / np.sqrt(e ** 2 - delta ** 2 + 0j)))
        # Lock-in + thermal smearing as a Gaussian of width sqrt(mod^2 + kT^2).
        width = np.hypot(1.2 * params["modulation_mv"], 0.11)
        kern = np.exp(-0.5 * (np.arange(-150, 151) * (v[1] - v[0]) / width) ** 2)
        y = np.convolve(np.pad(dos, 150, mode="edge"), kern / kern.sum(), mode="valid")
        if self.frame >= 31:
            y = y * (1.0 + 0.09 * v) + 0.25 * _gauss(v, -2.6, 1.1)     # a changed tip
        noise = 0.012 / (params["modulation_mv"] * np.sqrt(params["averages"]))
        y = y + rng.normal(0.0, noise, v.size)
        return Frame(x=v, y=y, params=params, x_label="bias_mV", y_label="dIdV_norm",
                     meta={"line_position": self.frame},
                     truth={"gap_mev": delta, "tip": "changed" if self.frame >= 31 else "original"})


class SpectrumImageSeries(_Simulator):
    """Spectrum images (datacubes) of plasmonic particles during in-situ heating.

    Each frame is a 14 × 14 pixel spectrum image with 160 energy channels: a
    plasmon resonance on a decaying background whose energy varies smoothly
    across the field (a continuous field, as real maps are) and red-shifts as
    the sample heats, 1.5 meV per frame. At frame 13 a second, weaker mode
    near 0.95 eV appears over one corner of the field (the lower right). ``dwell_ms`` sets the
    noise (it falls as 1/sqrt(dwell)) and the dose; ``binning`` trades spatial
    detail for signal.
    """

    name = "spectrum_image_series"
    modality = "hyperspectral"
    system_info = {"technique": "STEM-EELS spectrum imaging (low-loss), in-situ heating",
                   "experiment_type": "electron energy loss spectroscopy",
                   "sample": ("plasmonic nanocrystal array on a heating chip; one spectrum image "
                              "per temperature step, so the resonance may move from frame to frame"),
                   "energy_range": {"start": 0.30, "end": 1.20, "units": "eV"},
                   "x_axis": "energy loss (eV)", "y_axis": "counts"}
    schema = InstrumentSchema.from_dict({
        "dwell_ms": {"low": 1.0, "high": 50.0, "units": "ms",
                     "description": "pixel dwell time; noise falls as 1/sqrt(dwell), dose rises with it"},
        "binning": {"kind": "int", "low": 1, "high": 2,
                    "description": "spatial binning; 2 halves the map size and the noise"}})
    defaults = {"dwell_ms": 10.0, "binning": 1}
    outputs = {"plasmon_energy": "energy of the main plasmon resonance maximum, per pixel (eV)"}
    targets = ["plasmon resonance energy map"]
    events = [{"frame": 1, "what": "plasmon near 0.62 eV, red-shifting 1.5 meV per frame"},
              {"frame": 13, "what": "a second mode near 0.95 eV appears over the lower right corner"},
              {"frame": 24, "what": "end of the ramp"}]
    N = 24

    def energy(self, frame: int) -> float:
        return float(0.62 - 0.0015 * (frame - 1))

    def _acquire(self, params, rng) -> Frame:
        n, side = 160, 14 // int(params["binning"])
        e = np.linspace(0.30, 1.20, n)
        u, v = np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")
        centre = self.energy(self.frame) + 0.020 * u + 0.012 * v * v        # a continuous field
        amp = 100.0 * (1.0 + 0.25 * np.cos(2.0 * u) * np.cos(1.5 * v))
        cube = amp[..., None] * np.exp(-0.5 * ((e - centre[..., None]) / 0.055) ** 2)
        cube = cube + 60.0 * np.exp(-(e - 0.30) / 0.22)                       # zero-loss tail
        if self.frame >= 13:
            corner = np.clip(0.5 * (u + v), 0.0, None) ** 1.5
            cube = cube + (45.0 * corner)[..., None] * np.exp(-0.5 * ((e - 0.95) / 0.04) ** 2)
        noise = 9.0 / np.sqrt(params["dwell_ms"] / 10.0) / int(params["binning"])
        cube = cube + rng.normal(0.0, noise, cube.shape)
        return Frame(x=e, y=cube.reshape(-1, n).mean(axis=0), cube=cube, params=params,
                     x_label="energy_eV", y_label="mean_counts",
                     meta={"temperature_step": self.frame},
                     truth={"plasmon_energy": float(centre.mean()),
                            "second_mode": bool(self.frame >= 13)})


SIMULATORS = {cls.name: cls for cls in (BeamlineXRD, InSituRaman, AFMForceCurve, STMdIdV,
                                        SpectrumImageSeries)}


def get_simulator(name: str, seed: int = 0) -> Instrument:
    try:
        return SIMULATORS[name](seed=seed)
    except KeyError:
        raise ValueError(f"unknown simulator {name!r}; available: {sorted(SIMULATORS)}") from None
