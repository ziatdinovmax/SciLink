#!/usr/bin/env python3
"""
scilink live-simulator — synthetic data emitter for live-mode demos and tests.

Pretends to be a diffractometer (or other 1-D spectroscopy instrument)
that's actively measuring. Writes data to disk every N seconds in one
of three layouts so you can exercise each LiveDataSource shape
without needing a real instrument:

  --mode rewrite     single CSV file, rewritten in full each tick
                     (pair with `scilink live ... --source-kind mtime_poll`)
  --mode append      single CSV file, intensity rows appended each tick
                     (pair with `scilink live ... --source-kind append_only`)
  --mode directory   one file per tick into a folder
                     (pair with `scilink live ... --source-kind directory_watch`)

Typical use: open two terminals.

  TERMINAL A:
    scilink live-simulator ~/sim_scan --mode directory --interval 2

  TERMINAL B:
    scilink live ~/sim_scan --skill xrd \\
        --source-kind directory_watch --pattern '*.csv'

Or with a single rewriting file:

  TERMINAL A:
    scilink live-simulator ~/sim_scan.csv --mode rewrite --duration 60

  TERMINAL B:
    scilink live ~/sim_scan.csv --skill xrd

Five Si Fd-3m Cu Kα Bragg peaks emerge over the duration. Defaults
match the original demo: 60-second total scan, Si (111) emerges at
~5 s, (220) at 15 s, (311) at 25 s, (400) at 35 s, (331) at 45 s.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# Reference Si Fd-3m, a = 5.43 Å, Cu Kα — five peaks in 20–80° 2θ.
# (hkl, 2θ position [deg], relative intensity, time-of-emergence [s])
SI_REFLECTIONS = (
    ("(111)", 28.44, 100.0,  5.0),
    ("(220)", 47.30,  60.0, 15.0),
    ("(311)", 56.12,  35.0, 25.0),
    ("(400)", 69.13,  20.0, 35.0),
    ("(331)", 76.38,  18.0, 45.0),
)


def lorentzian(grid: np.ndarray, x0: float, amp: float, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return amp * gamma ** 2 / ((grid - x0) ** 2 + gamma ** 2)


def _current_pattern(
    *, elapsed_s: float, grid: np.ndarray, noise_frac: float,
    rng: np.random.Generator, zero_shift: float, fwhm: float,
) -> tuple[np.ndarray, list[str]]:
    intensity = np.zeros_like(grid)
    visible: list[str] = []
    for hkl, pos, amp, t_emerge in SI_REFLECTIONS:
        if elapsed_s < t_emerge:
            continue
        ramp = min(1.0, (elapsed_s - t_emerge) / 6.0)
        intensity += lorentzian(grid, pos + zero_shift, amp * ramp, fwhm)
        if ramp > 0.05:
            visible.append(hkl)
    if noise_frac > 0:
        scale = max(noise_frac * intensity.max(), 0.5)
        intensity = intensity + rng.normal(scale=scale, size=intensity.shape)
    intensity = np.clip(intensity, 0.0, None)
    return intensity, visible


def _write_full_csv(path: Path, grid: np.ndarray, intensity: np.ndarray) -> None:
    """Write two-column CSV atomically (tmp + rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr = np.column_stack([grid, intensity])
    np.savetxt(tmp, arr, delimiter=",", header="two_theta,intensity", comments="")
    tmp.replace(path)


def _ensure_append_header(path: Path) -> None:
    """Create the file with header on the first append."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("two_theta,intensity\n")


def _append_pattern(path: Path, grid: np.ndarray, intensity: np.ndarray) -> None:
    """Append intensity rows to a single CSV; on first call writes the
    full pattern, on each subsequent call appends only the new column.

    NOTE: append_only as a data source returns the just-appended bytes.
    A user pairing this simulator's --mode append with --source-kind
    append_only on the consumer side will see the full pattern as the
    first chunk and then... an empty appended chunk on subsequent ticks
    (because the pattern hasn't grown — only intensity values change).
    The 'rewrite' mode is therefore the right pairing for most scans
    where intensity counts accumulate at the same 2θ positions. We
    still support 'append' for the niche case of XRD that writes one
    NEW 2θ value per step (vs. accumulating counts at existing positions).
    """
    _ensure_append_header(path)
    with path.open("a") as f:
        writer = csv.writer(f)
        for x, y in zip(grid, intensity):
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])


def _write_to_directory(directory: Path, step: int, grid: np.ndarray,
                         intensity: np.ndarray) -> Path:
    """Write the current pattern as scan_NNNN.csv in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"scan_{step:04d}.csv"
    _write_full_csv(path, grid, intensity)
    return path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scilink live-simulator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output",
                    help="File (--mode rewrite|append) or directory "
                         "(--mode directory) to write to.")
    p.add_argument("--mode", choices=("rewrite", "append", "directory"),
                    default="rewrite",
                    help="Output layout. See module docstring for which "
                         "live-mode source to pair each with.")
    p.add_argument("--duration", type=float, default=60.0,
                    help="Total simulation duration in seconds (default 60).")
    p.add_argument("--interval", type=float, default=2.0,
                    help="Seconds between emissions (default 2.0).")
    p.add_argument("--two-theta-min", type=float, default=20.0)
    p.add_argument("--two-theta-max", type=float, default=80.0)
    p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.05,
                    help="Gaussian noise as fraction of max intensity (default 0.05).")
    p.add_argument("--zero-shift", type=float, default=0.12,
                    help="Constant 2θ zero-shift in degrees (default 0.12 — a "
                         "realistic-looking sample displacement).")
    p.add_argument("--fwhm", type=float, default=0.18,
                    help="Peak FWHM in degrees (default 0.18).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose logging (DEBUG level).")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format="[sim %(asctime)s.%(msecs)03d %(levelname)5s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("scilink-live-sim")

    out_path = Path(args.output).expanduser().resolve()
    grid = np.arange(args.two_theta_min, args.two_theta_max, args.grid_step)
    rng = np.random.default_rng(args.seed)

    log.info("Black-box experiment simulator")
    log.info("  output:   %s", out_path)
    log.info("  mode:     %s", args.mode)
    log.info("  duration: %.1fs, emission interval: %.1fs",
             args.duration, args.interval)
    log.info("  grid:     %d points, %.2f–%.2f° 2θ at %.2f° step",
             grid.size, args.two_theta_min, args.two_theta_max, args.grid_step)
    log.info("  schedule: %s",
             ", ".join(f"{hkl}@{t:.0f}s" for hkl, _, _, t in SI_REFLECTIONS))

    # Prepare output target
    if args.mode == "directory":
        out_path.mkdir(parents=True, exist_ok=True)
    elif args.mode == "append":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Start fresh — remove any stale file
        if out_path.exists():
            out_path.unlink()
    else:  # rewrite
        out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    step = 0
    last_visible_count = -1
    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed > args.duration:
                break
            step += 1
            intensity, visible = _current_pattern(
                elapsed_s=elapsed, grid=grid, noise_frac=args.noise,
                rng=rng, zero_shift=args.zero_shift, fwhm=args.fwhm,
            )
            if args.mode == "directory":
                target = _write_to_directory(out_path, step, grid, intensity)
                tag = target.name
            elif args.mode == "append":
                _append_pattern(out_path, grid, intensity)
                tag = "appended"
            else:  # rewrite
                _write_full_csv(out_path, grid, intensity)
                tag = "rewritten"

            if len(visible) != last_visible_count:
                log.info("t=%5.1fs step=%03d  visible peaks: %s  (%s, Imax=%.1f)",
                         elapsed, step, visible or "(none)",
                         tag, float(intensity.max()))
                last_visible_count = len(visible)
            else:
                log.debug("t=%5.1fs step=%03d  Imax=%.1f", elapsed, step,
                          float(intensity.max()))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted at t=%.1fs (step %d).", time.monotonic() - t0, step)
        return 0
    log.info("Done. Simulated %.1fs, %d emissions.",
             time.monotonic() - t0, step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
