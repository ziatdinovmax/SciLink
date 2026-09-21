"""A graded, model-free change signal for a stream of 1D measurements.

The first drift signal compared a peak-counting fingerprint of each frame with
ONE anchor frame against a fixed bar. Independent review on real spectra showed
what that does: on a rich multi-peak pattern it saturates (every frame past the
bar, no magnitude), on a noisy one it wobbles below any usable bar, and because
its calibration came from the locked recipe's residuals the SAME series flagged
most frames under one recipe and none under another.

This one reads the data only — never the recipe — and answers a different
question: *how much of this frame can the frames seen so far not describe?*

- Frames are put on a common grid and the accepted ones span a low-rank subspace
  (SVD, plus a constant and a slope). Amplitude changes, a background that scales
  differently from the peaks, and small shifts all live inside that subspace once
  a few frames have shown them; a new feature does not.
- ``fraction`` is the share of the frame's structure left outside the subspace,
  with the noise taken out: 0 = nothing new, 1 = nothing in common. It is a
  magnitude, comparable between frames and between runs.
- ``score`` is the same residual in units of what is normal for THIS stream (the
  residual-to-noise ratio of accepted frames), so correlated detector noise does
  not read as change.
- A frame is ``suspected`` when it is both material (``fraction``) and unusual
  (``score``) — on its own, or together with the frames just before it: a weak
  change that persists is averaged over the last few frames, where noise cancels
  and the change does not (a tip change that is 15 % of the structure and only
  twice the noise per frame is unmistakable over three). A single-frame glitch
  (a cosmic ray) is judged alone and kept out of that average.
- Only accepted frames teach, in a rolling window, so a slow drift is followed
  and an abrupt or sustained change is not absorbed. Suspected frames are HELD:
  when the caller decides the new state is real (the fit is good and an audit
  agrees, or a new recipe was adopted) ``adopt()`` makes them the new normal.
- A state that was accepted once is remembered. The rolling window forgets, and
  a line scan crossing the same kind of region again would otherwise be asked
  about again: a frame that an earlier accepted state describes is not new.
- A frame measured over a different window (a force curve that ends earlier on a
  stiffer sample) is compared on the part the windows share.
- ``from_reference`` is the same measure against the reference frames alone: how
  far the stream has moved since the recipe was locked. Information, not a
  verdict — the fit gate says whether the recipe still fits.

Deterministic, milliseconds per frame. The same monitor can watch the mean
spectrum of a datacube.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_GRID = 256
FRACTION_BAR = 0.10
SCORE_BAR = 3.0


def _point_noise(y: np.ndarray) -> float:
    d = np.diff(y, n=2)
    if d.size < 3:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6.0))


class DriftMonitor:
    """See the module docstring.

    Args:
        fraction_bar: a frame is material when more than this share of its
            structure is new. LOWER to catch weaker new features, RAISE if
            ordinary variation between frames is flagged.
        score_bar: and unusual when its residual is this many times what is
            normal for the stream. LOWER for more sensitivity on noisy data,
            RAISE if a stream with occasional rough frames is flagged.
        warmup: accepted frames learned before anything is judged. One frame
            gives no idea of what varies.
        window: accepted frames kept. Older ones are forgotten, which is what
            lets a slow drift be followed.
        max_rank: size of the learned subspace. RAISE for streams whose normal
            variation has many independent parts (several peaks moving
            separately).
    """

    def __init__(self, *, fraction_bar: float = FRACTION_BAR, score_bar: float = SCORE_BAR,
                 warmup: int = 4, window: int = 40, max_rank: int = 4, n_grid: int = N_GRID) -> None:
        self.fraction_bar, self.score_bar = float(fraction_bar), float(score_bar)
        self.warmup, self.window = int(warmup), int(window)
        self.max_rank, self.n_grid = int(max_rank), int(n_grid)
        self._x: Optional[np.ndarray] = None          # the first curve's x: every frame goes onto it
        self._rows: List[np.ndarray] = []              # accepted frames on the grid (rolling)
        self._seed: List[np.ndarray] = []              # the reference frames (kept)
        self._ratios: List[float] = []                 # residual/noise of accepted frames, at judge time
        self._basis: Optional[np.ndarray] = None
        self._seed_basis: Optional[np.ndarray] = None
        self._states: List[np.ndarray] = []            # bases of states already accepted (kept)
        self.max_states = 8
        self._recent: List[np.ndarray] = []            # residuals of the last unsuspected frames
        self._held: List[np.ndarray] = []              # suspected frames, waiting for adopt()
        self._held_resid: List[np.ndarray] = []        # ...and the part of each nothing explained
        self._held_partial: List[Tuple[np.ndarray, np.ndarray]] = []   # suspected frames on a shorter window
        self.coherent_frames = (3, 6)                  # runs over which a weak change is averaged

    # ------------------------------------------------------------------ grid
    def _to_grid(self, x: Any, y: Any, partial: bool = False):
        """The curve on the monitor's grid. With ``partial`` also the mask of
        bins the curve actually covers (``None`` when it covers too little)."""
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size < 16:
            return (None, None) if partial else None
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        if self._x is None:
            self._x = x
        ref = self._x
        block = max(1, ref.size // self.n_grid)
        n = (ref.size // block) * block
        covered = np.ones(n // block, dtype=bool)
        if x.size != ref.size or not np.allclose(x, ref, rtol=0, atol=1e-9 * max(1.0, float(np.ptp(ref)))):
            lo, hi = max(x[0], ref[0]), min(x[-1], ref[-1])
            share = (hi - lo) / float(ref[-1] - ref[0]) if ref[-1] > ref[0] else 0.0
            if share < (0.4 if partial else 0.9):
                return (None, None) if partial else None
            y = np.interp(ref, x, y)
            inside = (ref >= lo) & (ref <= hi)
            covered = inside[:n].reshape(-1, block).all(axis=1)
        g = y[:n].reshape(-1, block).mean(axis=1)
        return (g, covered) if partial else (g if covered.all() else None)

    # ----------------------------------------------------------------- basis
    def _make_basis(self, rows: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if not rows:
            return None
        n = rows[0].size
        t = np.linspace(-1.0, 1.0, n)
        fixed = np.vstack([np.ones(n), t])
        q_fixed, _ = np.linalg.qr(fixed.T)
        mat = np.vstack([r / (np.linalg.norm(r) or 1.0) for r in rows])
        mat = mat - (mat @ q_fixed) @ q_fixed.T       # what the baseline terms do not explain
        _, s, vt = np.linalg.svd(mat, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            return q_fixed
        # Keep a component while it carries more than the rows' own noise does.
        noise = np.median([_point_noise(r) / (np.linalg.norm(r) or 1.0) for r in rows]) * np.sqrt(n)
        keep = [i for i in range(min(self.max_rank, s.size)) if s[i] > max(2.0 * noise, 1e-9 * s[0])]
        keep = keep or [0]
        basis, _ = np.linalg.qr(np.vstack([q_fixed.T, vt[keep]]).T)
        return basis

    @staticmethod
    def _structure(g: np.ndarray, sigma: float) -> float:
        """Energy of the frame's structure: what a constant and a slope leave,
        minus what noise alone would leave."""
        n = g.size
        line, _ = np.linalg.qr(np.vstack([np.ones(n), np.linspace(-1.0, 1.0, n)]).T)
        rest = g - line @ (line.T @ g)
        return max(float(rest @ rest) - sigma ** 2 * max(n - 2, 1), 1e-300)

    def _measure(self, g: np.ndarray, basis: np.ndarray, covered: Optional[np.ndarray] = None):
        """(fraction of structure outside the basis, residual-to-noise ratio,
        residual vector or None when the frame covers only part of the grid)."""
        full = covered is None or bool(covered.all())
        if not full:
            g = g[covered]
            basis, _ = np.linalg.qr(basis[covered])
        n, k = g.size, basis.shape[1]
        resid = g - basis @ (basis.T @ g)
        sigma = _point_noise(g)
        e_res, e_noise = float(resid @ resid), sigma ** 2 * max(n - k, 1)
        fraction = float(np.sqrt(max(e_res - e_noise, 0.0) / self._structure(g, sigma)))
        ratio = e_res / e_noise if e_noise > 0 else float("inf")
        return min(fraction, 1.0), ratio, (resid if full else None)

    # ------------------------------------------------------------------- API
    def seed(self, curves: Sequence[Tuple[Any, Any]], keep: bool = False) -> int:
        """Start from reference frames (setup, or a re-anchor's window). They are
        learned, never judged. With ``keep`` (a new recipe for the SAME stream)
        what the monitor already knows stays: the frames held while the recipe
        was being rebuilt and the states accepted before — a new recipe changes
        how frames are fitted, not what the stream has looked like. Returns how
        many reference frames were usable."""
        held, states, x = (self._held, self._states, self._x) if keep else ([], [], None)
        if keep and self._basis is not None and self._rows:
            states = (states + [self._basis])[-self.max_states:]
        self._x, self._rows, self._ratios = x, [], []
        self._seed, self._recent, self._held, self._states = [], [], [], states
        self._held_resid, self._held_partial = [], []
        for cx, cy in curves:
            g = self._to_grid(cx, cy)
            if g is not None:
                self._seed.append(g)
        self._rows = list(self._seed)
        self._basis = self._make_basis(self._rows)
        self._seed_basis = self._make_basis(self._seed)
        if held:
            self._extend([h for h in held if self._rows and h.size == self._rows[0].size])
        return len(self._seed)

    def judge(self, x: Any, y: Any) -> Dict[str, Any]:
        """The change signal for one frame. Does not learn."""
        if not self._rows:                           # nothing seeded: the first frame is the reference
            self.seed([(x, y)])
            return {"available": True, "fraction": 0.0, "score": 1.0, "suspected": False,
                    "learning": True, "from_reference": 0.0}
        g, covered = self._to_grid(x, y, partial=True)
        if g is None or self._basis is None:
            return {"available": False}
        fraction, ratio, resid = self._measure(g, self._basis, covered)
        known = None
        for i, state in enumerate(self._states):     # a state accepted before is not new
            f_s, r_s, _ = self._measure(g, state, covered)
            if f_s < fraction:
                fraction, ratio, resid, known = f_s, r_s, None, i
        typical = max(float(np.median(self._ratios)) if len(self._ratios) >= 3 else 1.0, 1.0)
        score = ratio / typical
        learning = len(self._rows) < self.warmup
        alone = bool(fraction > self.fraction_bar and score > self.score_bar)
        together = False
        if resid is not None and not alone:
            # The same change, frame after frame: noise cancels in the mean, the
            # change does not. A short run catches it soon, a longer one catches
            # a weaker change later.
            sigma = _point_noise(g)
            for m in self.coherent_frames:
                if len(self._recent) < m - 1:
                    break
                mean = np.mean(self._recent[-(m - 1):] + [resid], axis=0)
                e_noise = sigma ** 2 * max(g.size - self._basis.shape[1], 1) / m
                e_mean = float(mean @ mean)
                frac_m = float(np.sqrt(max(e_mean - e_noise, 0.0) / self._structure(g, sigma)))
                score_m = (e_mean / e_noise if e_noise > 0 else float("inf")) / typical
                if frac_m > self.fraction_bar and score_m > self.score_bar:
                    together = True
                    fraction, score = max(fraction, frac_m), max(score, score_m)
                    break
        out = {"available": True, "fraction": round(min(fraction, 1.0), 4),
               "score": round(float(score), 3),
               "suspected": bool(not learning and (alone or together)),
               "learning": learning, "_ratio": ratio, "_grid": g if covered.all() else None,
               "_resid": resid}
        if known is not None:
            out["known_state"] = known
        if not covered.all():
            out["window_share"] = round(float(covered.mean()), 3)
        if self._seed_basis is not None:
            out["from_reference"] = round(self._measure(g, self._seed_basis, covered)[0], 4)
        # A frame judged alone as a glitch stays out of the running mean; a
        # suspected frame is held until the caller says the new state is real.
        if resid is not None and not alone:
            self._recent = (self._recent + [resid])[-max(self.coherent_frames):]
        if out["suspected"] and out["_grid"] is not None:
            self._held = (self._held + [out["_grid"]])[-self.window:]
            if resid is not None:
                self._held_resid = (self._held_resid + [resid])[-self.window:]
        elif out["suspected"]:                       # a changed frame on a shorter window is held too
            self._held_partial = (self._held_partial + [(g, covered)])[-self.window:]
        return out

    def learn(self, x: Any, y: Any, verdict: Optional[Dict[str, Any]] = None) -> None:
        """Accept a frame (the loop calls this for clean frames only)."""
        g = (verdict or {}).get("_grid")
        if g is None:
            g = self._to_grid(x, y)
        if g is None:
            return
        ratio = (verdict or {}).get("_ratio")
        if ratio is not None and np.isfinite(ratio):
            self._ratios = (self._ratios + [float(ratio)])[-self.window:]
        self._held, self._held_resid, self._held_partial = [], [], []   # an accepted frame ends a run of suspected ones
        self._extend([g])

    def _extend(self, grids: Sequence[np.ndarray]) -> None:
        keep = max(self.window - len(self._seed), 1)
        recent = [r for r in self._rows[len(self._seed):]] + list(grids)
        self._rows = list(self._seed) + recent[-keep:]
        self._basis = self._make_basis(self._rows)

    @property
    def n_held(self) -> int:
        return len(self._held) + len(self._held_partial)

    def _held_common(self):
        """The held frames where ALL of them were measured: (grids cut to the
        common bins, the mask of those bins), or ``(None, None)``."""
        entries = [(g, np.ones(g.size, dtype=bool)) for g in self._held] + list(self._held_partial)
        if not entries or len({g.size for g, _ in entries}) != 1:
            return None, None
        common = np.logical_and.reduce([c for _, c in entries])
        if common.sum() < 16:
            return None, None
        return [g[common] for g, _ in entries], common

    def held_agree(self) -> bool:
        """Do the held frames look like each other (a new stable state) rather
        than like a stream that is still changing? Frames on a shorter window
        are compared where they overlap."""
        grids, _ = self._held_common()
        if grids is None:
            return False
        if len(grids) < 2:
            return True
        basis = self._make_basis(grids[:-1])
        return self._measure(grids[-1], basis)[0] <= self.fraction_bar

    def _bin_x(self) -> Optional[np.ndarray]:
        if self._x is None:
            return None
        block = max(1, self._x.size // self.n_grid)
        n = (self._x.size // block) * block
        return self._x[:n].reshape(-1, block).mean(axis=1)

    def locate(self, max_regions: int = 3) -> List[Dict[str, Any]]:
        """WHERE the held frames are new, from the part of them nothing seen so
        far explains — no model, no fit. The held residuals are averaged (noise
        cancels, the new structure does not) and contiguous stretches that stand
        clear of the noise are reported, strongest first:

        - ``kind``: ``"new"`` (intensity that was not there), ``"missing"``
          (intensity that is gone), ``"shifted"`` (a gain right next to a loss of
          similar size — a feature that moved), or ``"broad"`` (most of the
          window: a background or overall shape change);
        - ``x_from`` / ``x_to`` / ``x_peak`` in the data's own x units;
        - ``share``: this stretch's share of everything unexplained.

        Frames that cover a shorter window than the stream did are located
        where they overlap it, and the part of the axis that is no longer
        measured is reported after the rest as ``kind="window"`` (``share``: its
        share of the axis) — a scan that ends early is itself something that
        changed, and often the consequence of what is new inside it.
        Empty when nothing is held."""
        xs_all = self._bin_x()
        grids, common = self._held_common()
        if grids is None or xs_all is None or self._basis is None or xs_all.size != common.size:
            return []
        lost = []
        if not common.all():
            idx = np.flatnonzero(common)
            for a, b in ((0, idx[0] - 1), (idx[-1] + 1, common.size - 1)):
                if b >= a:
                    lost.append({"kind": "window", "x_from": round(float(xs_all[a]), 6),
                                 "x_to": round(float(xs_all[b]), 6),
                                 "x_peak": round(float(xs_all[a if a else b]), 6),
                                 "share": round(float(b - a + 1) / common.size, 6)})
        return self._locate_in(grids, xs_all[common], self._basis[common], max_regions) + lost

    def _locate_in(self, grids: Sequence[np.ndarray], xs: np.ndarray, basis: np.ndarray,
                   max_regions: int) -> List[Dict[str, Any]]:
        if basis.shape[0] != self._basis.shape[0]:
            basis, _ = np.linalg.qr(basis)
        # A least-squares projection smears a change over everything it touches
        # (take one of two peaks away and the other appears to grow). What is new
        # is usually confined to part of the axis, so the fit is made robust: the
        # bulk of the frame decides how the known shapes are scaled, and what is
        # new stands out where it is.
        mean = np.mean([self._robust_residual(g, basis) for g in grids], axis=0)
        k = max(3, xs.size // 64)
        smooth = np.convolve(mean, np.ones(k) / k, mode="same")
        sigma = (_point_noise(mean) / np.sqrt(k)) or float(np.std(smooth)) or 1.0
        hot = np.abs(smooth) > 4.0 * sigma
        total = max(float(np.sum(smooth[hot] ** 2)), 1e-300)
        regions, i = [], 0
        while i < hot.size:
            if not hot[i]:
                i += 1
                continue
            j = i
            while j + 1 < hot.size and hot[j + 1] and np.sign(smooth[j + 1]) == np.sign(smooth[i]):
                j += 1
            seg = slice(i, j + 1)
            peak = i + int(np.argmax(np.abs(smooth[seg])))
            regions.append({"i": i, "j": j, "sign": float(np.sign(smooth[peak])),
                            "x_from": float(xs[i]), "x_to": float(xs[j]), "x_peak": float(xs[peak]),
                            "share": float(np.sum(smooth[seg] ** 2)) / total})
            i = j + 1
        if not regions:
            return []
        if float(hot.mean()) > 0.5:                  # most of the window: shape or background
            return [{"kind": "broad", "x_from": round(float(xs[0]), 6), "x_to": round(float(xs[-1]), 6),
                     "x_peak": round(float(xs[int(np.argmax(np.abs(smooth)))]), 6), "share": 1.0}]
        # A gain next to a loss of similar size is one feature that moved.
        merged, used = [], set()
        for a in range(len(regions)):
            if a in used:
                continue
            r = regions[a]
            b = regions[a + 1] if a + 1 < len(regions) else None
            if b is not None and b["sign"] != r["sign"] \
                    and b["i"] - r["j"] <= k + 0.5 * min(r["j"] - r["i"], b["j"] - b["i"]) \
                    and 0.25 < r["share"] / max(b["share"], 1e-12) < 4.0:
                used.add(a + 1)
                merged.append({"kind": "shifted", "x_from": r["x_from"], "x_to": b["x_to"],
                               "x_peak": (r["x_peak"] + b["x_peak"]) / 2.0,
                               "share": r["share"] + b["share"]})
            else:
                merged.append({"kind": "new" if r["sign"] > 0 else "missing", "x_from": r["x_from"],
                               "x_to": r["x_to"], "x_peak": r["x_peak"], "share": r["share"]})
        merged.sort(key=lambda r: -r["share"])
        return [{k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in r.items()}
                for r in merged[:max_regions] if r["share"] > 0.1]

    @staticmethod
    def _robust_residual(g: np.ndarray, basis: np.ndarray, iterations: int = 8) -> np.ndarray:
        """``g`` minus the known shapes, scaled by the bulk of the frame (Huber
        weights) rather than by all of it."""
        w = np.ones(g.size)
        resid = g - basis @ (basis.T @ g)
        for _ in range(iterations):
            scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1e-300
            w = 1.0 / np.maximum(1.0, np.abs(resid) / (1.345 * scale))
            coef, *_ = np.linalg.lstsq(basis * w[:, None], g * w, rcond=None)
            resid = g - basis @ coef
        return resid

    def adopt(self) -> int:
        """The held frames are the new normal. Returns how many were adopted."""
        n = len(self._held)
        if self._held_partial:                       # a different window: the caller restarts on it
            n = 0
        elif n:
            if self._basis is not None:              # remember the state being left
                self._states = (self._states + [self._basis])[-self.max_states:]
            self._extend(self._held)
        self._held, self._held_resid, self._held_partial, self._recent = [], [], [], []
        return n

    @property
    def n_learned(self) -> int:
        return len(self._rows)

    # ----------------------------------------------------------- persistence
    def to_state(self) -> Dict[str, Any]:
        return {"x": None if self._x is None else [float(v) for v in self._x],
                "seed": [[float(v) for v in r] for r in self._seed],
                "rows": [[float(v) for v in r] for r in self._rows[len(self._seed):]],
                "ratios": [float(v) for v in self._ratios],
                "states": [[[float(v) for v in col] for col in b.T] for b in self._states]}

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state or state.get("x") is None:
            return
        self._x = np.asarray(state["x"], dtype=float)
        self._seed = [np.asarray(r, dtype=float) for r in state.get("seed") or []]
        self._rows = self._seed + [np.asarray(r, dtype=float) for r in state.get("rows") or []]
        self._ratios = [float(v) for v in state.get("ratios") or []]
        self._states = [np.asarray(b, dtype=float).T for b in state.get("states") or []]
        self._basis = self._make_basis(self._rows)
        self._seed_basis = self._make_basis(self._seed)


class DriftBank:
    """Several curves per frame, one :class:`DriftMonitor` each.

    A spectrum is one curve. A datacube is more than its mean spectrum: a change
    confined to part of the field is diluted in the mean by the area it covers
    (observed on a simulated spectrum-image series: a new mode over one corner
    was 3 % of the mean spectrum and went unseen). So a frame may be given as
    named curves — the whole field and its regions — and is as changed as its
    MOST changed curve. The verdict and ``locate()`` say which region that was.

    The interface is the monitor's, with ``signals`` (``{name: (x, y)}``) where
    the monitor takes ``x, y``. A single unnamed curve behaves exactly like one
    monitor, and a single monitor's saved state loads as the curve ``"signal"``.
    """

    SINGLE = "signal"

    def __init__(self, **monitor_kwargs: Any) -> None:
        self._kw = monitor_kwargs
        self.monitors: Dict[str, DriftMonitor] = {}
        self._leading: Optional[str] = None       # the region the current run of changed frames is in

    @classmethod
    def as_signals(cls, curve: Any) -> Dict[str, Tuple[Any, Any]]:
        if isinstance(curve, dict):
            return curve
        return {cls.SINGLE: (curve[0], curve[1])}

    def _monitor(self, name: str) -> DriftMonitor:
        if name not in self.monitors:
            self.monitors[name] = DriftMonitor(**self._kw)
        return self.monitors[name]

    def seed(self, frames: Sequence[Any], keep: bool = False) -> int:
        frames = [self.as_signals(f) for f in frames]
        names = list(dict.fromkeys(n for f in frames for n in f))
        if not keep:
            self.monitors = {}
        self._leading = None
        return max([self._monitor(n).seed([f[n] for f in frames if n in f], keep=keep)
                    for n in names] or [0])

    def judge(self, signals: Any) -> Dict[str, Any]:
        signals = self.as_signals(signals)
        verdicts = {n: self._monitor(n).judge(x, y) for n, (x, y) in signals.items()}
        usable = {n: v for n, v in verdicts.items() if v.get("available")}
        if not usable:
            return {"available": False}
        # The whole field speaks first when it sees the change; a region is
        # named only when the change is confined to it.
        order = sorted(usable, key=lambda n: (not usable[n]["suspected"], n != next(iter(signals)),
                                              -usable[n]["fraction"]))
        worst = order[0]
        out = dict(usable[worst])
        out["_all"] = verdicts
        if len(signals) > 1:
            out["region"] = worst
            if out["suspected"]:
                self._leading = worst
        return out

    def learn(self, signals: Any, verdict: Optional[Dict[str, Any]] = None) -> None:
        signals = self.as_signals(signals)
        per = (verdict or {}).get("_all") or {}
        for n, (x, y) in signals.items():
            self._monitor(n).learn(x, y, per.get(n))
        self._leading = None

    def _held(self) -> List[DriftMonitor]:
        return [m for m in self.monitors.values() if m.n_held]

    @property
    def n_held(self) -> int:
        return max([m.n_held for m in self.monitors.values()] or [0])

    def held_agree(self) -> bool:
        held = self._held()
        return bool(held) and all(m.held_agree() for m in held)

    def locate(self, max_regions: int = 3) -> List[Dict[str, Any]]:
        name = self._leading if self._leading in self.monitors else None
        if name is None:
            held = sorted(self.monitors, key=lambda n: -self.monitors[n].n_held)
            name = held[0] if held else None
        if name is None:
            return []
        where = self.monitors[name].locate(max_regions)
        if len(self.monitors) > 1:
            where = [{**w, "region": name} for w in where]
        return where

    def adopt(self) -> int:
        """The held frames are the new normal, in every region. ``0`` when a
        region held frames it could not adopt (a different window): the caller
        restarts the bank on the new window, as with one monitor."""
        held = [bool(m.n_held) for m in self.monitors.values()]
        n = [m.adopt() for m in self.monitors.values()]
        self._leading = None
        if any(h and v == 0 for h, v in zip(held, n)):
            return 0
        return max(n or [0])

    @property
    def n_learned(self) -> int:
        return max([m.n_learned for m in self.monitors.values()] or [0])

    def to_state(self) -> Dict[str, Any]:
        return {"monitors": {n: m.to_state() for n, m in self.monitors.items()}}

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if "monitors" not in state:                  # a single monitor's state, from before
            state = {"monitors": {self.SINGLE: state}}
        for n, st in (state.get("monitors") or {}).items():
            self._monitor(n).load_state(st)
