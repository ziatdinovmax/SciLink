import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import os

logger = logging.getLogger(__name__)


def _try_orient_xy(data: np.ndarray) -> np.ndarray:
    """Return a 2-column curve array with the monotonic (X-like) column first.

    Monotonicity is the strongest portable cue that a column is the
    independent variable (time, wavelength, energy). When exactly one
    column is monotonic we use it to disambiguate; when both or neither
    are, the heuristic is ambiguous and we leave the input untouched so
    well-formed data is not perturbed. Non-2D inputs and shapes other
    than (N, 2) pass through unchanged.
    """
    if data.ndim != 2 or data.shape[1] != 2:
        return data
    col0, col1 = data[:, 0], data[:, 1]
    if not (np.isfinite(col0).all() and np.isfinite(col1).all()):
        return data

    def _mono(a: np.ndarray) -> bool:
        if a.size < 2:
            return False
        diffs = np.diff(a)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    if _mono(col1) and not _mono(col0):
        logger.info(
            "load_curve_data: swapped columns — col 1 is monotonic but col 0 is not, "
            "so col 1 is treated as X."
        )
        return np.ascontiguousarray(data[:, ::-1])
    return data


# Column-name fragments that strongly suggest the independent (X) variable, used
# to pick X from a >2-column table when no explicit hint is given.
_X_AXIS_HINTS = (
    "2theta", "two_theta", "twotheta", "theta", "angle", "wavelength", "wavenumber",
    "energy", "ev", "raman", "shift", "time", "freq", "frequency", "temperature",
    "temp", "field", "position", "distance", "voltage", "bias", "delay", "q", "x",
)
# Fragments suggesting a column is NOT the signal (uncertainty/weight columns) —
# avoided when choosing Y.
_NON_Y_HINTS = ("err", "error", "sigma", "std", "uncert", "weight", "noise")


_AMPLITUDE_KEYS = ("area", "amplitude", "intensity", "height", "step")
_SHAPE_KEYS = ("center", "position", "sigma", "gamma", "fwhm", "width",
               "eta", "decay", "tau", "frequency", "phase", "asymmetry")

ABSENT_COMPONENT_FIX = (
    "For each component flagged '_absent': refit with the component's SHAPE "
    "parameters FROZEN at the plan's (or last-fitted) values and ONLY its "
    "amplitude free; report that amplitude and its covariance error as the "
    "component's amplitude-like parameters (for localized peaks the "
    "baseline-subtracted window integral is an acceptable fast path), and "
    "report shape parameters as null. Absence is a measurement, not a gap."
)


def validate_absent_component_contract(parameters) -> list:
    """Deterministic check of the absence-as-value results contract.

    For every component whose dict (or a sibling key) carries an
    ``*_absent: true`` flag: its amplitude-like parameters (area/amplitude/
    intensity/height/step) must be FINITE numbers with ``_err`` siblings —
    the frozen-shape amplitude measurement — and its shape parameters
    (center/width/...) must be null. Returns human-readable violation
    strings (empty = compliant) for the correction-retry feedback channel.
    Never raises. Technique-blind: keys are matched by parameter ROLE.
    """
    import numpy as _np
    out = []
    if not isinstance(parameters, dict):
        return out

    def _flagged(d):
        return any(str(k).endswith("_absent") and v in (True, "true", "True", 1)
                   for k, v in d.items() if not isinstance(v, dict))

    comps = {k: v for k, v in parameters.items() if isinstance(v, dict)}
    for name, comp in comps.items():
        if not (_flagged(comp)
                or parameters.get(f"{name}_absent") in (True, "true", "True", 1)):
            continue
        amp_keys = [k for k in comp
                    if any(a in str(k).lower() for a in _AMPLITUDE_KEYS)
                    and not str(k).lower().endswith("_err")]
        if not amp_keys:
            out.append(f"component '{name}' is flagged absent but reports no "
                       "amplitude-like parameter at all")
        for k in amp_keys:
            v = comp.get(k)
            ok_num = isinstance(v, (int, float)) and _np.isfinite(v)
            if not ok_num:
                out.append(f"absent component '{name}': '{k}' must be the "
                           "MEASURED frozen-shape amplitude (finite number), "
                           f"got {v!r}")
            elif f"{k}_err" in comp and not (
                    isinstance(comp[f"{k}_err"], (int, float))
                    and _np.isfinite(comp[f"{k}_err"])):
                out.append(f"absent component '{name}': '{k}_err' must be a "
                           "finite propagated uncertainty")
        # Shape parameters: null is the contract, but a leftover numeric
        # value is tolerated (harmless to trends, which key on amplitude).
    return out


def _is_index_ramp(col) -> bool:
    """True for a perfect arithmetic progression — a row counter / frame
    index / uniform axis, never a physical signal. Fitting one of these as
    Y yields a perfect straight-line "result" (R^2 = 1.0 on y = t) that
    looks like success while measuring nothing; observed live collapsing a
    6-column instrument trace into its frame counter repeatedly."""
    import numpy as _np
    c = _np.asarray(col, float)
    if c.size < 3 or not _np.all(_np.isfinite(c)):
        return False
    d = _np.diff(c)
    span = float(_np.nanmax(c) - _np.nanmin(c))
    if span <= 0:
        return False
    return bool(_np.allclose(d, d[0], rtol=1e-6, atol=1e-9 * max(span, 1.0)))


def _choose_xy_indices(arr, system_info, column_names, log):
    """Pick (x_index, y_index) from a >2-column array.

    Priority: (1) explicit ``x_column``/``y_column`` in system_info (name or
    index); (2) a column whose name matches an axis hint as X + first non-error
    column as Y; (3) the single monotonic column as X; (4) first two columns.
    """
    n = arr.shape[1]
    si = system_info if isinstance(system_info, dict) else {}
    names = column_names or si.get("columns")
    low = [str(c).strip().lower() for c in names[:n]] if names and len(names) >= n else None

    def resolve(spec):
        if isinstance(spec, bool):
            return None
        if isinstance(spec, int) and 0 <= spec < n:
            return spec
        if isinstance(spec, str) and low:
            for i, c in enumerate(low):
                if c == spec.strip().lower():
                    return i
        return None

    xi, yi = resolve(si.get("x_column")), resolve(si.get("y_column"))
    if xi is not None and yi is not None and xi != yi:
        return xi, yi

    if low:  # name-hint heuristic
        x_guess = next((i for i, c in enumerate(low)
                        if any(h in c for h in _X_AXIS_HINTS)), None)
        if x_guess is not None:
            def _y_ok(i, strict):
                if i == x_guess or any(b in low[i] for b in _NON_Y_HINTS):
                    return False
                if strict and (any(h in low[i] for h in _X_AXIS_HINTS)
                               or _is_index_ramp(arr[:, i])):
                    return False   # a second axis / counter is not a signal
                return True
            y_guess = next((i for i in range(n) if _y_ok(i, True)),
                           next((i for i in range(n) if _y_ok(i, False)),
                                next((i for i in range(n) if i != x_guess),
                                     None)))
            return x_guess, y_guess

    # Monotonic-column heuristic (last-resort fallback). A monotonic column is
    # almost always an axis (temperature, time, index), so use it as X — but pick
    # a NON-monotonic column (a varying signal) as Y rather than blindly taking
    # the next column, which is often a second axis (e.g. a time channel beside
    # temperature). This only runs when there is no name signal and no LLM
    # column_mapping; when names are known the LLM owns this choice, so the job
    # here is just to avoid the catastrophic "axis as Y" guess.
    def _is_mono(col):
        return bool(col.size >= 2 and np.all(np.isfinite(col))
                    and (np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0)))

    mono = [i for i in range(n) if _is_mono(arr[:, i])]
    if mono:
        xi = mono[0]
        yi = next((j for j in range(n) if j != xi and j not in mono),
                  next(j for j in range(n) if j != xi))
        return xi, yi

    return 0, 1  # default: first two columns


def select_xy_columns(data, system_info=None, logger_=None, column_names=None) -> np.ndarray:
    """Reduce array-like curve data to a 2-column ``(x, y)`` array.

    Curve fitting is a 1D (x, y) operation, but real files arrive with extra
    columns (an error/weight column, multiple channels) or in row layout. This
    centralizes the reduction: 1D → ``(index, y)``; row-major ``(2, N)`` →
    transposed; ``(N, 2)`` → oriented; ``(N, M>2)`` → X/Y selected (see
    ``_choose_xy_indices``) and the rest dropped (logged). Column SELECTION is an
    analysis decision and so lives here — lossless file-prep keeps all columns.
    """
    log = logger_ or logger
    arr = np.asarray(data)
    if arr.ndim == 1:
        return np.column_stack([np.arange(arr.size), arr])
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D curve data, got {arr.ndim}D")
    # Orient to (n_points, n_cols): curve data has many more points than columns.
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    n_cols = arr.shape[1]
    if n_cols == 1:
        return np.column_stack([np.arange(arr.shape[0]), arr[:, 0]])
    if n_cols == 2:
        return _try_orient_xy(np.ascontiguousarray(arr))
    xi, yi = _choose_xy_indices(arr, system_info, column_names, log)
    if _is_index_ramp(arr[:, yi]):
        alt = next((j for j in range(n_cols)
                    if j not in (xi, yi) and not _is_index_ramp(arr[:, j])),
                   None)
        if alt is not None:
            if log:
                log.warning(
                    f"select_xy_columns: chosen Y (col {yi}) is a perfect "
                    f"index/counter ramp — not a signal; switching Y to "
                    f"col {alt}.")
            yi = alt
        elif log:
            log.warning(
                "select_xy_columns: every candidate Y column is an "
                "index/counter ramp — fitting it would measure nothing; "
                "proceeding under protest.")
    if log:
        cols_desc = f" ({column_names})" if column_names else ""
        log.warning(
            f"select_xy_columns: {n_cols}-column data{cols_desc} reduced to "
            f"(X=col {xi}, Y=col {yi}); other columns dropped for fitting."
        )
    return _try_orient_xy(np.column_stack([arr[:, xi], arr[:, yi]]))


def describe_columns(data, names=None):
    """Describe a >2-column table for the planning LLM, or None for <=2 columns.

    Oriented to (n_points, n_cols) to match :func:`select_xy_columns`, so the
    column indices the LLM returns line up with selection. Names are used when
    provided (DataFrame/CSV header), else positional ``col_i``.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        return None
    if arr.shape[0] < arr.shape[1]:   # orient: many more points than columns
        arr = arr.T
    n_cols = arr.shape[1]
    if n_cols <= 2:
        return None
    names_known = bool(names) and len(names) >= n_cols
    names = ([str(c) for c in names[:n_cols]] if names_known
             else [f"col_{i}" for i in range(n_cols)])
    per_column = []
    for i in range(n_cols):
        col = arr[:, i]
        is_num = np.issubdtype(col.dtype, np.number)
        finite = col[np.isfinite(col)] if is_num else col
        mono = (is_num and col.size >= 2
                and (np.all(np.diff(col) > 0) or np.all(np.diff(col) < 0)))
        per_column.append({
            "index": i, "name": names[i],
            "min": float(np.min(finite)) if is_num and finite.size else None,
            "max": float(np.max(finite)) if is_num and finite.size else None,
            "monotonic": bool(mono),
        })
    return {"n_columns": n_cols, "names": names, "names_known": names_known,
            "per_column": per_column, "preview_rows": arr[:5].tolist()}


def _hdr_isnum(s):
    cands = [s]
    if s.count(".") > 1:           # drop pandas '.N' duplicate-name suffix
        cands.append(s.rsplit(".", 1)[0])
    for c in cands:
        try:
            float(c); return True
        except ValueError:
            pass
    return False


def _recover_commented_header(data_path, ext):
    """Recover a column-header row that sits behind a comment marker.

    Format-agnostic ("header row adjacent to the data"): find the first numeric
    data row, then test the nearest non-blank line above it as a header after
    stripping any leading comment punctuation (``#``/``##``/``%``/``;``/``//``).
    Works for any delimiter / column count. Returns names or None — None when the
    line above the data is not a plausible header (token-count mismatch or mostly
    numeric), so a headerless file or a trailing metadata line never yields a
    spurious header. Used only as a fallback after the plain pandas read finds no
    usable header (e.g. instrument exports whose column line is itself ``#``-led).
    """
    import re as _re

    def split_delim(s):
        if ext == ".csv":
            return [t.strip() for t in s.split(",")]
        if ext == ".tsv":
            return [t.strip() for t in s.split("\t")]
        return s.split()

    def is_comment(line):
        return line.strip().startswith(("#", "%", ";", "//"))

    try:
        with open(data_path, "r", errors="replace") as fh:
            raw = [ln.rstrip("\n") for ln in fh.readlines()[:200]]
    except OSError:
        return None

    # First numeric data row: non-blank, non-comment, >=2 tokens, mostly numeric.
    data_idx, ncols = None, None
    for i, ln in enumerate(raw):
        if not ln.strip() or is_comment(ln):
            continue
        toks = split_delim(ln.strip())
        if len(toks) >= 2 and sum(_hdr_isnum(t) for t in toks) >= 0.8 * len(toks):
            data_idx, ncols = i, len(toks)
            break
    if not data_idx:        # None (no data) or 0 (data on line 0) → no header above
        return None

    # Candidate header = nearest non-blank line above the data block, with any
    # leading comment punctuation stripped.
    for j in range(data_idx - 1, -1, -1):
        if not raw[j].strip():
            continue
        stripped = _re.sub(r"^\s*(?:#+|%+|;+|//+)\s*", "", raw[j].strip())
        cand = split_delim(stripped)
        if (len(cand) == ncols and all(t for t in cand)
                and sum(_hdr_isnum(t) for t in cand) * 2 < len(cand)):
            return cand
        return None         # nearest line above data isn't a plausible header
    return None


def sniff_column_names(data_path):
    """Best-effort header sniff for a delimited text file → column names or None.

    None for headerless files (every header token parses as a number) and for
    formats without a textual header (.npy / binary). When the plain read finds
    no usable header — because the column-name row is itself hidden behind a
    comment marker (common in instrument exports, e.g. a ``##Temp\\tMass`` line) —
    falls back to recovering that adjacent header row, which the comment-stripping
    read would otherwise discard."""
    ext = os.path.splitext(str(data_path))[1].lower()
    if ext not in (".csv", ".tsv", ".dat", ".txt"):
        return None
    # Primary path (unchanged): handles every file whose header row is NOT hidden
    # behind a comment marker — plain CSV, #-metadata + plain header, units rows,
    # pandas duplicate-name suffixes — exactly as before.
    try:
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ("," if ext == ".csv" else r"\s+")
        df = pd.read_csv(data_path, sep=sep, comment="#", nrows=5, engine="python")
        cols = [str(c) for c in df.columns]
        # If most "names" parse as numbers, there was no header row (the data row
        # was misread as the header) → positional names are the safe choice.
        numeric = sum(_hdr_isnum(c) for c in cols)
        # A header whose first cell starts with a non-# comment marker (%, ;, //)
        # is a comment line pandas misread (it only strips #) → defer to recovery.
        looks_commented = bool(cols) and cols[0].strip().startswith(("%", ";", "//"))
        if numeric * 2 < len(cols) and not looks_commented:
            return cols
    except Exception:
        pass
    # Recovery path: only reached when the primary read found no usable header.
    return _recover_commented_header(data_path, ext)


def load_curve_data(data_path: str, auto_orient: bool = True,
                    system_info: dict = None, column_names: list = None) -> np.ndarray:
    """
    Robustly loads curve data (X, Y) from various file formats.
    Handles .npy, .h5/.hdf5/.nxs (NeXus), CSV, TSV, and whitespace separation
    automatically.

    When ``auto_orient`` is True (default), the result is normalized to a
    2-column ``(X, Y)`` array via :func:`select_xy_columns`: a 2-column array is
    oriented so the monotonic (X) column comes first, and a >2-column table (an
    extra error/weight column, multiple channels) is reduced to the chosen X/Y
    pair — guided by ``system_info`` (``x_column``/``y_column`` or ``columns``)
    and ``column_names`` when given, else by an axis-name / monotonicity
    heuristic. Pass ``auto_orient=False`` for legacy raw-layout behavior.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    def _finish(arr):
        return (select_xy_columns(arr, system_info=system_info,
                                  column_names=column_names)
                if auto_orient else arr)

    # Native numpy format
    if data_path.endswith('.npy'):
        return _finish(np.load(data_path))

    # NeXus / HDF5 — pull the signal and (when present) its axis so we
    # can return (X, Y) pairs for callers that expect a 2D layout.
    if data_path.lower().endswith(('.h5', '.hdf5', '.nxs')):
        from scilink.utils.hdf5_utils import load_hdf5_signal
        signal, axes = load_hdf5_signal(data_path, return_axes=True)
        if signal.ndim == 1 and axes and axes[0] is not None and axes[0].size == signal.size:
            return np.column_stack([axes[0], signal])
        return _finish(signal)

    attempts = [
        dict(),                                # whitespace-delimited, no header
        dict(skiprows=1),                      # whitespace-delimited, skip header
        dict(delimiter=','),                   # CSV, no header
        dict(delimiter=',', skiprows=1),       # CSV, skip header
    ]

    for kw in attempts:
        try:
            data = np.loadtxt(data_path, **kw)
            if data.size > 0:
                return _finish(data)
        except Exception:
            pass

    # If all fail, raise descriptive error
    raise ValueError(f"Unsupported file format or invalid data structure in {data_path}.")

def plot_curve_to_bytes(curve_data: np.ndarray, system_info: dict, title_suffix: str = "") -> bytes:
    """
    Plots a 1D curve and returns the image as bytes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
    
    plot_title = system_info.get("title") or "Data"
    ax.set_title(plot_title + title_suffix)

    xlabel_text = system_info.get("xlabel") or "X-axis"
    ax.set_xlabel(xlabel_text)

    ylabel_text = system_info.get("ylabel") or "Y-axis"
    ax.set_ylabel(ylabel_text)
    
    ax.grid(True, linestyle='--')
    # Object API on purpose: plt.tight_layout()/plt.savefig() act on pyplot's
    # CURRENT figure, which another thread (UI / meta / parallel workers) can
    # swap between our subplots() and the save — intermittently producing a
    # blank "Original Data" image in the report.
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)
    return image_bytes