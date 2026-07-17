"""Codegen-backed file preparation: split a combined data+metadata file into
separate data + metadata files BEFORE delegation to a specialist. ``prepare_inputs``
handles one file; ``prepare_inputs_batch`` handles many, generating one verified
split per structural cluster and reusing it (round-trip verified per file) across
structurally identical siblings.

STRICT SCOPE — this is the meta agent's ONLY code-generation surface, and it is
restricted to **lossless file repackaging**. The generated script may separate
existing data from metadata and reconstruct the original; it must NOT transform,
scale, filter, fit, resample, denoise, or analyze anything. Analysis is always
delegated.

Enforcement is layered:
  1. narrow prompt (read input → write data + metadata, values unchanged);
  2. ROUND-TRIP verification — the script must also reconstruct the original from
     its two outputs, and we check reconstruction ≈ original. This confirms a
     faithful reconstruction PATH exists; it does NOT, by itself, prove the data
     leg (`data_path`, the file actually delegated) is untransformed, because the
     reconstruction is produced by the same generated script and is not forced to
     derive solely from the two split outputs. The data leg's fidelity rests on
     the prompt + import guard; the round-trip is a strong corroborating signal,
     not a hermetic proof. If the round-trip can't be verified, the split is
     REJECTED;
  3. static import guard (no analysis libraries — note numpy/pandas are allowed
     and can transform, so this narrows but does not eliminate the surface);
  4. explicit documentation on the tool.
"""

import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

# Modules that signal analysis/computation rather than IO/parsing. A generated
# prep script importing any of these (or a submodule of one) is rejected before
# execution. Entries are matched by dotted PREFIX, so `scipy.optimize` denies
# `scipy.optimize.*` but leaves IO-only `scipy.io` / `scipy.sparse` available —
# the granularity is deliberate (classic MATLAB .mat needs `scipy.io`).
_ANALYSIS_IMPORT_DENYLIST = (
    "sklearn", "scipy.optimize", "scipy.signal", "scipy.ndimage",
    "scipy.interpolate", "scipy.stats", "scipy.fft", "scipy.fftpack",
    "lmfit", "torch", "tensorflow", "cv2", "skimage", "statsmodels",
    "matplotlib",
)

_PROMPT = '''You prepare a scientific data file for downstream analysis by
SEPARATING its data from its metadata. This is a LOSSLESS REPACKAGING step ONLY —
you are NOT analyzing anything.

PATHS ARE PROVIDED AT RUNTIME — DO NOT HARDCODE ANY FILE PATH. A dict named
`_PREP` is already defined ABOVE your code; read every path from it:
  _PREP["input"]        # the combined file to read
  _PREP["out_dir"]      # directory to write the data file into
  _PREP["metadata_out"] # exact path to write the metadata JSON
  _PREP["recon_out"]    # exact path to write the reconstruction
Your script MUST reference `_PREP[...]` for ALL paths and MUST NOT contain any
literal path string. (This lets the SAME script be reused across structurally
identical files — so write it generically against the probe, not this one file.)

Structural probe of a representative input (evidence — do not assume beyond it):
{probe}

Common layouts you may encounter (illustrative, not exhaustive — let the probe
decide):
- A `.dat`/`.txt`/`.csv` with a metadata HEADER block (comment lines, `key: value`
  or `key = value` pairs, a units row) ABOVE the numeric data rows.
- A MATLAB `.mat` (or `.npz`/HDF5) where some keys are arrays (data) and others
  are scalars/strings (metadata).
- An HDF5/NeXus dataset carrying metadata in attributes alongside the array.
- A `.tif`/`.tiff` carrying acquisition metadata in TIFF tags / ImageDescription
  (ImageJ `key=value` block or OME-XML) alongside the pixel array.
Identify which bytes/keys/lines are data vs metadata from the probe, then:

Write a Python script that:
1. Reads the input file at `_PREP["input"]`.
2. Splits it into (a) the primary numerical DATA (the array / table / signal) and
   (b) the METADATA — everything that is NOT the data values: headers, attributes,
   comments, units, calibration, axes, acquisition parameters, etc.
3. Writes the DATA to a file UNDER `_PREP["out_dir"]` (choose a basename and join
   it with `_PREP["out_dir"]`) — use `.npy` for an array/cube or `.csv` for a table.
4. Writes the METADATA to `_PREP["metadata_out"]` as a single JSON object. Put the
   HUMAN-MEANINGFUL scientific metadata at the TOP LEVEL with clean keys
   (technique, instrument, sample, units, axes, wavelength, ...). If you need
   extra bookkeeping to reconstruct the original byte-for-byte (line endings,
   dtypes, column order, key order), nest ALL of it under a single
   "_reconstruction" key so downstream sees clean metadata. A value can be both
   meaningful AND needed for reconstruction — when scientific metadata lives in a
   container's attributes/keys (HDF5 attrs, .npz/.mat keys), surface it at the
   TOP LEVEL and, if needed, also record its placement under "_reconstruction";
   do not relegate it to "_reconstruction" alone.
5. RECONSTRUCTS the original file from those two outputs and writes it to
   `_PREP["recon_out"]` (same format as the input). We will verify it matches.
6. Prints exactly one line (the ACTUAL data path you wrote):
   PREP_RESULT_JSON:{{"data_out": "<abs path to the data file you wrote>"}}

HARD CONSTRAINTS (a violation fails verification):
- DO NOT transform, scale, normalize, filter, fit, resample, denoise, crop, or
  analyze the data. The data you write MUST be the original values, unchanged.
- The reconstruction MUST reproduce the original file's data and metadata.
- Reference paths ONLY via `_PREP[...]`; no literal path strings anywhere.
- Allowed libraries ONLY: numpy, pandas, json, csv, struct, io, re, h5py,
  scipy.io (for MATLAB .mat read/write ONLY), PIL/Pillow (for TIFF tags /
  ImageDescription), os, pathlib. Do NOT import analysis libraries — sklearn,
  lmfit, torch, cv2, skimage, matplotlib, or any other scipy submodule
  (optimize, signal, ndimage, interpolate, stats, fft, ...).
- No network, no plotting.

Respond with a JSON object: {{"script": "<the full python script>"}}
'''


def _extract_script(text: str) -> str:
    """Pull the script string out of the model response (JSON or fenced)."""
    if not text:
        return ""
    # Prefer a {"script": "..."} JSON object.
    try:
        m = re.search(r'\{.*"script".*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0)).get("script", "")
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to a fenced code block.
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _denied_module(name: str) -> str | None:
    """Return the denylist entry a dotted module name matches by prefix, else None.

    ``scipy.optimize`` matches ``scipy.optimize`` and ``scipy.optimize.foo`` but
    NOT ``scipy.io``; ``sklearn`` matches ``sklearn`` and any ``sklearn.*``.
    """
    parts = name.split(".")
    for entry in _ANALYSIS_IMPORT_DENYLIST:
        ep = entry.split(".")
        if parts[: len(ep)] == ep:
            return entry
    return None


def _static_guard(script: str) -> str | None:
    """Return a rejection reason if the script imports a denied analysis module.

    Parses the script and inspects ``import``/``from`` statements so the dotted
    denylist is honored precisely (e.g. ``from scipy import optimize`` is caught
    while ``from scipy import io`` is allowed). Not a sandbox — dynamic imports
    (``__import__``) are out of scope; the round-trip check is the real net.
    """
    try:
        tree = ast.parse(script)
    except SyntaxError:
        # Unparseable code can't run, so it can't do harm; let execution surface
        # the syntax error. Conservative root-level fallback just in case.
        for entry in _ANALYSIS_IMPORT_DENYLIST:
            root = entry.split(".")[0]
            if re.search(rf"^\s*(import|from)\s+{re.escape(root)}\b", script, re.MULTILINE):
                return f"Generated prep script imports a forbidden module ('{root}')."
        return None

    def reason(hit: str) -> str:
        return (
            f"Generated prep script imports a forbidden analysis module "
            f"('{hit}'). File prep must use IO/parsing libraries only."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _denied_module(alias.name)
                if hit:
                    return reason(hit)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — not a third-party analysis lib
                continue
            mod = node.module or ""
            hit = _denied_module(mod)
            if hit:
                return reason(hit)
            for alias in node.names:  # from pkg import submod
                hit = _denied_module(f"{mod}.{alias.name}" if mod else alias.name)
                if hit:
                    return reason(hit)
    return None


def _members_equal(a, b) -> bool:
    """Compare two arrays type-appropriately: numeric within tolerance, else exact.

    Combined containers (e.g. ``.npz``) routinely hold non-numeric members —
    string/object metadata keys — alongside the numeric data, so a blanket
    ``np.allclose`` would raise on those and falsely fail the round-trip.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _attrs_equal(a: dict, b: dict) -> bool:
    """Compare two attribute dicts (HDF5 attrs / .mat fields) member-wise."""
    if set(a) != set(b):
        return False
    return all(_members_equal(a[k], b[k]) for k in a)


def _h5_equal(original: Path, recon: Path) -> bool:
    """Content-aware HDF5 comparison: a faithful re-write is rarely byte-identical
    (chunking, compression, attribute/dataset order, timestamps all vary), so we
    compare the group/dataset tree and attributes instead. ``h5py`` is already an
    allowed prep library; if it is unavailable we cannot verify → reject.
    """
    try:
        import h5py
    except Exception:
        return False

    def collect(f):
        items = {"/": ("group", None, dict(f.attrs))}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items[name] = ("dataset", obj[()], dict(obj.attrs))
            else:  # group
                items[name] = ("group", None, dict(obj.attrs))

        f.visititems(visit)
        return items

    try:
        with h5py.File(original, "r") as fa, h5py.File(recon, "r") as fb:
            a, b = collect(fa), collect(fb)
            if set(a) != set(b):
                return False
            for k in a:
                kind_a, val_a, attrs_a = a[k]
                kind_b, val_b, attrs_b = b[k]
                if kind_a != kind_b or not _attrs_equal(attrs_a, attrs_b):
                    return False
                if kind_a == "dataset" and not _members_equal(val_a, val_b):
                    return False
            return True
    except Exception:
        return False


def _mat_equal(original: Path, recon: Path) -> bool:
    """Content-aware MATLAB .mat comparison. Classic v5/6/7 .mat go through
    ``scipy.io.loadmat`` (an IO module, allowed); v7.3 .mat are HDF5 and fall back
    to :func:`_h5_equal`. Comparison ignores loadmat's ``__header__`` /
    ``__version__`` / ``__globals__`` bookkeeping (the header carries a timestamp).
    """
    try:
        from scipy.io import loadmat
    except Exception:
        return _h5_equal(original, recon)  # scipy absent → try HDF5 (v7.3 .mat)
    try:
        a, b = loadmat(original), loadmat(recon)
    except NotImplementedError:
        return _h5_equal(original, recon)  # v7.3 .mat is HDF5 under the hood
    except Exception:
        return False
    skip = {"__header__", "__version__", "__globals__"}
    ka = {k for k in a if k not in skip}
    kb = {k for k in b if k not in skip}
    if ka != kb:
        return False
    return all(_members_equal(a[k], b[k]) for k in ka)


def _tif_equal(original: Path, recon: Path) -> bool:
    """Content-aware TIFF comparison: pixel array(s) numeric-tolerant, tags exact.

    A faithful TIFF rewrite is rarely byte-identical (encoder, strip/tile layout,
    tag order), so compare decoded pixels per page plus the tag dict. ``PIL`` is
    an allowed prep library; if unavailable we cannot verify → reject.
    """
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        with Image.open(original) as ia, Image.open(recon) as ib:
            na, nb = getattr(ia, "n_frames", 1), getattr(ib, "n_frames", 1)
            if na != nb:
                return False
            for i in range(na):
                ia.seek(i)
                ib.seek(i)
                if not _members_equal(np.asarray(ia), np.asarray(ib)):
                    return False
                ta = dict(getattr(ia, "tag_v2", {}) or {})
                tb = dict(getattr(ib, "tag_v2", {}) or {})
                if set(ta) != set(tb):
                    return False
                # Tags must be reproduced exactly (metadata losslessness).
                if any(str(ta[k]) != str(tb.get(k)) for k in ta):
                    return False
            return True
    except Exception:
        return False


def _roundtrip_ok(original: Path, recon: Path) -> bool:
    """Tolerant lossless check: does ``recon`` reproduce ``original``?"""
    try:
        ob = original.read_bytes()
        rb = recon.read_bytes()
    except OSError:
        return False
    if ob == rb:
        return True  # byte-identical
    ext = original.suffix.lower()
    try:
        if ext == ".npy":
            return _members_equal(np.load(original, allow_pickle=True),
                                  np.load(recon, allow_pickle=True))
        if ext == ".npz":
            a, b = np.load(original, allow_pickle=True), np.load(recon, allow_pickle=True)
            return set(a.files) == set(b.files) and all(
                _members_equal(a[k], b[k]) for k in a.files
            )
        if ext in (".h5", ".hdf5", ".nxs", ".nx"):
            return _h5_equal(original, recon)
        if ext == ".mat":
            return _mat_equal(original, recon)
        if ext in (".tif", ".tiff"):
            return _tif_equal(original, recon)
        if ext in (".csv", ".tsv", ".txt", ".dat"):
            return _text_equal_tolerant(original.read_text(errors="replace"),
                                        recon.read_text(errors="replace"))
    except Exception:
        return False
    return False  # unknown/binary that wasn't byte-identical → cannot verify


def _text_equal_tolerant(a: str, b: str) -> bool:
    """Token-wise compare: numbers within tolerance, other tokens exact."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if np.isclose(float(x), float(y), equal_nan=True):
                continue
        except ValueError:
            pass
        return False
    return True


def _verify(input_path: Path, data_out: Path, meta_out: Path,
            recon_out: Path) -> tuple[bool, str]:
    """Verify the split is well-formed AND lossless. Returns (ok, reason).

    ``data_out`` is the model's chosen data path (pulled from PREP_RESULT_JSON);
    ``meta_out`` / ``recon_out`` are the template-fixed paths the caller owns, so
    we verify (and later return) those rather than trusting the model's echo.
    """
    if not data_out.exists():
        return False, f"data_out not written: {data_out}"
    try:
        if data_out.suffix.lower() == ".npy":
            arr = np.load(data_out, allow_pickle=False)
            if arr.size == 0:
                return False, "data_out array is empty"
        else:  # tabular
            if data_out.stat().st_size == 0:
                return False, "data_out file is empty"
    except Exception as e:
        return False, f"data_out does not load as data: {e}"

    if not meta_out.exists():
        return False, f"metadata_out not written: {meta_out}"
    try:
        meta = json.loads(meta_out.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return False, f"metadata_out is not valid JSON: {e}"
    # Reject empty metadata: a genuinely metadata-light file fails here and the
    # meta bounces it back to the user rather than delegating a no-op split.
    if not isinstance(meta, dict) or not meta:
        return False, "metadata_out is empty or not a JSON object"

    if not recon_out.exists():
        return False, f"recon_out (round-trip check) not written: {recon_out}"
    if not _roundtrip_ok(input_path, recon_out):
        return False, (
            "round-trip FAILED: reconstruction does not match the original — the "
            "split is not lossless (or the data was transformed). Rejected."
        )
    return True, "lossless split verified"


def _file_paths(input_path: Path, output_dir: Path) -> dict:
    """Per-file output paths in a collision-safe subdir keyed on the absolute path.

    Two uploads sharing a basename (runA/scan.tif and runB/scan.tif) get distinct
    subdirs so they never clobber each other's data/metadata/recon outputs.
    """
    stem = re.sub(r"[^0-9A-Za-z_-]", "_", input_path.stem) or "input"
    key = hashlib.sha1(str(input_path.resolve()).encode()).hexdigest()[:8]
    file_out = Path(output_dir) / f"{stem}_{key}"
    file_out.mkdir(parents=True, exist_ok=True)
    return {
        "input": str(input_path),
        "out_dir": str(file_out),
        "metadata_out": str(file_out / f"{stem}_metadata.json"),
        "recon_out": str(file_out / f"{stem}_reconstructed{input_path.suffix}"),
    }


# Extensions whose bytes are human-readable text worth showing the split LLM
# verbatim. Binary containers (.npy/.npz/.h5/.mat/.tif) surface their metadata
# structurally in the probe, so a "raw head" is meaningless for them.
_TEXT_HEAD_EXTS = {".csv", ".tsv", ".dat", ".txt", ".asc", ".xy", ".prn"}
_HEAD_MAX_LINES = 60
_HEAD_MAX_CHARS = 4000


def _raw_text_head(input_path: Path) -> str:
    """First lines of a text-like file, verbatim, for the split prompt.

    Lets the codegen LLM SEE the metadata/comment block and the column-name row
    (which the structural probe omits), so the generated split can separate data
    from metadata and preserve column names instead of writing the script blind.
    Returns "" for binary/container formats (their metadata is in the probe) or
    on read error. Capped by lines AND chars so the head stays small regardless
    of file size — the LLM reads STRUCTURE here, never bulk data values."""
    if input_path.suffix.lower() not in _TEXT_HEAD_EXTS:
        return ""
    try:
        lines = []
        with open(input_path, "r", errors="replace") as fh:
            for _ in range(_HEAD_MAX_LINES):
                ln = fh.readline()
                if not ln:
                    break
                lines.append(ln.rstrip("\n"))
        return "\n".join(lines)[:_HEAD_MAX_CHARS]
    except OSError:
        return ""


def _build_prompt(probe, input_path: Path) -> str:
    probe_str = json.dumps(probe, indent=2, default=str) if probe else \
        f"(no probe) extension={input_path.suffix}, size={input_path.stat().st_size} bytes"
    head = _raw_text_head(input_path)
    if head:
        probe_str += (
            "\n\nRaw file head (first lines, verbatim — use it to locate the "
            "metadata/comment block, the column-name row, and where the numeric "
            "data starts, and to preserve column names; do NOT assume the data "
            "continues in this exact form beyond the head):\n" + head
        )
    return _PROMPT.format(probe=probe_str)


def _prep_header(paths: dict) -> str:
    """Runtime `_PREP` path bindings prepended to a generated split body, so one
    generically-written body can be reused verbatim across structurally identical
    files (only this header changes per file)."""
    return "_PREP = " + json.dumps({k: str(v) for k, v in paths.items()}) + "\n"


def _generate_split_script(prompt: str, model) -> tuple:
    """One codegen attempt → (script_body, "") or (None, rejection_reason)."""
    try:
        script = _extract_script(model.generate_content(prompt).text)
    except Exception as e:  # noqa: BLE001
        return None, f"code generation failed: {e}"
    if not script:
        return None, "model returned no script"
    guard = _static_guard(script)
    if guard:
        return None, guard
    if "_PREP" not in script:
        return None, ("script must read paths from the _PREP dict, not hardcode "
                      "them — reference _PREP['input'/'out_dir'/'metadata_out'/'recon_out']")
    return script, ""


def _apply_and_verify(body: str, input_path: Path, paths: dict, executor) -> tuple:
    """Run a (path-parameterized) body for one file and verify losslessness.

    Returns ``(True, {data_path, metadata_path})`` or ``(False, reason)``. No LLM
    call — this is what lets a verified body be reused across a batch cheaply.
    """
    script = _prep_header(paths) + body
    exec_res = executor.execute_script(script, working_dir=paths["out_dir"])
    if exec_res.get("status") != "success":
        return False, f"script execution failed: {exec_res.get('message', '')[:600]}"
    m = re.search(r"PREP_RESULT_JSON:(\{.*\})", exec_res.get("stdout", ""))
    if not m:
        return False, "script did not print PREP_RESULT_JSON with the data path"
    try:
        result = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        return False, f"PREP_RESULT_JSON not valid JSON: {e}"
    data_out = Path(result.get("data_out", ""))
    ok, reason = _verify(input_path, data_out,
                         Path(paths["metadata_out"]), Path(paths["recon_out"]))
    if not ok:
        return False, reason
    Path(paths["recon_out"]).unlink(missing_ok=True)  # only needed for the check
    return True, {"data_path": str(data_out), "metadata_path": paths["metadata_out"]}


def _prepare_one(input_path: Path, paths: dict, base_prompt: str, model, executor,
                 logger, max_retries: int) -> tuple:
    """Generate+verify a split for one file, with retries feeding back failures.

    Returns ``(result_dict, verified_body_or_None)`` — the body is handed back so
    a batch can reuse it on structurally identical siblings.
    """
    prompt = base_prompt
    last_reason = ""
    for attempt in range(1, max_retries + 2):
        if logger:
            logger.info(f"📦 File prep (attempt {attempt}): generating split for {input_path.name}")
        body, reason = _generate_split_script(prompt, model)
        if body is None:
            last_reason = reason
            if logger:
                logger.warning(f"📦 File prep rejected: {reason}")
            prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{reason}\nFix it."
            continue
        ok, res = _apply_and_verify(body, input_path, paths, executor)
        if ok:
            if logger:
                logger.info(f"✅ File prep OK: data={res['data_path']}")
            return {"status": "success", **res, "attempts": attempt}, body
        last_reason = res
        if logger:
            logger.warning(f"📦 File prep verification failed: {res}")
        prompt = base_prompt + f"\n\n### PREVIOUS ATTEMPT FAILED\n{res}\nProduce a lossless split."

    Path(paths["recon_out"]).unlink(missing_ok=True)
    return {
        "status": "error",
        "message": f"Could not produce a verified lossless split after "
                   f"{max_retries + 1} attempt(s): {last_reason}",
        "attempts": max_retries + 1,
    }, None


def prepare_inputs(input_path, model, executor, output_dir, probe=None,
                   logger=None, max_retries: int = 1) -> dict:
    """Split ONE combined data+metadata file into data + metadata files (codegen).

    Returns ``{status:"success", data_path, metadata_path, attempts}`` or
    ``{status:"error", message, attempts}``. The reconstruction used for the
    round-trip check is deleted once verified. The caller decides how to handle an
    error — the meta surfaces it to the user rather than delegating an unsplit file.
    """
    input_path = Path(input_path)
    paths = _file_paths(input_path, Path(output_dir))
    base_prompt = _build_prompt(probe, input_path)
    result, _ = _prepare_one(input_path, paths, base_prompt, model, executor,
                             logger, max_retries)
    return result


def _signature(probe):
    """Structural fingerprint for clustering — same signature ⇒ one split script
    can serve all. Keyed on STRUCTURE (ndim/dtype, columns, container keys), not
    exact dimensions, so same-kind/different-size files still cluster. Returns
    None when there's no structural confidence (→ caller treats the file as solo).
    """
    if not probe:
        return None
    ext, kind = probe.get("ext"), probe.get("kind")
    if kind == "array":
        return (ext, "array", len(probe.get("shape") or []), probe.get("dtype"))
    if kind == "image":
        return (ext, "image", probe.get("mode"), probe.get("n_frames"))
    if kind == "table":
        return (ext, "table", tuple(probe.get("columns") or []),
                tuple(sorted((probe.get("dtypes") or {}).items())))
    if kind == "npz":
        return (ext, "npz", tuple(sorted(probe.get("keys") or [])))
    if kind == "mat":
        return (ext, "mat", tuple(sorted(probe.get("keys") or [])))
    if kind == "hdf5":
        return (ext, "hdf5", tuple(sorted(probe.get("datasets") or [])))
    if kind == "json":
        return (ext, "json", tuple(sorted(probe.get("top_level_keys") or [])))
    return None


def prepare_inputs_batch(input_paths, model, executor, output_dir, probes=None,
                         logger=None, max_retries: int = 1) -> dict:
    """Split MANY combined files, reusing one verified split per structural cluster.

    Files are clustered by :func:`_signature`; the cluster's first file drives a
    codegen, and the verified body is reused (no LLM) on its siblings — each still
    round-trip verified. A sibling the shared body can't reproduce falls back to
    its own codegen, so a heterogeneous "batch" still splits correctly.

    Returns ``{status, results:[{file, status, ...}], summary:{...}}`` where
    ``status`` is ``success`` (all ok), ``partial`` (some failed), or ``error``
    (all failed). ``probes`` (parallel to ``input_paths``) sharpen clustering;
    without them files default to per-file (no reuse).
    """
    input_paths = [Path(p) for p in input_paths]
    if probes is None:
        probes = [None] * len(input_paths)
    output_dir = Path(output_dir)

    clusters: dict = {}
    for i, pr in enumerate(probes):
        sig = _signature(pr)
        if sig is None:                       # no structural confidence → solo
            sig = ("__solo__", i)
        clusters.setdefault(sig, []).append(i)

    results = [None] * len(input_paths)
    n_codegens = n_reused = n_fallback = 0

    for idxs in clusters.values():
        first = idxs[0]
        base_prompt = _build_prompt(probes[first], input_paths[first])
        res, body = _prepare_one(input_paths[first], _file_paths(input_paths[first], output_dir),
                                 base_prompt, model, executor, logger, max_retries)
        n_codegens += 1
        results[first] = {"file": str(input_paths[first]), "reused": False, **res}

        for j in idxs[1:]:
            paths_j = _file_paths(input_paths[j], output_dir)
            if body is not None:
                ok, r = _apply_and_verify(body, input_paths[j], paths_j, executor)
                if ok:
                    n_reused += 1
                    results[j] = {"file": str(input_paths[j]), "status": "success",
                                  "reused": True, "attempts": 0, **r}
                    if logger:
                        logger.info(f"♻️  Reused split for {input_paths[j].name}")
                    continue
                if logger:
                    logger.warning(f"📦 Reuse failed for {input_paths[j].name} ({r}); regenerating")
            # Fallback: this sibling isn't actually the same shape — solo codegen.
            bp = _build_prompt(probes[j], input_paths[j])
            res_j, _ = _prepare_one(input_paths[j], paths_j, bp, model, executor,
                                    logger, max_retries)
            n_codegens += 1
            n_fallback += 1
            results[j] = {"file": str(input_paths[j]), "reused": False, **res_j}

    n_failed = sum(1 for r in results if r.get("status") != "success")
    status = ("success" if n_failed == 0
              else "error" if n_failed == len(results) else "partial")
    return {
        "status": status,
        "results": results,
        "summary": {
            "n_files": len(input_paths), "n_clusters": len(clusters),
            "n_codegens": n_codegens, "n_reused": n_reused,
            "n_fallback": n_fallback, "n_failed": n_failed,
        },
    }


def stage_pairs_flat(results: list, flat_dir) -> dict:
    """Stage a batch's verified (data, metadata) pairs into ONE flat directory
    with stem-matched names, so a series consumer can ingest the whole batch as a
    directory and pair each data file with its sidecar by stem.

    The per-file split machinery writes each pair into its own collision-safe
    subdir with mismatched stems (``<stem>_data.csv`` + ``<stem>_metadata.json``)
    — correct for the split/verify step but unusable by the directory-based series
    loader, which excludes ``.json`` as metadata and pairs sidecars by EXACT stem
    (``spec_00.csv`` ↔ ``spec_00.json``). This copies (never moves — the originals
    and their round-trip provenance stay intact) each successful pair to
    ``flat_dir/<input_stem><data_suffix>`` + ``flat_dir/<input_stem>.json``, using
    the ORIGINAL input filename as the stem (meaningful ``unit`` rows downstream)
    and disambiguating only on collision. Failed splits are skipped.

    Returns ``{"staged_dir": str, "pairs": [{"data", "metadata"}], "n": int}``.
    """
    flat = Path(flat_dir)
    flat.mkdir(parents=True, exist_ok=True)
    pairs: list = []
    used: set = set()
    for r in results or []:
        if r.get("status") != "success":
            continue
        dp, mp = Path(r.get("data_path", "")), Path(r.get("metadata_path", ""))
        if not (dp.is_file() and mp.is_file()):
            continue
        base = re.sub(r"[^0-9A-Za-z_-]", "_", Path(r.get("file") or dp).stem) or "input"
        stem, n = base, 1
        while (stem in used or (flat / f"{stem}{dp.suffix}").exists()
               or (flat / f"{stem}.json").exists()):
            n += 1
            stem = f"{base}_{n}"
        used.add(stem)
        data_dst, meta_dst = flat / f"{stem}{dp.suffix}", flat / f"{stem}.json"
        shutil.copy2(dp, data_dst)
        shutil.copy2(mp, meta_dst)
        pairs.append({"data": str(data_dst), "metadata": str(meta_dst)})
    return {"staged_dir": str(flat), "pairs": pairs, "n": len(pairs)}
