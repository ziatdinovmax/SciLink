import json
import logging
import os

from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params
from ...auth import get_internal_proxy_key


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the JSON Schema as a Dictionary
METADATA_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        # --- Core Identification ---
        "experiment_type": {
            "type": "string",
            "description": "General type of experiment (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis)."
        },
        # --- Detailed Experiment Info ---
        "experiment": {
            "type": "object",
            "description": "Details about the specific experimental setup.",
            "properties": {
                "technique": {
                    "type": "string",
                    "description": "Specific experimental technique used (e.g., HAADF-STEM, TEM, Photoluminescence Spectroscopy, XRD)."
                },
                "date": {
                    "type": ["string", "null"], # Optional
                    "description": "Date of the experiment (if available, YYYY-MM-DD)."
                },
                "instrument": {
                    "type": ["string", "null"], # Optional
                    "description": "Instrument used for the experiment."
                },
                "details": {
                    "type": ["string", "null"], # Optional
                    "description": "Other relevant conditions or parameters (e.g., voltage, temperature, excitation source, probe current)."
                }
            },
            "required": ["technique"] # Technique is crucial
        },
        # --- Sample Info ---
        "sample": {
            "type": "object",
            "description": "Details about the sample.",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "Specific material name, composition, or formula (e.g., MoS2, Gallium Nitride (GaN))."
                },
                "description": {
                    "type": ["string", "null"], # Optional
                    "description": "Additional description (e.g., form, substrate, synthesis method)."
                }
            },
            "required": ["material"] # Specific material is crucial
        },
        # --- Microscopy Specific (Conditional) ---
        "spatial_info": {
            "type": ["object", "null"], # Optional object
            "description": "Spatial dimensions/scale (Primarily for Microscopy).",
            "properties": {
                "field_of_view_x": {"type": ["number", "null"]},
                "field_of_view_y": {"type": ["number", "null"]},
                "field_of_view_units": {"type": ["string", "null"], "description": "Units like 'nm', 'um', 'pixels'."}
            },
        },
        # --- Spectroscopy/Hyperspectral Specific (Conditional) ---
        "energy_range": {
            "type": ["object", "null"], # Optional object
            "description": "Spectral or energy range covered (Primarily for Spectroscopy/Hyperspectral).",
            "properties": {
                "start": {"type": ["number", "null"]},
                "end": {"type": ["number", "null"]},
                "units": {"type": ["string", "null"], "description": "Units like 'nm', 'eV', 'cm^-1'."}
            },
        },
        # --- Generic Axis Spec (Optional, for non-energy 3D hyperspectral) ---
        # Used when the dataset is 3D but the axes are not the standard
        # (x_spatial, y_spatial, energy) layout — e.g. (force, distance, frequency),
        # (x, y, voltage), (x, y, time). When absent, downstream code synthesizes
        # axis_spec from legacy spatial_info + energy_range for backward compatibility.
        "axis_spec": {
            "type": ["object", "null"],
            "description": (
                "Optional per-axis description for 3D hyperspectral data. Keys "
                "are 'axis_0', 'axis_1', 'axis_2'. Each value is an object with "
                "'name' (e.g. 'x', 'time', 'voltage'), 'units' (e.g. 'nm', 's', "
                "'V'), 'kind' (one of 'spatial', 'parameter', 'signal', 'time'), "
                "and for the signal-like axis also 'start' and 'end' (physical "
                "range endpoints). When absent, the system uses legacy "
                "spatial_info + energy_range as if the layout were "
                "(x_spatial, y_spatial, energy)."
            ),
            "properties": {
                "axis_0": {"type": ["object", "null"]},
                "axis_1": {"type": ["object", "null"]},
                "axis_2": {"type": ["object", "null"]},
                "signal_is_nonnegative": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether the signal-like axis is physically non-negative "
                        "(e.g. photon counts). Defaults to true. Set to false for "
                        "signed signals like derivative or difference spectra so "
                        "preprocessing does not clip negative values."
                    ),
                },
            },
        },
        # --- 1D Curve Specific (Conditional) ---
        "title": {
            "type": ["string", "null"], # Optional
            "description": "A descriptive title for the data/plot (Primarily for 1D Curves)."
        },
        "data_columns": {
            "type": ["array", "null"], # Optional array
            "description": "Description of data columns, typically X and Y (Primarily for 1D Curves).",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name like 'Wavelength', 'Angle', 'Energy', 'Intensity', 'Counts'."},
                    "units": {"type": "string"}
                },
                "required": ["name", "units"]
            }
        },
        "xlabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested X-axis label, including units (Primarily for 1D Curves)."
        },
        "ylabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested Y-axis label, including units (Primarily for 1D Curves)."
        },
        "custom_processing_instruction": {
            "type": ["string", "null"],
            "description": (
                "A natural language instruction for custom data preprocessing. "
                "Use when the user describes a non-standard preprocessing step such as "
                "baseline division/subtraction using an external reference file, "
                "custom normalization, background correction with a specific method, etc. "
                "Should include file paths if referencing external data. "
                "Example: 'Divide each spectrum by the baseline in /data/reference.npy and set inf/nan to 0.'"
            )
        }
    },
    # Universally required fields at the top level
    "required": ["experiment_type", "experiment", "sample"]
}

# Convert schema dictionary to a JSON string for embedding in the prompt
schema_json_string_for_prompt = json.dumps(METADATA_SCHEMA_DICT, indent=2)

METADATA_GENERATION_PROMPT = f"""
You are an expert scientific assistant. Your task is to read the provided plain text description of a scientific experiment (which could be microscopy, spectroscopy/hyperspectral, or 1D curve data like PL/XRD) and extract key metadata into a structured JSON format.

Format your output STRICTLY as a valid JSON object conforming *exactly* to the following structure:

```json
{schema_json_string_for_prompt}
```

Instructions:

Identify Experiment Type: First, determine the general experiment_type (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis) and the specific experiment.technique (e.g., HAADF-STEM, TEM, PL, XRD, Absorption). Also identify sample.material.

Fill Core Fields: Populate the required fields: experiment_type, experiment (including technique), and sample (including material). Extract other details like instrument or conditions into experiment.details or sample.description if available.

Faithful Extraction — DO NOT FABRICATE (CRITICAL): Record ONLY information stated in the description. Do NOT add, infer, or enrich the metadata with domain knowledge the source text does not contain — in particular, do NOT invent expected/characteristic peak positions, known phase transitions, typical parameter values, or literature constants for the named material. Downstream analysis treats sample.description and experiment.details as factual experimental context and will try to act on any peaks/features named there, so fabricated expectations actively mislead it. If the author explicitly provided such details (expected peaks, transitions, prior values), preserve them verbatim — the rule is to neither invent nor drop. When unsure whether a detail came from the text or from your own knowledge, leave it out.

Fill Conditional Fields based on Type:

If Microscopy: Focus on extracting spatial_info (field of view and units). Omit or use null for spectroscopy/curve fields if not relevant.

If Spectroscopy/Hyperspectral: Focus on extracting energy_range (start, end, units). Omit or use null for microscopy/curve fields if not relevant. If the third (channel) axis of a 3D hyperspectral dataset is NOT energy/wavelength (e.g. it is time, voltage, force, frequency, or another parameter), populate the optional axis_spec object (with axis_0, axis_1, axis_2 entries — each having name, units, kind, and for the signal axis also start/end) instead of energy_range. Use axis_spec.signal_is_nonnegative=false when the signal can be signed (e.g. derivative spectra, difference spectra).

If 1D Curve Data (PL, XRD, etc.): Focus on extracting title, data_columns (determining X and Y column names/units), xlabel, and ylabel. energy_range might also apply if the x-axis represents energy. Omit or use null for microscopy fields.

Handle Missing Info: Use null for any optional fields (like experiment.date) or omit entire optional objects (spatial_info, energy_range, data_columns) if the information is missing or clearly not applicable to the described experiment type.

Custom Preprocessing: If the user describes any non-standard preprocessing steps (e.g., "divide by
a baseline file", "subtract the dark reference at /path/to/dark.npy", "apply rubber-band baseline
correction", "normalize to the peak at 520 cm^-1"), capture the FULL instruction verbatim in the
custom_processing_instruction field. Preserve any file paths exactly as written. If no custom
preprocessing is mentioned, set this field to null.

Required Fields: For universally required fields (experiment_type, experiment.technique, sample.material) that are truly missing even after careful reading, use the string "N/A".

Strict Formatting: Only include fields defined in the schema. Output ONLY the valid JSON object without markdown formatting.

Ensure the output JSON accurately reflects the information present in the text description and adheres to the conditional logic based on the experiment type.
"""


# =====================================================================
# Alias mappings for deterministic normalization (Tier 1)
# =====================================================================
# Each canonical field maps to a list of common aliases (case-insensitive)
_TECHNIQUE_ALIASES = [
    "technique", "method", "experimental technique", "measurement_type",
    "experimental_technique", "measurement_method",
]
_MATERIAL_ALIASES = [
    "material", "sample_material", "composition", "formula",
    "sample_name", "compound",
]
_EXPERIMENT_TYPE_ALIASES = [
    "type", "data_type", "exp_type", "experiment_type",
]
_ENERGY_START_ALIASES = [
    "energy_start", "spectral_start", "range_start", "start_wavelength",
    "start_energy", "x_start",
]
_ENERGY_END_ALIASES = [
    "energy_end", "spectral_end", "range_end", "end_wavelength",
    "end_energy", "x_end",
]
_ENERGY_UNITS_ALIASES = [
    "energy_units", "spectral_units", "x_units", "wavelength_units",
]
_SPATIAL_FOV_ALIASES = ["field_of_view", "fov"]
_PIXEL_SIZE_ALIASES = [
    "pixel_size", "pixel_size_nm", "nm_per_pixel",
    "scale_nm_per_pixel", "pixelSize", "pixel_scale", "resolution_nm",
]
_PIXEL_SIZE_UNIT_ALIASES = [
    "pixel_size_unit", "pixel_size_units", "spatial_units",
]


def _ci_pop(d: dict, aliases: list[str]):
    """Case-insensitive pop: find the first matching alias in *d* and remove it."""
    lower_map = {k.lower(): k for k in d}
    for alias in aliases:
        real_key = lower_map.get(alias.lower())
        if real_key is not None and d[real_key] is not None:
            return d.pop(real_key)
    return None


def _describes_spectral_imaging(metadata: dict) -> bool:
    """True when the metadata describes a spectral-imaging / hyperspectral
    datacube — a dataset whose defining third axis is an energy/spectral axis.

    For such data the energy axis is *required* (downstream builds the channel
    axis from it); for 1D curves or plain microscopy it is optional. Heuristic
    on the free-text ``experiment_type`` + ``experiment.technique`` so a dict
    that only names the technique in prose still routes correctly.
    """
    exp = metadata.get("experiment")
    technique = exp.get("technique", "") if isinstance(exp, dict) else ""
    blob = f"{metadata.get('experiment_type') or ''} {technique}".lower()
    if "hyperspectral" in blob or "datacube" in blob:
        return True
    # "spectral/spectroscopy ... imaging/mapping/spectrum-image/cube"
    return ("spectr" in blob
            and any(w in blob for w in ("imag", "map", "cube", "spectrum image")))


def _has_resolvable_energy_axis(metadata: dict) -> bool:
    """True when an energy/signal axis with both endpoints can be resolved —
    either legacy ``energy_range`` or ``axis_spec.axis_2`` carries start+end."""
    er = metadata.get("energy_range")
    if isinstance(er, dict) and er.get("start") is not None and er.get("end") is not None:
        return True
    ax = metadata.get("axis_spec")
    if isinstance(ax, dict):
        a2 = ax.get("axis_2")
        if isinstance(a2, dict) and a2.get("start") is not None and a2.get("end") is not None:
            return True
    return False


def check_schema_conformance(metadata: dict) -> tuple[bool, list[str]]:
    """Check whether *metadata* conforms to the canonical schema.

    Returns ``(is_conformant, issues)`` where *issues* is a list of
    human-readable strings describing what is missing or wrong.
    Fast-path: returns ``(True, [])`` for already-conformant dicts.
    """
    issues: list[str] = []

    # Required top-level keys
    for key in ("experiment_type", "experiment", "sample"):
        if key not in metadata:
            issues.append(f"Missing required top-level key: '{key}'")

    # Nested required keys
    exp = metadata.get("experiment")
    if isinstance(exp, dict):
        if "technique" not in exp:
            issues.append("Missing 'experiment.technique'")
    elif exp is not None:
        issues.append("'experiment' should be a dict")

    sample = metadata.get("sample")
    if isinstance(sample, dict):
        if "material" not in sample:
            issues.append("Missing 'sample.material'")
    elif sample is not None:
        issues.append("'sample' should be a dict")

    # Type checks for optional nested objects
    for key in ("energy_range", "spatial_info"):
        val = metadata.get(key)
        if val is not None and not isinstance(val, dict):
            issues.append(f"'{key}' should be a dict or null")

    # A spectral-imaging / hyperspectral datacube MUST carry a resolvable energy
    # axis (energy_range.start/end or axis_spec.axis_2.start/end). Without it the
    # axis silently falls back to channel indices, so the spectral axis is only
    # in prose — flag it non-conformant so the LLM normalizer re-extracts it.
    if _describes_spectral_imaging(metadata) and not _has_resolvable_energy_axis(metadata):
        issues.append(
            "Spectral-imaging/hyperspectral metadata is missing a resolvable "
            "energy axis (energy_range.start/end or axis_spec.axis_2.start/end)"
        )

    return (len(issues) == 0, issues)


def normalize_metadata_dict(metadata: dict) -> tuple[dict, bool]:
    """Tier 1 deterministic normalizer — map common aliases to canonical form.

    Returns ``(normalized_dict, was_modified)``.  If no changes were needed
    the *same* dict object is returned with ``was_modified=False``.
    All unrecognized keys are preserved at the top level.
    """
    # Fast-path: already conformant → no-op
    is_ok, _ = check_schema_conformance(metadata)
    if is_ok:
        return metadata, False

    # Work on a shallow copy so the original is not mutated
    d = dict(metadata)
    modified = False

    # --- experiment_type ---
    if "experiment_type" not in d:
        val = _ci_pop(d, _EXPERIMENT_TYPE_ALIASES)
        if val is not None:
            d["experiment_type"] = val
            modified = True

    # --- experiment.technique ---
    exp = d.get("experiment")
    if not isinstance(exp, dict):
        exp = {}
    if "technique" not in exp:
        val = _ci_pop(d, _TECHNIQUE_ALIASES)
        if val is not None:
            exp = dict(exp)  # copy if we're modifying
            exp["technique"] = val
            d["experiment"] = exp
            modified = True
    if exp and "experiment" not in d:
        d["experiment"] = exp
        modified = True

    # --- sample.material ---
    sample = d.get("sample")
    if not isinstance(sample, dict):
        sample = {}
    if "material" not in sample:
        val = _ci_pop(d, _MATERIAL_ALIASES)
        if val is not None:
            sample = dict(sample)
            sample["material"] = val
            d["sample"] = sample
            modified = True
    if sample and "sample" not in d:
        d["sample"] = sample
        modified = True

    # --- energy_range ---
    er = d.get("energy_range")
    if not isinstance(er, dict):
        er = {}
    er_modified = False
    if "start" not in er:
        val = _ci_pop(d, _ENERGY_START_ALIASES)
        if val is not None:
            er["start"] = val
            er_modified = True
    if "end" not in er:
        val = _ci_pop(d, _ENERGY_END_ALIASES)
        if val is not None:
            er["end"] = val
            er_modified = True
    if "units" not in er:
        val = _ci_pop(d, _ENERGY_UNITS_ALIASES)
        if val is not None:
            er["units"] = val
            er_modified = True
    if er_modified:
        d["energy_range"] = er
        modified = True

    # --- spatial_info ---
    si = d.get("spatial_info")
    if not isinstance(si, dict):
        si = {}
    si_modified = False

    # field_of_view → field_of_view_x (scalar fov treated as x)
    if "field_of_view_x" not in si:
        val = _ci_pop(d, _SPATIAL_FOV_ALIASES)
        if val is not None:
            if isinstance(val, dict):
                # e.g. {"x": 100, "y": 100, "units": "nm"}
                si.update({
                    "field_of_view_x": val.get("x"),
                    "field_of_view_y": val.get("y"),
                    "field_of_view_units": val.get("units"),
                })
            else:
                si["field_of_view_x"] = val
            si_modified = True

    # pixel_size → nm_per_pixel at top level (kept for spatial_info enrichment)
    ps_val = _ci_pop(d, _PIXEL_SIZE_ALIASES)
    if ps_val is not None:
        si["nm_per_pixel"] = ps_val
        ps_unit = _ci_pop(d, _PIXEL_SIZE_UNIT_ALIASES)
        if ps_unit is not None:
            si["pixel_size_unit"] = ps_unit
        si_modified = True

    if si_modified:
        d["spatial_info"] = si
        modified = True

    return (d, True) if modified else (metadata, False)


# =====================================================================
# Axis spec resolution (used by hyperspectral pipeline)
# =====================================================================
# The hyperspectral analysis agent traditionally assumed a (x, y, energy)
# layout. axis_spec generalizes that to 3D datasets where the axes can be
# anything (e.g. (force, distance, frequency)). resolve_axis_spec is the
# single point of truth: every downstream consumer reads through it so
# legacy energy-style metadata and new generic metadata coexist.

_AXIS_SPATIAL_DEFAULTS = [
    {"name": "x", "units": "pixel", "kind": "spatial"},
    {"name": "y", "units": "pixel", "kind": "spatial"},
]
_AXIS_SIGNAL_DEFAULT = {"name": "energy", "units": "eV", "kind": "signal"}


def resolve_axis_spec(system_info: dict | None) -> dict:
    """Return a fully-populated axis_spec for a 3D hyperspectral dataset.

    If ``system_info['axis_spec']`` is present, missing per-axis fields are
    filled with defaults. Otherwise the spec is synthesized from legacy
    ``spatial_info`` and ``energy_range`` so that an existing (x, y, energy)
    call site sees identical behavior.

    The returned dict always has keys ``axis_0``, ``axis_1``, ``axis_2``
    (each an object with at least ``name``, ``units``, ``kind``) plus
    ``signal_is_nonnegative`` (bool). The signal-like axis additionally
    carries ``start`` and ``end`` when known; downstream callers raise if
    they need those values and they are absent — same behavior as today.
    """
    system_info = system_info or {}
    provided = system_info.get("axis_spec") or {}

    spec: dict = {}

    # axis_0, axis_1 — default to spatial; legacy spatial_info supplies units
    spatial_info = system_info.get("spatial_info") or {}
    spatial_units = spatial_info.get("field_of_view_units") or _AXIS_SPATIAL_DEFAULTS[0]["units"]
    for idx in (0, 1):
        key = f"axis_{idx}"
        user_axis = provided.get(key) or {}
        default = dict(_AXIS_SPATIAL_DEFAULTS[idx])
        if spatial_units:
            default["units"] = spatial_units
        spec[key] = {
            "name": user_axis.get("name", default["name"]),
            "units": user_axis.get("units", default["units"]),
            "kind": user_axis.get("kind", default["kind"]),
        }

    # axis_2 — default to signal (energy). Legacy energy_range supplies
    # start/end/units when axis_spec.axis_2 doesn't.
    user_axis_2 = provided.get("axis_2") or {}
    energy_range = system_info.get("energy_range") or {}
    axis_2 = {
        "name": user_axis_2.get("name", _AXIS_SIGNAL_DEFAULT["name"]),
        "units": user_axis_2.get("units", energy_range.get("units") or _AXIS_SIGNAL_DEFAULT["units"]),
        "kind": user_axis_2.get("kind", _AXIS_SIGNAL_DEFAULT["kind"]),
    }
    start = user_axis_2.get("start", energy_range.get("start"))
    end = user_axis_2.get("end", energy_range.get("end"))
    if start is not None:
        axis_2["start"] = start
    if end is not None:
        axis_2["end"] = end
    spec["axis_2"] = axis_2

    # Signal sign assumption — default True (counts-like) for backward compat
    nonneg = provided.get("signal_is_nonnegative")
    if nonneg is None:
        nonneg = True
    spec["signal_is_nonnegative"] = bool(nonneg)

    return spec


def describe_axes_for_prompt(shape, axis_spec: dict) -> str:
    """Render a short human-friendly axes description for LLM prompts.

    For the legacy (x_spatial, y_spatial, energy) layout this returns the
    historical wording ("{h}x{w} spatial pixels, {e} spectral channels") so
    existing prompts are unchanged. For generic layouts it switches to an
    axis-name-driven phrasing.
    """
    n0, n1, n2 = shape
    a0, a1, a2 = axis_spec["axis_0"], axis_spec["axis_1"], axis_spec["axis_2"]
    legacy_layout = (
        a0.get("kind") == "spatial"
        and a1.get("kind") == "spatial"
        and a2.get("kind") == "signal"
        and a2.get("name") == "energy"
    )
    if legacy_layout:
        return f"{n0}x{n1} spatial pixels, {n2} spectral channels"
    return (
        f"{n0}x{n1} samples along ({a0['name']}, {a1['name']}), "
        f"{n2} channels along {a2['name']}"
    )


def _run_metadata_llm(text: str, model) -> dict | None:
    """Shared LLM call + JSON parsing for metadata generation.

    Sends *text* through the ``METADATA_GENERATION_PROMPT`` using *model*
    and returns the parsed dict, or ``None`` on failure.
    """
    prompt_parts = [
        METADATA_GENERATION_PROMPT,
        "\n--- Plain Text Description ---",
        text,
        "\n--- Extracted JSON Metadata ---",
    ]

    try:
        response = model.generate_content(contents=prompt_parts)

        if not hasattr(response, "text"):
            if hasattr(response, "candidates"):
                raw_text = response.candidates[0].content.parts[0].text
            else:
                raw_text = str(response)
        else:
            raw_text = response.text

        # Clean markdown fences
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]

        return json.loads(raw_text.strip())

    except Exception as e:
        logger.error(f"Error in LLM metadata generation: {e}", exc_info=True)
        return None


def normalize_metadata_dict_with_llm(
    metadata: dict,
    model,
    ext_logger=None,
) -> dict | None:
    """Tier 2 LLM normalizer — serialize dict as JSON text and run through LLM.

    Reuses the existing ``METADATA_GENERATION_PROMPT``.  Returns the
    normalized dict or ``None`` on failure.
    """
    log = ext_logger or logger
    text = json.dumps(metadata, indent=2, default=str)
    log.info("Running LLM-based metadata normalization")
    return _run_metadata_llm(text, model)


def generate_metadata_json_from_text(
    input_text_filepath: str,           
    api_key: str | None = None,
    model_name: str = "claude-opus-4-6",
    base_url: str | None = None,
    # Deprecated
    google_api_key: str | None = None,
    local_model: str | None = None
) -> dict | None:
    """
    Reads a plain text experimental description and converts it to metadata JSON.
    """
    
    # 1. Normalize Parameters
    api_key, base_url = normalize_params(
        api_key=api_key,
        google_api_key=google_api_key,
        base_url=base_url,
        local_model=local_model,
        source="generate_metadata_json_from_text"
    )

    # 2. Input Validation
    if not input_text_filepath or not os.path.exists(input_text_filepath):
        logger.error(f"Input text file not found: {input_text_filepath}")
        return None
    try:
        with open(input_text_filepath, 'r', encoding='utf-8') as f:
            text_description = f.read()
        if not text_description.strip(): return None
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return None

    base_name = os.path.splitext(input_text_filepath)[0]
    output_json_filepath = f"{base_name}.json"

    # 3. Model Initialization
    model = None
    
    if base_url:
        if 'gguf' in base_url:
             # Support GGUF here for completeness
             from scilink.wrappers.llama_wrapper import LocalLlamaModel
             model = LocalLlamaModel(base_url)
        else:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                # Fallback or error based on preference, here we log warning
                logger.warning("No API key found for proxy, attempting connection anyway...")

            logger.info(f"🏛️ Using OpenAI-compatible agent: {base_url}")
            model = OpenAIAsGenerativeModel(model=model_name, api_key=api_key, base_url=base_url)
    else:
        logger.info(f"☁️ Using LiteLLM agent: {model_name}")
        model = LiteLLMGenerativeModel(model=model_name, api_key=api_key)

    # 4. Generate via shared helper
    metadata_dict = _run_metadata_llm(text_description, model)

    if metadata_dict is None:
        return None

    # 5. Save
    try:
        with open(output_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=4)
        logger.info(f"Saved metadata to: {output_json_filepath}")
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}", exc_info=True)

    return metadata_dict