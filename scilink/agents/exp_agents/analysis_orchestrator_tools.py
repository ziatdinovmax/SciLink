"""
Tool definitions and schemas for the AnalysisOrchestratorAgent.
Supports both OpenAI (JSON schemas) and LiteLLM formats.

Each analysis run creates a unique output directory to ensure traceability
and prevent output collisions when analyzing multiple datasets.

Per-file JSON sidecar metadata
------------------------------
When a data directory contains JSON files whose stems match data files
(e.g. ``spec_5K.csv`` ↔ ``spec_5K.json``), they are treated as *sidecar
metadata* rather than global metadata.  ``run_analysis`` will attempt to
extract the series control variable from the sidecars automatically via
LLM reasoning.  If extraction fails, the user is prompted and shown the
sidecar contents to help them specify the variable manually.
"""

import fnmatch
import glob
import hashlib
import json
import logging
import os
import re
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional


# Bound on the metadata echoed back into a tool result. The FULL metadata is
# always stored on the session (self._replace_metadata) and on disk — only
# the conversational echo is bounded. Observed live: load_metadata pointed at
# a large results JSON echoed 7.7MB into the history, overflowing the model's
# context on every later call of the session.
_METADATA_ECHO_MAX_CHARS = 20_000


def _bounded_metadata_echo(metadata):
    """The metadata object itself when small; a summary stub when huge."""
    try:
        s = json.dumps(metadata, default=str)
    except Exception:  # noqa: BLE001 - echo must never break the tool
        return metadata
    if len(s) <= _METADATA_ECHO_MAX_CHARS:
        return metadata
    keys = (", ".join(list(metadata)[:40]) if isinstance(metadata, dict)
            else type(metadata).__name__)
    return {"_truncated_echo": (
        f"metadata too large to echo in conversation ({len(s)} chars); it is "
        f"stored IN FULL for the analysis. Top-level keys: {keys}")}

from .data_preparation import detect_raw_instrument as _detect_raw_instrument
from .metadata_converter import (
    generate_metadata_json_from_text,
    METADATA_SCHEMA_DICT,
    check_schema_conformance,
    normalize_metadata_dict,
    normalize_metadata_dict_with_llm,
)
from ..lit_agents import OwlLiteratureAgent, NoveltyScorer, FittingModelLiteratureAgent
from ..lit_agents.optimize_query_for_analysis import optimize_query_for_analysis
from .recommendation_agent import RecommendationAgent
from .feature_table import write_feature_table, describe_feature_table
from ._locked_exec import CANDIDATES_DIR_NAME
from ...utils.text_io import write_text_utf8
from ...skills.loader import list_skills, list_all_skills, load_skill
# Note: the simulation pipeline (`run_complete_workflow`) is imported lazily
# inside `run_dft_workflow` to avoid pulling in the optional [sim] extras
# (ase, atomate2, pymatgen) on every AnalysisOrchestratorAgent instantiation.


# Full-text extraction for the read_document tool — a few documents read
# straight into the LLM context, with no embeddings / chunking / vector store
# (that is the knowledge KB's job, for large corpora).
_READ_DOC_MAX_CHARS = 200_000  # ~50k tokens; longer documents are truncated


def _extract_document_text(path: Path, ocr_model: Any = None) -> Dict[str, Any]:
    """Extract plain text from a PDF / DOCX / Markdown / text file.

    Thin wrapper over the shared ``scilink.parsers.extract_text`` (which adds
    table-aware PDF extraction); it applies read_document's character cap.
    When ``ocr_model`` is supplied, scanned/sparse PDF pages are transcribed
    via the vision-OCR fallback. Returns a dict with ``text`` plus metadata
    (page/paragraph count, ``n_chars``, ``truncated``, ``n_ocr_pages``).
    Raises ValueError for an unsupported extension; reader errors propagate
    to the caller.
    """
    from scilink.parsers import extract_text

    info = extract_text(path, ocr_model=ocr_model)
    text = info.get("text", "")
    info["truncated"] = len(text) > _READ_DOC_MAX_CHARS
    if info["truncated"]:
        text = text[:_READ_DOC_MAX_CHARS]
        info["text"] = text
        info["n_chars"] = len(text)
    return info


def _build_skill_description(agent_registry: dict = None,
                              custom_skills: dict = None) -> str:
    """Build the ``skill`` parameter description for ``run_analysis``.

    Auto-discovers built-in skill domains and inspects the agent registry
    to determine which agents accept a ``skill`` parameter.
    """
    import inspect

    parts = [
        "Domain skill name(s) or path to a custom .md skill file — a single "
        "string or a list of strings (to load several at once). DEFAULT: omit "
        "this. Every analysis agent inspects the actual data (image pixels, the "
        "fitted curve, the spectrum-image metadata) and auto-selects the "
        "relevant skill(s) itself, from a richer signal than you have here. "
        "Pass a skill ONLY when the user explicitly named a skill or technique "
        "— in this request or earlier in the conversation (this includes a "
        "user asking to use a specific custom skill they uploaded). Custom "
        "skills are auto-selected by the agent just like built-ins, so you do "
        "NOT need to pass one merely because it was uploaded. Otherwise leave "
        "it unset and let the agent select."
    ]

    # Discover which agents support skills from their analyze() signature.
    # Registry entries use lazy class loading — resolve class_path if needed.
    if agent_registry:
        supported = []
        for entry in agent_registry.values():
            cls = entry.get("class")
            if cls is None and "class_path" in entry:
                try:
                    module_path, cls_name = entry["class_path"].rsplit(".", 1)
                    import importlib
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, cls_name)
                except Exception:
                    continue
            if cls is None:
                continue
            try:
                sig = inspect.signature(cls.analyze)
                if "skill" in sig.parameters:
                    supported.append(entry["name"])
            except (ValueError, TypeError):
                continue
        if supported:
            parts.append(f"Supported by: {', '.join(supported)}.")

    # Auto-discover all built-in skill domains with descriptions.
    # Prefer the frontmatter `description` field when present; fall back
    # to the first line of the overview section.
    for domain, names in list_all_skills().items():
        if domain == "data_preparation":
            continue   # preparation skills belong to prepare_data, not run_analysis
        skill_descs = []
        for name in names:
            try:
                parsed = load_skill(name, domain=domain)
                meta = parsed.get("meta") or {}
                # Provisional (auto-distilled) skills are kept out of the
                # auto-routing menu — they stay loadable via an explicit
                # `skill=` until reviewed/promoted (see scilink memory).
                if meta.get("provisional") is True:
                    continue
                desc = meta.get("description")
                if not desc:
                    desc = parsed.get("overview", "").split("\n")[0].strip()
                # Trim trailing punctuation so the join with ". " below
                # doesn't produce ".." or ".;".
                desc = desc.rstrip(".;,") if desc else desc
                # Surface the declared measurement technique(s) explicitly so the
                # selector can technique-match instead of inferring from prose.
                techs = meta.get("technique")
                if isinstance(techs, str):
                    techs = [techs]
                tech_tag = f" [technique: {', '.join(map(str, techs))}]" if techs else ""
                body = f"{desc}" if desc else ""
                skill_descs.append(f"'{name}'{tech_tag}" + (f" — {body}" if body else ""))
            except Exception:
                skill_descs.append(f"'{name}'")
        if skill_descs:
            parts.append(f"Built-in {domain} skills: {'; '.join(skill_descs)}.")

    if custom_skills:
        parts.append(f"Custom skills: {sorted(custom_skills.keys())}.")

    parts.append(
        "When you DO pass a skill (user-requested or custom), choose only one "
        "whose measurement technique matches the data's technique — compare "
        "against each skill's `[technique: …]` tag above; do not substitute the "
        "nearest-sounding skill. Curve-fitting techniques are mutually "
        "exclusive — pass at most one. If unsure, omit `skill` and let the "
        "agent decide."
    )

    return " ".join(parts)


# Names that are always treated as global (directory-level) metadata files,
# never as per-file sidecars, even if their stem happens to match a data file.
_GLOBAL_METADATA_NAMES = frozenset([
    "metadata.json", "meta.json", "info.json", "experiment.json",
])

# Best-of-N defaults per agent class when the LLM passes no n_candidates.
# Both image and curve fitting fan out by default *in escalation mode* —
# attempt 0 runs alone and is fast-accepted when strong, so the extra cost is
# paid only when the first attempt is weak. Image fans out because of run-to-
# run variance; curve fitting because, with no domain skill to pin the
# technique, a weak fit has high plan variance and independent candidates can
# recover it (the agent suppresses the fan-out to 1 when a skill IS active —
# the candidates would just converge on the mandated model). Agents absent
# from this dict default to a single attempt. Deterministic policy in code
# (not prompt prose) — an explicit tool param still overrides exactly.
_DEFAULT_N_CANDIDATES_BY_AGENT_CLASS = {
    "ImageAnalysisAgent": 3,
    "CurveFittingAgent": 3,
}


def _resolve_n_candidates(agent: Any, requested: Any) -> Optional[int]:
    """Resolve the effective best-of-N count for this agent.

    Returns ``None`` when the agent's ``analyze()`` does not accept
    ``n_candidates`` (caller should skip forwarding, optionally logging);
    otherwise the explicit ``requested`` value, or the per-agent-class
    default (1 when the class is not listed).
    """
    import inspect as _inspect
    try:
        params = _inspect.signature(agent.analyze).parameters
    except (TypeError, ValueError):
        return None
    if "n_candidates" not in params:
        return None
    if requested is not None:
        return max(1, int(requested))
    return _DEFAULT_N_CANDIDATES_BY_AGENT_CLASS.get(
        type(agent).__name__, 1
    )


def _resolve_candidate_escalation(agent: Any, requested: Any) -> bool:
    """True only when the best-of-N DEFAULT applies.

    An explicit ``n_candidates`` from the user means "run exactly N" —
    escalation (attempt 0 alone, fan out only if weak) is reserved for the
    auto-injected default, and only for agents whose ``analyze()`` accepts
    ``candidate_escalation``.
    """
    if requested is not None:
        return False
    import inspect as _inspect
    try:
        params = _inspect.signature(agent.analyze).parameters
    except (TypeError, ValueError):
        return False
    return "candidate_escalation" in params


_GLOB_MAGIC = ("*", "?", "[")


def _is_glob(data_path) -> bool:
    """True when data_path is a glob PATTERN rather than an existing path.

    Strictly a fallback test: an existing file or directory (even one whose
    name literally contains a bracket) is never treated as a pattern, so
    directory / single-file inputs behave exactly as before.
    """
    if not isinstance(data_path, str):
        return False
    try:
        if Path(data_path).exists():
            return False
    except OSError:  # noqa: PERF203 - absurdly long string is not a path
        return False
    return any(c in data_path for c in _GLOB_MAGIC)


def _resolve_glob_files(pattern: str) -> tuple[list[Path], list[Path]]:
    """Expand a glob into (data_files, all_files).

    ``data_files`` are the matched non-metadata files; ``all_files`` adds the
    stem-matched JSON sidecars sitting beside them (which the pattern itself
    typically does not match, e.g. ``series_*.txt``), so sidecar-based series
    metadata keeps working for a globbed subset of a directory.
    """
    matched = sorted((Path(f) for f in glob.glob(pattern) if os.path.isfile(f)),
                     key=lambda p: p.name)
    data_files = [
        f for f in matched
        if f.suffix.lower() != ".json"
        and "metadata" not in f.name.lower()
        and f.name.lower() not in ("info.txt", "description.txt",
                                   "readme.txt", "readme.md")
    ]
    all_files = list(data_files)
    for f in data_files:
        sidecar = f.with_suffix(".json")
        if sidecar.is_file():
            all_files.append(sidecar)
    return data_files, all_files


def _dataset_key(data_path: str) -> str:
    """Canonical form of a dataset reference for ownership comparison.

    Globs keep their pattern (resolved to an absolute base) so two different
    patterns over the same folder stay distinct datasets; real paths resolve
    to their absolute form.
    """
    if _is_glob(data_path):
        p = Path(data_path)
        try:
            return str(p.parent.resolve() / p.name)
        except OSError:
            return data_path
    try:
        return str(Path(data_path).resolve())
    except OSError:
        return data_path


def _same_dataset(owner: str, requested: str) -> bool:
    """Does metadata bound to *owner* also cover *requested*?

    True when they are the same path, when *requested* drills into the
    owner directory (a member file, or a glob selecting files within it),
    or when *requested* is a file the owner glob matches. A broader or
    sibling dataset is NOT covered — reuse must be re-resolved.
    """
    owner_key = _dataset_key(owner)
    req_key = _dataset_key(requested)
    if owner_key == req_key:
        return True

    owner_is_glob = _is_glob(owner)
    req_is_glob = _is_glob(requested)

    if not owner_is_glob:
        owner_p = Path(owner_key)
        if owner_p.is_dir():
            # Member file/dir of the owner folder, or a glob over it.
            probe = Path(req_key).parent if req_is_glob else Path(req_key)
            try:
                probe.relative_to(owner_p)
                return True
            except ValueError:
                return False
        return False

    # Owner is a glob: a single file it matches belongs to its dataset.
    if not req_is_glob:
        return fnmatch.fnmatch(req_key, owner_key)
    return False


def _detect_sidecar_jsons(
    data_files: list[Path],
    all_files: list[Path],
) -> tuple[dict[str, Path], list[Path]]:
    """Identify JSON files that are stem-matched sidecars for data files.

    A JSON file is a *sidecar* when its stem matches a data file's stem
    (e.g. ``spec_5K.json`` ↔ ``spec_5K.csv``).  Files whose names are in
    ``_GLOBAL_METADATA_NAMES`` are always treated as global metadata.

    Returns
    -------
    sidecar_map : dict[str, Path]
        ``{data_filename: sidecar_Path}`` for every matched pair.
    global_jsons : list[Path]
        JSON files that are **not** sidecars (global metadata or unmatched).
    """
    data_stems = {f.stem: f for f in data_files}
    json_files = [f for f in all_files if f.suffix.lower() == ".json"]

    sidecar_map: dict[str, Path] = {}
    global_jsons: list[Path] = []

    for jf in json_files:
        if jf.name.lower() in _GLOBAL_METADATA_NAMES:
            global_jsons.append(jf)
        elif jf.stem in data_stems:
            sidecar_map[data_stems[jf.stem].name] = jf
        else:
            global_jsons.append(jf)

    return sidecar_map, global_jsons


def _parse_key_list_response(raw: str, valid_keys: list[str]) -> list[str]:
    """Extract an ordered list of key names from a (possibly noisy) LLM
    response. The LLM is asked for the control-variable field names
    comma-separated, most primary first; ``NONE`` means none qualifies.

    Returns the matched keys in response order (deduped), or ``[]``.
    """
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()

    if text.strip().strip("\"'`*. ").upper() == "NONE":
        return []

    # Ideal case: comma / newline / semicolon separated tokens.
    ordered: list[str] = []
    for token in re.split(r"[,\n;]+", text):
        tok = token.strip().strip("\"'`*. ")
        if tok in valid_keys and tok not in ordered:
            ordered.append(tok)
    if ordered:
        return ordered

    # Fallback: scan for any valid key as a whole word, in order of appearance.
    positions: list[tuple[int, str]] = []
    for key in valid_keys:
        m = re.search(rf"\b{re.escape(key)}\b", text)
        if m:
            positions.append((m.start(), key))
    positions.sort()
    return [k for _, k in positions]


def _llm_identify_control_variables(
    varying_keys: list[str],
    sidecar_data: dict[str, dict],
    model,
    logger: logging.Logger,
    experimental_context: dict | None = None,
) -> list[str]:
    """Use the LLM to identify which varying sidecar fields are genuine
    independent control variables, ordered most-primary-first.

    An experiment may deliberately vary several variables at once (a
    factorial / grid design). The LLM returns the full set, so the caller
    can record a primary axis plus secondary variables rather than being
    forced into a single pick.

    Parameters
    ----------
    varying_keys : list[str]
        Numeric sidecar keys whose values differ across files (≥ 1).
    sidecar_data : dict[str, dict]
        ``{data_filename: sidecar_dict}`` for every file.
    model
        LLM wrapper with a ``generate_content`` method.
    logger
        Logger instance.
    experimental_context : dict | None
        Optional dict with keys ``"objective"``, ``"hints"``, and/or
        ``"metadata"`` providing broader experimental context.

    Returns
    -------
    list[str]
        Control-variable key names, primary first; ``[]`` when none of the
        candidates is a genuine control variable.
    """
    # Build a summary of each candidate key and its per-file values
    candidates_summary = {}
    for key in varying_keys:
        candidates_summary[key] = {
            fname: d[key] for fname, d in sidecar_data.items()
        }

    # Assemble context lines
    context_parts = []
    if experimental_context:
        if experimental_context.get("objective"):
            context_parts.append(
                f"Analysis objective: {experimental_context['objective']}"
            )
        if experimental_context.get("hints"):
            context_parts.append(
                f"User hints: {experimental_context['hints']}"
            )
        meta = experimental_context.get("metadata")
        if isinstance(meta, dict):
            # Include high-level experiment info, not the full blob
            for k in ("experiment_type", "experiment", "sample"):
                if k in meta:
                    context_parts.append(f"{k}: {json.dumps(meta[k])}")

    context_block = "\n".join(context_parts) if context_parts else "None provided."

    keys_list = ", ".join(varying_keys)
    prompt = (
        "You are a scientific data analysis assistant. A user has a series of "
        "data files, each accompanied by a JSON sidecar containing per-file "
        "metadata. The following numeric fields change across the files.\n\n"
        "Identify which of these fields are genuine **independent control "
        "variables** — physical or experimental quantities the experimenter "
        "intentionally varied across measurements (e.g. temperature, "
        "concentration, voltage, pressure, dose, time).\n\n"
        "An experiment may deliberately vary MORE THAN ONE at once — a "
        "factorial / grid design varies several parameters across the "
        "measurements. Report ALL genuine control variables; do NOT force a "
        "single choice and do NOT answer 'uncertain' just because there are "
        "several.\n\n"
        "For in-situ, time-resolved, or kinetic experiments, elapsed time "
        "or total time IS a control variable — do not dismiss it as mere "
        "acquisition metadata. "
        "Instrument and acquisition parameters that happen to differ "
        "(e.g. laser_power, integration_time, slit_width, probe_current) "
        "are NOT control variables — exclude them. Note: 'integration_time' "
        "(detector exposure per scan) is an acquisition setting, but 'total "
        "time' or 'elapsed time' (cumulative experiment duration) is a real "
        "control variable.\n\n"
        "IDENTIFIERS are never control variables: a run / workflow / sample / "
        "file number or index varies across files but encodes no condition, and "
        "sorting by it does not order the measurements physically. When a "
        "field that encodes the experimental STATE or CONDITION is present "
        "(e.g. a state code, an applied field or position, has_transition, a "
        "stimulus level), prefer it over any identifier.\n\n"
        "It is also possible that NONE of the listed fields is a true "
        "control variable — for example the real one was set manually and "
        "not recorded in the sidecar metadata.\n\n"
        f"Experimental context:\n{context_block}\n\n"
        f"Candidate fields and their per-file values:\n"
        f"{json.dumps(candidates_summary, indent=2)}\n\n"
        "RESPONSE FORMAT: a single line, comma-separated, listing the control "
        "variable field names MOST PRIMARY FIRST (the dominant experimental "
        "axis first, the rest after). No explanations, no other text. If "
        "none of the fields is a genuine control variable, respond with "
        "exactly NONE.\n"
        f"Choose only from: {keys_list}\n"
    )

    try:
        response = model.generate_content(contents=[prompt])
        raw = (
            response.text
            if hasattr(response, "text")
            else str(response)
        ).strip()

        chosen = _parse_key_list_response(raw, varying_keys)

        if not chosen:
            logger.info(
                "LLM identified no series control variable "
                "(response: %r, candidates: %s)",
                raw,
                varying_keys,
            )
            return []

        logger.info(
            "LLM identified control variable(s) %s from candidates %s",
            chosen,
            varying_keys,
        )
        return chosen

    except Exception as exc:
        logger.warning("LLM control-variable identification failed: %s", exc)
        return []


# Keys produced by the LLM metadata normalization schema.
_CANONICAL_SCHEMA_KEYS = frozenset({
    "experiment_type", "experiment", "sample", "spatial_info",
    "energy_range", "title", "data_columns", "xlabel", "ylabel",
    "custom_processing_instruction",
})

# Keys managed by the sidecar / series extraction pipeline.
_INTERNAL_KEYS = frozenset({"per_file_metadata", "series"})


def _structure_metadata_for_save(metadata: dict) -> dict:
    """Restructure flat current_metadata into grouped sections for saving.

    The runtime ``current_metadata`` dict is kept flat so that agents can
    access keys directly (``system_info.get("title")``).  For the saved
    ``metadata_used.json`` we reorganise into three clear groups:

    * **global** — normalised experiment-level fields from the canonical
      schema (experiment, sample, title, xlabel, …).
    * **per_file_metadata** / **series** — kept at top level as-is.
    * **raw_instrument** — remaining passthrough fields from sidecar
      synthesis that were not consumed by the LLM normalisation.
    """
    if not isinstance(metadata, dict):
        return metadata

    global_section: dict = {}
    raw_section: dict = {}
    internal_section: dict = {}

    for key, value in metadata.items():
        if key in _INTERNAL_KEYS:
            internal_section[key] = value
        elif key in _CANONICAL_SCHEMA_KEYS:
            global_section[key] = value
        else:
            raw_section[key] = value

    # Build result with global first for readability
    result: dict = {}
    if global_section:
        result["global"] = global_section
    result.update(internal_section)
    if raw_section:
        result["raw_instrument"] = raw_section

    return result


def _extract_series_from_sidecars(
    sidecar_map: dict[str, Path],
    data_files: list[Path],
    logger: logging.Logger,
    model=None,
    experimental_context: dict | None = None,
) -> tuple[dict | None, dict[str, dict]]:
    """Try to auto-build series metadata from per-file sidecar JSONs.

    The algorithm:

    1. Load every sidecar; bail out if coverage is incomplete.
    2. Collect top-level numeric keys common to **all** sidecars.
    3. Keep only keys whose values **differ** across files — this naturally
       eliminates constant instrument settings (e.g. integration_time=1.0
       in every file).
    4. If *model* is provided, ask the LLM to evaluate the remaining
       candidates (even if there is only one) and decide whether any is a
       true independent control variable.  A single varying key might still
       be just an acquisition setting; the real control variable may not be
       recorded in the sidecars at all.
    5. Fall back to ``None`` (user prompt) if no model or the LLM cannot
       decide.

    Returns
    -------
    series_meta : dict | None
        ``{"variable": ..., "values": {fname: val}, "unit": ...}`` for the
        primary control variable, plus an optional ``"secondary_variables"``
        list of the same shape for any others that co-vary (grid designs).
        ``None`` when extraction is not possible.
    per_file_meta : dict[str, dict]
        Full sidecar contents keyed by data filename (always returned,
        even when ``series_meta`` is ``None``).
    """
    per_file_meta: dict[str, dict] = {}
    sidecar_data: dict[str, dict] = {}

    # 1. Load all sidecars
    for fname, jpath in sidecar_map.items():
        try:
            with open(jpath, "r") as f:
                content = json.load(f)
            if isinstance(content, dict):
                sidecar_data[fname] = content
                per_file_meta[fname] = content
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load sidecar %s: %s", jpath.name, exc)

    if not sidecar_data:
        return None, per_file_meta

    # 2. Only proceed when every data file has a sidecar
    data_fnames = {f.name for f in data_files}
    if set(sidecar_data.keys()) != data_fnames:
        logger.info(
            "Sidecar coverage incomplete: %d/%d files",
            len(sidecar_data),
            len(data_fnames),
        )
        return None, per_file_meta

    # 3. Common top-level numeric keys across all sidecars
    all_keys: set[str] | None = None
    for d in sidecar_data.values():
        numeric_keys = {
            k for k, v in d.items() if isinstance(v, (int, float))
        }
        all_keys = numeric_keys if all_keys is None else all_keys & numeric_keys

    if not all_keys:
        return None, per_file_meta

    # 4. Keep only keys whose values differ across files
    varying_keys = []
    for key in all_keys:
        vals = [d[key] for d in sidecar_data.values()]
        if len(set(vals)) > 1:
            varying_keys.append(key)

    if not varying_keys:
        return None, per_file_meta

    # 5. Ask the LLM which varying fields are genuine control variables
    #    (even a single one could be just an acquisition setting; a grid
    #    design may have several).
    if model is not None:
        print(
            f"    Varying fields in sidecars: {varying_keys}. "
            f"Asking LLM to identify the control variable(s)..."
        )
        control_vars = _llm_identify_control_variables(
            varying_keys, sidecar_data, model, logger,
            experimental_context=experimental_context,
        )
    else:
        control_vars = []

    if not control_vars:
        logger.info(
            "Could not identify a series control variable from "
            "sidecar candidates: %s",
            varying_keys,
        )
        return None, per_file_meta

    # 6. Build series metadata. The primary control variable becomes
    #    variable/values/unit (the axis used for file ordering, scouting and
    #    trend analysis); any others are recorded as secondary_variables so
    #    the analysis is told the full set of conditions that co-vary.
    def _meta_for(var: str) -> dict:
        vals = {fname: sidecar_data[fname][var] for fname in sidecar_data}
        unit = ""
        sample_sidecar = next(iter(sidecar_data.values()))
        for unit_key in (f"{var}_unit", f"{var}_units", "unit", "units"):
            if unit_key in sample_sidecar:
                unit = str(sample_sidecar[unit_key])
                break
        return {"variable": var, "values": vals, "unit": unit}

    series_meta = _meta_for(control_vars[0])
    if len(control_vars) > 1:
        series_meta["secondary_variables"] = [
            _meta_for(v) for v in control_vars[1:]
        ]
        logger.info(
            "Series control variables: primary=%s, secondary=%s",
            control_vars[0], control_vars[1:],
        )

    return series_meta, per_file_meta



def _effective_full_result(record: dict) -> dict:
    """The record's full_result with the latest literature-refined
    interpretation swapped in when one exists (refine_interpretation,
    issue #323). Returns a copy; the record is never mutated."""
    full_result = record.get("full_result") or {}
    revisions = record.get("interpretation_revisions") or []
    if revisions and full_result:
        full_result = dict(full_result)
        full_result["detailed_analysis"] = revisions[-1]["revised_analysis"]
    return full_result

class AnalysisOrchestratorTools:
    """
    Manages tool definitions, schemas, and execution for the AnalysisOrchestratorAgent.
    """

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: Reference to the parent AnalysisOrchestratorAgent
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        # Agent display dicts — populated from the orchestrator's live registry
        # so that custom agents registered via register_agent() appear here too.
        self.AGENT_NAMES: Dict[int, str] = {}
        self.AGENT_DESCRIPTIONS: Dict[int, str] = {}
        self._sync_from_registry()

        # Build function map and schemas
        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        self._register_all_tools()
        # Bounded access to this session's own action history (#462) —
        # registered after the mode's own tools so schemas append cleanly.
        from ...session_events import register_history_tools
        register_history_tools(
            self._register_tool,
            lambda: Path(self.orch.base_dir) / "events.jsonl",
        )

    def _sync_from_registry(self) -> None:
        """Rebuild AGENT_NAMES and AGENT_DESCRIPTIONS from the orchestrator's registry."""
        registry = getattr(self.orch, "_agent_registry", {})
        self.AGENT_NAMES = {aid: e["name"] for aid, e in registry.items()}
        self.AGENT_DESCRIPTIONS = {aid: e["description"] for aid, e in registry.items()}

    def _get_human_feedback_enabled(self) -> bool:
        """Get current human feedback setting from orchestrator."""
        return getattr(self.orch, '_enable_human_feedback', True)

    def _check_instruction_redundancy(self, existing: str, new: str) -> bool:
        """Ask the LLM whether two preprocessing instructions are redundant.

        Returns ``True`` if the instructions describe essentially the same
        processing operation (even if worded differently), ``False`` if they
        are genuinely distinct steps.  Falls back to ``False`` on any error
        so that the caller never blocks on a failed check.
        """
        prompt = (
            "You are a scientific data processing expert. Determine whether "
            "the following two preprocessing instructions describe the SAME "
            "operation (even if worded differently) or genuinely DIFFERENT "
            "processing steps.\n\n"
            f"Instruction A: {existing}\n\n"
            f"Instruction B: {new}\n\n"
            "If they are essentially the same operation (e.g. both describe "
            "baseline division, both describe the same normalization, etc.), "
            "respond with exactly: REDUNDANT\n"
            "If they are genuinely different processing steps, respond with "
            "exactly: DISTINCT\n\n"
            "Respond with a SINGLE word — REDUNDANT or DISTINCT."
        )
        try:
            response = self.orch.model.generate_content(contents=[prompt])
            raw = (
                response.text if hasattr(response, "text") else str(response)
            ).strip().upper()
            return "REDUNDANT" in raw
        except Exception as e:
            self.logger.warning(f"LLM redundancy check failed: {e}")
            return False

    def _replace_metadata(self, new_metadata: dict) -> dict | None:
        """Replace current_metadata, preserving any custom_processing_instruction.

        If the old metadata had a custom_processing_instruction that would be
        lost (i.e. the new metadata doesn't include one), it is carried
        forward automatically.

        Returns a warning dict if both old and new metadata contain
        *different* custom_processing_instructions (the new one wins, but
        the caller should surface the conflict).  Returns ``None`` otherwise.
        """
        _CPI = "custom_processing_instruction"
        old_instruction = None
        if self.orch.current_metadata:
            old_instruction = self.orch.current_metadata.get(_CPI)

        self.orch.current_metadata = new_metadata
        # A fresh document is unbound: it binds to the first dataset
        # run_analysis consumes it for (issue #411).
        self.orch.current_metadata_owner = None

        if not old_instruction:
            return None  # Nothing to preserve

        new_instruction = new_metadata.get(_CPI)

        if not new_instruction:
            # New metadata has no instruction — carry forward the old one
            new_metadata[_CPI] = old_instruction
            return None

        # Both have instructions — keep the new one but warn
        if old_instruction.strip() != new_instruction.strip():
            return {
                "preprocessing_warning": (
                    "New metadata contains a custom_processing_instruction that "
                    "differs from a previously set one. The new instruction is "
                    "being used. Review to ensure correctness."
                ),
                "previous_instruction": old_instruction,
                "current_instruction": new_instruction,
            }
        return None

    def _examine_hdf5(self, path: Path, result: dict) -> None:
        """Populate *result* with shape/dimensions/metadata for an
        HDF5 file.

        Uses the official ``nexusformat`` library for NeXus-conformant
        files (handles NXentry/NXdata/signal/axes per the standard,
        including older conventions).  Falls back to a generic ``h5py``
        walk for non-NeXus HDF5 so the LLM still gets useful structural
        info (group/dataset shapes, dtypes, attrs).
        """
        try:
            import h5py  # noqa: F401  (used in the fallback walk)
        except ImportError:
            result["data_type"] = "unknown"
            result["suggested_agents"] = []
            result["message"] = (
                "Reading HDF5 files requires h5py. Install with: "
                "pip install h5py"
            )
            return

        # Try NeXus parse first.
        parsed = None
        nexus_error = None
        try:
            import nexusformat.nexus as nx
            try:
                nx_root = nx.nxload(str(path), mode="r")
                parsed = self._parse_nexus(nx_root)
            except Exception as exc:
                nexus_error = str(exc)
        except ImportError:
            nexus_error = (
                "nexusformat not installed; falling back to a generic "
                "HDF5 walk. Install with: pip install nexusformat"
            )

        # Always run the generic content harvester — it surfaces auxiliary
        # metadata groups (sidpy_metadata, hyperspy metadata, …) that the
        # NeXus standards-aware parse doesn't touch, and is the bridge
        # source for ``convert_metadata`` when an h5 happens to embed
        # producer-specific metadata.  Producer-agnostic; bounded payload.
        try:
            harvested = self._harvest_dataset_contents(path)
        except Exception as exc:
            harvested = None
            result.setdefault("hdf5_walk_error", str(exc))

        if parsed is not None:
            self._apply_nexus_to_result(parsed, result)
            if harvested is not None:
                # Don't duplicate the signal/axis arrays — the NeXus parse
                # already surfaced shape/dtype/units for the primary
                # data.  Auxiliary content (everything else) is what's
                # genuinely new here.
                result["root_attrs"] = harvested["root_attrs"]
                result["hdf5_datasets"] = harvested["datasets"][:50]
                result["dataset_count"] = harvested["dataset_count"]
                if harvested["truncated"]:
                    result["hdf5_content_truncated"] = True
            return

        # Generic fallback — no NeXus structure.
        if nexus_error:
            result["nexus_parse_error"] = nexus_error
        if harvested is not None:
            self._apply_harvested_to_result(harvested, result)
        else:
            result["data_type"] = "unknown"
            result["suggested_agents"] = []
            result.setdefault("message", "Failed to read HDF5")

    @staticmethod
    def _parse_nexus(nx_root) -> "dict | None":
        """Extract NeXus-standard structure (signal, axes, dimensions)
        from an HDF5 file via ``nexusformat``'s standards-aware accessors.

        Producer-specific metadata layouts (sidpy's ``sidpy_metadata``
        group, HyperSpy's ``signal.metadata`` tree, …) are *not*
        interpreted here — bridging third-party metadata dialects to
        SciLink's canonical schema is the job of ``convert_metadata``,
        which is producer-agnostic.

        Returns ``None`` if the file has no NXentry/NXdata or no signal.
        """
        entries = list(nx_root.NXentry)
        if not entries:
            return None
        default_entry = nx_root.attrs.get("default")
        entry = next(
            (e for e in entries if e.nxname == default_entry),
            entries[0],
        )

        nxdatas = list(entry.NXdata)
        if not nxdatas:
            return None
        default_nxdata = entry.attrs.get("default")
        nxdata = next(
            (d for d in nxdatas if d.nxname == default_nxdata),
            nxdatas[0],
        )

        signal = nxdata.nxsignal
        if signal is None:
            return None

        def _attr(field, key, default=""):
            v = field.attrs.get(key, default)
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return v

        title = _attr(signal, "title") or _attr(signal, "long_name") or ""
        units = _attr(signal, "units") or ""

        try:
            axes = list(nxdata.nxaxes or [])
        except Exception:
            axes = []

        dims_info = []
        for i, dim_len in enumerate(signal.shape):
            entry_d = {"index": i, "length": int(dim_len)}
            if i < len(axes):
                ax = axes[i]
                ax_name = getattr(ax, "nxname", None)
                if ax_name:
                    entry_d["name"] = ax_name
                entry_d["units"] = _attr(ax, "units") or ""
                try:
                    vals = np.asarray(ax.nxdata)
                    if vals.size >= 2:
                        diffs = np.diff(vals)
                        if diffs.size and np.allclose(diffs, diffs[0]):
                            entry_d["start"] = float(vals[0])
                            entry_d["stop"] = float(vals[-1])
                            entry_d["step"] = float(diffs[0])
                except Exception:
                    pass
            dims_info.append(entry_d)

        return {
            "shape": list(signal.shape),
            "dtype": str(signal.dtype),
            "title": title,
            "units": units,
            "dimensions": dims_info,
        }

    @staticmethod
    def _apply_nexus_to_result(parsed: dict, result: dict) -> None:
        """Copy the parsed NeXus payload into the examine_data result and
        derive a shape-based agent suggestion."""
        shape = parsed["shape"]

        result["shape"] = shape
        result["dtype"] = parsed["dtype"]
        result["title"] = parsed["title"]
        result["units"] = parsed["units"]
        result["dimensions"] = parsed["dimensions"]

        if len(shape) == 1:
            result["data_type"] = "1d_data"
            result["suggested_agents"] = [0]
            result["primary_suggestion"] = 0
        elif len(shape) == 2:
            result["data_type"] = "image"
            result["suggested_agents"] = [1]
            result["primary_suggestion"] = 1
        elif len(shape) == 3:
            result["data_type"] = "hyperspectral"
            result["suggested_agents"] = [2]
            result["primary_suggestion"] = 2
        else:
            result["data_type"] = "nd_data"
            result["suggested_agents"] = []

        result["note"] = (
            f"NeXus dataset: shape={shape}, "
            f"{len(parsed['dimensions'])} dimension(s)"
        )

    # Per-dataset and total caps for embedded text/JSON content surfaced
    # to the LLM.  Numeric arrays are *never* included — only their shape
    # and dtype.  Object-dtype contents that don't decode cleanly as
    # UTF-8 are dropped (avoids pickle hazard and binary mojibake).
    _HDF5_PER_DATASET_CONTENT_CAP = 32 * 1024
    _HDF5_TOTAL_CONTENT_BUDGET = 128 * 1024

    @staticmethod
    def _decode_h5_value(v):
        """Best-effort decode of an h5py attribute or scalar value into a
        JSON-serialisable Python object."""
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8")
            except UnicodeDecodeError:
                return v.decode("utf-8", errors="replace")
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("S", "O"):
                return [
                    item.decode("utf-8", errors="replace")
                    if isinstance(item, bytes) else item
                    for item in v.tolist()
                ]
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    @classmethod
    def _try_decode_dataset_content(cls, raw):
        """Decode a dataset value to a JSON object or text string, or
        return ``None`` if it can't be safely decoded.

        Refuses bytes that don't decode as UTF-8 (avoids surfacing
        pickled Python objects or binary blobs to the LLM).
        """
        # Bytes scalar: try UTF-8 strictly (no replace — we want to know
        # if it's actually text).
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
        elif isinstance(raw, str):
            text = raw
        elif isinstance(raw, np.ndarray):
            if raw.dtype.kind in ("S", "O"):
                pieces = []
                items = raw.tolist() if raw.shape else [raw.item()]
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if isinstance(item, bytes):
                        try:
                            pieces.append(item.decode("utf-8"))
                        except UnicodeDecodeError:
                            return None
                    elif isinstance(item, str):
                        pieces.append(item)
                    else:
                        return None
                text = "\n".join(pieces)
            elif raw.dtype.kind == "U":
                items = raw.tolist() if raw.shape else [raw.item()]
                text = "\n".join(items if isinstance(items, list) else [items])
            else:
                return None  # numeric — caller skips
        else:
            return None

        # Try JSON parse first; if it parses, the structured form is more
        # useful to the LLM than a giant text blob.
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    @classmethod
    def _harvest_dataset_contents(cls, path: Path) -> dict:
        """Walk an HDF5 file and return its structure plus the contents
        of small text/JSON datasets, bounded by per-dataset and total
        byte budgets.

        Returns
        -------
        dict with keys: ``root_attrs``, ``datasets`` (list of descriptors,
        each with ``path/shape/dtype/attrs`` and optionally ``content``),
        ``dataset_count`` (total found), ``truncated`` (True if budget hit).
        """
        import h5py

        per_cap = cls._HDF5_PER_DATASET_CONTENT_CAP
        total_cap = cls._HDF5_TOTAL_CONTENT_BUDGET
        spent = 0
        truncated = False
        descriptors: list[dict] = []

        with h5py.File(str(path), "r") as f:
            root_attrs = {
                k: cls._decode_h5_value(v) for k, v in f.attrs.items()
            }

            def visit(name, obj):
                nonlocal spent, truncated
                if not isinstance(obj, h5py.Dataset):
                    return
                d = {
                    "path": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": {
                        k: cls._decode_h5_value(v) for k, v in obj.attrs.items()
                    },
                }

                # Only consider surfacing contents for text/object dtypes.
                # Numeric arrays are intentionally skipped (would dump
                # signal data).
                if obj.dtype.kind in ("S", "U", "O"):
                    # Estimate byte size; refuse large datasets entirely
                    # before reading them.
                    try:
                        nbytes = int(obj.nbytes) if obj.size else 0
                    except Exception:
                        nbytes = 0
                    if nbytes <= per_cap and spent < total_cap:
                        try:
                            raw = obj[()]
                            decoded = cls._try_decode_dataset_content(raw)
                        except Exception:
                            decoded = None

                        if decoded is not None:
                            # Approximate cost as JSON-encoded length so
                            # the budget tracks LLM token consumption.
                            try:
                                cost = len(
                                    json.dumps(decoded, default=str).encode("utf-8")
                                )
                            except Exception:
                                cost = per_cap  # be conservative
                            if cost <= per_cap and spent + cost <= total_cap:
                                d["content"] = decoded
                                spent += cost
                            else:
                                truncated = True

                descriptors.append(d)

            f.visititems(visit)

        return {
            "root_attrs": root_attrs,
            "datasets": descriptors,
            "dataset_count": len(descriptors),
            "truncated": truncated,
        }

    @staticmethod
    def _apply_harvested_to_result(harvested: dict, result: dict) -> None:
        """Populate the examine_data result from a generic harvest (used
        when no NeXus structure was found)."""
        datasets = harvested["datasets"]
        result["root_attrs"] = harvested["root_attrs"]
        result["hdf5_datasets"] = datasets[:50]
        result["dataset_count"] = harvested["dataset_count"]
        if harvested["truncated"]:
            result["hdf5_content_truncated"] = True
        result["note"] = (
            f"HDF5 file with {harvested['dataset_count']} dataset(s); "
            "non-NeXus or unparseable as NeXus."
        )

        # Best-effort agent suggestion from a single top-level dataset.
        if harvested["dataset_count"] == 1:
            shp = datasets[0]["shape"]
            if len(shp) == 1:
                result["data_type"] = "1d_data"
                result["suggested_agents"] = [0]
                result["primary_suggestion"] = 0
            elif len(shp) == 2:
                result["data_type"] = "image"
                result["suggested_agents"] = [1]
                result["primary_suggestion"] = 1
            elif len(shp) == 3:
                result["data_type"] = "hyperspectral"
                result["suggested_agents"] = [2]
                result["primary_suggestion"] = 2
            else:
                result["data_type"] = "nd_data"
                result["suggested_agents"] = []
        else:
            result["data_type"] = "unknown"
            result["suggested_agents"] = []

    def _register_all_tools(self):
        """Register all tools with OpenAI format."""
        
        # =====================================================================
        # 1. EXAMINE DATA
        # =====================================================================
        def examine_data(data_path: str) -> str:
            """
            Examine a data file or directory to determine its type and
            characteristics.  Supports single files and directories
            containing multiple spectra.

            For directories, JSON files whose stems match a data file
            (e.g. ``spec_5K.json`` ↔ ``spec_5K.csv``) are reported as
            per-file sidecar metadata in ``sidecar_json_files``, separate
            from global ``metadata_files``.

            ``data_path`` may also be a glob PATTERN (e.g.
            ``/data/series_*C.txt``) selecting ONE dataset out of a directory
            that holds several; it is examined exactly like a directory
            restricted to the matched files.
            """
            print(f"  ⚡ Tool: Examining data at {data_path}...")

            path = Path(data_path)
            is_glob_input = _is_glob(data_path)
            if not is_glob_input and not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File not found: {data_path}"
                })

            result = {
                "status": "success",
                "path": str(path.absolute()),
            }

            try:
                # ============================================================
                # DIRECTORY or GLOB PATTERN: Multiple files (series)
                # ============================================================
                if path.is_dir() or is_glob_input:
                    if is_glob_input:
                        _dfiles, files = _resolve_glob_files(data_path)
                        if not _dfiles:
                            return json.dumps({
                                "status": "error",
                                "message": (f"No data files match the pattern "
                                            f"'{data_path}'.")
                            })
                        result["is_pattern"] = True
                        result["pattern"] = data_path
                        result["pattern_hint"] = (
                            f"This pattern selects {len(_dfiles)} file(s) out of "
                            f"{Path(data_path).parent}. Pass the pattern itself as "
                            "`data_path` to run_analysis — NOT the parent directory, "
                            "which may hold other datasets."
                        )
                    else:
                        files = list(path.iterdir())
                        files = [f for f in files if f.is_file() and not f.name.startswith('.')]

                    result["is_directory"] = True
                    result["file_count"] = len(files)

                    if not files:
                        result["status"] = "error"
                        result["message"] = "Directory is empty"
                        return json.dumps(result)
                    
                    # Look for metadata files — distinguish sidecar JSONs from
                    # global metadata so the orchestrator knows sidecars exist.
                    non_json_stems = {
                        f.stem for f in files if f.suffix.lower() != ".json"
                    }
                    sidecar_jsons = [
                        f for f in files
                        if f.suffix.lower() == ".json"
                        and f.name.lower() not in _GLOBAL_METADATA_NAMES
                        and f.stem in non_json_stems
                    ]
                    global_meta_files = [
                        f for f in files
                        if (
                            (f.suffix.lower() == ".json" and f not in sidecar_jsons)
                            or "metadata" in f.name.lower()
                            or f.name.lower() in ["info.txt", "description.txt", "readme.txt"]
                        )
                    ]

                    if global_meta_files:
                        result["metadata_files"] = [f.name for f in global_meta_files]
                        result["metadata_hint"] = (
                            f"Found potential metadata file(s): "
                            f"{[f.name for f in global_meta_files]}"
                        )
                    if sidecar_jsons:
                        result["sidecar_json_files"] = [f.name for f in sidecar_jsons]
                        result["sidecar_hint"] = (
                            f"Found {len(sidecar_jsons)} per-file JSON sidecar(s) "
                            f"(may contain series variable values)"
                        )

                    # Get data file extensions (excluding metadata and sidecars)
                    excluded = set(global_meta_files) | set(sidecar_jsons)
                    data_files = [f for f in files if f not in excluded]
                    extensions = set(f.suffix.lower() for f in data_files)
                    result["extensions"] = list(extensions)
                    
                    # Categorize by extension
                    csv_files = [f for f in data_files if f.suffix.lower() in ['.csv', '.txt', '.tsv'] 
                                 and 'metadata' not in f.name.lower() 
                                 and f.name.lower() not in ['info.txt', 'description.txt', 'readme.txt']]
                    npy_files = [f for f in data_files if f.suffix.lower() == '.npy']
                    image_files = [f for f in data_files if f.suffix.lower() in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']]
                    
                    if csv_files:
                        result["data_type"] = "tabular_series"
                        result["series_count"] = len(csv_files)
                        result["suggested_agents"] = [0]  # CurveFitting
                        result["primary_suggestion"] = 0
                        result["data_files"] = sorted([f.name for f in csv_files[:10]])
                        if len(csv_files) > 10:
                            result["data_files"].append(f"... and {len(csv_files) - 10} more")
                        result["note"] = f"Directory contains {len(csv_files)} tabular data files (CSV/TXT) - curves, spectra, time series, etc."
                        
                    elif npy_files:
                        # Check first NPY to determine type
                        first_npy = np.load(str(npy_files[0]))
                        if first_npy.ndim == 1:
                            result["data_type"] = "tabular_series"
                            result["suggested_agents"] = [0]
                            result["primary_suggestion"] = 0
                            result["note"] = f"Directory contains {len(npy_files)} NPY files (1D data)"
                        elif first_npy.ndim == 2:
                            # Distinguish images from tabular data
                            is_image = (
                                min(first_npy.shape) >= 64
                                and max(first_npy.shape) / min(first_npy.shape) <= 4
                            )
                            if is_image:
                                result["data_type"] = "image_series"
                                result["suggested_agents"] = [1]  # ImageAnalysis
                                result["primary_suggestion"] = 1
                                result["note"] = (
                                    f"Directory contains {len(npy_files)} NPY files "
                                    f"({first_npy.shape[0]}x{first_npy.shape[1]}, "
                                    f"{first_npy.dtype}) — detected as image series"
                                )
                            else:
                                result["data_type"] = "tabular_series"
                                result["suggested_agents"] = [0]
                                result["primary_suggestion"] = 0
                                result["note"] = f"Directory contains {len(npy_files)} NPY files (2D tabular data)"
                        else:
                            result["data_type"] = "hyperspectral_series"
                            result["suggested_agents"] = [2]
                            result["primary_suggestion"] = 2
                            result["note"] = f"Directory contains {len(npy_files)} NPY files (3D datacubes)"
                        
                        result["series_count"] = len(npy_files)
                        result["data_files"] = sorted([f.name for f in npy_files[:10]])
                        if len(npy_files) > 10:
                            result["data_files"].append(f"... and {len(npy_files) - 10} more")
                        
                    elif image_files:
                        result["data_type"] = "image_series"
                        result["series_count"] = len(image_files)
                        result["suggested_agents"] = [1]  # ImageAnalysis
                        result["primary_suggestion"] = 1
                        result["data_files"] = sorted([f.name for f in image_files[:10]])
                        if len(image_files) > 10:
                            result["data_files"].append(f"... and {len(image_files) - 10} more")
                        result["note"] = f"Directory contains {len(image_files)} image files - microscopy, photos, etc."
                    
                    else:
                        result["data_type"] = "unknown"
                        result["message"] = f"Directory contains unsupported file types: {extensions}"
                    
                    # ---- raw-instrument override -------------------------------------
                    # A container whose sidecar / manifest / embedded contract says it
                    # must be reconstructed first (e.g. a raw hologram stack) is never
                    # handed to an analysis agent: route it to prepare_data instead.
                    _raw = _detect_raw_instrument(path)
                    if _raw:
                        result["data_type"] = "raw_instrument"
                        result["suggested_agents"] = []
                        result["primary_suggestion"] = None
                        result["preparation_required"] = True
                        result["next_tool"] = "prepare_data"
                        result["raw_instrument_evidence"] = _raw.get("evidence")
                        if _raw.get("manifest"):
                            result["manifest"] = _raw["manifest"]
                        result["note"] = ("Raw instrument container: it must be transformed into "
                                          "analysis-ready products with `prepare_data` (which "
                                          "applies the matching data-preparation skill) BEFORE any "
                                          "run_analysis. Do NOT pass it to an analysis agent.")
                    # Store in orchestrator state
                    self.orch.current_data_path = str(path.absolute())
                    self.orch.current_data_type = result.get("data_type")
                    
                    return json.dumps(result)
                
                # ============================================================
                # SINGLE FILE
                # ============================================================
                file_size = path.stat().st_size
                extension = path.suffix.lower()
                
                result["is_directory"] = False
                result["file_name"] = path.name
                result["file_size_bytes"] = file_size
                result["extension"] = extension
                
                # Determine data type based on extension and content
                if extension in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']:
                    result["data_type"] = "microscopy"
                    result["suggested_agents"] = [1]  # ImageAnalysis
                    
                    # Try to load and get shape
                    try:
                        from ...skills._shared.image_processor import load_image
                        img = load_image(str(path))
                        result["shape"] = list(img.shape)
                        result["dtype"] = str(img.dtype)
                        
                        # Suggest based on image characteristics
                        if len(img.shape) == 2:
                            h, w = img.shape
                        else:
                            h, w = img.shape[:2]
                        
                        result["image_size"] = f"{w}x{h}"
                        result["primary_suggestion"] = 1  # ImageAnalysis
                            
                    except Exception as e:
                        result["load_error"] = str(e)
                
                elif extension == '.npy':
                    # Could be 1D data, 2D data/image, series, or hyperspectral
                    data = np.load(str(path))
                    result["shape"] = list(data.shape)
                    result["dtype"] = str(data.dtype)
                    
                    if data.ndim == 1:
                        result["data_type"] = "1d_data"
                        result["suggested_agents"] = [0]  # CurveFitting
                        result["primary_suggestion"] = 0
                        result["n_points"] = data.shape[0]
                        result["note"] = "Single 1D array - curve, spectrum, time series, etc."
                        
                    elif data.ndim == 2:
                        # Check if it's a series (N x points) or single data (points x 2) or image
                        if data.shape[1] == 2:
                            # Single data with x,y columns
                            result["data_type"] = "1d_data"
                            result["suggested_agents"] = [0]
                            result["primary_suggestion"] = 0
                            result["n_points"] = data.shape[0]
                            result["note"] = "Single dataset with (x, y) columns"
                        elif data.shape[0] == 2:
                            # Single data with x,y rows
                            result["data_type"] = "1d_data"
                            result["suggested_agents"] = [0]
                            result["primary_suggestion"] = 0
                            result["n_points"] = data.shape[1]
                            result["note"] = "Single dataset with (x, y) rows"
                        elif data.shape[0] > 2 and data.shape[1] > 2:
                            # Could be series of 1D data OR 2D image
                            # Heuristic: if one dimension is much smaller, likely a series
                            if data.shape[0] < 100 and data.shape[1] > 100:
                                # Likely N datasets of M points each
                                result["data_type"] = "1d_series"
                                result["suggested_agents"] = [0]
                                result["primary_suggestion"] = 0
                                result["series_count"] = data.shape[0]
                                result["n_points"] = data.shape[1]
                                result["note"] = f"Series of {data.shape[0]} datasets, each with {data.shape[1]} points"
                            elif data.shape[1] < 100 and data.shape[0] > 100:
                                # Likely M points x N datasets (transposed)
                                result["data_type"] = "1d_series"
                                result["suggested_agents"] = [0]
                                result["primary_suggestion"] = 0
                                result["series_count"] = data.shape[1]
                                result["n_points"] = data.shape[0]
                                result["note"] = f"Series of {data.shape[1]} datasets, each with {data.shape[0]} points (may need transpose)"
                            elif (
                                min(data.shape) >= 64
                                and max(data.shape) / min(data.shape) <= 4
                                and data.dtype in (
                                    np.uint8, np.uint16, np.int16,
                                    np.float32, np.float64,
                                )
                            ):
                                # Large, roughly-square array — almost certainly an image
                                result["data_type"] = "image"
                                result["suggested_agents"] = [1]
                                result["primary_suggestion"] = 1
                                result["note"] = (
                                    f"2D array ({data.shape[0]}x{data.shape[1]}, "
                                    f"{data.dtype}) — detected as image"
                                )
                            else:
                                # Ambiguous - could be image or data matrix
                                # Try to infer from metadata if available
                                result["data_type"] = "2d_data_ambiguous"
                                result["suggested_agents"] = [0, 1]
                                result["primary_suggestion"] = None  # No clear suggestion
                                result["note"] = (
                                    f"Ambiguous 2D array ({data.shape[0]}x{data.shape[1]}). Could be:\n"
                                    f"  - Microscopy image → Agent 1 (ImageAnalysisAgent)\n"
                                    f"  - Series of 1D data (rows or columns) → Agent 0 (CurveFittingAgent)\n"
                                    f"  - 2D spectral slice → Agent 2 (HyperspectralAnalysisAgent)\n"
                                    f"Check metadata or ask user to clarify."
                                )
                                result["disambiguation_needed"] = True
                                result["disambiguation_questions"] = [
                                    "Is this a microscopy/image?",
                                    "Is this a matrix where each row (or column) is a separate spectrum/curve?",
                                    "What technique was used to acquire this data?"
                                ]
                            
                    elif data.ndim == 3:
                        n_channels = data.shape[2]
                        if n_channels in (2, 3, 4):
                            # Few channels: multi-channel image (2-ch AFM, RGB, RGBA)
                            result["data_type"] = "image"
                            result["suggested_agents"] = [1]  # ImageAnalysis
                            result["primary_suggestion"] = 1
                            result["channels"] = n_channels
                            result["note"] = (
                                f"3D array ({data.shape[0]}x{data.shape[1]}, "
                                f"{n_channels} channels, {data.dtype}) — "
                                f"detected as multi-channel image"
                            )
                        else:
                            # Many channels: spectral datacube
                            result["data_type"] = "hyperspectral"
                            result["suggested_agents"] = [2]  # Hyperspectral
                            result["primary_suggestion"] = 2
                            result["spatial_shape"] = list(data.shape[:2])
                            result["spectral_channels"] = n_channels
                            result["note"] = (
                                f"3D datacube: {data.shape[0]}x{data.shape[1]} "
                                f"spatial, {n_channels} channels"
                            )
                    
                    else:
                        result["data_type"] = "nd_data"
                        result["note"] = f"{data.ndim}D array - may need custom handling"
                        result["suggested_agents"] = []
                
                elif extension in ['.csv', '.txt', '.tsv']:
                    result["data_type"] = "tabular"
                    result["suggested_agents"] = [0]  # CurveFitting
                    result["primary_suggestion"] = 0

                    # Try to peek at the file and count rows
                    try:
                        import csv
                        with open(path, 'r') as f:
                            # Read first few lines for preview
                            first_lines = [f.readline().strip() for _ in range(5)]
                            result["preview"] = first_lines

                            # Count total lines (approximate row count)
                            f.seek(0)
                            row_count = sum(1 for _ in f) - 1  # Subtract header
                            result["n_points"] = row_count
                            result["note"] = f"Tabular data with ~{row_count} data points"
                    except Exception as e:
                        result["preview_error"] = str(e)

                elif extension in ['.h5', '.hdf5', '.nxs']:
                    # NeXus / SID-style HDF5 — produced by the SciFiReaders
                    # MCP server (read_scifireaders_file) or any sidpy
                    # pipeline. ``.nxs`` is the official NeXus extension;
                    # all three route through the same HDF5+NeXus reader.
                    # Surface shape, dimensions, and metadata so the LLM
                    # has enough to route to the right agent without
                    # needing the raw array inline.
                    self._examine_hdf5(path, result)

                else:
                    result["data_type"] = "unknown"
                    result["message"] = f"Unknown file extension: {extension}"
                    result["suggested_agents"] = []
                    # Hint when the user uploaded a vendor format that needs
                    # to be converted via the SciFiReaders MCP server before
                    # SciLink can examine it.
                    from ...ui.config import VENDOR_DATA_EXTENSIONS
                    if extension in VENDOR_DATA_EXTENSIONS:
                        result["hint"] = (
                            f"'{extension}' is a vendor format. If the "
                            "SciFiReaders MCP server is connected, call "
                            "read_scifireaders_file(file_path) to convert "
                            "to NeXus HDF5, then re-run examine_data on "
                            "the resulting '.nxs.h5' file to get shape, "
                            "dimensions, and metadata."
                        )
                
                # ---- raw-instrument override -------------------------------------
                # A container whose sidecar / manifest / embedded contract says it
                # must be reconstructed first (e.g. a raw hologram stack) is never
                # handed to an analysis agent: route it to prepare_data instead.
                _raw = _detect_raw_instrument(path)
                if _raw:
                    result["data_type"] = "raw_instrument"
                    result["suggested_agents"] = []
                    result["primary_suggestion"] = None
                    result["preparation_required"] = True
                    result["next_tool"] = "prepare_data"
                    result["raw_instrument_evidence"] = _raw.get("evidence")
                    if _raw.get("manifest"):
                        result["manifest"] = _raw["manifest"]
                    result["note"] = ("Raw instrument container: it must be transformed into "
                                      "analysis-ready products with `prepare_data` (which "
                                      "applies the matching data-preparation skill) BEFORE any "
                                      "run_analysis. Do NOT pass it to an analysis agent.")
                # Store in orchestrator state
                self.orch.current_data_path = str(path.absolute())
                self.orch.current_data_type = result.get("data_type")
                
                return json.dumps(result)
                
            except Exception as e:
                self.logger.error(f"Error examining data: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=examine_data,
            name="examine_data",
            description=(
                "Examine a data file to determine its type and characteristics. "
                "Returns data type, shape, and suggested analysis agents. "
                "For directories, also detects per-file JSON sidecar metadata "
                "(stem-matched to data files) and reports them separately from "
                "global metadata files."
                " Reports data_type='raw_instrument' (preparation_required=true) for a raw "
                "container that must go through prepare_data before any analysis."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": (
                        "Path to the data file or directory to examine. May also "
                        "be a glob PATTERN (e.g. '/data/series_*C.txt') selecting "
                        "one dataset out of a directory holding several."
                    )
                }
            },
            required=["data_path"]
        )
        
        # =====================================================================
        # 2. CONVERT METADATA
        # =====================================================================
        def convert_metadata(
            text_input: str = None,
            text_file_path: str = None
        ) -> str:
            """
            Convert natural language description to structured metadata JSON.
            """
            print(f"  ⚡ Tool: Converting metadata...")
            
            if text_file_path:
                path = Path(text_file_path)
                if not path.exists():
                    return json.dumps({
                        "status": "error",
                        "message": f"File not found: {text_file_path}"
                    })
                
                # Use the metadata converter
                try:
                    metadata = generate_metadata_json_from_text(
                        input_text_filepath=str(path),
                        api_key=self.orch.api_key,
                        model_name=self.orch.model_name,
                        base_url=self.orch.base_url
                    )
                    
                    if metadata:
                        cpi_warning = self._replace_metadata(metadata)
                        output_path = self.orch.base_dir / "metadata.json"
                        with open(output_path, 'w', encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)

                        result = {
                            "status": "success",
                            "metadata": _bounded_metadata_echo(metadata),
                            "saved_to": str(output_path)
                        }
                        if cpi_warning:
                            result.update(cpi_warning)
                        return json.dumps(result)
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": "Failed to convert metadata"
                        })

                except Exception as e:
                    self.logger.error(f"Metadata conversion error: {e}", exc_info=True)
                    return json.dumps({
                        "status": "error",
                        "message": str(e)
                    })

            elif text_input:
                # Create temporary file and convert
                temp_path = self.orch.base_dir / "temp_metadata_input.txt"
                with open(temp_path, 'w', encoding="utf-8") as f:
                    f.write(text_input)
                
                try:
                    metadata = generate_metadata_json_from_text(
                        input_text_filepath=str(temp_path),
                        api_key=self.orch.api_key,
                        model_name=self.orch.model_name,
                        base_url=self.orch.base_url
                    )
                    
                    # Clean up temp file
                    temp_path.unlink()
                    
                    if metadata:
                        cpi_warning = self._replace_metadata(metadata)
                        output_path = self.orch.base_dir / "metadata.json"
                        with open(output_path, 'w', encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)

                        result = {
                            "status": "success",
                            "metadata": _bounded_metadata_echo(metadata),
                            "saved_to": str(output_path)
                        }
                        if cpi_warning:
                            result.update(cpi_warning)
                        return json.dumps(result)
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": "Failed to convert metadata"
                        })

                except Exception as e:
                    if temp_path.exists():
                        temp_path.unlink()
                    return json.dumps({
                        "status": "error",
                        "message": str(e)
                    })

            else:
                return json.dumps({
                    "status": "error",
                    "message": "Must provide either text_input or text_file_path"
                })
        
        # ------------------------------------------------------------------
        # prepare_data — raw instrument container -> analysis-ready products
        # ------------------------------------------------------------------
        def prepare_data(data_path: str = None, task: str = None, skill=None,
                         max_attempts: int = 3, timeout_s: int = 1800,
                         llm_verify: bool = True) -> str:
            """Transform a raw instrument container into analysis-ready products."""
            from ...executors import ScriptExecutor, require_sandbox_approval
            from ...skills.loader import load_skill, list_skills
            from ...skills._shared._registry import format_tool_inventory
            from .data_preparation import (build_prep_context, run_preparation,
                                            DOMAIN as _PREP_DOMAIN)
            data_path = data_path or self.orch.current_data_path
            if not data_path:
                return json.dumps({"status": "error",
                                   "message": "No data_path given and none examined yet."})
            path = Path(data_path)
            if not path.exists():
                return json.dumps({"status": "error", "message": f"Path not found: {data_path}"})
            print(f"  🧰 Tool: Preparing raw data: {path.name}")
            if not require_sandbox_approval(
                    context="Data preparation (generated reconstruction / reduction script)"):
                return json.dumps({"status": "aborted", "reason": "sandbox_declined",
                                   "message": "Code execution declined; the raw data cannot be prepared."})
            detection = _detect_raw_instrument(path)
            task = task or ("Prepare this raw instrument data for analysis: reconstruct / "
                            "reduce it into analysis-ready products with sidecars and QC.")
            # --- skill: explicit (name or custom .md path) or auto-selected ---
            skill_names = [skill] if isinstance(skill, str) else list(skill or [])
            available = list_skills(domain=_PREP_DOMAIN)
            context = build_prep_context(path, detection)
            if not skill_names and available:
                try:
                    from ...skills._shared._skill_selector import select_relevant_skills
                    from ...skills._shared._graduation import parse_json_response as _pjr
                    _parse = lambda resp: (_pjr(getattr(resp, "raw_text", None) or getattr(resp, "text", None) or str(resp)), None)  # noqa: E731
                    skill_names = select_relevant_skills(
                        model=self._internal_model(), parse_fn=_parse, domain=_PREP_DOMAIN,
                        context_parts=[f"TASK: {task}", context[:6000]],
                        max_skills=1, exclusive=True,
                        custom_skills=self.orch._custom_skills or None, logger=logging)
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"prepare_data: skill auto-selection skipped: {e}")
                    skill_names = []
            loaded = None
            for name in skill_names:
                ref = self.orch._custom_skills.get(name, name) if self.orch._custom_skills else name
                try:
                    loaded = load_skill(ref, domain=_PREP_DOMAIN); break
                except Exception as e:  # noqa: BLE001
                    logging.warning(f"prepare_data: could not load skill '{name}': {e}")
            active = [loaded["name"]] if loaded else []
            inventory = format_tool_inventory(_PREP_DOMAIN, active_skills=active)
            if loaded:
                print(f"     Using preparation skill: {loaded['name']}")
            # --- output dir (results/prepare_<stem>_<ts>_<n>) ---
            stem = re.sub(r"[^\w\-]", "_", path.name if path.is_dir() else path.stem)[:30]
            self.orch._analysis_run_counter += 1
            prep_id = f"{stem}_prepare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.orch._analysis_run_counter:03d}"
            out_dir = self.orch.results_dir / f"prepare_{prep_id}"
            scratch = out_dir / "_scratch"
            executor = ScriptExecutor(timeout=int(timeout_s))
            res = run_preparation(
                model=self._internal_model(), executor=executor, data_path=str(path),
                task=task, out_dir=out_dir, scratch_dir=scratch, context=context,
                skill=loaded, tool_inventory=inventory, logger=logging,
                max_attempts=int(max_attempts), llm_verify=bool(llm_verify),
                parse_json=getattr(self.orch, "_parse_json_response", None))
            record = {"analysis_id": prep_id, "timestamp": datetime.now().isoformat(),
                      "data_path": str(path), "agent_id": "prepare", "agent_name": "DataPreparation",
                      "status": res.get("status"), "output_directory": str(out_dir),
                      "full_result": res, "novelty_assessment": None}
            self.orch.analysis_results.append(record)
            if res.get("status") != "success":
                return json.dumps({"status": "error", "prepare_id": prep_id,
                                   "output_directory": str(out_dir),
                                   **{k: v for k, v in res.items() if k != "status"}}, default=str)
            products = res["products"]
            files = [p["path"] for p in products]
            groups = {}
            for p in products:
                groups.setdefault(p.get("group") or p.get("kind") or "other", []).append(p["path"])
            analysis_groups = {g: paths for g, paths in groups.items()
                               if any(Path(x).suffix.lower() in (".npy", ".csv") for x in paths)
                               and not all(str(x).endswith("wrapped_phase.npy") for x in paths)}
            return json.dumps({
                "product_groups": {g: len(v) for g, v in groups.items()},
                "analysis_groups": analysis_groups,
                "status": "success", "prepare_id": prep_id, "output_directory": str(out_dir),
                "skill_used": loaded["name"] if loaded else None, "attempts": res["attempts"],
                "summary": res.get("summary"), "qc": res.get("qc"),
                "products": products, "files_produced": files,
                "receipt": res.get("receipt"), "script_path": res.get("script_path"),
                "next_steps": ("The products are ordinary analysis inputs with same-stem JSON "
                               "sidecars. Each ANALYSIS GROUP answers a different question and "
                               "gets its OWN run_analysis: a time-series/curve group (WHEN and HOW "
                               "MUCH — series via their directory or glob) and a map/image group "
                               "(WHERE — spatial structure, localization, reversal between "
                               "sibling maps). Analyze every group before concluding; do not stop "
                               "after one. Load each sidecar with load_metadata. Do not analyze "
                               "the raw container again."),
            }, default=str)

        self._register_tool(
            func=prepare_data,
            name="prepare_data",
            description=(
                "Transform a RAW INSTRUMENT container into analysis-ready products BEFORE "
                "analysis — e.g. reconstruct a raw off-axis hologram / interferogram stack "
                "(HDF5, frame x y x) into phase maps and phase-vs-time curves, or reduce a "
                "raw detector container that its sidecar/manifest says must not be routed "
                "to an image or spectral agent. A data-preparation skill (auto-selected, or "
                "given by name / custom .md path) supplies the recipe, its QC gate and its "
                "helper tools; a script is generated, run in the sandbox, and verified "
                "(products exist with sidecars, QC gate passed, skill validation rules met). "
                "Products land in results/prepare_<id>/ with same-stem JSON sidecars; then "
                "run examine_data / run_analysis on THEM. Use when examine_data reports "
                "data_type='raw_instrument' (preparation_required=true), or when the user "
                "says the data must be reconstructed / reduced first. Long-running."
            ),
            parameters={
                "data_path": {"type": "string",
                              "description": "Raw file or bundle directory (default: the path last examined). For a bundle, pass the directory that holds the manifest / raw files."},
                "task": {"type": "string",
                         "description": "What to prepare and for which downstream question — which runs, which products (maps, time traces), condition columns to join, QC to enforce. Quote the user's goal."},
                "skill": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                          "description": "Preparation skill name (e.g. 'mmzi_hologram_reconstruction') or a custom .md path. Omit to auto-select from the container's evidence."},
                "max_attempts": {"type": "integer", "description": "Generate-run-verify attempts before giving up (default 3)."},
                "timeout_s": {"type": "integer", "description": "Wall-clock limit for one script execution in seconds (default 1800; raise for many large files)."},
                "llm_verify": {"type": "boolean", "description": "Also verify against the skill's validation rules with the model (default true)."},
            },
            required=[],
        )

        self._register_tool(
            func=convert_metadata,
            name="convert_metadata",
            description=(
                "Convert natural language description to structured metadata JSON. "
                "Accepts either direct text input or a path to a text file. "
                "Use this when user provides experimental description in plain text."
            ),
            parameters={
                "text_input": {
                    "type": "string",
                    "description": "Direct text description of the experiment (alternative to file)"
                },
                "text_file_path": {
                    "type": "string",
                    "description": "Path to a .txt file containing experiment description"
                }
            },
            required=[]
        )
        
        # =====================================================================
        # 3. LOAD METADATA
        # =====================================================================
        def load_metadata(json_path: str) -> str:
            """
            Load existing JSON metadata file.

            Can accept either a direct path to a JSON file, or a directory
            path (will search for metadata.json or similar files in the
            directory).  Per-file sidecar JSONs (whose stem matches a data
            file) are excluded from the search so they are not mistakenly
            loaded as global metadata.
            """
            print(f"  ⚡ Tool: Loading metadata from {json_path}...")
            
            path = Path(json_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File/directory not found: {json_path}"
                })
            
            # If directory, search for metadata file
            if path.is_dir():
                # Look for common metadata file names
                metadata_candidates = [
                    path / "metadata.json",
                    path / "meta.json",
                    path / "info.json",
                    path / "experiment.json",
                ]
                
                # Also look for any .json file, but exclude sidecar JSONs
                # (files whose stem matches a data file, e.g. spec_5K.json ↔ spec_5K.csv)
                json_files = list(path.glob("*.json"))
                _data_exts = {
                    ".csv", ".txt", ".tsv", ".xlsx",
                    ".npy", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
                }
                _data_stems = {
                    f.stem
                    for f in path.iterdir()
                    if f.is_file() and f.suffix.lower() in _data_exts
                }
                non_sidecar_jsons = [
                    jf for jf in json_files if jf.stem not in _data_stems
                ]

                # Find the first existing metadata file
                metadata_path = None
                for candidate in metadata_candidates:
                    if candidate.exists():
                        metadata_path = candidate
                        break

                # If no standard name found, use first non-sidecar .json file
                if metadata_path is None and non_sidecar_jsons:
                    metadata_path = non_sidecar_jsons[0]
                
                if metadata_path is None:
                    # ---------------------------------------------------------
                    # Synthesize global metadata from sidecar JSONs
                    # ---------------------------------------------------------
                    # When there is no dedicated metadata file but per-file
                    # sidecars exist, extract fields that are identical across
                    # ALL sidecars as shared (global) metadata.  This lets
                    # users skip writing a separate metadata.json when the
                    # sidecars already contain experiment/sample information.
                    sidecar_paths = [
                        jf for jf in json_files if jf.stem in _data_stems
                    ]
                    if sidecar_paths:
                        try:
                            all_sidecar_dicts = []
                            for sp in sidecar_paths:
                                with open(sp, "r") as _f:
                                    all_sidecar_dicts.append(json.load(_f))

                            if all_sidecar_dicts:
                                # Collect keys shared by every sidecar
                                shared_keys = set(all_sidecar_dicts[0].keys())
                                for sd in all_sidecar_dicts[1:]:
                                    shared_keys &= sd.keys()

                                # Keep only fields whose value is the same in
                                # every sidecar (these describe the experiment,
                                # not the varying control variable).
                                synthesized: dict = {}
                                for key in shared_keys:
                                    values = [sd[key] for sd in all_sidecar_dicts]
                                    ref = values[0]
                                    if all(v == ref for v in values):
                                        synthesized[key] = ref

                                if synthesized:
                                    # Normalize to canonical schema
                                    is_conformant, _ = check_schema_conformance(synthesized)
                                    if not is_conformant:
                                        normed, _ = normalize_metadata_dict(synthesized)
                                        re_ok, _ = check_schema_conformance(normed)
                                        if not re_ok:
                                            try:
                                                llm_result = normalize_metadata_dict_with_llm(
                                                    synthesized, self.orch.model, self.logger
                                                )
                                                if llm_result:
                                                    for k, v in synthesized.items():
                                                        if k not in llm_result:
                                                            llm_result[k] = v
                                                    synthesized = llm_result
                                            except Exception:
                                                synthesized = normed
                                        else:
                                            synthesized = normed

                                    cpi_warning = self._replace_metadata(synthesized)
                                    output_path = self.orch.base_dir / "metadata.json"
                                    with open(output_path, 'w', encoding="utf-8") as f:
                                        json.dump(synthesized, f, indent=2)
                                    print(
                                        f"    Synthesized global metadata from "
                                        f"{len(sidecar_paths)} sidecar JSON(s)"
                                    )

                                    required_fields = ["experiment_type", "experiment", "sample"]
                                    missing = [f for f in required_fields if f not in synthesized]
                                    status = "warning" if missing else "success"
                                    result_payload = {
                                        "status": status,
                                        "source": "synthesized_from_sidecars",
                                        "num_sidecars": len(sidecar_paths),
                                        "metadata": synthesized,
                                        "experiment_type": synthesized.get("experiment_type"),
                                        "technique": (
                                            synthesized.get("experiment", {}).get("technique")
                                            if isinstance(synthesized.get("experiment"), dict)
                                            else synthesized.get("technique")
                                        ),
                                        "material": (
                                            synthesized.get("sample", {}).get("material")
                                            if isinstance(synthesized.get("sample"), dict)
                                            else synthesized.get("material")
                                        ),
                                    }
                                    if missing:
                                        result_payload["message"] = (
                                            f"Metadata synthesized from sidecar JSONs "
                                            f"but missing recommended fields: {missing}"
                                        )
                                    if cpi_warning:
                                        result_payload.update(cpi_warning)
                                    return json.dumps(result_payload)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to synthesize metadata from sidecars: {e}"
                            )

                    # Look for .txt description files
                    txt_candidates = [
                        path / "metadata.txt",
                        path / "description.txt",
                        path / "info.txt",
                    ]
                    for candidate in txt_candidates:
                        if candidate.exists():
                            return json.dumps({
                                "status": "info",
                                "message": f"Found text description file: {candidate.name}. Use convert_metadata to convert it to JSON.",
                                "text_file": str(candidate)
                            })

                    return json.dumps({
                        "status": "error",
                        "message": f"No metadata file found in directory: {json_path}"
                    })
                
                path = metadata_path
                print(f"    Found metadata file: {path.name}")
            
            try:
                with open(path, 'r') as f:
                    metadata = json.load(f)
                
                # Normalize metadata to canonical schema if needed
                is_conformant, issues = check_schema_conformance(metadata)
                if not is_conformant:
                    normalized, was_modified = normalize_metadata_dict(metadata)  # Tier 1
                    re_ok, _ = check_schema_conformance(normalized)
                    if not re_ok:
                        # Tier 2: LLM normalization for remaining gaps
                        try:
                            llm_result = normalize_metadata_dict_with_llm(
                                metadata, self.orch.model, self.logger
                            )
                            if llm_result:
                                # Preserve non-schema keys from the original
                                for k, v in metadata.items():
                                    if k not in llm_result:
                                        llm_result[k] = v
                                metadata = llm_result
                        except Exception as e:
                            self.logger.warning(f"LLM metadata normalization failed: {e}")
                            if was_modified:
                                metadata = normalized
                    else:
                        metadata = normalized

                # Always store metadata (possibly normalized)
                cpi_warning = self._replace_metadata(metadata)

                # Validate basic structure
                required_fields = ["experiment_type", "experiment", "sample"]
                missing = [f for f in required_fields if f not in metadata]

                if missing:
                    result = {
                        "status": "warning",
                        "message": f"Metadata loaded but missing recommended fields: {missing}",
                        "metadata_file": path.name,
                        "metadata": _bounded_metadata_echo(metadata),
                        "experiment_type": metadata.get("experiment_type"),
                        "technique": metadata.get("experiment", {}).get("technique") if isinstance(metadata.get("experiment"), dict) else metadata.get("technique"),
                        "material": metadata.get("sample", {}).get("material") if isinstance(metadata.get("sample"), dict) else metadata.get("material")
                    }
                    if cpi_warning:
                        result.update(cpi_warning)
                    return json.dumps(result)

                result = {
                    "status": "success",
                    "metadata_file": path.name,
                    "metadata": _bounded_metadata_echo(metadata),
                    "experiment_type": metadata.get("experiment_type"),
                    "technique": metadata.get("experiment", {}).get("technique"),
                    "material": metadata.get("sample", {}).get("material")
                }
                if cpi_warning:
                    result.update(cpi_warning)
                return json.dumps(result)
                
            except json.JSONDecodeError as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid JSON: {e}"
                })
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=load_metadata,
            name="load_metadata",
            description=(
                "Load experiment metadata. "
                "Can accept a direct path to a .json file OR a directory path. "
                "When given a directory it will: (1) look for a dedicated metadata file "
                "(metadata.json, meta.json, info.json, etc.), or (2) if none exists, "
                "automatically synthesize global metadata from per-file sidecar JSONs "
                "by extracting fields that are shared across all sidecars and "
                "normalizing them into the canonical schema. "
                "Use this for any directory containing metadata — whether as a "
                "single file or as per-file sidecars."
            ),
            parameters={
                "json_path": {
                    "type": "string",
                    "description": "Path to JSON metadata file OR directory containing metadata"
                }
            },
            required=["json_path"]
        )
        
        # =====================================================================
        # 4. SELECT AGENT
        # =====================================================================
        def select_agent(
            agent_id: int,
            reasoning: str = None
        ) -> str:
            """
            Set the selected analysis agent. The chat LLM decides which agent to use
            based on data type, metadata, and image preview (if applicable).
            
            Agent IDs:
                0: CurveFittingAgent - 1D curves, spectra
                1: ImageAnalysisAgent - all image types
                2: HyperspectralAnalysisAgent - spectral datacubes
            """
            print(f"  ⚡ Tool: Setting agent to {agent_id}...")
            
            if agent_id not in self.AGENT_NAMES:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid agent_id: {agent_id}. Valid IDs: {list(self.AGENT_NAMES.keys())}"
                })
            
            self.orch.selected_agent_id = agent_id
            
            return json.dumps({
                "status": "success",
                "agent_id": agent_id,
                "agent_name": self.AGENT_NAMES.get(agent_id),
                "description": self.AGENT_DESCRIPTIONS.get(agent_id),
                "reasoning": reasoning or "Selected by user/LLM"
            })
        
        self._register_tool(
            func=select_agent,
            name="select_agent",
            description=(
                "Set the analysis agent to use. Call this after examining data and metadata. "
                "Agent IDs: 0=CurveFitting (1D data), 1=ImageAnalysis (all images), 2=Hyperspectral (3D datacubes)"
            ),
            parameters={
                "agent_id": {
                    "type": "integer",
                    "description": "Agent ID to use (0=CurveFitting, 1=ImageAnalysis, 2=Hyperspectral)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this agent was chosen"
                }
            },
            required=["agent_id"]
        )
        
        # =====================================================================
        # 4b. PREVIEW IMAGE (for microscopy agent selection)
        # =====================================================================
        def preview_image(image_path: str = None) -> str:
            """
            Load and return a preview of a microscopy image for the LLM to analyze.
            Use this to visually inspect the image before analysis.
            """
            print(f"  ⚡ Tool: Loading image preview...")
            
            if image_path is None:
                image_path = self.orch.current_data_path
            
            if image_path is None:
                return json.dumps({
                    "status": "error",
                    "message": "No image path provided. Use examine_data first."
                })
            
            path = Path(image_path)
            if not path.exists():
                return json.dumps({
                    "status": "error", 
                    "message": f"File not found: {image_path}"
                })
            
            # Accept standard image formats, plus the array-container formats
            # load_image can read (.npy/.h5/.hdf5/.nxs). The latter are generic
            # containers — a .npy may hold a spectrum or datacube, not an
            # image — so the loaded array's shape is validated below.
            image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
            array_extensions = ['.npy', '.h5', '.hdf5', '.nxs']
            suffix = path.suffix.lower()
            if suffix not in image_extensions and suffix not in array_extensions:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Unsupported file type: {path.suffix}. preview_image "
                        "accepts .tif/.tiff/.png/.jpg/.jpeg/.bmp, or "
                        ".npy/.h5/.hdf5/.nxs holding a 2-D or RGB image array."
                    )
                })

            try:
                from ...skills._shared.image_processor import load_image
                import base64
                from io import BytesIO
                from PIL import Image

                # load_image handles .npy/.h5 too, and normalizes them to uint8
                img_array = load_image(str(path))

                # An array container may hold a non-image. Only a 2-D array,
                # or 3-D with 3/4 colour channels, is a previewable image.
                if img_array.ndim == 3 and img_array.shape[-1] == 1:
                    img_array = img_array[:, :, 0]   # singleton channel -> 2-D
                is_image = (
                    img_array.ndim == 2
                    or (img_array.ndim == 3 and img_array.shape[-1] in (3, 4))
                )
                if not is_image:
                    if img_array.ndim == 1:
                        guess = "a 1-D signal / spectrum"
                    elif img_array.ndim == 3:
                        guess = (f"a {img_array.shape[-1]}-channel datacube "
                                 "(e.g. hyperspectral)")
                    else:
                        guess = f"a {img_array.ndim}-D array"
                    return json.dumps({
                        "status": "error",
                        "message": (
                            f"{path.name} holds an array of shape "
                            f"{list(img_array.shape)} — not a previewable "
                            f"image; it looks like {guess}. preview_image "
                            "renders 2-D or RGB images only."
                        )
                    })

                # Get basic stats
                shape = img_array.shape
                dtype = str(img_array.dtype)

                # Convert to PIL for resizing and encoding
                pil_img = Image.fromarray(img_array)

                # Resize for preview (max 512px)
                max_dim = 512
                if max(pil_img.size) > max_dim:
                    ratio = max_dim / max(pil_img.size)
                    new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
                    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                pil_img.convert('RGB').save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return json.dumps({
                    "status": "success",
                    "image_path": str(path),
                    "shape": list(shape),
                    "dtype": dtype,
                    "preview_size": list(pil_img.size),
                    "image_base64": img_base64,
                    "guidance": (
                        "Examine this image. For any microscopy/image data, use "
                        "ImageAnalysisAgent (ID: 1)."
                    )
                })
                
            except Exception as e:
                self.logger.error(f"Image preview error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=preview_image,
            name="preview_image",
            description=(
                "Load a microscopy image preview for visual inspection. "
                "Accepts .png/.tif/.jpg/.bmp and .npy/.h5/.hdf5/.nxs array "
                "files that hold a 2-D or RGB image (a non-image array — "
                "spectrum, datacube — is rejected with an explanation). "
                "Returns the image as base64 for you to examine."
            ),
            parameters={
                "image_path": {
                    "type": "string",
                    "description": "Path to image file (uses current data path if not specified)"
                }
            },
            required=[]
        )
        
        # =====================================================================
        # 5. RUN ANALYSIS
        # =====================================================================
        def run_analysis(
            data_path: str = None,
            agent_id: int = None,
            analysis_goal: str = None,
            objective: str = None,
            hints: str = None,
            auxiliary_data = None,  # str | list[str] | None (#226 multi-aux)
            auxiliary_label = None,  # str | list[str] | None
            skill = None,  # str | list[str] | None (PR 3 multi-skill)
            skill_hint = None,  # str | list[str] | None — non-binding suggestion
            series_metadata: str = None,
            task_mode: str = None,
            prior_analysis_paths: List[str] = None,
            reuse_locked_script: bool = False,
            script_edits: List[dict] = None,
            profile: str = None,
            literature_file: str = None,
            r2_threshold: float = None,
            max_verification_iterations: int = None,
            max_series_refits: int = None,
            starting_annealing_level: int = None,
            n_candidates: int = None,
            executor_timeout: int = None,
        ) -> str:
            """
            Execute analysis with the selected or specified agent.

            Each analysis run creates a unique output directory under results/
            to ensure traceability and prevent output collisions when analyzing
            multiple datasets with the same agent.

            For agents that execute LLM-generated code (CurveFitting, Hyperspectral),
            a sandbox check is performed. If no sandbox is detected, the user is
            prompted to confirm before proceeding.

            auxiliary_data / auxiliary_label provide one or several companion
            datasets (str or list). DUAL ROLE: always shown to the agent as
            context, AND — when a companion is shape-aligned with the primary —
            offered to the generated code as an OPTIONAL named numerical operand
            (subtract a baseline, divide by an I0 reference, mask/normalize with a
            co-registered channel); the label is the operand key. Supported by
            CurveFitting, Hyperspectral, and Image agents.

            Series metadata resolution (in priority order):

            1. **Explicit ``series_metadata`` parameter** — a JSON string
               describing the independent variable.  When ``values`` is a
               dict mapping filenames to values, files are automatically
               sorted by value for correct physical ordering and the dict
               is converted to a sorted list before passing to the agent.
               Expected format::

                   {"variable": "temperature",
                    "values": {"spec_5K.csv": 5, ...}, "unit": "K"}

            2. **Per-file JSON sidecars** — if the data directory contains
               JSON files whose stems match data files (e.g.
               ``spec_5K.json`` ↔ ``spec_5K.csv``), the system loads them,
               identifies numeric fields that vary across files, and uses
               LLM reasoning (with experimental context from *objective*,
               *hints*, and loaded metadata) to decide which field, if any,
               is the true control variable.  Full sidecar contents are
               stored in ``current_metadata["per_file_metadata"]`` for
               downstream agent access.

            3. **User prompt** — if neither of the above yields series
               metadata, returns a ``needs_series_metadata`` status.  When
               sidecars were loaded but the LLM could not identify a
               control variable, the sidecar contents are included in the
               response so the orchestrator can show them to the user.

            Output directory format: results/analysis_{dataset_name}_{timestamp}_{counter}/
            """
            print(f"  ⚡ Tool: Running analysis...")
            
            # Use current state if not provided
            if data_path is None:
                data_path = self.orch.current_data_path
            
            if agent_id is None:
                agent_id = self.orch.selected_agent_id
            
            # Validate inputs
            if data_path is None:
                return json.dumps({
                    "status": "error",
                    "message": "No data path provided. Use examine_data first."
                })
            
            _agent_inferred = None
            if agent_id is None:
                # Tool-only callers (MCP clients, run_task) do not run the
                # examine_data -> select_agent preamble a chat turn would.
                # Probe the data ourselves and, when exactly one agent fits,
                # use it — the same decision the orchestrator would make.
                try:
                    _probe = json.loads(self.execute_tool("examine_data", data_path=data_path))
                except Exception as _e:  # noqa: BLE001 - fall through to the error below
                    _probe = {"error": str(_e)}
                _sugg = _probe.get("suggested_agents") or []
                if len(_sugg) == 1 and _sugg[0] in self.AGENT_NAMES:
                    agent_id = int(_sugg[0])
                    self.orch.selected_agent_id = agent_id
                    _agent_inferred = self.AGENT_NAMES.get(agent_id)
                    print(f"  🤖 Agent inferred from data probe "
                          f"({_probe.get('data_type')}): {_agent_inferred}")
                else:
                    return json.dumps({
                        "status": "error",
                        "message": (
                            "No agent selected and the data probe could not pick "
                            f"one unambiguously (data_type={_probe.get('data_type')!r}, "
                            f"suggested_agents={_sugg}). Pass agent_id "
                            "(0=CurveFitting, 1=ImageAnalysis, 2=Hyperspectral) or "
                            "use select_agent first."),
                    })
            
            # Dataset-aware metadata resolution (issue #411). The loaded
            # document may serve this run only if it is unbound (freshly
            # loaded) or bound to this same dataset; metadata bound to a
            # *different* dataset is stale and must be re-resolved — never
            # silently reused across techniques.
            _stale_owner = None
            _metadata_minimal = False
            if self.orch.current_metadata is not None:
                _owner = getattr(self.orch, "current_metadata_owner", None)
                if _owner and not _same_dataset(_owner, data_path):
                    _stale_owner = _owner
            if self.orch.current_metadata is None or _stale_owner:
                # When a data directory contains sidecar JSONs, allow
                # run_analysis to proceed so the sidecar extraction code
                # below can populate metadata automatically.
                data_p = Path(data_path)
                has_sidecars = False
                if data_p.is_dir():
                    _all = [f for f in data_p.iterdir() if f.is_file() and not f.name.startswith('.')]
                    _data = [f for f in _all if f.suffix.lower() != ".json"]
                    _smap, _ = _detect_sidecar_jsons(_data, _all)
                    has_sidecars = bool(_smap)
                elif _is_glob(data_path):
                    # A glob selects one dataset out of a directory holding
                    # several (e.g. one in-situ series per pattern); its files'
                    # sidecars supply metadata exactly as in the directory case.
                    _data, _all = _resolve_glob_files(data_path)
                    _smap, _ = _detect_sidecar_jsons(_data, _all)
                    has_sidecars = bool(_smap)
                if has_sidecars:
                    if _stale_owner:
                        print(f"  📎 Metadata loaded for '{_stale_owner}' does not "
                              f"cover this dataset; using its sidecar JSONs instead")
                    self.orch.current_metadata = {}
                    self.orch.current_metadata_owner = None
                else:
                    # Single-file backstop: if a stem-matched sidecar JSON sits
                    # next to the data file (foo.npy -> foo.json), auto-load it
                    # instead of erroring. It never overrides an explicit load
                    # *for this dataset* (that case is not re-resolved at all);
                    # it converts a hard error into a usable run and makes
                    # metadata forwarding deterministic for the meta fan-out
                    # path.
                    _resolved_from_sidecar = False
                    if data_p.is_file():
                        sidecar = data_p.with_suffix(".json")
                        if sidecar.exists():
                            try:
                                with open(sidecar) as _fh:
                                    _md = json.load(_fh)
                                if isinstance(_md, dict) and _md:
                                    self.orch.current_metadata = _md
                                    self.orch.current_metadata_owner = None
                                    _resolved_from_sidecar = True
                                    print(f"  📎 Auto-loaded sidecar metadata: {sidecar.name}")
                            except Exception:  # noqa: BLE001 - bad sidecar -> fall through to error
                                pass
                    if not _resolved_from_sidecar:
                        if _stale_owner:
                            # Refuse rather than reuse: a plausible fit against
                            # another dataset's schema is worse than a stopped
                            # run. Recovery is one tool call.
                            return json.dumps({
                                "status": "error",
                                "message": (
                                    f"The loaded metadata belongs to a different "
                                    f"dataset ('{_stale_owner}') and no sidecar "
                                    f"metadata was found for '{data_path}'. Use "
                                    f"load_metadata or convert_metadata for this "
                                    f"dataset before analyzing it."
                                )
                            })
                        if self.orch.current_metadata is None:
                            _goal_bits = {k: v for k, v in (
                                ("analysis_goal", analysis_goal),
                                ("objective", objective), ("hints", hints)) if v}
                            if _goal_bits:
                                # Minimal metadata from the call itself: a tool-only
                                # caller with no metadata file still stated what the
                                # data is and what to do with it. Better a run with
                                # thin context than a hard stop nobody can answer.
                                self.orch.current_metadata = {
                                    **_goal_bits,
                                    "_source": "run_analysis arguments (no metadata "
                                               "file or sidecar provided)",
                                }
                                self.orch.current_metadata_owner = None
                                _metadata_minimal = True
                                print("  📎 No metadata file/sidecar; using the call's "
                                      "analysis_goal/objective/hints as minimal metadata")
                            else:
                                return json.dumps({
                                    "status": "error",
                                    "message": (
                                        "No metadata available. Use load_metadata or "
                                        "convert_metadata first, drop a stem-matched JSON "
                                        "sidecar next to the data, or state what the data "
                                        "is in analysis_goal."),
                                })
            # Bind unbound metadata to the dataset now consuming it.
            if (self.orch.current_metadata is not None
                    and not getattr(self.orch, "current_metadata_owner", None)):
                self.orch.current_metadata_owner = _dataset_key(data_path)
            
            try:
                # === Handle directory / glob input - filter out metadata files ===
                path = Path(data_path)
                actual_data_input = data_path  # Default: pass as-is
                is_glob_input = _is_glob(data_path)
                all_files: list[Path] = []
                data_files: list[Path] = []

                if path.is_dir() or is_glob_input:
                    if is_glob_input:
                        # A glob names ONE dataset inside a directory that holds
                        # several (distinct in-situ series in a single upload
                        # folder); only its matched files are analyzed.
                        data_files, all_files = _resolve_glob_files(data_path)
                        if not data_files:
                            return json.dumps({
                                "status": "error",
                                "message": (f"No data files match the pattern "
                                            f"'{data_path}'.")
                            })
                        print(f"    Pattern '{Path(data_path).name}' matched "
                              f"{len(data_files)} data file(s)")
                    else:
                        # Get all files excluding metadata
                        all_files = [f for f in path.iterdir() if f.is_file() and not f.name.startswith('.')]

                        # Filter out metadata files
                        data_files = []
                        for f in all_files:
                            is_metadata = (
                                f.suffix.lower() == '.json' or
                                'metadata' in f.name.lower() or
                                f.name.lower() in ['info.txt', 'description.txt', 'readme.txt', 'readme.md']
                            )
                            if not is_metadata:
                                data_files.append(f)

                        if not data_files:
                            return json.dumps({
                                "status": "error",
                                "message": "No data files found in directory (only metadata files present)"
                            })

                        print(f"    Found {len(data_files)} data files (excluded metadata)")

                    # Sort for consistent ordering
                    data_files = sorted(data_files, key=lambda x: x.name)

                    # Pass as list of file paths for series analysis
                    actual_data_input = [str(f) for f in data_files]

                    # If only one file, pass as string (single spectrum mode)
                    if len(actual_data_input) == 1:
                        actual_data_input = actual_data_input[0]
                        print(f"    Single file, using single spectrum mode")
                    else:
                        print(f"    Series mode: passing {len(actual_data_input)} files")
                        for i, fp in enumerate(actual_data_input[:3]):
                            print(f"      [{i}] {Path(fp).name}")
                        if len(actual_data_input) > 3:
                            print(f"      ... and {len(actual_data_input) - 3} more")
                
                # === Handle series metadata ===
                is_series = isinstance(actual_data_input, list) and len(actual_data_input) > 1
                has_series_meta = (
                    isinstance(self.orch.current_metadata, dict)
                    and "series" in self.orch.current_metadata
                )

                if series_metadata is not None:
                    # Parse and inject series metadata from the tool call
                    try:
                        parsed_series = json.loads(series_metadata) if isinstance(series_metadata, str) else series_metadata
                        self.orch.current_metadata["series"] = parsed_series
                        has_series_meta = True
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"Failed to parse series_metadata: {e}")

                # === Try to extract series metadata from sidecar JSON files ===
                if is_series and not has_series_meta and (path.is_dir() or is_glob_input):
                    sidecar_map, _global_jsons = _detect_sidecar_jsons(
                        data_files, all_files
                    )
                    if sidecar_map:
                        print(
                            f"    Found {len(sidecar_map)} sidecar JSON file(s) "
                            f"paired with data files"
                        )
                        extracted_series, per_file_meta = (
                            _extract_series_from_sidecars(
                                sidecar_map,
                                data_files,
                                self.logger,
                                model=self.orch.model,
                                experimental_context={
                                    "objective": objective,
                                    "hints": hints,
                                    "metadata": self.orch.current_metadata,
                                },
                            )
                        )
                        # Store per-file metadata for agent access
                        if per_file_meta:
                            self.orch.current_metadata[
                                "per_file_metadata"
                            ] = per_file_meta

                            # Synthesize normalized global metadata from
                            # shared invariant fields across all sidecars,
                            # unless current_metadata already contains
                            # normalized top-level sections.
                            _has_global = any(
                                k in self.orch.current_metadata
                                for k in ("experiment", "sample", "instrument")
                            )
                            if not _has_global:
                                _all_dicts = list(per_file_meta.values())
                                _shared = set(_all_dicts[0].keys())
                                for _sd in _all_dicts[1:]:
                                    _shared &= _sd.keys()
                                _synth: dict = {}
                                for _k in _shared:
                                    _vals = [_sd[_k] for _sd in _all_dicts]
                                    if all(v == _vals[0] for v in _vals):
                                        _synth[_k] = _vals[0]
                                if _synth:
                                    try:
                                        _ok, _ = check_schema_conformance(_synth)
                                        if not _ok:
                                            _normed, _ = normalize_metadata_dict(_synth)
                                            _re_ok, _ = check_schema_conformance(_normed)
                                            if not _re_ok:
                                                _llm = normalize_metadata_dict_with_llm(
                                                    _synth, self.orch.model, self.logger
                                                )
                                                if _llm:
                                                    for _k2, _v2 in _synth.items():
                                                        if _k2 not in _llm:
                                                            _llm[_k2] = _v2
                                                    _synth = _llm
                                                else:
                                                    _synth = _normed
                                            else:
                                                _synth = _normed
                                        # Merge normalized global fields into
                                        # current_metadata without overwriting
                                        # per_file_metadata or series.
                                        for _k3, _v3 in _synth.items():
                                            if _k3 not in self.orch.current_metadata:
                                                self.orch.current_metadata[_k3] = _v3
                                        print(
                                            f"    Synthesized global metadata from "
                                            f"{len(per_file_meta)} sidecar(s)"
                                        )
                                    except Exception as _e:
                                        self.logger.warning(
                                            "Failed to synthesize global metadata "
                                            "in run_analysis: %s", _e
                                        )

                        # Auto-populate series metadata if extraction succeeded
                        if extracted_series is not None:
                            self.orch.current_metadata["series"] = extracted_series
                            has_series_meta = True
                            print(
                                f"    Auto-extracted series variable "
                                f"'{extracted_series['variable']}' from sidecar JSONs"
                            )
                            # In co-pilot / autopilot modes, let the user
                            # know which control variable was extracted and
                            # give them a chance to confirm or correct it
                            # before proceeding with the analysis.
                            mode = self.orch.analysis_mode.value
                            if mode in ("co-pilot", "autopilot"):
                                values = extracted_series.get("values", {})
                                unit = extracted_series.get("unit", "")
                                # Build a readable summary of the mapping
                                sample_items = list(values.items())[:5]
                                mapping_lines = [
                                    f"  {fname}: {val}"
                                    for fname, val in sample_items
                                ]
                                if len(values) > 5:
                                    mapping_lines.append(
                                        f"  ... and {len(values) - 5} more"
                                    )
                                mapping_str = "\n".join(mapping_lines)
                                return json.dumps({
                                    "status": "series_variable_extracted",
                                    "message": (
                                        f"Auto-extracted series control variable "
                                        f"'{extracted_series['variable']}'"
                                        f"{(' (' + unit + ')') if unit else ''} "
                                        f"from per-file sidecar JSON metadata. "
                                        f"File-to-value mapping:\n{mapping_str}\n\n"
                                        f"Present this to the user and ask them "
                                        f"to confirm it is correct before "
                                        f"proceeding.\n"
                                        f"- If the user CONFIRMS: re-call "
                                        f"run_analysis with the same parameters "
                                        f"(no series_metadata needed — it is "
                                        f"already stored).\n"
                                        f"- If the user DISAGREES or wants a "
                                        f"different variable: ask them for the "
                                        f"correct variable name, values, and "
                                        f"unit, then re-call run_analysis with "
                                        f"an explicit series_metadata parameter "
                                        f"containing the corrected mapping. "
                                        f"The explicit parameter will override "
                                        f"the auto-extracted one."
                                    ),
                                    "variable": extracted_series["variable"],
                                    "unit": unit,
                                    "values": values,
                                    "num_files": len(values),
                                })

                if is_series and not has_series_meta:
                    num_files = len(actual_data_input)

                    # If per-file sidecar metadata was loaded, include it in
                    # the prompt so the orchestrator LLM can show the user
                    # what each file already contains.
                    per_file = self.orch.current_metadata.get(
                        "per_file_metadata"
                    )
                    sidecar_note = ""
                    if per_file:
                        sidecar_note = (
                            " Per-file JSON sidecar metadata was found but "
                            "none of the recorded fields could be "
                            "confidently identified as the control variable. "
                            "Show the user the sidecar contents below and "
                            "ask them to confirm which field (if any) is the "
                            "control variable, or to specify it manually."
                        )

                    prompt_payload = {
                        "status": "needs_series_metadata",
                        "message": (
                            f"Detected {num_files} spectra (series mode) but "
                            "no series metadata found. "
                            "Series metadata describes the experimental "
                            "variable that changes across spectra "
                            "(e.g. temperature, concentration, voltage). "
                            "Ask the user what variable changes across the "
                            "spectra, the range or values, and the units. "
                            "The user can describe this naturally — e.g. "
                            "'temperature from 300 to 500 K in 50 K steps' "
                            "or 'concentration: 0.1, 0.2, 0.5 mM'. "
                            "Use the filenames and the user's response to "
                            "build the values dict mapping each filename to "
                            "its value, then re-call run_analysis with the "
                            "series_metadata parameter. "
                            "Files will be sorted by value automatically for "
                            "correct trend analysis."
                            + sidecar_note
                        ),
                        "num_spectra": num_files,
                        "expected_format": {
                            "variable": "<variable name, e.g. temperature>",
                            "values": {
                                "<filename>": "<value>",
                                "...": "...",
                            },
                            "unit": "<unit string, e.g. K, mM, V>",
                        },
                        "files": [
                            Path(f).name for f in actual_data_input
                        ],
                    }
                    if per_file:
                        prompt_payload["per_file_sidecar_metadata"] = per_file

                    return json.dumps(prompt_payload)

                # Sort files by series values for correct physical ordering
                if is_series and has_series_meta:
                    series_info = self.orch.current_metadata.get("series", {})
                    values = series_info.get("values")
                    if isinstance(values, dict):
                        # Map filenames to full paths
                        name_to_path = {Path(f).name: f for f in actual_data_input}
                        # Build sorted (path, value) pairs by value
                        paired = []
                        for fname, val in values.items():
                            full_path = name_to_path.get(fname)
                            if full_path is not None:
                                try:
                                    paired.append((full_path, float(val)))
                                except (TypeError, ValueError):
                                    paired.append((full_path, val))
                        # Sort by value (numeric sort when possible)
                        try:
                            paired.sort(key=lambda x: x[1])
                        except TypeError:
                            pass  # mixed types, keep original order
                        if paired:
                            actual_data_input = [p[0] for p in paired]
                            sorted_values = [p[1] for p in paired]
                            # Replace dict with sorted list for agent consumption
                            series_info["values"] = sorted_values
                            self.orch.current_metadata["series"] = series_info

                # === Generate unique analysis output directory ===
                # Deferred until after early-return checks (series variable
                # confirmation, missing series metadata) to avoid creating
                # orphan directories that never receive analysis results.
                analysis_id = self.orch.generate_analysis_id(data_path, agent_id)
                analysis_output_dir = self.orch.results_dir / f"analysis_{analysis_id}"
                analysis_output_dir.mkdir(parents=True, exist_ok=True)

                print(f"    Analysis ID: {analysis_id}")
                print(f"    Output directory: {analysis_output_dir}")

                # === Save metadata copy for traceability ===
                metadata_copy_path = analysis_output_dir / "metadata_used.json"
                with open(metadata_copy_path, 'w', encoding="utf-8") as f:
                    json.dump({
                        "analysis_id": analysis_id,
                        "data_path": data_path,
                        "agent_id": agent_id,
                        "agent_name": self.AGENT_NAMES.get(agent_id),
                        "analysis_goal": analysis_goal,
                        "timestamp": datetime.now().isoformat(),
                        "metadata": _structure_metadata_for_save(
                            self.orch.current_metadata
                        ),
                    }, f, indent=2)

                # === Create agent with unique output directory ===
                # NOTE: Code-executing agents may prompt the user
                # for sandbox approval and raise RuntimeError if declined.
                try:
                    agent = self.orch.create_agent_for_analysis(
                        agent_id, str(analysis_output_dir),
                        executor_timeout=executor_timeout)
                except RuntimeError as e:
                    # Handle sandbox rejection or other init failures
                    error_msg = str(e)

                    if "sandbox" in error_msg.lower() or "declined" in error_msg.lower():
                        # Clean up the output directory we created
                        import shutil
                        if analysis_output_dir.exists():
                            shutil.rmtree(analysis_output_dir)

                        return json.dumps({
                            "status": "aborted",
                            "reason": "sandbox_declined",
                            "message": "Analysis aborted: User declined to proceed without sandbox protection.",
                            "agent_id": agent_id,
                            "agent_name": self.AGENT_NAMES.get(agent_id),
                            "recommendation": (
                                "This agent executes AI-generated Python code and requires a secure environment.\n\n"
                                "Please run SciLink in one of the following:\n"
                                "  1. Docker container (recommended)\n"
                                "  2. Virtual machine (VMware, VirtualBox, cloud VM)\n"
                                "  3. Google Colab\n\n"
                                "See the documentation for setup instructions."
                            )
                        })
                    else:
                        # Some other initialization error
                        raise

                print(f"    Using agent: {type(agent).__name__}")
                print(f"    Data: {data_path}")

                # === Run analysis ===
                analyze_kwargs = {
                    "data": actual_data_input,
                    "system_info": self.orch.current_metadata,
                }
                if objective is not None:
                    analyze_kwargs["objective"] = objective
                if hints is not None:
                    analyze_kwargs["hints"] = hints
                if auxiliary_data is not None:
                    analyze_kwargs["auxiliary_data"] = auxiliary_data
                if auxiliary_label is not None:
                    analyze_kwargs["auxiliary_label"] = auxiliary_label
                if skill is not None:
                    # PR 3: ``skill`` may be a single name or a list. Resolve
                    # any custom-skill names to their registered file paths
                    # so the agent's load_skill() can locate them.
                    custom_skills = getattr(self.orch, "_custom_skills", {})

                    def _resolve_one(s):
                        return custom_skills[s] if s in custom_skills else s

                    if isinstance(skill, str):
                        analyze_kwargs["skill"] = _resolve_one(skill)
                    elif isinstance(skill, (list, tuple)):
                        analyze_kwargs["skill"] = [_resolve_one(s) for s in skill]
                    else:
                        analyze_kwargs["skill"] = skill
                if skill_hint is not None:
                    # Non-binding suggestion: the agent's auto-selector uses it
                    # as a prior but decides from the data (agent has final
                    # authority). Only forward to agents whose analyze() accepts
                    # it; others accept **kwargs and ignore it.
                    import inspect as _inspect
                    if "skill_hint" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["skill_hint"] = skill_hint
                # Make user-registered custom skills auto-selectable by the
                # agent (not only passable as authoritative `skill`): forward
                # the {name: path} registry so the agent-side selector folds
                # them into its catalog (#256 fix #1).
                _custom = getattr(self.orch, "_custom_skills", None)
                if _custom:
                    import inspect as _inspect
                    if "custom_skills" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["custom_skills"] = dict(_custom)
                if task_mode is not None:
                    # Currently consumed by CurveFittingAgent; other agents
                    # accept **kwargs and silently ignore unknown parameters,
                    # matching the existing pattern for `hints`.
                    analyze_kwargs["task_mode"] = task_mode
                if self.orch.active_knowledge:
                    analyze_kwargs["prior_knowledge"] = self.orch.active_knowledge
                if prior_analysis_paths:
                    analyze_kwargs["prior_analysis_paths"] = prior_analysis_paths
                if reuse_locked_script:
                    analyze_kwargs["reuse_locked_script"] = True
                if script_edits:
                    # Surgical follow-up: only curve fitting supports it.
                    # An unsupported agent must REFUSE, not silently drop
                    # the caller's explicitly requested change (same
                    # principle as the multi-regime guard).
                    import inspect as _inspect
                    if "script_edits" in _inspect.signature(
                            agent.analyze).parameters:
                        analyze_kwargs["script_edits"] = script_edits
                    else:
                        return json.dumps({
                            "status": "error",
                            "message": (
                                f"script_edits is not supported by "
                                f"{type(agent).__name__} — it applies "
                                "surgical edits to a prior run's saved "
                                "script, currently supported for curve "
                                "fitting and image analysis. Re-run "
                                "without script_edits.")})
                if profile:
                    # Operating profile (#346): forward only to agents whose
                    # analyze() accepts it (all three do; introspection keeps
                    # this robust for custom agents).
                    import inspect as _inspect
                    if "profile" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["profile"] = profile
                if literature_file:
                    analyze_kwargs["literature_file"] = literature_file
                if r2_threshold is not None:
                    # Experienced-user R² override. Only forward to agents whose
                    # analyze() accepts it (currently CurveFitting); others have
                    # no R² gate, so ignore it rather than error.
                    import inspect as _inspect
                    if "r2_threshold" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["r2_threshold"] = float(r2_threshold)
                    else:
                        self.logger.info(
                            f"   r2_threshold={r2_threshold} ignored: "
                            f"{self.AGENT_NAMES.get(agent_id, 'agent')} has no R² gate."
                        )
                if max_verification_iterations is not None:
                    # Thoroughness / turnaround override. 0 bypasses LLM
                    # verification (fast/in-situ), higher = more refinement
                    # (thorough/post-experiment). Forwarded by signature
                    # introspection (all three analysis agents accept it).
                    import inspect as _inspect
                    if "max_verification_iterations" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["max_verification_iterations"] = int(max_verification_iterations)
                    else:
                        self.logger.info(
                            f"   max_verification_iterations={max_verification_iterations} ignored: "
                            f"{self.AGENT_NAMES.get(agent_id, 'agent')} has no verification loop."
                        )
                if max_series_refits is not None:
                    # Wall-clock budget for per-unit re-analysis in a series;
                    # forwarded only to agents whose analyze() has the knob.
                    import inspect as _inspect
                    if "max_series_refits" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["max_series_refits"] = int(max_series_refits)
                    else:
                        self.logger.info(
                            f"   max_series_refits={max_series_refits} ignored: "
                            f"{self.AGENT_NAMES.get(agent_id, 'agent')} has no series refit stage."
                        )
                if starting_annealing_level is not None:
                    # Annealing-schedule override for a RE-RUN: start the
                    # constraint-relaxation schedule higher (e.g. hot) so the
                    # agent does not repeat early stages a prior run already
                    # found inadequate. Default/first run is None -> unchanged
                    # (schedule starts frozen at T=0). Only forward to agents
                    # whose analyze() accepts it (CurveFitting, Image).
                    import inspect as _inspect
                    if "starting_annealing_level" in _inspect.signature(agent.analyze).parameters:
                        analyze_kwargs["starting_annealing_level"] = int(starting_annealing_level)
                    else:
                        self.logger.info(
                            f"   starting_annealing_level={starting_annealing_level} ignored: "
                            f"{self.AGENT_NAMES.get(agent_id, 'agent')} has no annealing schedule."
                        )
                # Best-of-N: deterministic per-agent default (image=3, others 1)
                # unless explicitly requested; forwarded only to agents whose
                # analyze() accepts it.
                resolved_n = _resolve_n_candidates(agent, n_candidates)
                if resolved_n is None:
                    if n_candidates is not None:
                        self.logger.info(
                            f"   n_candidates={n_candidates} ignored: "
                            f"{self.AGENT_NAMES.get(agent_id, 'agent')} has no "
                            f"best-of-N support."
                        )
                elif resolved_n > 1:
                    analyze_kwargs["n_candidates"] = resolved_n
                    if _resolve_candidate_escalation(agent, n_candidates):
                        analyze_kwargs["candidate_escalation"] = True
                        self.logger.info(
                            f"   Best-of-{resolved_n} (escalation): attempt 0 "
                            f"runs alone; fans out to {resolved_n} only if it "
                            f"is weak. An LLM judge selects the winner."
                        )
                    else:
                        self.logger.info(
                            f"   Best-of-{resolved_n}: anchor analyses run in "
                            f"parallel; an LLM judge selects the winner."
                        )
                result = agent.analyze(**analyze_kwargs)
                
                # === Store result ===
                analysis_record = {
                    "analysis_id": analysis_id,
                    "timestamp": datetime.now().isoformat(),
                    "data_path": data_path,
                    "agent_id": agent_id,
                    "agent_name": self.AGENT_NAMES.get(agent_id),
                    "status": result.get("status"),
                    "output_directory": str(analysis_output_dir),
                    "literature_file": literature_file,
                    "full_result": result,
                    "novelty_assessment": None
                }
                self.orch.analysis_results.append(analysis_record)
                
                # === Format response ===
                # "partial" is hyperspectral's salvage outcome (approximate /
                # withheld maps with honest caveats) — a usable, degraded
                # result. It gets the full success-shaped payload with its
                # caveats attached, NOT the error branch (which used to
                # surface it as an error with an empty error object).
                if result.get("status") in ("success", "partial"):
                    # Find main visualization
                    viz_path = None
                    for candidate in analysis_output_dir.rglob("*_analysis.png"):
                        if CANDIDATES_DIR_NAME in candidate.parts:
                            continue  # best-of-N loser attempt, not canonical
                        viz_path = str(candidate)
                        break
                    if not viz_path:
                        for candidate in analysis_output_dir.rglob("*.png"):
                            if CANDIDATES_DIR_NAME in candidate.parts:
                                continue
                            if "report" not in candidate.name.lower():
                                viz_path = str(candidate)
                                break

                    response = {
                        "status": result.get("status"),
                        "analysis_id": analysis_id,
                        "agent_used": self.AGENT_NAMES.get(agent_id),
                        "output_directory": str(analysis_output_dir),
                        "detailed_analysis": (result.get("detailed_analysis") or "")[:2000],
                        "claims_count": len(result.get("scientific_claims") or []),
                        "full_result_available": True,
                        "note": f"All outputs saved to: {analysis_output_dir}",
                        "next_steps": "Use assess_novelty to check literature for these claims, or get_recommendations for follow-up experiments.",
                    }
                    if _agent_inferred:
                        response["agent_inferred"] = _agent_inferred
                    if _metadata_minimal:
                        response["metadata_note"] = (
                            "No metadata file or sidecar was available; the run used "
                            "the call's analysis_goal/objective/hints as minimal "
                            "metadata. Provide load_metadata/convert_metadata or a "
                            "JSON sidecar for richer context (units, technique, "
                            "sample).")
                    if result.get("status") == "partial":
                        response["confidence"] = result.get("confidence")
                        response["warnings"] = result.get("warnings") or []
                        if result.get("degraded_outputs"):
                            response["degraded_outputs"] = result[
                                "degraded_outputs"]
                        response["note"] = (
                            "PARTIAL result: some outputs are approximate or "
                            "were withheld (see warnings). "
                            + response["note"]
                        )
                    if viz_path:
                        response["visualization_path"] = viz_path
                    if result.get("tier2_results"):
                        response["tier2_ran"] = True
                        t2 = result["tier2_results"]
                        response["tier2_focus"] = t2.get(
                            "analysis_approach", "deeper analysis"
                        )
                    # Emit a flat feature table (per-unit conditions + extracted
                    # scalar features) so downstream planning / BO can ingest the
                    # results as a file rather than re-typed prose.
                    feature_table = write_feature_table(analysis_output_dir)
                    if feature_table:
                        response["feature_table"] = feature_table
                        # Describe the table in the response itself: a client
                        # that cannot open server files (remote MCP) — or the
                        # orchestrator LLM choosing BO inputs/targets — needs
                        # the column names and where the holes are.
                        _desc = describe_feature_table(feature_table)
                        if _desc:
                            response["feature_columns"] = _desc["columns"]
                            response["feature_rows"] = _desc["n_rows"]
                            if _desc["missing"]:
                                response["feature_missing"] = _desc["missing"]
                    # #172: surface the locked-script reuse verdict so the
                    # orchestrator can act on a non-`good` outcome (a poorly
                    # fitting reused recipe, or a re-derived schema).
                    if result.get("refit_skipped_by_budget"):
                        response["refit_skipped_by_budget"] = result["refit_skipped_by_budget"]
                    reuse_validity = result.get("reuse_validity")
                    if reuse_validity:
                        response["reuse_validity"] = reuse_validity
                        if reuse_validity.get("verdict") != "good":
                            response["reuse_warning"] = reuse_validity.get(
                                "message", ""
                            )
                    # Hyperspectral locked-replay summary (#509/#518): carries
                    # verbatim-ness and any degraded-harmonization scoping
                    # warnings — a harmonized fan-out follower must narrate a
                    # scoping it could not reproduce, so fusion can discount.
                    if result.get("script_reuse"):
                        response["script_reuse"] = result["script_reuse"]
                        if result["script_reuse"].get("scope_degraded"):
                            response["reuse_warning"] = (
                                (response.get("reuse_warning", "") + " ").lstrip()
                                + "DEGRADED HARMONIZATION: "
                                + " | ".join(result["script_reuse"].get(
                                    "scope_warnings") or []))
                    # Best-of-N: compact candidate table + judge reasoning so
                    # the orchestrator can narrate the comparison.
                    if result.get("anchor_candidates"):
                        response["anchor_candidates"] = result[
                            "anchor_candidates"
                        ]
                        judge = result.get("anchor_judge") or {}
                        if judge.get("reasoning"):
                            response["anchor_judge_reasoning"] = judge[
                                "reasoning"
                            ][:300]
                    return json.dumps(response)
                else:
                    return json.dumps({
                        "status": "error",
                        "analysis_id": analysis_id,
                        "error": result.get("error", {}),
                        "agent_used": self.AGENT_NAMES.get(agent_id),
                        "output_directory": str(analysis_output_dir)
                    })
                    
            except Exception as e:
                self.logger.error(f"Analysis error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=run_analysis,
            name="run_analysis",
            description=(
                "Execute analysis with the selected or specified agent. "
                "Each run creates a unique output directory (analysis_{dataset_name}_{timestamp}) "
                "for traceability. Requires data path and metadata to be set. "
                "For series analysis, the system resolves the control variable in order: "
                "(1) explicit series_metadata parameter, "
                "(2) automatic extraction from per-file JSON sidecars via LLM reasoning, "
                "(3) user prompt. "
                "Optional objective provides a high-level scientific question to frame the analysis "
                "(e.g. 'Determine the oxidation state of Ti'). "
                "Optional hints provide tactical guidance to steer the analysis "
                "(e.g. 'focus on the Ti L-edge around 460 eV'). "
                "Optional auxiliary_data (str or list) provides companion dataset(s) "
                "— shown as context AND, when shape-aligned, usable by generated code "
                "as an optional numerical operand (subtract a baseline, divide by an "
                "I0 reference, mask/normalize with a co-registered channel); "
                "auxiliary_label is the operand key. Supported by CurveFitting, "
                "Hyperspectral, Image. "
                "Optional skill provides domain-specific knowledge "
                "(e.g. 'xps', 'xrd') for improved fitting and interpretation. "
                "Returns analysis_id and output_directory for reference."
            ),
            parameters={
                "data_path": {
                    "type": "string",
                    "description": (
                        "Path to a data file, a directory (analyzed as a series), "
                        "or a glob PATTERN like '/data/series_*C.txt' selecting one "
                        "dataset out of a directory that holds several. When you were "
                        "given a pattern, pass it VERBATIM — substituting its parent "
                        "directory would pull in the other datasets. Uses current if "
                        "not specified."
                    )
                },
                "agent_id": {
                    "type": "integer",
                    "description": (
                        "Agent ID to use (0=CurveFitting, 1=ImageAnalysis, "
                        "2=Hyperspectral). Optional: uses the selected agent, else "
                        "the one the data probe suggests when that is unambiguous "
                        "(the response then reports agent_inferred).")
                },
                "analysis_goal": {
                    "type": "string",
                    "description": (
                        "Specific analysis objective (saved with results for "
                        "traceability). When no metadata file or JSON sidecar is "
                        "available it also serves as the run's minimal metadata, "
                        "so say what the data is (technique, units, sample) as "
                        "well as what to extract.")
                },
                "objective": {
                    "type": "string",
                    "description": (
                        "High-level scientific objective that frames the analysis "
                        "(e.g. 'Determine whether the sample underwent a phase transition', "
                        "'Quantify relative concentration of anatase vs rutile'). "
                        "Unlike hints, this tells the agent *why* the analysis is being "
                        "performed and *what question* to answer."
                    )
                },
                "hints": {
                    "type": "string",
                    "description": (
                        "Tactical guidance to steer the analysis "
                        "(e.g. 'focus on the Ti L-edge around 460 eV', "
                        "'pay attention to peaks between 280-300 nm') AND/OR "
                        "figure-presentation preferences, which reach the "
                        "generated plotting code (e.g. 'place the legend "
                        "outside the axes — it covers the data', 'use a log "
                        "intensity scale'). Route any user request about how "
                        "plots should LOOK through this parameter. Supported "
                        "by CurveFitting, Image, and Hyperspectral agents."
                    )
                },
                "auxiliary_data": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Path (or list of paths) to companion dataset(s) — e.g. a "
                        "reference/baseline spectrum, an incident-beam/I0 reference, "
                        "or a co-registered channel. DUAL ROLE: always shown to the "
                        "agent as visual "
                        "context, AND — when a companion is shape-aligned with the "
                        "primary — offered to the generated code as an OPTIONAL named "
                        "numerical operand it may subtract / divide by / mask with "
                        "(the matching auxiliary_label is the operand name). Pass "
                        "several as a list with parallel auxiliary_label entries. Only "
                        "use for genuinely related data (same sample/region or "
                        "complementary techniques) — do not chain unrelated analyses. "
                        "Supported by CurveFitting, Hyperspectral, and Image agents."
                    )
                },
                "auxiliary_label": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Label(s) for the auxiliary dataset(s), parallel to "
                        "auxiliary_data. Each label is BOTH the human description and "
                        "the operand key the generated code addresses, e.g. "
                        "'empty-sample baseline', 'incident beam I0', 'reference channel'. "
                        "Use a list when auxiliary_data is a list."
                    )
                },
                "skill": {
                    # Accepts a single skill name (string) or a list of names
                    # for multi-skill loading. Schema permits both shapes via
                    # JSON Schema ``oneOf``.
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": _build_skill_description(
                        getattr(self.orch, "_agent_registry", None),
                        getattr(self.orch, "_custom_skills", None),
                    ),
                },
                "skill_hint": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Non-binding skill SUGGESTION — use this (instead of "
                        "`skill`) for your OWN autonomous guess that a skill may "
                        "apply, when the user did not explicitly request one. The "
                        "agent inspects the actual data and decides: it may "
                        "confirm your hint, add complementary skills, or override "
                        "it. Use `skill` (authoritative) only for an explicit "
                        "user request or a custom skill; use `skill_hint` for "
                        "everything you infer yourself. Same name(s) as `skill`."
                    ),
                },
                "series_metadata": {
                    "type": "string",
                    "description": (
                        "JSON string describing the experimental variable that changes across "
                        "spectra in a series. Takes highest priority — overrides automatic "
                        "extraction from per-file JSON sidecars. "
                        "Values is a dict mapping each filename to its value — files are "
                        "automatically sorted by value for correct trend analysis. "
                        "Format: {\"variable\": \"<variable>\", \"values\": {\"<filename>\": <value>, ...}, \"unit\": \"<units>\"}. "
                        "Example: {\"variable\": \"temperature\", \"values\": {\"spec_5K.csv\": 5, \"spec_10K.csv\": 10, \"spec_20K.csv\": 20}, \"unit\": \"K\"}"
                    )
                },
                "task_mode": {
                    "type": "string",
                    "enum": ["fitting", "identification"],
                    "description": (
                        "CurveFitting agent only. Set to 'identification' when the user "
                        "is asking the agent to help identify what material or phase the "
                        "spectrum is from (no sample identity known), rather than to fit "
                        "and interpret a known material. Identity unknown means use "
                        "identification mode even when the elemental composition IS known "
                        "(e.g. from EDX/metadata) — element hints constrain candidates, "
                        "they do not name the material. In identification mode the planner "
                        "uses a generic flexible model and the interpreter enumerates "
                        "ranked candidate materials with discriminating peaks instead of "
                        "asserting a single answer. Leave unset (defaults to 'fitting') "
                        "for standard analyses where the sample is known."
                    )
                },
                "prior_analysis_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of folder or file paths from previous analyses whose "
                        "outputs the new run can consume. Use this when a follow-up "
                        "analysis needs to load artifacts (masks, positions, "
                        "feature tables, abundance maps) from a prior run rather "
                        "than recomputing them. Directory paths typically come "
                        "from `list_results()` (the `output_directory` field). "
                        "For each path, the agent's code generator receives the "
                        "file listing (loadable via absolute path); for paths "
                        "containing `analysis_results.json`, the planner also "
                        "receives a state summary (pipeline, quality score, "
                        "extracted features, scientific claims, saved-arrays "
                        "catalog). Consumed by the image-analysis agent and "
                        "the curve-fitting agent — for a prior curve-fit run "
                        "the saved fitting script and fit summary are surfaced "
                        "to its planning and script-generation stages. The "
                        "hyperspectral agent consumes it ONLY together with "
                        "`reuse_locked_script=true` (locked replay of the "
                        "prior run's approved per-pixel script).\n"
                        "By default the prior run is REFERENCE MATERIAL: the "
                        "agent decides for itself whether to reuse, adapt, or "
                        "rewrite the prior script given the goal. Pass this for "
                        "ANY follow-up that should build on a prior run — "
                        "verification, deeper analysis, or extending a series. "
                        "For a verification / re-examination the agent derives "
                        "the result independently (re-running the script that "
                        "produced a result cannot verify it). To FORCE verbatim "
                        "reuse of the prior locked extraction script, ALSO set "
                        "`reuse_locked_script=true` (see that parameter)."
                    )
                },
                "reuse_locked_script": {
                    "type": "boolean",
                    "description": (
                        "Opt-in, default false. Set true ONLY when the new data "
                        "is the NEXT MEASUREMENT of the SAME series as a run in "
                        "`prior_analysis_paths` — the same kind of unit, only "
                        "the control parameters differ (a new point in a "
                        "Bayesian-optimization / closed-loop campaign). It forces "
                        "the agent to reuse that run's locked extraction script "
                        "VERBATIM instead of re-deriving the model, guaranteeing "
                        "the new feature row has the SAME columns as the campaign "
                        "(which the planning-side feature-table append strictly "
                        "requires). The run then reports a `reuse_validity` "
                        "verdict (`good` / `poor` / `script_failed`) — read it "
                        "and act on a non-`good` verdict. Do NOT set it for "
                        "verification or deeper-analysis follow-ups, or for a "
                        "different kind of measurement — leave it false so the "
                        "agent decides how to use the prior run as reference. "
                        "ALSO valid for hyperspectral: with a prior "
                        "hyperspectral run dir it replays that run's approved "
                        "dynamic-analysis script(s) VERBATIM on the new cube "
                        "(harmonized re-run across sibling datasets — same "
                        "technique, e.g. one cube per experimental condition) "
                        "so extracted magnitudes are method-comparable; "
                        "decomposition and fresh planning are skipped."
                    )
                },
                "script_edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["old_text", "new_text"],
                    },
                    "description": (
                        "Surgical single-knob follow-up (curve fitting and "
                        "image analysis; requires `prior_analysis_paths` + "
                        "`reuse_locked_script=true`): exact old/new snippet "
                        "pairs applied to the prior run's saved script BEFORE "
                        "it re-executes, so the rerun differs from the prior "
                        "run in EXACTLY the requested change — a controlled "
                        "one-variable-at-a-time comparison. Use when the user "
                        "asks to redo an accepted analysis with one parameter "
                        "changed (a threshold, a bound, a window) — NOT for a "
                        "different model or a re-derivation, where the agent "
                        "should plan freely. Copy old_text VERBATIM from the "
                        "prior run's saved script (scripts/*.py in its output "
                        "dir); each edit must match exactly once. All edits "
                        "apply atomically or the run refuses with a per-edit "
                        "report before any execution. The result records "
                        "`script_edits_applied` and the usual "
                        "`reuse_validity` R² verdict (signal, not gate — the "
                        "knob change is intentional; judge the outcome "
                        "yourself)."
                    )
                },
                "profile": {
                    "type": "string",
                    "enum": ["thorough", "realtime"],
                    "description": (
                        "Operating profile. Omit (or 'thorough') for the normal "
                        "full-quality analysis. 'realtime' is the per-frame "
                        "in-situ mode for curve data: it executes the locked "
                        "script from `prior_analysis_paths` with ZERO LLM calls "
                        "— requires `prior_analysis_paths` + "
                        "`reuse_locked_script=true`, and the anchor frame must "
                        "have been analyzed thoroughly first. The result carries "
                        "`reuse_validity` with the gate verdict plus a `drift` "
                        "field ('none'/'suspected') from a fingerprint "
                        "comparison against the anchor frame — 'suspected' "
                        "means the DATA changed (e.g. a phase transition) even "
                        "if the fit still passes; recommend a thorough "
                        "re-analysis of such frames. Use for high-cadence "
                        "measurement streams; interpretation is deferred to a "
                        "post-experiment sweep."
                    )
                },
                "literature_file": {
                    "type": "string",
                    "description": (
                        "Path to a markdown file of literature / document "
                        "context — produced by `search_literature` (external "
                        "search) or `read_document` (user-provided papers). "
                        "When provided, its contents are injected into the "
                        "planner so the proposed analysis plan is grounded in "
                        "that literature. Skipped for the curve-fitting agent's "
                        "`task_mode='identification'` to preserve the unbiased "
                        "fit; in that case the literature still informs Stage-2 "
                        "candidate enumeration."
                    )
                },
                "r2_threshold": {
                    "type": "number",
                    "description": (
                        "CurveFitting agent only. Experienced-user override of the "
                        "R² acceptance threshold (between 0 and 1, e.g. 0.90). Use "
                        "ONLY when the user explicitly asks to loosen or tighten the "
                        "fit-quality bar. It overrides the default and any active "
                        "skill's R² gate (a warning is logged on conflict), but it "
                        "will NOT replace a skill that scores by a non-R² metric. "
                        "Leave unset for standard analyses."
                    )
                },
                "executor_timeout": {
                    "type": "integer",
                    "description": (
                        "All analysis agents. Per-run script-execution timeout "
                        "in seconds (default 600; the adaptive escalation "
                        "ladder retries a timed-out script at 2x up to 1800 s "
                        "before any LLM correction). RAISE it when a previous "
                        "run of the same data failed with 'Script execution "
                        "timed out' or when the dataset is very large and a "
                        "single fit legitimately needs longer; LOWER it (e.g. "
                        "30-60) for real-time/quick-look turnaround where a "
                        "slow fit should fail fast. Leave unset otherwise."
                    )
                },
                "max_series_refits": {
                    "type": "integer",
                    "description": (
                        "Curve fitting, SERIES runs only. Caps how many flagged "
                        "spectra (fits below the R² threshold with the locked "
                        "model) get an independent full re-analysis after the "
                        "series pass — each such refit is a complete LLM "
                        "planning/codegen/verification loop (minutes per unit). "
                        "Worst fits are re-analyzed first; the rest keep their "
                        "locked-model result (a valid, schema-consistent row) and "
                        "are listed under refit_skipped_by_budget. Set for large "
                        "series when turnaround matters (0 = no refits, e.g. a "
                        "closed loop that only needs the locked features); leave "
                        "unset for the standard behaviour (re-analyze every "
                        "flagged spectrum)."
                    )
                },
                "max_verification_iterations": {
                    "type": "integer",
                    "description": (
                        "All analysis agents. Controls how thorough the "
                        "verification/refinement loop is — the speed-vs-rigor "
                        "knob. Set it ONLY when the user asks about turnaround or "
                        "checking depth; leave unset for the standard (thorough) "
                        "default. Map the user's intent: a fast / quick / in-situ / "
                        "real-time / 'good enough' turnaround → 1 (a single check, "
                        "no refinement loop); 'skip/bypass/no verification' → 0 "
                        "(accept the first successful result with no verification "
                        "or retry loop at all; a failed result still enters the "
                        "recovery path); an explicit count (e.g. 'two verification "
                        "steps') → that integer; thorough / careful / publication / "
                        "post-experiment analysis → leave unset (defaults: 7 for "
                        "curve/image verification passes, 4 retries for "
                        "hyperspectral codegen). Often paired with r2_threshold "
                        "for curve fits (e.g. 'use R²=0.98 and a single "
                        "verification step')."
                    )
                },
                "starting_annealing_level": {
                    "type": "integer",
                    "description": (
                        "CurveFitting and Image analysis agents. The constraint-"
                        "annealing schedule normally starts FROZEN (skill rules "
                        "and model locked) and only relaxes toward full freedom "
                        "across verification iterations. Set this to start the "
                        "schedule HIGHER on a RE-RUN, so the agent does not waste "
                        "iterations repeating early constraint stages that a prior "
                        "run already showed to be inadequate. Levels: 0 = frozen "
                        "(default / first run — ALWAYS leave UNSET on a first "
                        "analysis), 1 = warm (constraints loosened), 2 = hot (full "
                        "freedom, fresh generation from scratch). Set it ONLY when "
                        "the user is re-running after an unsatisfactory fit and "
                        "wants to skip the early restrictive stages (e.g. 'this "
                        "didn't work — try again without the earlier constraints' "
                        "→ 2). Leave UNSET otherwise."
                    )
                },
                "n_candidates": {
                    "type": "integer",
                    "description": (
                        "Image analysis and curve fitting agents. Number of "
                        "independent anchor-analysis attempts run in PARALLEL; "
                        "an LLM judge compares the finished attempts "
                        "(verification scores/R² + output visualizations) and "
                        "locks the winner — reduces run-to-run variance. Leave "
                        "UNSET for the per-agent default: image analysis and "
                        "curve fitting auto-run in ESCALATION mode (the first "
                        "attempt is accepted immediately if strong; the rest "
                        "launch only if it is weak — so the extra cost is paid "
                        "only on a weak first attempt). Curve fitting suppresses "
                        "this to a single fit when a domain skill is active (the "
                        "candidates would converge on the mandated model); other "
                        "agents run 1. Set ONLY when the user explicitly asks "
                        "for more/fewer parallel attempts (e.g. 'try 5 "
                        "candidates' → 5; 'single attempt / cheapest run' → 1); "
                        "an explicit value forces exactly N parallel attempts "
                        "(no escalation, and not skill-suppressed)."
                    )
                }
            },
            required=[]
        )

        # =====================================================================
        # 6. LIST RESULTS
        # =====================================================================
        def list_results() -> str:
            """
            List analysis results in the session directory.
            Shows all analysis runs with their IDs and output directories.
            """
            print(f"  ⚡ Tool: Listing results...")
            
            results = []
            
            # List analysis directories in results folder
            results_dir = self.orch.results_dir
            if results_dir.exists():
                # Find all analysis directories
                analysis_dirs = sorted(
                    [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("analysis_")],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True  # Most recent first
                )
                
                for analysis_dir in analysis_dirs:
                    analysis_info = {
                        "directory": analysis_dir.name,
                        "path": str(analysis_dir),
                        "files": [],
                        "has_novelty_assessment": False
                    }
                    
                    # Check for metadata_used.json to get analysis details
                    metadata_file = analysis_dir / "metadata_used.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                meta = json.load(f)
                            analysis_info["analysis_id"] = meta.get("analysis_id")
                            analysis_info["data_path"] = meta.get("data_path")
                            analysis_info["agent_name"] = meta.get("agent_name")
                            analysis_info["timestamp"] = meta.get("timestamp")
                        except Exception:
                            pass
                    
                    # Check for novelty assessment
                    novelty_file = analysis_dir / "literature_assessment" / "novelty_report.json"
                    if novelty_file.exists():
                        analysis_info["has_novelty_assessment"] = True
                    
                    # List files in directory
                    for f in analysis_dir.iterdir():
                        if f.is_file():
                            analysis_info["files"].append(f.name)
                    
                    results.append(analysis_info)
            
            # Also include in-memory analysis history
            return json.dumps({
                "status": "success",
                "session_directory": str(self.orch.base_dir),
                "results_directory": str(results_dir),
                "total_analyses": len(results),
                "analyses": results,
                "in_memory_history": [
                    {
                        "analysis_id": r.get("analysis_id"),
                        "data_path": r.get("data_path"),
                        "agent_name": r.get("agent_name"),
                        "status": r.get("status"),
                        "output_directory": r.get("output_directory"),
                        "has_novelty_assessment": r.get("novelty_assessment") is not None,
                        # discoverability: downstream consumers already prefer
                        # the latest revision; this flag lets the agent SEE
                        # that a literature-refined interpretation exists
                        "interpretation_revisions": len(r.get("interpretation_revisions") or []),
                        "novelty_assessment_stale": bool((r.get("novelty_assessment") or {}).get("stale"))
                    }
                    for r in self.orch.analysis_results
                ]
            })
        
        self._register_tool(
            func=list_results,
            name="list_results",
            description=(
                "List all analysis results in the session. "
                "Shows analysis IDs, data paths, agents used, and output directories."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 7. SAVE CHECKPOINT
        # =====================================================================
        def save_checkpoint() -> str:
            """
            Save session state for later resumption.
            """
            print(f"  ⚡ Tool: Saving checkpoint...")
            
            try:
                checkpoint_data = {
                    "timestamp": datetime.now().isoformat(),
                    "current_metadata": self.orch.current_metadata,
                    "current_metadata_owner": self.orch.current_metadata_owner,
                    "current_data_path": self.orch.current_data_path,
                    "current_data_type": self.orch.current_data_type,
                    "selected_agent_id": self.orch.selected_agent_id,
                    "analysis_results": self.orch.analysis_results,
                    "analysis_run_counter": self.orch._analysis_run_counter,
                    "message_count": self.orch.message_count,
                    "analysis_mode": self.orch.analysis_mode.value,
                    "active_knowledge": self.orch.active_knowledge,
                    "graduated_skill_sources": self.orch._graduated_skill_sources,
                }

                with open(self.orch.checkpoint_path, 'w', encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, indent=2)

                return json.dumps({
                    "status": "success",
                    "checkpoint_path": str(self.orch.checkpoint_path),
                    "timestamp": checkpoint_data["timestamp"],
                    "analyses_saved": len(self.orch.analysis_results)
                })
                
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=save_checkpoint,
            name="save_checkpoint",
            description=(
                "Save complete session state including metadata, analysis history, "
                "and conversation. Use this to enable session resumption."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 8. SHOW AVAILABLE AGENTS
        # =====================================================================
        def show_available_agents() -> str:
            """
            Show list of available analysis agents and their capabilities,
            plus any custom tools registered via register_tools().
            """
            print(f"  ⚡ Tool: Showing available agents...")

            agents = []
            for agent_id in sorted(self.AGENT_NAMES.keys()):
                agents.append({
                    "id": agent_id,
                    "name": self.AGENT_NAMES[agent_id],
                    "description": self.AGENT_DESCRIPTIONS[agent_id]
                })

            available_skills = list_all_skills()
            custom_skills = getattr(self.orch, "_custom_skills", {})
            if custom_skills:
                available_skills["custom"] = sorted(custom_skills.keys())

            result = {
                "status": "success",
                "agents": agents,
                "current_selection": self.orch.selected_agent_id,
                "available_skills": available_skills,
            }

            external_tools = getattr(self.orch, "_external_tools", [])
            if external_tools:
                result["custom_tools"] = external_tools
                result["custom_tools_note"] = (
                    "These tools are callable directly by name and operate on "
                    "the current data file (set via examine_data)."
                )

            return json.dumps(result)

        self._register_tool(
            func=show_available_agents,
            name="show_available_agents",
            description=(
                "Show list of available analysis agents and their capabilities, "
                "plus any custom tools registered for the current session."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 9. GET METADATA SCHEMA
        # =====================================================================
        def get_metadata_schema() -> str:
            """
            Get the metadata JSON schema for reference.
            """
            print(f"  ⚡ Tool: Getting metadata schema...")
            
            return json.dumps({
                "status": "success",
                "schema": METADATA_SCHEMA_DICT,
                "required_fields": ["experiment_type", "experiment", "sample"],
                "hint": "Use convert_metadata to create metadata from natural language"
            })
        
        self._register_tool(
            func=get_metadata_schema,
            name="get_metadata_schema",
            description=(
                "Get the metadata JSON schema showing required and optional fields. "
                "Use this to understand what metadata is needed."
            ),
            parameters={},
            required=[]
        )
        
        # =====================================================================
        # 10. GET MEASUREMENT RECOMMENDATIONS (UPDATED)
        # =====================================================================
        def get_recommendations(analysis_id: str = None, analysis_index: int = -1) -> str:
            """
            Get measurement recommendations from a completed analysis.
            Can specify by analysis_id or by index in the history.
            
            UPDATED: Now incorporates novelty assessment results to prioritize
            recommendations based on scientific novelty.
            """
            print(f"  ⚡ Tool: Getting measurement recommendations...")
            
            if not self.orch.analysis_results:
                return json.dumps({
                    "status": "error",
                    "message": "No analyses completed yet. Run an analysis first."
                })
            
            try:
                # Find the analysis record
                record = None
                
                if analysis_id:
                    # Search by analysis_id
                    for r in self.orch.analysis_results:
                        if r.get("analysis_id") == analysis_id:
                            record = r
                            break
                    if record is None:
                        return json.dumps({
                            "status": "error",
                            "message": f"Analysis not found: {analysis_id}"
                        })
                else:
                    # Use index
                    record = self.orch.analysis_results[analysis_index]
                
                agent_id = record.get("agent_id")
                if agent_id is None:
                    return json.dumps({
                        "status": "error",
                        "message": "Analysis record missing agent_id"
                    })
                
                # Get the stored analysis result
                full_result = record.get("full_result")
                if full_result is not None:
                    full_result = _effective_full_result(record)
                if full_result is None:
                    return json.dumps({
                        "status": "error",
                        "message": "Analysis result not stored. Please run the analysis again."
                    })
                
                # Get novelty assessment if available
                novelty_assessment = record.get("novelty_assessment")
                
                # Create agent for recommendations (uses same output dir)
                output_dir = record.get("output_directory", str(self.orch.results_dir / "temp"))
                agent = self.orch.create_agent_for_analysis(agent_id, output_dir)
                
                # Call recommend_measurements with the stored result AND novelty assessment
                result = agent.recommend_measurements(
                    data=record.get("data_path"),
                    system_info=self.orch.current_metadata,
                    analysis_result=full_result,
                    novelty_assessment=novelty_assessment  # NEW: Pass novelty data
                )
                
                response = {
                    "status": result.get("status", "success"),
                    "analysis_id": record.get("analysis_id"),
                    "recommendations": result.get("measurement_recommendations", []),
                    "analysis_integration": result.get("analysis_integration", ""),
                    "novelty_informed": novelty_assessment is not None,
                    "novelty_assessment_stale": bool((novelty_assessment or {}).get("stale"))
                }
                
                # Add novelty-specific recommendations if available
                if novelty_assessment:
                    response["novelty_summary"] = {
                        "total_claims_assessed": len(novelty_assessment.get("assessments", [])),
                        "high_novelty_claims": len(novelty_assessment.get("high_novelty_claims", [])),
                        "novelty_driven_recommendations": result.get("novelty_recommendations", [])
                    }
                
                return json.dumps(response)
                
            except Exception as e:
                self.logger.error(f"Recommendations error: {e}", exc_info=True)
                return json.dumps({
                    "status": "error",
                    "message": str(e)
                })
        
        self._register_tool(
            func=get_recommendations,
            name="get_recommendations",
            description=(
                "Get measurement recommendations based on a completed analysis. "
                "Specify by analysis_id or use analysis_index (-1 for most recent). "
                "Returns suggested follow-up experiments and measurements. "
                "If assess_novelty was run first, recommendations are prioritized "
                "based on scientific novelty (high-novelty claims get validation experiments)."
            ),
            parameters={
                "analysis_id": {
                    "type": "string",
                    "description": "Specific analysis ID to get recommendations for"
                },
                "analysis_index": {
                    "type": "integer",
                    "description": "Index of analysis in history (-1 for most recent)"
                }
            },
            required=[]
        )

        # =====================================================================
        # 10a2. RECONCILE SERIES — couple a profile-fitting pass with an ID pass
        # =====================================================================
        def reconcile_series(profile_analysis: int = None,
                             identification_analysis: int = None,
                             profile_id: str = None,
                             identification_id: str = None,
                             tol: float = None,
                             regime_window_frac: float = 0.33,
                             crossover_threshold: float = 0.5) -> str:
            """Reconcile two PRIOR series analyses of the same frames: a
            profile-fitting pass (peak/line fitting — HOW the structure
            evolves; fits a lineshape model but is database-independent, it
            needs no reference) and an identification pass (WHICH phases, by
            reference matching). Attributes the fitted feature-evolution
            trends to the identified labels and cross-checks the transition
            the two find independently. Technique-agnostic — works for any
            series where one pass fits peaks and the other identifies.
            Reference each prior analysis by index (position in history) or
            analysis_id."""
            print("  ⚡ Tool: Reconciling profile-fit + identification series...")
            from scilink.skills._shared._reconcile import reconcile_analysis_dirs

            def _find(idx, aid, role):
                if aid:
                    rec = next((r for r in self.orch.analysis_results
                                if r.get("analysis_id") == aid), None)
                    if rec is None:
                        raise ValueError(f"{role} analysis id not found: {aid}")
                    return rec
                if idx is None:
                    raise ValueError(f"provide {role} analysis by index or id")
                return self.orch.analysis_results[idx]

            if not self.orch.analysis_results:
                return json.dumps({"status": "error",
                                   "message": "No analyses yet. Run the "
                                   "profile-fitting and identification passes "
                                   "first, then reconcile."})
            try:
                prof = _find(profile_analysis, profile_id, "profile")
                idr = _find(identification_analysis, identification_id, "identification")
                fig = str(self.orch.results_dir / "reconciled_series.png")
                report = str(self.orch.results_dir / "reconciled_series_report.html")
                svar = ((getattr(self.orch, "current_metadata", None) or {}).get("series_variable")
                        or (getattr(self.orch, "current_metadata", None) or {}).get("variable")
                        or "series variable")
                r = reconcile_analysis_dirs(
                    prof.get("output_directory"), idr.get("output_directory"),
                    output_figure=fig, output_report=report,
                    series_variable=str(svar),
                    tol=(float(tol) if tol is not None else None),
                    regime_window_frac=float(regime_window_frac),
                    crossover_threshold=float(crossover_threshold))
                r["status"] = "success"
                # Persist the result so finalize_reconcile_report can re-render
                # the report with the LLM's interpretation WITHOUT recomputing.
                # (Persist BEFORE attaching the figure bytes so the cache stays
                # lean — finalize does not need the base64.)
                r["series_variable"] = str(svar)
                try:
                    (self.orch.results_dir / "reconciled_series_result.json").write_text(
                        json.dumps(r, default=str), encoding="utf-8")
                except Exception:
                    pass
                # Attach the reconciled figure so the orchestrator LLM sees the
                # plot (providers that render tool-result images) before it
                # writes the interpretation — figure-grounded, not number-only.
                try:
                    figpath = r.get("figure") or fig
                    if figpath and Path(figpath).is_file():
                        import base64 as _b64
                        r["image_base64"] = _b64.b64encode(
                            Path(figpath).read_bytes()).decode()
                except Exception:
                    pass
                # The report currently carries only the computed numbers + the
                # deterministic note — no synthesis, unlike the component
                # profile-fitting and identification reports. Ask the LLM to
                # supply one and finalize, so the coupled report is not the only
                # one missing an interpretation.
                r["next_step"] = (
                    "Write a 2-4 sentence scientific interpretation of this "
                    "reconciliation — what transforms into what, whether the two "
                    "independent transition estimates corroborate each other "
                    "(agreement verdict above), and any caveat the numbers imply "
                    "(a divergent verdict, an unidentified regime, a possible "
                    "multi-step process) — then call finalize_reconcile_report("
                    "interpretation=...) to embed it as the report's Interpretation "
                    "section. That report is the coupled deliverable to surface.")
                return json.dumps(r, default=str)
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=reconcile_series,
            name="reconcile_series",
            description=(
                "Couple two PRIOR series analyses of the same frames — a "
                "profile-fitting pass (peak/line fitting) and an identification "
                "pass — into the combined view: phase/species-labeled "
                "feature-evolution trends with a cross-validated transition. "
                "Run BOTH passes first (run_analysis with the profile-fitting "
                "skill, then with the identification skill), then call this. "
                "The fitted trends are database-independent (they work even "
                "where identification cannot name a phase — organics, novel "
                "products); a regime the ID pass could not name stays honestly "
                "'unidentified'. Agreement of the two transitions is "
                "corroboration; divergence is a flag. Reference each analysis "
                "by index or analysis_id. Produces a self-contained HTML "
                "report (transition summary, phase labels, embedded figure, "
                "tracked-feature table) and a figure; their paths are in the "
                "returned 'report' / 'figure' fields. The report carries the "
                "computed numbers but no synthesis yet — after this returns, "
                "write a short interpretation and call finalize_reconcile_report "
                "to embed it, then surface that report as the coupled deliverable."
            ),
            parameters={
                "profile_analysis": {"type": "integer", "description": "History index of the profile-fitting (peak/line-fit) pass."},
                "identification_analysis": {"type": "integer", "description": "History index of the identification pass."},
                "profile_id": {"type": "string", "description": "analysis_id of the profile-fitting pass (alternative to index)."},
                "identification_id": {"type": "string", "description": "analysis_id of the identification pass (alternative to index)."},
                "tol": {"type": "number", "description": "Feature-tracking position tolerance in the data's x-units (2θ° / ppm / eV). Omit to auto-scale from the peak spacing (recommended — no technique-specific default). Set explicitly to RAISE for features that drift a lot across the series, or LOWER to keep close features apart."},
                "regime_window_frac": {"type": "number", "description": "Fraction of frames at each end used to classify start- vs end-phase features (default 0.33). LOWER when endpoint frames are pure; RAISE toward 0.5 for a gradual transformation."},
                "crossover_threshold": {"type": "number", "description": "End-phase weight-share level defining the transition (default 0.5 ≈ 50% conversion). Change only to mark a different conversion fraction."},
            },
            required=[]
        )

        # =====================================================================
        # 10a3. FINALIZE RECONCILE REPORT — embed the LLM synthesis
        # =====================================================================
        def finalize_reconcile_report(interpretation: str) -> str:
            """Embed a scientific interpretation into the most recent reconcile
            report. Call this AFTER reconcile_series, passing your own synthesis
            of the coupled result (what transforms into what, whether the two
            transition estimates corroborate, any caveat). Re-renders the report
            in place with an Interpretation section — no recomputation — so the
            coupled deliverable reads like the component reports, which carry a
            narrative rather than only numbers."""
            print("  ⚡ Tool: Finalizing reconcile report with interpretation...")
            from scilink.skills._shared._reconcile import render_reconcile_report
            try:
                cache = self.orch.results_dir / "reconciled_series_result.json"
                if not cache.is_file():
                    return json.dumps({"status": "error", "message":
                                       "No reconcile result to finalize. Run "
                                       "reconcile_series first."})
                result = json.loads(cache.read_text())
                report_path = result.get("report") or str(
                    self.orch.results_dir / "reconciled_series_report.html")
                render_reconcile_report(
                    result, report_path,
                    series_variable=result.get("series_variable", "series variable"),
                    interpretation=interpretation)
                return json.dumps({"status": "success", "report": report_path,
                                   "message": "Interpretation embedded; this "
                                   "report is the coupled deliverable to surface "
                                   "to the user."})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=finalize_reconcile_report,
            name="finalize_reconcile_report",
            description=(
                "Embed your scientific interpretation into the reconcile report "
                "produced by reconcile_series. Call this immediately after "
                "reconcile_series with a 2-4 sentence synthesis of the coupled "
                "result — what transforms into what, whether the two independent "
                "transition estimates corroborate (the agreement verdict), and "
                "any caveat (divergence, an unidentified regime, a possible "
                "multi-step process). Re-renders the report in place with an "
                "Interpretation section so the coupled deliverable carries a "
                "narrative like the component reports do. The transition numbers "
                "stay computed — this narrates them, it does not change them."
            ),
            parameters={
                "interpretation": {"type": "string", "description": "Your 2-4 sentence scientific synthesis of the reconciled result. Narrate the computed transition/agreement; do not invent numbers."},
            },
            required=["interpretation"]
        )

        # =====================================================================
        # 10b. SEARCH LITERATURE (preparatory — call BEFORE run_analysis)
        # =====================================================================
        def search_literature(query: str) -> str:
            """
            Search scientific literature for context to inform an upcoming
            analysis. Returns a file path that should be passed as
            `literature_file` to the next `run_analysis` call so the planner
            can produce a literature-informed plan.
            """
            print(f"  ⚡ Tool: Searching literature for '{query[:80]}...'")

            if not self.orch.futurehouse_api_key:
                return json.dumps({
                    "status": "error",
                    "message": "No FutureHouse/Edison API Key provided in Orchestrator initialization."
                })

            try:
                # 50-min ceiling matches plan mode's `LiteratureSearchAgent`
                # default; Edison CROW jobs can take 20-30 min for harder queries.
                lit_agent = FittingModelLiteratureAgent(
                    api_key=self.orch.futurehouse_api_key, max_wait_time=3000
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Failed to init Literature Agent: {e}"})

            # Refine the orchestrator-LLM's draft query using the loaded data
            # preview + metadata (visual + experimental specifics). Best-effort:
            # falls back to the raw query on any failure.
            refined_query = optimize_query_for_analysis(
                raw_query=query,
                data_type=getattr(self.orch, "current_data_type", None),
                data_path=getattr(self.orch, "current_data_path", None),
                metadata=getattr(self.orch, "current_metadata", None),
                model=self.orch.model,
            )
            if refined_query != query:
                print(f"  🔍 Refined query: {refined_query}")
                logging.info(f"search_literature: refined query → {refined_query}")
            else:
                print(f"  🔍 Using raw query (no refinement applied)")

            try:
                result = lit_agent.query_for_models(refined_query)
            except Exception as e:
                logging.error(f"Literature search error: {e}", exc_info=True)
                return json.dumps({"status": "error", "message": str(e)})

            if result.get("status") != "success":
                return json.dumps({
                    "status": result.get("status", "error"),
                    "message": result.get("message", "Literature search did not succeed")
                })

            content = result.get("formatted_answer", "") or ""

            # Hash the raw query for an idempotent, collision-free filename:
            # same raw query → same file (re-runnable), different queries → different files.
            query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
            lit_path = self.orch.base_dir / f"literature_search_{query_hash}.md"
            header = f"# Literature Search Results\n\n**Draft query:** {query}\n"
            if refined_query != query:
                header += f"**Refined query:** {refined_query}\n"
            write_text_utf8(lit_path, f"{header}\n{content}")

            print(f"  ✅ Literature search completed. Saved to {lit_path.name}")

            preview = content[:500] + "..." if len(content) > 500 else content
            return json.dumps({
                "status": "success",
                "file_path": str(lit_path),
                "content_preview": preview,
                "hint": "Pass file_path as literature_file to the next run_analysis() call."
            })

        self._register_tool(
            func=search_literature,
            name="search_literature",
            description=(
                "Search scientific literature via the FutureHouse Edison API to gather "
                "context that will inform an upcoming analysis. Call BEFORE run_analysis(); "
                "pass the returned file_path as `literature_file` to run_analysis() so the "
                "planner produces a literature-informed plan."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "A focused research question (e.g., 'methods for detecting grain boundaries in HRTEM images of 2D materials')."
                }
            },
            required=["query"]
        )


        # =====================================================================
        # REFINE INTERPRETATION (issue #323 — Channel B; feature-conditioned
        # literature. Curve + image + hyperspectral — feature surfacing per
        # issue #327 phase 2.)
        # =====================================================================
        def refine_interpretation(analysis_id: str = None, analysis_index: int = -1,
                                  focus: str = None) -> str:
            """
            Post-analysis, feature-conditioned literature refinement (Channel B).
            Searches the literature FROM the fitted features (peak positions,
            trends) of a completed analysis and revises its interpretation
            against what is published. Append-only: the original interpretation
            is preserved; revisions accumulate on the analysis record.
            """
            print(f"  ⚡ Tool: Refining interpretation against feature-conditioned literature...")

            if not self.orch.futurehouse_api_key:
                return json.dumps({
                    "status": "error",
                    "message": "No FutureHouse/Edison API Key provided in Orchestrator initialization."
                })

            # 1. Retrieve the analysis record (same contract as assess_novelty).
            record = None
            record_index = None
            if analysis_id:
                for i, r in enumerate(self.orch.analysis_results):
                    if r.get("analysis_id") == analysis_id:
                        record, record_index = r, i
                        break
                if record is None:
                    return json.dumps({"status": "error",
                                       "message": f"Analysis ID not found: {analysis_id}"})
            else:
                if not self.orch.analysis_results:
                    return json.dumps({"status": "error",
                                       "message": "No analysis history available."})
                record_index = (analysis_index if analysis_index >= 0
                                else len(self.orch.analysis_results) + analysis_index)
                record = self.orch.analysis_results[record_index]

            full_result = record.get("full_result") or {}
            detailed = (full_result.get("detailed_analysis")
                        or full_result.get("full_analysis") or "")
            if not detailed.strip():
                return json.dumps({"status": "error",
                                   "message": "Analysis record has no interpretation text to refine."})

            # 2. Surface the measured features.
            # Curve: series runs are trend-conditioned — once a parameter trend
            # exists, static per-spectrum peaks barely matter (issue #323 §5.2).
            features = {}
            for key in ("model_type", "fit_quality"):
                if full_result.get(key):
                    features[key] = full_result[key]
            trends = full_result.get("parameter_trends")
            if trends:
                features["parameter_trends"] = trends
                if full_result.get("flagged_spectra_analysis"):
                    features["flagged_spectra_analysis"] = full_result["flagged_spectra_analysis"]
                locked = (full_result.get("summary") or {}).get("locked_model") if isinstance(full_result.get("summary"), dict) else None
                if locked:
                    features["locked_model"] = locked
            elif full_result.get("fitting_parameters"):
                features["fitting_parameters"] = full_result["fitting_parameters"]
            # Image + hyperspectral: both now surface extracted_features at the
            # top level of the result (issue #327 phase 2) — this is what lifts
            # the tool from "curve fitting v1" to all three modalities.
            if full_result.get("extracted_features"):
                features["extracted_features"] = full_result["extracted_features"]
            if not features:
                return json.dumps({"status": "error",
                                   "message": "No fitted features surfaced on this record — nothing to condition the search on."})
            features_text = json.dumps(features, default=str)[:4000]

            # 3. Feature → query builder: an LLM micro-call, because feature
            # MEANING is technique-dependent (peaks at 1349/1582 cm^-1 only
            # read as D/G bands with domain context). Falls back to a plain
            # template on any failure.
            focus_line = f"\nUser focus: {focus}" if focus else ""
            builder_prompt = (
                "You are building ONE focused scientific-literature search query.\n"
                "A spectral/curve fit produced these measured features:\n"
                f"{features_text}\n"
                f"Experimental context: {json.dumps(getattr(self.orch, 'current_metadata', None) or {}, default=str)[:800]}"
                f"{focus_line}\n\n"
                "Write a single natural-language query (<= 40 words) that asks what "
                "materials/phases/processes are reported for these SPECIFIC measured "
                "features (positions, widths, trends) — not a generic method query. "
                "Return ONLY the query text."
            )
            try:
                q_resp = self._internal_model().generate_content(contents=[builder_prompt])
                query = (q_resp.text or "").strip().strip('"')
                assert 10 < len(query) < 400
            except Exception:
                query = f"Interpretation of measured features: {features_text[:200]}"
            print(f"  🔍 Feature-conditioned query: {query[:100]}")

            # 4. Literature search (CROW backend — "what is reported for these
            # features", not the novelty question).
            try:
                lit_agent = FittingModelLiteratureAgent(
                    api_key=self.orch.futurehouse_api_key, max_wait_time=3000
                )
                result = lit_agent.query_for_models(query)
            except Exception as e:
                logging.error(f"refine_interpretation literature error: {e}", exc_info=True)
                return json.dumps({"status": "error", "message": str(e)})
            if result.get("status") != "success":
                return json.dumps({"status": result.get("status", "error"),
                                   "message": result.get("message", "Literature search did not succeed")})
            content = result.get("formatted_answer", "") or ""

            # Persist the search output (inspectable, same shape as search_literature).
            query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
            lit_path = self.orch.base_dir / f"interpretation_lit_{query_hash}.md"
            write_text_utf8(
                lit_path,
                f"# Feature-Conditioned Literature (Channel B)\n\n"
                f"**Analysis:** {record.get('analysis_id')}\n"
                f"**Query:** {query}\n\n{content}")

            # 5. Tier-A refinement: text-in/text-out — revise the existing
            # interpretation against the literature. No pixel/state re-invoke.
            refine_prompt = (
                "You are revising the interpretation of a completed analysis "
                "against literature that was searched from its MEASURED features.\n\n"
                f"## Current interpretation\n{detailed[:6000]}\n\n"
                f"## Measured features\n{features_text}\n\n"
                f"## Feature-conditioned literature\n{content[:8000]}\n\n"
                "Rewrite the interpretation: keep every conclusion the data still "
                "supports, revise or qualify what the literature contradicts, and "
                "add what it newly explains (cite the literature inline where used). "
                "Where the literature identifies the material/phase from these "
                "features, say so explicitly. Do not soften data-supported "
                "conclusions merely because the literature is silent. Return ONLY "
                "the revised interpretation text."
            )
            try:
                r_resp = self._internal_model().generate_content(contents=[refine_prompt])
                revised = (r_resp.text or "").strip()
                if not revised:
                    raise ValueError("empty refinement")
            except Exception as e:
                return json.dumps({"status": "error",
                                   "message": f"Literature saved to {lit_path} but refinement failed: {e}"})

            # Append the revised interpretation to the same document so the
            # revision is human-readable on disk, not only in session state.
            write_text_utf8(
                lit_path,
                "\n\n---\n\n## Literature-Refined Interpretation\n\n" + revised,
                append=True)

            # Companion HTML next to the agent's original report (append-only:
            # a separate document, mirroring how assess_novelty writes its own
            # doc rather than rewriting the analysis). Best-effort.
            report_html = None
            out_dir = record.get("output_directory")
            if out_dir and Path(out_dir).is_dir():
                try:
                    import html as _html
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_html = Path(out_dir) / f"Interpretation_Revision_{ts}.html"
                    paras = "".join(f"<p>{_html.escape(x)}</p>"
                                    for x in revised.split("\n\n") if x.strip())
                    # Same fixed palette as the agent's CurveFitting report so
                    # the two render identically in the UI's light AND dark
                    # modes (the report theme is deliberately self-contained,
                    # not inherited from the embedder).
                    report_html.write_text(
                        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                        "<title>Literature-Refined Interpretation</title><style>"
                        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;"
                        " line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto;"
                        " padding: 20px; background-color: #f4f4f9; }"
                        ".container { background-color: #fff; padding: 40px; border-radius: 8px;"
                        " box-shadow: 0 2px 10px rgba(0,0,0,0.1); }"
                        "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }"
                        ".metadata-box { background-color: #ecf0f1; padding: 15px; border-radius: 5px;"
                        " border-left: 5px solid #3498db; margin-bottom: 20px; }"
                        ".analysis-text { white-space: pre-wrap; background-color: #fafafa;"
                        " padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }"
                        ".footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }"
                        "</style></head><body><div class='container'>"
                        "<h1>Literature-Refined Interpretation</h1>"
                        "<div class='metadata-box'>"
                        f"<b>Analysis:</b> {_html.escape(str(record.get('analysis_id')))}<br>"
                        f"<b>Query:</b> {_html.escape(query)}<br>"
                        f"<b>Literature:</b> {_html.escape(lit_path.name)}</div>"
                        f"<div class='analysis-text'>{paras}</div>"
                        "<div class='footer'>Post-fit revision (feature-conditioned "
                        "literature); the original report is unchanged.</div>"
                        "</div></body></html>", encoding="utf-8")
                except Exception as e:
                    logging.warning(f"Companion revision HTML failed: {e}")
                    report_html = None

            # Revise the scientific claims to match the new interpretation
            # (optional per plan — skipped on any parse failure). Without
            # this, a post-revision novelty re-run would re-assess claims
            # anchored to the superseded reading.
            revised_claims = None
            orig_claims = full_result.get("scientific_claims") or []
            if orig_claims:
                claims_prompt = (
                    "An analysis interpretation was revised against literature. "
                    "Update its scientific claims to match the REVISED interpretation: "
                    "keep claims still supported, revise ones the new reading changes, "
                    "drop invalidated ones, add clearly warranted new ones.\n\n"
                    f"## Revised interpretation\n{revised[:6000]}\n\n"
                    f"## Original claims (JSON)\n{json.dumps(orig_claims)[:4000]}\n\n"
                    "Return ONLY a JSON array with the SAME schema per claim "
                    "(claim, has_anyone_question, keywords, scientific_impact)."
                )
                try:
                    c_resp = self._internal_model().generate_content(contents=[claims_prompt])
                    raw = (c_resp.text or "").strip()
                    m = re.search(r"\[.*\]", raw, re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else None
                    if (isinstance(parsed, list) and parsed
                            and all(isinstance(c, dict) and c.get("claim") for c in parsed)):
                        revised_claims = parsed
                except Exception as e:
                    logging.warning(f"Claims revision skipped (parse failure): {e}")

            # 6. Append-only storage (mirrors novelty_assessment attach).
            revision = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "literature_file": str(lit_path),
                "revised_analysis": revised,
            }
            if revised_claims:
                revision["revised_claims"] = revised_claims
            if report_html:
                revision["report_html"] = str(report_html)
            self.orch.analysis_results[record_index].setdefault(
                "interpretation_revisions", []).append(revision)

            # Ripple: a prior novelty assessment was made against claims that
            # predate this revision — mark it stale so consumers and the agent
            # know it needs a re-run to be current.
            novelty_stale = False
            prior_novelty = self.orch.analysis_results[record_index].get("novelty_assessment")
            if prior_novelty:
                prior_novelty["stale"] = True
                prior_novelty["staled_by_revision"] = revision["timestamp"]
                novelty_stale = True

            print(f"  ✅ Interpretation refined ({len(revised)} chars). Literature: {lit_path.name}")
            return json.dumps({
                "status": "success",
                "analysis_id": record.get("analysis_id"),
                "query": query,
                "literature_file": str(lit_path),
                "revised_interpretation_preview": revised[:600],
                "report_html": str(report_html) if report_html else None,
                "claims_revised": bool(revised_claims),
                "prior_novelty_assessment_stale": novelty_stale,
                "note": "Original interpretation preserved; revision appended to the record."
                        + (" A prior novelty assessment predates this revision — "
                           "re-run assess_novelty to update it." if novelty_stale else "")
            })

        self._register_tool(
            func=refine_interpretation,
            name="refine_interpretation",
            description=(
                "Refine a completed analysis's interpretation against literature searched "
                "from its FITTED features (peak positions/widths, parameter trends) — the "
                "post-fit counterpart of search_literature. Call AFTER run_analysis when the "
                "interpretation would materially benefit from published context. Strongly "
                "recommended after identification-mode runs (they are literature-free in-run; "
                "this step is what identifies the material from the fitted bands). Append-only: "
                "the original interpretation is preserved."
            ),
            parameters={
                "analysis_id": {"type": "string",
                                "description": "ID of the completed analysis (default: most recent)."},
                "analysis_index": {"type": "integer",
                                   "description": "Alternative: index into the analysis history (default -1, most recent)."},
                "focus": {"type": "string",
                          "description": "Optional steer for the literature query (e.g., 'candidate phases for the 742 cm-1 band')."},
            },
            required=[]
        )

        # =====================================================================
        # RE-ENTER INTERPRETATION (issue #322 — synthesis re-entry with a
        # human/orchestrator critique; the non-literature sibling of
        # refine_interpretation, built on SynthesisReEntryController.)
        # =====================================================================
        def reenter_interpretation(critique: str, analysis_id: str = None,
                                   analysis_index: int = -1) -> str:
            """
            Re-run ONLY the interpretation of a completed analysis with an
            injected critique — no per-unit re-analysis, no literature search.
            Append-only: the original interpretation is preserved; revisions
            accumulate on the analysis record.
            """
            print(f"  ⚡ Tool: Re-entering interpretation with critique...")

            if not critique or not str(critique).strip():
                return json.dumps({"status": "error",
                                   "message": "A non-empty critique is required."})

            # 1. Retrieve the analysis record (same contract as assess_novelty /
            # refine_interpretation).
            record = None
            record_index = None
            if analysis_id:
                for i, r in enumerate(self.orch.analysis_results):
                    if r.get("analysis_id") == analysis_id:
                        record, record_index = r, i
                        break
                if record is None:
                    return json.dumps({"status": "error",
                                       "message": f"Analysis ID not found: {analysis_id}"})
            else:
                if not self.orch.analysis_results:
                    return json.dumps({"status": "error",
                                       "message": "No analysis history available."})
                record_index = (analysis_index if analysis_index >= 0
                                else len(self.orch.analysis_results) + analysis_index)
                record = self.orch.analysis_results[record_index]

            full_result = record.get("full_result") or {}
            detailed = (full_result.get("detailed_analysis")
                        or full_result.get("full_analysis") or "")
            if not detailed.strip():
                return json.dumps({"status": "error",
                                   "message": "Analysis record has no interpretation text to revise."})

            # 2. Build the payload + surfaced features, and revise via the
            # shared Tier-A re-entry controller (one mechanism for all
            # critique producers — see analysis_qc_unification_plan.md §4).
            from ._critique import CritiquePayload
            from .base_agent import BaseAnalysisAgent, LLMAgentMixin
            from .controllers.base_controllers import SynthesisReEntryController

            payload = CritiquePayload(source="human", critique=str(critique))
            features = BaseAnalysisAgent.surface_features_for_reentry(full_result)
            features_block = (json.dumps(features, indent=2, default=str)[:4000]
                              if features else "")

            class _Parse(LLMAgentMixin):
                def __init__(self):
                    self.logger = logging.getLogger("reenter_interpretation")

            controller = SynthesisReEntryController(
                model=self._internal_model(),
                logger=logging.getLogger("reenter_interpretation"),
                generation_config=None,
                safety_settings=None,
                parse_fn=_Parse()._parse_llm_response,
            )
            # Use the CURRENT effective interpretation (latest revision if one
            # exists) so successive critiques compose instead of each starting
            # from the original text.
            effective = dict(full_result)
            prior_revs = record.get("interpretation_revisions") or []
            if prior_revs and prior_revs[-1].get("revised_analysis"):
                effective["detailed_analysis"] = prior_revs[-1]["revised_analysis"]
                if prior_revs[-1].get("revised_claims"):
                    effective["scientific_claims"] = prior_revs[-1]["revised_claims"]

            revision, error = controller.revise(
                effective, payload, features_block=features_block,
                system_info=self.orch.current_metadata
                if isinstance(getattr(self.orch, "current_metadata", None), dict)
                else None,
            )
            if error:
                return json.dumps({"status": "error",
                                   "message": f"Re-entry failed: {error}"})

            # 3. Append-only storage (same shape as refine_interpretation so
            # _effective_full_result overlays both kinds uniformly) + novelty
            # staleness ripple.
            entry = {
                "timestamp": datetime.now().isoformat(),
                "source": payload.source,
                "critique": payload.critique,
                "revised_analysis": revision["detailed_analysis"],
                "revision_summary": revision.get("revision_summary", ""),
            }
            claims = revision.get("scientific_claims") or []
            if (isinstance(claims, list) and claims
                    and all(isinstance(c, dict) and c.get("claim") for c in claims)):
                entry["revised_claims"] = claims
            self.orch.analysis_results[record_index].setdefault(
                "interpretation_revisions", []).append(entry)

            novelty_stale = False
            prior_novelty = self.orch.analysis_results[record_index].get("novelty_assessment")
            if prior_novelty:
                prior_novelty["stale"] = True
                prior_novelty["staled_by_revision"] = entry["timestamp"]
                novelty_stale = True

            print(f"  ✅ Interpretation revised ({len(entry['revised_analysis'])} chars).")
            return json.dumps({
                "status": "success",
                "analysis_id": record.get("analysis_id"),
                "revision_summary": entry["revision_summary"],
                "revised_interpretation_preview": entry["revised_analysis"][:600],
                "claims_revised": "revised_claims" in entry,
                "prior_novelty_assessment_stale": novelty_stale,
                "note": "Original interpretation preserved; revision appended to the record."
                        + (" A prior novelty assessment predates this revision — "
                           "re-run assess_novelty to update it." if novelty_stale else "")
            })

        self._register_tool(
            func=reenter_interpretation,
            name="reenter_interpretation",
            description=(
                "Revise a completed analysis's interpretation using a critique or extra "
                "context (from the user or your own review) WITHOUT re-running the "
                "analysis and WITHOUT a literature search — the cheap way to fix or "
                "sharpen conclusions when the fits/segmentations are fine but the "
                "reading of them should change. Successive calls compose (each revises "
                "the latest revision). Append-only: the original interpretation is "
                "preserved. For literature-driven revision use refine_interpretation "
                "instead."
            ),
            parameters={
                "critique": {"type": "string",
                             "description": "The critique/context to revise against (e.g., 'the trend has a break at 270 K — treat it as two regimes', or the user's correction)."},
                "analysis_id": {"type": "string",
                                "description": "ID of the completed analysis (default: most recent)."},
                "analysis_index": {"type": "integer",
                                   "description": "Alternative: index into the analysis history (default -1, most recent)."},
            },
            required=["critique"]
        )

        # =====================================================================
        # 11. ASSESS NOVELTY
        # =====================================================================
        def assess_novelty(analysis_id: str = None, analysis_index: int = -1) -> str:
            """
            Perform a literature search and novelty assessment on claims generated 
            by a previous analysis run.
            
            UPDATED: Now stores results in the analysis record for use by
            get_recommendations.
            """
            print(f"  ⚡ Tool: Assessing novelty for analysis...")

            if not self.orch.futurehouse_api_key:
                return json.dumps({
                    "status": "error",
                    "message": "No FutureHouse/Edison API Key provided in Orchestrator initialization."
                })

            # 1. Retrieve the analysis record
            record = None
            record_index = None
            
            if analysis_id:
                for i, r in enumerate(self.orch.analysis_results):
                    if r.get("analysis_id") == analysis_id:
                        record = r
                        record_index = i
                        break
                if record is None:
                    return json.dumps({"status": "error", "message": f"Analysis ID not found: {analysis_id}"})
            else:
                if not self.orch.analysis_results:
                    return json.dumps({"status": "error", "message": "No analysis history available."})
                record_index = analysis_index if analysis_index >= 0 else len(self.orch.analysis_results) + analysis_index
                record = self.orch.analysis_results[record_index]

            # 2. Extract Claims — prefer the latest revision's revised claims
            # (refine_interpretation, issue #323): assessing superseded claims
            # would produce verdicts about assertions the analysis no longer
            # makes.
            full_result = record.get("full_result", {})
            revisions = record.get("interpretation_revisions") or []
            claims, claims_source = full_result.get("scientific_claims", []), "original"
            for rev in reversed(revisions):
                if rev.get("revised_claims"):
                    claims, claims_source = rev["revised_claims"], "latest_revision"
                    break
            
            if not claims:
                return json.dumps({
                    "status": "warning",
                    "message": "No scientific claims found in this analysis to assess."
                })

            print(f"    Found {len(claims)} claims to assess from {record.get('analysis_id')}")

            # 3. Initialize Lit Agents
            try:
                owl_agent = OwlLiteratureAgent(api_key=self.orch.futurehouse_api_key, max_wait_time=600)
                
                # Use orchestrator's generic LLM config for the Scorer
                scorer = NoveltyScorer(
                    api_key=self.orch.api_key,
                    model_name=self.orch.model_name,
                    base_url=self.orch.base_url
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Failed to init Lit Agents: {e}"})

            # 4. Process Claims
            scored_results = []
            high_novelty_claims = []
            
            # Create a dedicated directory for lit results inside the analysis folder
            analysis_dir = Path(record.get("output_directory", self.orch.results_dir))
            lit_output_dir = analysis_dir / "literature_assessment"
            lit_output_dir.mkdir(exist_ok=True)

            print(f"    Output directory: {lit_output_dir}")

            for i, claim_obj in enumerate(claims):
                question = claim_obj.get("has_anyone_question")
                claim_text = claim_obj.get("claim")
                
                if not question:
                    continue

                print(f"    🔍 Searching claim {i+1}/{len(claims)}: {question[:60]}...")
                
                # Search (Owl)
                search_res = owl_agent.query_literature(question)
                
                if search_res.get("status") != "success":
                    print(f"       ⚠️ Search failed for claim {i+1}")
                    continue

                formatted_answer = search_res.get("formatted_answer", "")

                # Score (Scorer)
                print(f"       ⚖️ Scoring novelty...")
                score_res = scorer.score_novelty(question, formatted_answer)
                
                novelty_score = score_res.get("novelty_score", 0)
                
                result_entry = {
                    "claim_index": i,
                    "original_claim": claim_text,
                    "question": question,
                    "search_answer": formatted_answer,
                    "novelty_score": novelty_score,
                    "novelty_explanation": score_res.get("explanation"),
                    "sources": [s.url for s in getattr(search_res, 'sources', []) if hasattr(s, 'url')]
                }
                scored_results.append(result_entry)
                
                # Track high-novelty claims for recommendations
                if novelty_score >= 4:
                    high_novelty_claims.append(result_entry)
                
                # Pause briefly to be polite to APIs
                time.sleep(1)

            # 5. Build novelty assessment object
            novelty_assessment = {
                "timestamp": datetime.now().isoformat(),
                # ripple bookkeeping: which revision state these verdicts cover
                "assessed_at_revision": len(record.get("interpretation_revisions") or []),
                "claims_source": claims_source,
                "assessments": scored_results,
                "high_novelty_claims": high_novelty_claims,
                "summary_stats": {
                    "total_assessed": len(scored_results),
                    "high_novelty_count": len(high_novelty_claims),
                    "average_score": sum(r.get("novelty_score", 0) for r in scored_results) / len(scored_results) if scored_results else 0
                }
            }
            
            # 6. Store in the analysis record (KEY CHANGE)
            self.orch.analysis_results[record_index]["novelty_assessment"] = novelty_assessment
            
            # 7. Save Results to file
            output_file = lit_output_dir / "novelty_report.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "analysis_id": record.get("analysis_id"),
                    **novelty_assessment
                }, f, indent=2)

            # 8. Summarize for Chat
            summary_lines = []
            
            for res in scored_results:
                score = res['novelty_score']
                if score >= 4:
                    icon = "🌟"
                elif score == 3:
                    icon = "🤔"
                else:
                    icon = "📚"
                
                summary_lines.append(
                    f"{icon} [Score {score}/5] {res['original_claim'][:50]}... "
                    f"-> {res['novelty_explanation'][:80]}..."
                )

            return json.dumps({
                "status": "success",
                "total_assessed": len(scored_results),
                "high_novelty_count": len(high_novelty_claims),
                "average_novelty_score": novelty_assessment["summary_stats"]["average_score"],
                "summary_text": "\n".join(summary_lines),
                "report_path": str(output_file),
                "stored_for_recommendations": True,
                "note": "Novelty assessment stored. Use get_recommendations to get novelty-informed follow-up suggestions."
            })

        self._register_tool(
            func=assess_novelty,
            name="assess_novelty",
            description=(
                "Perform a literature search to assess the novelty of scientific claims "
                "generated by a previous analysis. Requires an analysis_id (from run_analysis). "
                "Returns novelty scores (1-5) and checks for prior art. "
                "Results are stored and used by get_recommendations for prioritized suggestions."
            ),
            parameters={
                "analysis_id": {
                    "type": "string",
                    "description": "The ID of the analysis run to assess (e.g. 'sample1_FFT_2023...')"
                },
                "analysis_index": {
                    "type": "integer",
                    "description": "Alternatively, use the index of the analysis in memory (-1 for most recent)"
                }
            },
            required=[]
        )

        # =====================================================================
        # 11b. RECOMMEND DFT STRUCTURES
        # =====================================================================
        def recommend_simulations(analysis_id: str = None,
                                  analysis_index: int = -1) -> str:
            """
            Generate scale-neutral simulation-study recommendations from a
            completed analysis, optionally informed by a prior novelty
            assessment.
            """
            print(f"  ⚡ Tool: Generating simulation recommendations...")

            # 1. Locate the analysis record
            record = None
            record_index = None

            if analysis_id:
                for i, r in enumerate(self.orch.analysis_results):
                    if r.get("analysis_id") == analysis_id:
                        record = r
                        record_index = i
                        break
                if record is None:
                    return json.dumps({"status": "error", "message": f"Analysis ID not found: {analysis_id}"})
            else:
                if not self.orch.analysis_results:
                    return json.dumps({"status": "error", "message": "No analysis history available."})
                record_index = analysis_index if analysis_index >= 0 else len(self.orch.analysis_results) + analysis_index
                record = self.orch.analysis_results[record_index]

            # 2. Extract analysis text (latest literature-refined
            # interpretation preferred — issue #323)
            full_result = _effective_full_result(record)
            analysis_text = (
                full_result.get("detailed_analysis")
                or full_result.get("full_analysis")
                or ""
            )
            if not analysis_text:
                return json.dumps({
                    "status": "error",
                    "message": "Analysis record has no detailed_analysis text to work from."
                })

            # 3. Pull novel-claim *strings* (high_novelty_claims entries are dicts)
            novelty = record.get("novelty_assessment") or {}
            novel_claim_dicts = novelty.get("high_novelty_claims", []) or []
            novel_claims = [
                c.get("original_claim", "") if isinstance(c, dict) else str(c)
                for c in novel_claim_dicts
            ]
            novel_claims = [c for c in novel_claims if c]

            # 4. Build novelty context (mirror DFTRecommender.run_from_data)
            if novel_claims:
                context = "Focus on these potentially novel findings:\n"
                for i, claim in enumerate(novel_claims, 1):
                    context += f"{i}. {claim}\n"
                context += "\nPrioritize simulation studies that can investigate these novel aspects."
            else:
                context = "No specific novel claims identified. Focus on most interesting aspects."

            # 5. Output dir nested under the analysis record's directory
            base_dir = Path(record.get("output_directory", self.orch.results_dir))
            out_dir = base_dir / "simulation_recommendations"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 6. Run RecommendationAgent directly (no need for the DFTRecommender wrapper)
            try:
                agent = RecommendationAgent(
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                )
                result = agent.generate_simulation_recommendations_from_text(
                    cached_detailed_analysis=analysis_text,
                    additional_prompt_context=context,
                    system_info=self.orch.current_metadata,
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Failed to generate recommendations: {e}"})

            if "error" in result:
                return json.dumps({"status": "error", "message": result.get("error")})

            recommendations = result.get("recommendations", []) or []
            reasoning = result.get("analysis_summary_or_reasoning", "")

            # 7. Persist sidecar JSON for parity with the standalone runner
            output_file = out_dir / "simulation_recommendations.json"
            try:
                with open(output_file, 'w', encoding="utf-8") as f:
                    json.dump({
                        "reasoning": reasoning,
                        "recommendations": recommendations,
                        "novel_claims": novel_claims,
                    }, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to write simulation recommendations sidecar: {e}")

            # 8. Persist on the analysis record for downstream tools.
            # Both keys carry the same list: "simulation_recommendations" is
            # the current name; "dft_recommendations" keeps pre-#45 readers
            # (run_dft_workflow's recommendation_index, external scripts)
            # working unchanged.
            self.orch.analysis_results[record_index]["simulation_recommendations"] = recommendations
            self.orch.analysis_results[record_index]["dft_recommendations"] = recommendations

            print(f"    Generated {len(recommendations)} simulation recommendations → {output_file}")

            return json.dumps({
                "status": "success",
                "count": len(recommendations),
                "recommendations": [
                    {
                        "priority": r.get("priority"),
                        "description": r.get("description"),
                        "research_goal": r.get("research_goal"),
                        "suggested_scale": r.get("suggested_scale"),
                        "scientific_interest": r.get("scientific_interest"),
                    }
                    for r in recommendations
                ],
                "output_file": str(output_file),
            })

        self._register_tool(
            func=recommend_simulations,
            name="recommend_simulations",
            description=(
                "Generate simulation-study recommendations from a completed "
                "analysis: a structure to model plus a research goal per "
                "recommendation, with a non-binding scale hint (periodic_dft "
                "/ molecular_qc / molecular_dynamics) — the "
                "simulation router makes the final engine/scale call. "
                "Specify by analysis_id or use analysis_index (-1 for most "
                "recent). If assess_novelty was run first, recommendations "
                "focus on novel claims. Stores recommendations on the "
                "analysis record; DFT-scale ones can feed run_dft_workflow "
                "by recommendation_index, and any of them can seed a "
                "simulation delegation."
            ),
            parameters={
                "analysis_id": {
                    "type": "string",
                    "description": "The ID of the analysis run to use (e.g. 'sample1_FFT_2023...')"
                },
                "analysis_index": {
                    "type": "integer",
                    "description": "Alternatively, use the index of the analysis in memory (-1 for most recent)"
                }
            },
            required=[]
        )
        # Back-compat alias: pre-#45 programmatic callers dispatch the old
        # name; only recommend_simulations appears in the LLM tool schema.
        self.functions_map["recommend_dft_structures"] = recommend_simulations

        # =====================================================================
        # 11c. RUN DFT WORKFLOW (run_complete_workflow, scale="periodic_dft")
        # =====================================================================
        def run_dft_workflow(structure_description: str = None,
                             analysis_id: str = None,
                             analysis_index: int = -1,
                             recommendation_index: int = None,
                             vasp_generator_method: str = "atomate2",
                             max_refinement_cycles: int = 4) -> str:
            """
            Run the DFT orchestrator to produce VASP-ready inputs (POSCAR, INCAR,
            KPOINTS) for a given structure description, or for a structure picked
            from the recommendations stored on a previous analysis record.
            """
            print(f"  ⚡ Tool: Running DFT workflow...")

            # 1. Locate the optional analysis record
            record = None
            record_index = None
            if analysis_id:
                for i, r in enumerate(self.orch.analysis_results):
                    if r.get("analysis_id") == analysis_id:
                        record = r
                        record_index = i
                        break
                if record is None:
                    return json.dumps({"status": "error", "message": f"Analysis ID not found: {analysis_id}"})
            elif self.orch.analysis_results:
                record_index = analysis_index if analysis_index >= 0 else len(self.orch.analysis_results) + analysis_index
                if 0 <= record_index < len(self.orch.analysis_results):
                    record = self.orch.analysis_results[record_index]

            # 2. Resolve the structure description
            if recommendation_index is not None:
                if record is None:
                    return json.dumps({"status": "error",
                                       "message": "recommendation_index requires an analysis_id or available analysis history."})
                recs = (record.get("simulation_recommendations")
                        or record.get("dft_recommendations") or [])
                if not (0 <= recommendation_index < len(recs)):
                    return json.dumps({"status": "error",
                                       "message": f"recommendation_index {recommendation_index} out of range (have {len(recs)})."})
                rec = recs[recommendation_index]
                # An MD-scale recommendation describes a system (large supercell,
                # finite temperature) that is intractable as a DFT calculation —
                # refuse loudly instead of generating unusable VASP inputs.
                # molecular_qc and unscored (None) recommendations pass through.
                if rec.get("suggested_scale") == "molecular_dynamics":
                    return json.dumps({
                        "status": "error",
                        "message": (
                            f"Recommendation {recommendation_index} is suggested for "
                            "molecular_dynamics, not periodic DFT — its system size / "
                            "finite-temperature goal is intractable as a DFT calculation. "
                            "Route it to simulate mode instead."),
                        "recommendation": {
                            "description": rec.get("description"),
                            "research_goal": rec.get("research_goal"),
                            "suggested_scale": rec.get("suggested_scale"),
                        },
                    })
                structure_description = rec.get("description") or structure_description
            if not structure_description:
                return json.dumps({"status": "error",
                                   "message": "structure_description is required (or pass recommendation_index with an analysis that has stored recommendations)."})

            # 3. Build output directory
            slug = re.sub(r'[^A-Za-z0-9_-]+', '_', structure_description)[:40].strip('_') or "structure"
            base_dir = Path(record.get("output_directory")) if record else Path(self.orch.results_dir)
            out_dir = base_dir / "dft" / slug
            out_dir.mkdir(parents=True, exist_ok=True)

            # 4. Run the pipeline (lazy import — keeps [sim] extras optional)
            try:
                from ..sim_agents.simulation_pipeline import run_complete_workflow
            except ImportError as e:
                # vasp_generator_method='llm' only needs ase; 'atomate2' also
                # needs pymatgen + atomate2. Both paths require ase for
                # structure validation.
                if vasp_generator_method == "atomate2":
                    hint = "pip install ase pymatgen atomate2  (or: pip install 'scilink[sim]')"
                else:
                    hint = "pip install ase  (sufficient for vasp_generator_method='llm')"
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"DFT workflow could not load required dependency. "
                        f"Install: {hint}. Original error: {e}"
                    ),
                })
            try:
                result = run_complete_workflow(
                    structure_description,
                    scale="periodic_dft",
                    software="vasp",
                    method=vasp_generator_method,
                    output_dir=str(out_dir),
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                    model_name=self.orch.model_name,
                    futurehouse_api_key=self.orch.futurehouse_api_key,
                    mp_api_key=None,  # auto-discovered via get_api_key downstream
                    max_refinement_cycles=max_refinement_cycles,
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"DFT workflow failed: {e}"})

            final_status = result.get("final_status")

            # 5. Persist on the analysis record
            if record is not None:
                self.orch.analysis_results[record_index].setdefault("dft_runs", []).append({
                    "description": structure_description,
                    "output_directory": str(out_dir),
                    "final_status": final_status,
                })

            # Surface refinement diagnostics to the agent. Without these the
            # agent sees only `ready_for_vasp: true` and reports success even
            # when the structure was accepted with substantial unresolved
            # validation issues (e.g., circuit-breaker fired on diverging
            # validator complaints).
            structure_gen = result.get("structure_generation", {}) or {}
            structure_warning = structure_gen.get("warning")
            cycles_used = structure_gen.get("cycles_used")
            val_result = structure_gen.get("validation_result", {}) or {}
            outstanding_issues = val_result.get("all_identified_issues", []) or []

            return json.dumps({
                "status": final_status if final_status else "error",
                "final_status": final_status,
                "output_directory": str(out_dir),
                "manifest_path": str(out_dir / "final_files_manifest.json"),
                "ready_for_vasp": final_status == "success",
                "structure_warning": structure_warning,
                "structure_refinement_cycles": cycles_used,
                "structure_outstanding_issues_count": len(outstanding_issues),
                "structure_outstanding_issues": outstanding_issues[:10],
            })

        self._register_tool(
            func=run_dft_workflow,
            name="run_dft_workflow",
            description=(
                "Run the DFT orchestrator to produce VASP-ready inputs (POSCAR, INCAR, "
                "KPOINTS) for a structure. Provide either an explicit structure_description, "
                "or an analysis_id + recommendation_index to pick a structure from prior "
                "recommend_simulations output (choose a DFT-scale recommendation; one "
                "suggested for molecular_dynamics is refused — route it to simulate mode). "
                "Does not run VASP itself; only generates inputs."
            ),
            parameters={
                "structure_description": {
                    "type": "string",
                    "description": "Free-text description of the structure to build (e.g. 'MoS2 monolayer with sulfur vacancy')."
                },
                "analysis_id": {
                    "type": "string",
                    "description": "Analysis run to attach this DFT job to (used for output dir + recommendation lookup)."
                },
                "analysis_index": {
                    "type": "integer",
                    "description": "Alternatively, index of the analysis in memory (-1 for most recent)."
                },
                "recommendation_index": {
                    "type": "integer",
                    "description": "Index into the analysis record's stored DFT recommendations to use as the structure description."
                },
                "vasp_generator_method": {
                    "type": "string",
                    "enum": ["llm", "atomate2"],
                    "description": "How to produce INCAR/KPOINTS. 'atomate2' is rule-based and fast; 'llm' is more flexible but slower."
                },
                "max_refinement_cycles": {
                    "type": "integer",
                    "description": "Maximum validator-guided structure refinement cycles."
                }
            },
            required=[]
        )

        # =====================================================================
        # 12. SET CUSTOM PREPROCESSING INSTRUCTION
        # =====================================================================
        def set_preprocessing_instruction(instruction: str, mode: str = "auto") -> str:
            """
            Add or update a custom preprocessing instruction in the current metadata.
            Metadata must already be loaded via load_metadata or convert_metadata.
            
            Modes:
                - "auto": If existing instruction found, return conflict for LLM to resolve
                - "replace": Overwrite existing instruction
                - "append": Append new instruction to existing one (blocked if redundant)
                - "force_append": Append without redundancy check
            """
            print(f"  ⚡ Tool: Setting custom preprocessing instruction...")
            
            if self.orch.current_metadata is None:
                return json.dumps({
                    "status": "error",
                    "message": "No metadata loaded. Use load_metadata or convert_metadata first."
                })

            # Treat empty/whitespace-only instruction as clearing custom preprocessing
            if not instruction or not instruction.strip():
                self.orch.current_metadata.pop("custom_processing_instruction", None)
                return json.dumps({
                    "status": "success",
                    "message": "Custom preprocessing instruction cleared (empty instruction)."
                })

            existing = self.orch.current_metadata.get("custom_processing_instruction")

            # Conflict detection
            if existing and mode == "auto":
                result = {
                    "status": "conflict",
                    "message": "Metadata already contains a custom preprocessing instruction.",
                    "existing_instruction": existing,
                    "new_instruction": instruction,
                    "options": [
                        "Call again with mode='replace' to overwrite the existing instruction.",
                        "Call again with mode='append' to combine both instructions.",
                    ],
                    "hint": "Ask the user which they prefer if unclear."
                }
                if self._check_instruction_redundancy(existing, instruction):
                    result["redundancy_warning"] = (
                        "These instructions appear to describe the same processing "
                        "operation. Appending would likely cause the same processing "
                        "to be applied twice, corrupting the data. Prefer 'replace' "
                        "unless you are certain they describe distinct steps."
                    )
                return json.dumps(result)

            if existing and mode == "append":
                if self._check_instruction_redundancy(existing, instruction):
                    return json.dumps({
                        "status": "conflict",
                        "message": (
                            "The new instruction appears to describe the same "
                            "processing as the existing one. Appending would likely "
                            "apply the same operation twice, corrupting the data."
                        ),
                        "existing_instruction": existing,
                        "new_instruction": instruction,
                        "options": [
                            "Call again with mode='replace' to use only the new instruction.",
                            "If you are certain these are distinct steps, call "
                            "again with mode='force_append' to combine them."
                        ],
                    })
                combined = f"{existing}\nThen: {instruction}"
                self.orch.current_metadata["custom_processing_instruction"] = combined
                return json.dumps({
                    "status": "success",
                    "message": "Appended new instruction to existing one.",
                    "final_instruction": combined
                })

            if existing and mode == "force_append":
                combined = f"{existing}\nThen: {instruction}"
                self.orch.current_metadata["custom_processing_instruction"] = combined
                return json.dumps({
                    "status": "success",
                    "message": "Force-appended new instruction to existing one.",
                    "final_instruction": combined
                })
            
            # mode == "replace" or no existing instruction
            self.orch.current_metadata["custom_processing_instruction"] = instruction
            
            result = {
                "status": "success",
                "message": "Custom preprocessing instruction set.",
                "instruction": instruction
            }
            if existing:
                result["note"] = f"Replaced previous instruction: '{existing}'"
            
            return json.dumps(result)

        self._register_tool(
            func=set_preprocessing_instruction,
            name="set_preprocessing_instruction",
            description=(
                "Add or update a custom DATA PREPROCESSING instruction in the currently loaded metadata. "
                "Use ONLY for raw data transformations BEFORE fitting: baseline division/subtraction, "
                "background correction, normalization, dark reference subtraction, smoothing, etc. "
                "Do NOT use for fitting model choices (e.g., 'use Lorentzian', 'fit with Fano', "
                "'fit the peak with a Voigt') — those go in the `hints` parameter of `run_analysis`. "
                "If metadata already has a preprocessing instruction, returns a conflict "
                "for you to resolve with the user. When appending, an LLM check detects "
                "redundant instructions to prevent double-processing. "
                "Supports modes: 'auto' (detect conflict), 'replace' (overwrite), "
                "'append' (combine both, with redundancy check), "
                "'force_append' (combine without redundancy check)."
            ),
            parameters={
                "instruction": {
                    "type": "string",
                    "description": (
                        "Natural language preprocessing instruction. Include file paths if "
                        "referencing external data."
                    )
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "How to handle existing instructions: "
                        "'auto' (default, detect conflicts), "
                        "'replace' (overwrite), "
                        "'append' (combine both, blocks if redundant), "
                        "'force_append' (combine without redundancy check)"
                    )
                }
            },
            required=["instruction"]
        )

        # =====================================================================
        # 13. SYNTHESIZE KNOWLEDGE
        # =====================================================================
        def synthesize_knowledge(analysis_ids: list, focus: str, synthesis_type: str = "reference") -> str:
            """
            Distill findings from completed analyses into reusable knowledge.
            The synthesized knowledge is automatically injected into all
            subsequent run_analysis calls.
            """
            from scilink.knowledge import synthesize_knowledge as _synthesize

            print(f"  ⚡ Tool: Synthesizing knowledge ({synthesis_type}) from {len(analysis_ids)} analyses...")

            # Collect result dicts by analysis ID
            results = []
            missing_ids = []
            for aid in analysis_ids:
                found = False
                for record in self.orch.analysis_results:
                    if record.get("analysis_id") == aid:
                        full_result = record.get("full_result", {})
                        full_result["analysis_id"] = aid
                        results.append(full_result)
                        found = True
                        break
                if not found:
                    missing_ids.append(aid)

            if missing_ids:
                return json.dumps({
                    "status": "error",
                    "message": f"Analysis IDs not found: {missing_ids}"
                })

            # Synthesize via the standalone function
            counter = len(self.orch.active_knowledge) + 1
            try:
                entry = _synthesize(
                    results, focus,
                    model=self.orch.model,
                    knowledge_id=f"knowledge_{counter:03d}",
                    synthesis_type=synthesis_type,
                )
            except (ValueError, RuntimeError) as e:
                return json.dumps({"status": "error", "message": str(e)})

            entry["source_analyses"] = analysis_ids
            self.orch.active_knowledge.append(entry)

            # Save to disk
            knowledge_dir = self.orch.base_dir / "knowledge"
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            knowledge_file = knowledge_dir / f"{entry['id']}.json"
            with open(knowledge_file, 'w', encoding="utf-8") as f:
                json.dump(entry, f, indent=2)

            response = {
                "status": "success",
                "knowledge_id": entry["id"],
                "focus": focus,
                "synthesis_type": synthesis_type,
                "summary": entry["summary"],
                "key_findings": entry["key_findings"],
                "saved_to": str(knowledge_file),
                "note": "This knowledge will be automatically injected into all subsequent run_analysis calls."
            }

            # Check if any graduated skill is linked to knowledge with same focus
            for skill_name, source_ids in self.orch._graduated_skill_sources.items():
                for kid in source_ids:
                    for k in self.orch.active_knowledge:
                        if k.get("id") == kid and k.get("focus", "").lower() == focus.lower():
                            response["skill_update_suggested"] = skill_name
                            response["skill_update_note"] = (
                                f"Graduated skill '{skill_name}' is linked to knowledge "
                                f"with the same focus area. Consider calling update_skill "
                                f"to incorporate the new findings."
                            )
                            break
                    if "skill_update_suggested" in response:
                        break
                if "skill_update_suggested" in response:
                    break

            return json.dumps(response)

        self._register_tool(
            func=synthesize_knowledge,
            name="synthesize_knowledge",
            description=(
                "Distill findings from completed analyses into reusable knowledge. "
                "Use when the user wants to learn from reference spectra, derive calibration, "
                "build a reference model, detect trends, learn from failures, or compare methods. "
                "The synthesized knowledge is automatically "
                "injected into all subsequent run_analysis calls as prior knowledge context."
            ),
            parameters={
                "analysis_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of analysis IDs to synthesize knowledge from"
                },
                "focus": {
                    "type": "string",
                    "description": "What to extract/learn (e.g., 'peak assignments for Ti 2p XPS', 'baseline behavior in DSC curves')"
                },
                "synthesis_type": {
                    "type": "string",
                    "enum": ["reference", "trend", "failure", "method"],
                    "description": (
                        "Type of synthesis: 'reference' (calibration/reference extraction, default), "
                        "'trend' (cross-sample trend detection), "
                        "'failure' (failure pattern learning), "
                        "'method' (method selection heuristics)"
                    )
                }
            },
            required=["analysis_ids", "focus"]
        )

        # =====================================================================
        # 14. LIST KNOWLEDGE
        # =====================================================================
        def list_knowledge() -> str:
            """List all active knowledge entries."""
            print(f"  ⚡ Tool: Listing active knowledge...")

            if not self.orch.active_knowledge:
                return json.dumps({
                    "status": "success",
                    "message": "No active knowledge entries.",
                    "entries": []
                })

            entries = []
            for entry in self.orch.active_knowledge:
                entries.append({
                    "id": entry["id"],
                    "focus": entry["focus"],
                    "source_count": len(entry.get("source_analyses", [])),
                    "findings_count": len(entry.get("key_findings", [])),
                    "timestamp": entry.get("timestamp")
                })

            return json.dumps({
                "status": "success",
                "total_entries": len(entries),
                "entries": entries
            })

        self._register_tool(
            func=list_knowledge,
            name="list_knowledge",
            description="Show all active knowledge entries synthesized from previous analyses.",
            parameters={},
            required=[]
        )

        # =====================================================================
        # 15. CLEAR KNOWLEDGE
        # =====================================================================
        def clear_knowledge(knowledge_id: str = None) -> str:
            """Remove active knowledge entries. If knowledge_id is None, removes all."""
            print(f"  ⚡ Tool: Clearing knowledge...")

            knowledge_dir = self.orch.base_dir / "knowledge"

            if knowledge_id is None:
                count = len(self.orch.active_knowledge)
                self.orch.active_knowledge.clear()
                # Remove all files
                if knowledge_dir.exists():
                    for f in knowledge_dir.glob("knowledge_*.json"):
                        f.unlink()
                return json.dumps({
                    "status": "success",
                    "message": f"Cleared all {count} knowledge entries."
                })

            # Find and remove specific entry
            for i, entry in enumerate(self.orch.active_knowledge):
                if entry["id"] == knowledge_id:
                    self.orch.active_knowledge.pop(i)
                    # Remove disk file
                    knowledge_file = knowledge_dir / f"{knowledge_id}.json"
                    if knowledge_file.exists():
                        knowledge_file.unlink()
                    return json.dumps({
                        "status": "success",
                        "message": f"Removed knowledge entry: {knowledge_id}"
                    })

            return json.dumps({
                "status": "error",
                "message": f"Knowledge ID not found: {knowledge_id}"
            })

        self._register_tool(
            func=clear_knowledge,
            name="clear_knowledge",
            description=(
                "Remove active knowledge entries. Specify a knowledge_id to remove a "
                "specific entry, or omit to clear all knowledge."
            ),
            parameters={
                "knowledge_id": {
                    "type": "string",
                    "description": "ID of knowledge entry to remove (omit to clear all)"
                }
            },
            required=[]
        )

        # =====================================================================
        # 16. GRADUATE TO SKILL
        # =====================================================================
        def graduate_to_skill(knowledge_id: str, skill_name: str, domain: str = "curve_fitting") -> str:
            """
            Convert a knowledge entry into a reusable skill (.md file).
            The skill is automatically registered for use in subsequent analyses.
            """
            from scilink.agents.exp_agents.instruct import (
                KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                SKILL_UPDATE_INSTRUCTIONS,
            )
            from scilink.skills._shared._graduation import graduate_to_skill_file

            print(f"  ⚡ Tool: Graduating knowledge '{knowledge_id}' to skill '{skill_name}'...")

            # Find the knowledge entry
            knowledge_entry = None
            for entry in self.orch.active_knowledge:
                if entry.get("id") == knowledge_id:
                    knowledge_entry = entry
                    break

            if knowledge_entry is None:
                return json.dumps({
                    "status": "error",
                    "message": f"Knowledge ID not found: {knowledge_id}"
                })

            # Collect source analysis details
            analysis_details_parts = []
            source_ids = knowledge_entry.get("source_analyses", [])
            for aid in source_ids:
                for record in self.orch.analysis_results:
                    if record.get("analysis_id") == aid:
                        full_result = _effective_full_result(record)
                        parts = [f"### Analysis: {aid}"]

                        da = full_result.get("detailed_analysis", "")
                        if da:
                            parts.append(da[:2000])  # Truncate for prompt size

                        fp = full_result.get("fitting_parameters")
                        if fp:
                            parts.append(f"Fitting parameters: {json.dumps(fp, indent=2, default=str)}")

                        hf = full_result.get("human_feedback", {})
                        if isinstance(hf, dict) and hf.get("user_feedback"):
                            parts.append(f"User feedback: {hf['user_feedback']}")

                        analysis_details_parts.append("\n".join(parts))
                        break

            analysis_details = "\n\n".join(analysis_details_parts) if analysis_details_parts else "No source analysis details available."

            # Fold focus/summary/findings + source detail into a single
            # knowledge_entry; the shared helper renders it into {knowledge_text}.
            findings = knowledge_entry.get("key_findings", []) or []
            distill_entry = {
                "focus": knowledge_entry.get("focus", ""),
                "summary": knowledge_entry.get("summary", ""),
                "key_findings": "\n".join(f"- {f}" for f in findings),
                "source_analysis_details": analysis_details,
            }

            def _llm_call(prompt: str) -> str:
                response = self.orch.model.generate_content(
                    contents=[prompt],
                    generation_config=None,
                    safety_settings=None,
                )
                return response.text if hasattr(response, "text") else str(response)

            # Write to the persistent store (~/.scilink/graduated_skills),
            # via the shared structured-JSON helper, so the skill survives
            # the session and a pip upgrade. The helper auto-detects
            # create-vs-update by whether the bundle already exists.
            try:
                result = graduate_to_skill_file(
                    knowledge_entry=distill_entry,
                    skill_name=skill_name,
                    domain=domain,
                    llm_call=_llm_call,
                    fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                    update_template=SKILL_UPDATE_INSTRUCTIONS,
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Graduation failed: {e}"})

            skill_path = result["skill_path"]

            # Register the skill so the live session can use it immediately.
            self.orch.register_skill(str(skill_path))

            # Track the link
            self.orch._graduated_skill_sources[skill_name] = [knowledge_id]

            return json.dumps({
                "status": "success",
                "skill_name": skill_name,
                "skill_path": str(skill_path),
                "method": result.get("method"),
                "source_knowledge_id": knowledge_id,
                "note": f"Skill '{skill_name}' has been registered (persistent memory) and will be available in run_analysis."
            })

        self._register_tool(
            func=graduate_to_skill,
            name="graduate_to_skill",
            description=(
                "Convert a knowledge entry into a reusable skill (.md file). "
                "The skill is organized into 5 sections (overview, planning, analysis, "
                "interpretation, validation) and automatically registered for use in "
                "subsequent analyses."
            ),
            parameters={
                "knowledge_id": {
                    "type": "string",
                    "description": "ID of the knowledge entry to graduate"
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name for the new skill (used as filename and reference)"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain/technique area (e.g., 'curve_fitting', 'xps', 'raman'). Default: 'curve_fitting'"
                }
            },
            required=["knowledge_id", "skill_name"]
        )

        # =====================================================================
        # 17. UPDATE SKILL
        # =====================================================================
        def update_skill(skill_name: str, knowledge_ids: list = None, domain: str = "curve_fitting") -> str:
            """
            Update a graduated skill with new knowledge entries.
            Merges into the persistent skill bundle in place.
            """
            from scilink.agents.exp_agents.instruct import (
                KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                SKILL_UPDATE_INSTRUCTIONS,
            )
            from scilink.skills._shared._graduation import graduate_to_skill_file
            from scilink.skills.loader import graduated_skills_dir

            print(f"  ⚡ Tool: Updating skill '{skill_name}'...")

            # The skill must already exist in the persistent store so the
            # helper takes its merge (update) branch.
            skill_path = graduated_skills_dir() / domain / skill_name / f"{skill_name}.md"
            if not skill_path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Graduated skill not found: {domain}/{skill_name}"
                })

            # Determine source knowledge IDs
            tracked_ids = self.orch._graduated_skill_sources.get(skill_name, [])
            if knowledge_ids:
                new_ids = knowledge_ids
            else:
                # Use all knowledge entries with matching focus
                focus_areas = set()
                for kid in tracked_ids:
                    for k in self.orch.active_knowledge:
                        if k.get("id") == kid:
                            focus_areas.add(k.get("focus", "").lower())
                new_ids = [
                    k["id"] for k in self.orch.active_knowledge
                    if k["id"] not in tracked_ids and k.get("focus", "").lower() in focus_areas
                ]

            if not new_ids:
                return json.dumps({
                    "status": "error",
                    "message": "No new knowledge entries found to update the skill with."
                })

            # Collect new knowledge texts
            new_knowledge_parts = []
            for kid in new_ids:
                for k in self.orch.active_knowledge:
                    if k.get("id") == kid:
                        part = f"### {kid}\n**Focus:** {k.get('focus', '')}\n"
                        part += f"**Summary:** {k.get('summary', '')}\n"
                        part += "**Key Findings:**\n"
                        for f in k.get("key_findings", []):
                            part += f"- {f}\n"
                        new_knowledge_parts.append(part)
                        break

            new_knowledge = "\n\n".join(new_knowledge_parts)

            def _llm_call(prompt: str) -> str:
                response = self.orch.model.generate_content(
                    contents=[prompt],
                    generation_config=None,
                    safety_settings=None,
                )
                return response.text if hasattr(response, "text") else str(response)

            # Delegate to the shared helper; since the bundle exists it
            # takes the structured-JSON merge (update) branch.
            try:
                result = graduate_to_skill_file(
                    knowledge_entry={"new_knowledge": new_knowledge},
                    skill_name=skill_name,
                    domain=domain,
                    llm_call=_llm_call,
                    fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                    update_template=SKILL_UPDATE_INSTRUCTIONS,
                )
            except Exception as e:
                return json.dumps({"status": "error", "message": f"Update failed: {e}"})

            # Update source tracking
            all_ids = list(set(tracked_ids + new_ids))
            self.orch._graduated_skill_sources[skill_name] = all_ids

            # Re-register the skill
            self.orch.register_skill(str(result["skill_path"]))

            return json.dumps({
                "status": "success",
                "skill_name": skill_name,
                "skill_path": str(result["skill_path"]),
                "method": result.get("method"),
                "new_knowledge_ids": new_ids,
                "total_source_ids": all_ids,
                "note": f"Skill '{skill_name}' has been updated in persistent memory."
            })

        self._register_tool(
            func=update_skill,
            name="update_skill",
            description=(
                "Update a graduated skill with new knowledge entries. "
                "Use when new knowledge has been synthesized and a linked skill "
                "should incorporate the new findings. The old version is preserved."
            ),
            parameters={
                "skill_name": {
                    "type": "string",
                    "description": "Name of the graduated skill to update"
                },
                "knowledge_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific knowledge IDs to incorporate (omit to auto-detect from matching focus area)"
                },
                "domain": {
                    "type": "string",
                    "description": "Domain the skill was graduated under (e.g., 'curve_fitting', 'xps'). Default: 'curve_fitting'"
                }
            },
            required=["skill_name"]
        )

        # =====================================================================
        # 18. SAVE FILE
        # =====================================================================
        def save_file(filename: str, content: str, subfolder: str = "") -> str:
            """
            Save text content (reports, summaries, tables, scripts) to a file
            in the session directory.
            """
            print(f"  ⚡ Tool: Saving file '{filename}'...")

            # Sanitise: strip path separators from filename to prevent traversal.
            safe_name = Path(filename).name
            if not safe_name:
                return json.dumps({
                    "status": "error",
                    "message": "Invalid filename.",
                })

            target_dir = self.orch.base_dir
            if subfolder:
                safe_sub = Path(subfolder).name
                target_dir = target_dir / safe_sub
            target_dir.mkdir(parents=True, exist_ok=True)
            dest = target_dir / safe_name

            try:
                dest.write_text(content, encoding="utf-8")
                print(f"    💾 Saved: {dest}")
                return json.dumps({
                    "status": "success",
                    "path": str(dest),
                    "size_bytes": dest.stat().st_size,
                })
            except Exception as e:
                logging.error(f"save_file failed: {e}")
                return json.dumps({
                    "status": "error",
                    "message": str(e),
                })

        self._register_tool(
            func=save_file,
            name="save_file",
            description=(
                "Save text content (reports, summaries, tables, scripts, notes) "
                "to a file in the session directory. Use this to persist "
                "synthesized knowledge summaries, analysis reports, exported "
                "results, or any text artifact the user requests."
            ),
            parameters={
                "filename": {
                    "type": "string",
                    "description": (
                        "Name of the file to create, e.g. 'analysis_report.md', "
                        "'peak_positions.csv', or 'summary.txt'."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write to the file.",
                },
                "subfolder": {
                    "type": "string",
                    "description": (
                        "Optional subfolder within the session directory, "
                        "e.g. 'reports' or 'exports'. Created if it doesn't exist."
                    ),
                },
            },
            required=["filename", "content"]
        )

        # =====================================================================
        # READ DOCUMENT
        # =====================================================================
        def read_document(paths) -> str:
            """Read one or more PDF/DOCX/MD/TXT documents; return the combined
            text and persist a literature_file for run_analysis."""
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                return json.dumps({
                    "status": "error",
                    "message": "No document path provided.",
                })
            print(f"  📄 Tool: Reading {len(paths)} document(s)...")
            docs, errors = [], []
            for p in paths:
                dp = Path(p)
                if not dp.is_file():
                    errors.append(f"Not a file: {p}")
                    continue
                try:
                    docs.append((dp, _extract_document_text(
                        dp, ocr_model=self.orch.model)))
                except ValueError as e:
                    errors.append(str(e))
                except Exception as e:
                    logging.error(f"read_document failed for {p}: {e}")
                    errors.append(f"Could not read {dp.name}: {e}")
            if not docs:
                return json.dumps({
                    "status": "error",
                    "message": "No documents could be read.",
                    "errors": errors,
                })
            combined = "\n\n---\n\n".join(
                f"## {dp.name}\n\n{info['text']}" for dp, info in docs
            )
            combined_truncated = len(combined) > _READ_DOC_MAX_CHARS
            if combined_truncated:
                combined = combined[:_READ_DOC_MAX_CHARS]
            # Persist as a literature_file so run_analysis can ground its plan
            # in these documents — the same channel search_literature uses.
            lit_path = None
            try:
                lit_dir = self.orch.base_dir / "literature"
                lit_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                lit_path = lit_dir / f"provided_documents_{ts}.md"
                write_text_utf8(lit_path, combined)
            except Exception as e:
                logging.error(f"read_document: could not save literature file: {e}")
            n_ocr = sum(info.get("n_ocr_pages", 0) for _, info in docs)
            return json.dumps({
                "status": "success",
                "file_path": str(lit_path) if lit_path else None,
                "n_documents": len(docs),
                "n_ocr_pages": n_ocr,
                "ocr_note": (
                    f"{n_ocr} scanned page(s) had no text layer and were "
                    "transcribed by vision-OCR — verify any figures/numerics."
                ) if n_ocr else None,
                "documents": [
                    {"name": dp.name,
                     **{k: v for k, v in info.items() if k != "text"}}
                    for dp, info in docs
                ],
                "errors": errors or None,
                "combined_truncated": combined_truncated,
                "text": combined,
                "hint": (
                    "Pass file_path as `literature_file` to the next "
                    "run_analysis() call so the planner produces a "
                    "document-informed plan."
                ),
            })

        self._register_tool(
            func=read_document,
            name="read_document",
            description=(
                "Read one or more documents the user provided — PDF, DOCX, "
                "Markdown, or text files (a methods paper, protocol, prior "
                "report, notes). Returns the combined text AND saves a "
                "literature file: pass the returned `file_path` as "
                "`literature_file` to run_analysis() so the provided documents "
                "drive the analysis plan, exactly as a search_literature "
                "result does. For a handful of documents read straight into "
                "context — it runs NO external literature search and builds no "
                "index (use search_literature for the wider literature). Pass "
                "absolute file paths."
            ),
            parameters={
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path(s) to the document(s) to read (.pdf, "
                        ".docx, .md, or .txt). Multiple documents are combined "
                        "into one literature file."
                    ),
                },
            },
            required=["paths"]
        )

    def _internal_model(self):
        """Tool-free sibling of the orchestrator's model for INTERNAL LLM calls.

        The orchestrator's chat wrapper binds its system prompt and the full
        tool schemas at construction (LiteLLM path,
        analysis_orchestrator.py:701-706). An internal call made from inside a
        tool function therefore behaves like the chat agent — the model may
        answer with a *tool call*, leaving ``.text`` empty and failing the
        parse ("Empty response from LLM"). Discovered live via the
        reenter_interpretation routing test; refine_interpretation's three
        internal calls had the same latent failure on the LiteLLM path (the
        proxy path constructs its wrapper without bound tools, masking it).

        Returns a plain generator with the same model/key/base_url and no
        tools or system instruction. Cached; degrades to the bound model if
        construction fails (better a maybe-empty response than a crashed tool).
        """
        cached = getattr(self, "_internal_model_cache", None)
        if cached is not None:
            return cached
        base = self.orch.model
        try:
            if getattr(self.orch, "use_openai", False):
                from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
                m = OpenAIAsGenerativeModel(
                    model=base.model, api_key=base.api_key,
                    base_url=getattr(base, "base_url", None),
                )
            else:
                from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
                m = LiteLLMGenerativeModel(
                    model=base.model, api_key=getattr(base, "api_key", None),
                    base_url=getattr(base, "base_url", None),
                )
        except Exception as e:  # noqa: BLE001 - degrade, don't crash the tool
            logging.warning(f"_internal_model fallback to bound model: {e}")
            m = base
        self._internal_model_cache = m
        return m

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None
    ):
        """Register a tool in OpenAI format."""
        self.functions_map[name] = func
        
        openai_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or []
                }
            }
        }
        self.openai_schemas.append(openai_schema)

    def _update_skill_description(self, custom_skills: dict) -> None:
        """Update the ``skill`` parameter description in ``run_analysis``
        to include newly registered custom skills."""
        for schema in self.openai_schemas:
            fn = schema.get("function", {})
            if fn.get("name") != "run_analysis":
                continue
            skill_prop = fn["parameters"]["properties"].get("skill")
            if skill_prop is None:
                break
            skill_prop["description"] = _build_skill_description(
                getattr(self.orch, "_agent_registry", None),
                custom_skills,
            )
            break

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given arguments."""
        result = self._dispatch_tool(tool_name, **kwargs)
        # One bounded line per tool call into the session's events.jsonl
        # (no-op when the thread has no bound log). See session_events.
        from ...session_events import append_event
        append_event(tool_name, kwargs, result)
        return result

    def _dispatch_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found"
            })
        
        try:
            return self.functions_map[tool_name](**kwargs)
        except Exception as e:
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name
            })