"""Parallel multi-dataset analysis (fan-out) + complementarity gating + fusion.

See CLAUDE.md "The meta agent". This is the meta's **fan-out primitive**: run
several analysis branches concurrently over GENUINELY COMPLEMENTARY datasets —
each branch sees the others as full-mesh auxiliary operands — then fuse their
findings into one cross-dataset narrative.

Two guards bracket the fan-out, because the failure mode here is not a crash
but a *plausible fabrication*:

1. **Entry gate (complementarity).** Before any branch launches, the datasets
   are assessed and PARTITIONED. The fan-out runs only over the complementary
   subset that shares a join axis; redundant duplicates and unrelated outliers
   are pruned out. Forcing the fusion template over unrelated data would
   manufacture a correlation that isn't there — the gate is what prevents that.
2. **Exit guard (anti-spurious-fusion).** Even on a clean gate, the fusion
   prompt states that "no correlation found" is a valid, valuable conclusion,
   so the synthesis reconciles the evidence rather than inventing a link.

Branches run AUTONOMOUS regardless of the meta's mode: concurrent `input()`
human-feedback prompts cannot interleave across threads, so a parallel branch
cannot pause for approval. The single up-front user confirmation (AUTOPILOT)
compensates for the per-branch approval the user would otherwise get.

The logic lives here as free functions taking the orchestrator instance; thin
``MetaOrchestratorAgent`` methods wrap them, matching how ``telemetry.py`` is a
sibling helper to the orchestrator.
"""

import glob
import io
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("meta_agent.fanout")

# Concurrency + sizing. The complementary SET (post-gate) is what these bound,
# not the raw input: the gate prunes first, so a 6-upload request with one
# complementary pair runs a 2-way mesh, not a 6-way one.
FANOUT_MAX_WORKERS = 4          # peak concurrent branches (rate-limit ceiling)
_FANOUT_POLL_S = 5              # how often to check for finished branches
_FANOUT_HEARTBEAT_S = 60        # min gap between "still running" ticks
FANOUT_SOFT_CAP = 5             # warn / confirm beyond this many fused branches
FANOUT_HARD_CAP = 8             # refuse beyond this — cost/quality cliff
# In AUTONOMOUS mode there is no human to confirm, so the verdict IS the gate:
# proceed only on a confident 'complementary' read.
AUTONOMOUS_CONFIDENCE_THRESHOLD = 0.6


# ======================================================================
# Complementarity gate
# ======================================================================

COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS = """You are a measurement scientist deciding whether several datasets are \
GENUINELY COMPLEMENTARY — i.e. whether fusing their analyses into one \
cross-dataset narrative is scientifically meaningful, or would instead \
manufacture a correlation that the data do not support.

Datasets are complementary only when ALL THREE hold:
1. SAME SUBJECT — they measure the same physical system / sample / region.
2. NON-REDUNDANT — they carry different information (different modality, \
observable, or condition); two measurements of the same thing the same way \
are redundant, not complementary.
3. JOINABLE — a concrete axis exists to reconcile them ON: spatial \
co-registration, a shared energy/time/parameter axis, or a shared \
sample/condition. A join does NOT require pixel-level co-registration: \
reconciling one modality's spatially-resolved or local measurement against \
another's bulk / area-averaged measurement of the SAME sample (e.g. microscopy \
phase fractions vs XPS/XRD/EDX composition) is itself a valid join — this \
bulk-vs-local reconciliation is a canonical multimodal case, not a \
manufactured correlation. Without any join there is nothing to fuse.

Partition the datasets accordingly. Each dataset carries an `id` — echo these \
`id` values VERBATIM in every output list below. Several distinct datasets may \
share one filesystem path (a directory holding more than one measurement \
series) and may look structurally alike (e.g. two 1-D curves): judge \
redundancy by what each dataset measures and its stated analysis task, not by \
path or structural shape. Put into `fanout_set` ONLY a subset that \
is mutually complementary on all three criteria and shares ONE join axis \
(>= 2 members to be worth running in parallel). Cluster exact-duplicate / \
same-information datasets in `redundant_clusters`. List datasets that belong \
to a different system or have no join axis in `unrelated`.

Be conservative: if you are not confident the datasets share a system and a \
join axis, prefer `uncertain` over `complementary`. A wrong `complementary` \
call produces a fabricated cross-dataset claim, which is worse than declining.

Respond in valid JSON with EXACTLY these keys:
{
  "verdict": "complementary" | "partially_complementary" | "redundant" | "unrelated" | "uncertain",
  "confidence": <float 0..1>,
  "rationale": "<one or two sentences: what the datasets are and why this verdict>",
  "join_axis": "<the shared axis the fanout_set reconciles on, or null>",
  "fanout_set": ["<id>", ...],
  "redundant_clusters": [["<id>", "<id>"], ...],
  "unrelated": ["<id>", ...],
  "excluded_notes": "<why anything was left out of fanout_set, or empty>"
}
"""


def _slug(text: str, maxlen: int = 32) -> str:
    """Filesystem-safe short slug from a label."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s[:maxlen] or "branch")


_LLM_JSON_ATTEMPTS = 3


def _structured_model(orch):
    """A TOOL-FREE generative model for the gate/fusion JSON calls.

    The meta's chat model is built WITH the delegation tool schemas, so a
    structured-output prompt about datasets intermittently comes back as an
    attempted tool-call with empty text instead of the JSON we asked for.
    A dedicated tool-less model (same provider routing / credentials) removes
    that failure mode. Cached on the orchestrator.
    """
    m = getattr(orch, "_fanout_structured_model", None)
    if m is not None:
        return m
    if orch.base_url:
        from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
        m = OpenAIAsGenerativeModel(
            model=orch.model_name, api_key=orch.api_key, base_url=orch.base_url)
    else:
        from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
        m = LiteLLMGenerativeModel(
            model=orch.model_name, api_key=orch.api_key,
            system_instruction="You output only valid JSON exactly as instructed.",
            tools=None)
    orch._fanout_structured_model = m
    return m


def _parse_json_block(text: str) -> Optional[dict]:
    """Parse a JSON object from raw model text — tolerant of ```json fences and
    surrounding prose (falls back to the first balanced ``{...}``)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except Exception:  # noqa: BLE001 - fall back to first balanced {...}
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None


def _llm_json(orch, prompt: str, extra_parts=None) -> Optional[dict]:
    """LLM call returning parsed JSON, or None after retries.

    Retries on an empty completion or an unparseable body — Bedrock
    intermittently returns an empty content block, and silently fail-closing
    the gate on that transient would wrongly decline a complementary set.
    Only a persistent failure returns None (which callers fail closed on).

    ``extra_parts`` is an optional list of additional prompt parts (label
    strings and/or ``{mime_type, data}`` image dicts) appended after the prompt
    — used to attach per-dataset figures to the (multimodal) fusion call.
    """
    model = _structured_model(orch)
    contents = [prompt] + list(extra_parts or [])
    for attempt in range(_LLM_JSON_ATTEMPTS):
        try:
            resp = model.generate_content(contents=contents)
            text = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"complementarity/fusion LLM call failed "
                           f"(attempt {attempt + 1}): {e}")
            continue
        parsed = _parse_json_block(text)
        if parsed is not None:
            return parsed
        logger.warning("complementarity/fusion LLM returned "
                       f"{'empty' if not text else 'unparseable'} response "
                       f"(attempt {attempt + 1}/{_LLM_JSON_ATTEMPTS}); retrying")
    return None


def _resolve_branch_files(data_path: str, pattern: Optional[str]) -> Optional[List[str]]:
    """Resolve a branch's own file set from its glob pattern (issue #326 Fix 3).

    Series data often lands as many files in one shared directory; the pattern
    is what selects THIS branch's files out of it. Returns a sorted list of
    matching file paths, or None when there is no pattern or nothing matches
    (the branch then stays directory-shaped, as before).
    """
    if not pattern:
        return None
    try:
        pat = (pattern if os.path.isabs(str(pattern))
               else os.path.join(str(data_path), str(pattern)))
        files = sorted(f for f in glob.glob(pat) if os.path.isfile(f))
    except Exception as e:  # noqa: BLE001 - a bad pattern must not kill the fan-out
        logger.warning(f"fan-out: could not resolve pattern {pattern!r}: {e}")
        return None
    if not files:
        logger.warning(f"fan-out: pattern {pattern!r} matched no files "
                       f"under {data_path}")
        return None
    return files


def _existing_json_path(value) -> Optional[Path]:
    """Interpret a metadata value as a metadata-JSON path if it plausibly is
    one, else None (the value is an inline description). Inline metadata can
    be arbitrarily long prose, and ``os.stat`` on such a string raises
    ``OSError: File name too long`` — so the probe must be defensive."""
    try:
        p = Path(str(value))
        if p.exists() and p.suffix.lower() == ".json":
            return p
    except OSError:
        pass
    return None


def _dataset_descriptor(path: str, role: Optional[str],
                        metadata: Optional[str],
                        task: Optional[str] = None,
                        label: Optional[str] = None,
                        dataset_id: Optional[str] = None,
                        files: Optional[List[str]] = None) -> dict:
    """Lightweight, router-tier descriptor of one dataset for the gate.

    Reuses the meta's content probe (shape/dtype, table columns, document /
    image dims) — the same evidence the meta routes on — plus any user-stated
    role and metadata. Deliberately does NOT load full arrays: the gate is a
    judgement over descriptors, consistent with the meta being a router.

    ``task`` / ``label`` / ``dataset_id`` carry a fan-out branch's analysis
    intent and identity (issue #326): when two branches share one path (a
    series directory) and one structural probe (two 1-D curves), the task is
    the only signal that separates a complementary pair from a duplicate.
    ``files`` is the branch's resolved file set (Fix 3): when present, the
    content probe runs on the branch's OWN first file instead of the shared
    directory, so same-directory branches get distinct structural signatures.
    """
    from .meta_orchestrator_tools import _probe_file

    p = Path(path)
    desc: Dict[str, Any] = {"id": str(dataset_id or path), "path": str(path)}
    if label:
        desc["label"] = str(label)
    if task:
        desc["analysis_task"] = str(task)
    if files:
        names = [Path(f).name for f in files]
        desc["n_files"] = len(names)
        desc["files_sample"] = (names[:3] + ["..."] + names[-1:]
                                if len(names) > 4 else names)
    if not p.exists():
        desc["note"] = "file not found"
        return desc
    try:
        target = Path(files[0]) if files else p
        desc["probe"] = _probe_file(target)
        if files:
            desc["probe_note"] = "probe of the first file in this branch's file set"
    except Exception as e:  # noqa: BLE001 - probe must not break the gate
        desc["note"] = f"probe failed: {e}"
    if role:
        desc["stated_role"] = role
    if metadata:
        mp = _existing_json_path(metadata)
        if mp is not None:
            try:
                with open(mp, "r", errors="replace") as fh:
                    desc["metadata"] = json.load(fh)
            except Exception:  # noqa: BLE001
                desc["metadata"] = str(metadata)
        else:
            desc["metadata"] = str(metadata)
    return desc


def assess_complementarity(orch, datasets: List[dict]) -> dict:
    """Partition datasets into complementary / redundant / unrelated.

    `datasets` is a list of ``{"path", "role"?, "metadata"?, "id"?, "task"?,
    "label"?}``. Identity is the per-dataset ``id`` — defaulting to ``path``,
    which keeps the standalone tool's path-keyed behavior — so ``run_fanout``
    can pass distinct ids for branches that share one path (issue #326); the
    verdict lists (``fanout_set`` etc.) contain ids. Returns the verdict dict
    (see COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS). Cached on the orchestrator
    by the frozenset of ids so the standalone tool and the internal gate in
    run_fanout don't double-spend the LLM call.
    """
    datasets = [d for d in (datasets or [])
                if isinstance(d, dict) and d.get("path")]
    ids = [str(d.get("id") or d["path"]) for d in datasets]
    if len(set(ids)) < 2:
        return {"verdict": "uncertain", "confidence": 0.0,
                "rationale": "Need at least two datasets to assess complementarity.",
                "join_axis": None, "fanout_set": [], "redundant_clusters": [],
                "unrelated": list(dict.fromkeys(ids)), "excluded_notes": ""}

    key = frozenset(ids)
    cached = orch._complementarity_cache.get(key)
    if cached is not None:
        return cached

    descriptors = [
        _dataset_descriptor(d["path"], d.get("role"), d.get("metadata"),
                            task=d.get("task"), label=d.get("label"),
                            dataset_id=i, files=d.get("files"))
        for d, i in zip(datasets, ids)
    ]
    prompt = (
        COMPLEMENTARITY_ASSESSMENT_INSTRUCTIONS
        + "\n\n--- DATASETS ---\n"
        + json.dumps(descriptors, indent=2, default=str)
    )
    verdict = _llm_json(orch, prompt)
    if not verdict or "verdict" not in verdict:
        # Fail closed: an unparseable assessment must not green-light a fusion.
        verdict = {
            "verdict": "uncertain", "confidence": 0.0,
            "rationale": "Complementarity assessment did not return a usable verdict.",
            "join_axis": None, "fanout_set": [], "redundant_clusters": [],
            "unrelated": list(ids), "excluded_notes": "",
        }
    # Constrain the model's fanout_set to the actually-requested ids.
    requested = set(ids)
    verdict["fanout_set"] = [p for p in (verdict.get("fanout_set") or [])
                             if p in requested]
    # Defensive: a clearly-negative verdict must never carry a runnable set,
    # even if the model inconsistently populated one — it means "do not fuse".
    if (verdict.get("verdict") or "").lower() in ("unrelated", "redundant"):
        verdict["fanout_set"] = []
    orch._complementarity_cache[key] = verdict
    return verdict


# ======================================================================
# Confirmation
# ======================================================================

def _confirm_fanout(orch, verdict: dict, fanout_set: List[str],
                    branches_by_id: Dict[str, dict]) -> tuple:
    """Decide whether to fire the fan-out. Returns (proceed: bool, reason: str).

    AUTOPILOT (human attached): show the verdict + the exact plan and ask the
    user to confirm. AUTONOMOUS (no human): the verdict is the gate — proceed
    only on a confident 'complementary' read within the soft cap.
    """
    n = len(fanout_set)
    n_aux = n * (n - 1)  # full-mesh: each branch sees the other n-1

    if n > FANOUT_HARD_CAP:
        return False, (f"Complementary set has {n} datasets (> hard cap "
                       f"{FANOUT_HARD_CAP}); refuse to fan out. Run them in "
                       "smaller complementary groups.")

    if not orch._enable_human_feedback:
        # AUTONOMOUS: verdict-gated, conservative.
        v = (verdict.get("verdict") or "").lower()
        conf = float(verdict.get("confidence") or 0.0)
        if v != "complementary" or conf < AUTONOMOUS_CONFIDENCE_THRESHOLD:
            return False, (f"Autonomous mode declines fan-out: verdict='{v}' "
                           f"confidence={conf:.2f} (needs 'complementary' >= "
                           f"{AUTONOMOUS_CONFIDENCE_THRESHOLD}). "
                           f"{verdict.get('rationale', '')}")
        if n > FANOUT_SOFT_CAP:
            return False, (f"Autonomous mode declines a {n}-way mesh (> soft cap "
                           f"{FANOUT_SOFT_CAP}); too expensive to fire unattended.")
        return True, "autonomous: confident complementary verdict"

    # AUTOPILOT: informed human confirmation.
    lines = [
        "",
        "=" * 78,
        "🔀 PARALLEL MULTI-DATASET ANALYSIS — confirm before launching",
        "=" * 78,
        f"  Complementarity verdict : {verdict.get('verdict')} "
        f"(confidence {verdict.get('confidence')})",
        f"  Join axis               : {verdict.get('join_axis')}",
        f"  Rationale               : {verdict.get('rationale')}",
        "",
        f"  Will run {n} branches concurrently, full-mesh "
        f"(~{n_aux} auxiliary loads):",
    ]
    for bid in fanout_set:
        b = branches_by_id.get(bid, {})
        name = Path(b.get("data_path") or bid).name
        lines.append(f"    • {b.get('label') or _slug(name)}  ({name})")
    if verdict.get("redundant_clusters"):
        lines.append(f"  Pruned as redundant     : {verdict['redundant_clusters']}")
    if verdict.get("unrelated"):
        lines.append(f"  Pruned as unrelated     : {verdict['unrelated']}")
    if n > FANOUT_SOFT_CAP:
        lines.append(f"  ⚠️  {n}-way mesh exceeds the soft cap ({FANOUT_SOFT_CAP}) "
                     "— this is expensive.")
    lines.append("  Branches run AUTONOMOUSLY (no per-branch approval pauses).")
    lines.append("=" * 78)
    print("\n".join(lines))

    try:
        ans = input("\n🤔 Launch this parallel analysis? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # No usable input channel in a mode that expects one → do not fire an
        # expensive parallel op on a guess.
        return False, "no confirmation received (declined)"
    if ans in ("y", "yes"):
        return True, "user confirmed"
    return False, "user declined"


# ======================================================================
# Branch execution
# ======================================================================

def _make_ephemeral_analysis_child(orch, base_dir: Path):
    """Build an isolated, one-shot analysis orchestrator for one branch.

    NOT registered in ``orch._children`` — these are ephemeral fan-out workers,
    not the persistent singleton, so they share no mutable state across threads
    and are never restored. Resting mode AUTONOMOUS; run_task pins it per call.
    """
    from ..exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    child = AnalysisOrchestratorAgent(
        base_dir=str(base_dir),
        api_key=orch.api_key,
        model_name=orch.model_name,
        base_url=orch.base_url,
        embedding_model=orch.embedding_model,
        embedding_api_key=orch.embedding_api_key,
        futurehouse_api_key=orch.futurehouse_api_key,
        restore_checkpoint=False,
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )
    child._agent_label = "Analysis branch"
    # Share skills / custom tools / MCP servers registered on the meta.
    orch._propagate_extensions_to_child(child)
    return child


def _branch_primary_path(branch: dict) -> str:
    """The path the branch hands to ``run_analysis`` as its `data_path`.

    For a pattern-scoped branch this is the GLOB (directory + pattern), not the
    shared directory: ``run_analysis`` expands a glob to exactly that branch's
    files, so a sibling series in the same folder is never pulled in (#326).
    """
    pattern = branch.get("pattern")
    if pattern and branch.get("files"):
        return (str(pattern) if os.path.isabs(str(pattern))
                else os.path.join(str(branch["data_path"]), str(pattern)))
    return str(branch["data_path"])


def _mesh_task(branch: dict, companions: List[dict]) -> str:
    """Compose a branch's self-contained task with its full-mesh companions.

    Companions are named as auxiliary datasets with distinct labels so the
    specialist passes them through ``run_analysis``'s ``auxiliary_data`` /
    ``auxiliary_label`` — the existing operand path — and the codegen may use
    a shape-aligned companion numerically (correlate / mask / normalize).
    """
    task = branch["task"].rstrip()
    # This is ONE branch of a joint multi-dataset analysis — novelty is assessed
    # ONCE on the fused cross-dataset interpretation, not per branch.
    block = ["", "",
             "NOTE — this is ONE branch of a JOINT multi-dataset analysis: do NOT "
             "run novelty / literature assessment (assess_novelty) on this branch's "
             "findings. Novelty is evaluated once on the FUSED cross-dataset "
             "interpretation afterward. Complete the analysis itself normally."]
    if companions:
        primary = _branch_primary_path(branch)
        block += ["", "",
                  f"PRIMARY dataset for THIS analysis: {primary} — pass it "
                  "VERBATIM as run_analysis's `data_path`. The companion(s) below "
                  "are AUXILIARY ONLY; do NOT analyze a companion as the primary."]
        if branch.get("pattern") and branch.get("files"):
            block += [f"That path is a FILE PATTERN selecting this branch's "
                      f"{len(branch['files'])} file(s) out of a directory that also "
                      "holds the companion datasets — pass the pattern itself, do "
                      "NOT replace it with the parent directory (that would pull in "
                      "the other datasets)."]
    # Forward the caller-supplied metadata so the branch USES it rather than
    # synthesizing metadata from the task prose (which loses technique-specific
    # fields the downstream skill needs, e.g. the EELS energy axis).
    meta = branch.get("metadata")
    if meta:
        mp = _existing_json_path(meta)
        if mp is not None:
            block += ["", f"Metadata for the primary dataset is at {mp} — call "
                          "`load_metadata` on this path before `run_analysis`; do "
                          "NOT synthesize metadata when this file is provided."]
        else:
            block += ["", f"Metadata for the primary dataset: {meta}"]
    if companions:
        block += ["",
                  "COMPANION DATASETS (complementary measurements of the SAME "
                  "system — pass each as auxiliary_data with the given label so your "
                  "generated code may correlate/mask/normalize against it where the "
                  "method benefits; they are optional operands, never required):"]
        for c in companions:
            note = f"; {c['note']}" if c.get("note") else ""
            block.append(f"  - auxiliary_data: {c['data_path']}  "
                         f"(auxiliary_label: '{c['label']}'{note})")
    if not block:
        return task
    return task + "\n".join(block)


def _run_one_branch(orch, branch: dict, companions: List[dict],
                    entry: dict) -> None:
    """Execute a single fan-out branch into its preallocated ledger slot.

    Each worker touches ONLY its own ``entry`` dict, so there is no shared
    mutation across threads. Never raises — failures are captured into the
    ledger slot like ``_delegate`` does.
    """
    index = entry["index"]
    slug = _slug(branch.get("label") or Path(branch["data_path"]).stem)
    base_dir = orch.fanout_dir / f"{index:02d}_{slug}"
    try:
        from ..exp_agents.analysis_orchestrator import AnalysisMode
        child = _make_ephemeral_analysis_child(orch, base_dir)
        result = child.run_task(
            _mesh_task(branch, companions),
            context=branch.get("context"),
            autonomy=AnalysisMode.AUTONOMOUS,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"fan-out branch {index} failed: {e}")
        result = {"status": "error", "error": str(e), "summary": "",
                  "key_findings": [], "files_produced": [],
                  "suggested_followups": [], "warnings": []}
    orch._close_delegation(entry, result)


def run_fanout(orch, branches: List[dict]) -> str:
    """Gate → confirm → run branches concurrently (full-mesh aux). Returns JSON.

    `branches` is a list of ``{"data_path", "task", "label", "metadata"?,
    "context"?, "pattern"?}``. ``pattern`` is a filename glob selecting the
    branch's own files when ``data_path`` is a directory holding several
    datasets (issue #326 Fix 3). The complementarity gate prunes to the
    complementary subset; only that subset runs, each branch seeing the
    others as auxiliary operands.
    """
    # --- normalize input ---
    norm: List[dict] = []
    seen_dup: set = set()
    n_dup_dropped = 0
    for b in (branches or []):
        if not isinstance(b, dict):
            continue
        dp, task = b.get("data_path"), b.get("task")
        if not dp or not task:
            continue
        # True-duplicate guard (issue #326): the same (data_path, pattern,
        # task) twice is an accidental repeat of ONE branch — running it twice
        # wastes a slot and previously masked a silently-dropped sibling
        # branch. Distinct tasks or patterns on one path (two series in one
        # upload dir) are distinct branches and are kept.
        pattern = b.get("pattern")
        dup_key = (str(dp), str(pattern or ""),
                   re.sub(r"\s+", " ", str(task)).strip().lower())
        if dup_key in seen_dup:
            n_dup_dropped += 1
            logger.warning("fan-out: dropping duplicate branch "
                           f"(same data_path AND task): {dp}")
            continue
        seen_dup.add(dup_key)
        norm.append({
            "data_path": dp, "task": task,
            "label": (b.get("label") or Path(dp).stem),
            "metadata": b.get("metadata"), "context": b.get("context"),
            "pattern": pattern,
            "files": _resolve_branch_files(dp, pattern),
        })
    if len(norm) < 2:
        msg = ("Fan-out needs at least two branches, each with a data_path "
               "and a task.")
        if n_dup_dropped:
            msg += (f" ({n_dup_dropped} duplicate branch(es) with identical "
                    "data_path + task were dropped.)")
        return json.dumps({"status": "error", "message": msg})

    # --- branch identity (issue #326): id, not path ---
    # Distinct branches may legitimately share one data_path (a series
    # directory holding e.g. an FTIR and an XRD series, distinguished only by
    # task), so keying on the path would collapse them last-write-wins. The
    # id IS the path when unique — keeping the gate cache shared with the
    # standalone assess_complementarity tool — and path#N for same-path
    # siblings (which also forces a fresh, task-aware gate call instead of
    # reusing a path-keyed verdict computed without task context).
    path_n = Counter(b["data_path"] for b in norm)
    nth: Dict[str, int] = {}
    for b in norm:
        dp = b["data_path"]
        if path_n[dp] == 1:
            b["branch_id"] = dp
        else:
            nth[dp] = nth.get(dp, 0) + 1
            b["branch_id"] = f"{dp}#{nth[dp]}"
    by_id = {b["branch_id"]: b for b in norm}

    # --- entry gate (reuses cached verdict if assess_complementarity ran) ---
    datasets = [{"path": b["data_path"], "metadata": b.get("metadata"),
                 "id": b["branch_id"], "task": b["task"], "label": b["label"],
                 "files": b.get("files")}
                for b in norm]
    verdict = assess_complementarity(orch, datasets)
    fanout_set = [i for i in (verdict.get("fanout_set") or []) if i in by_id]

    if len(fanout_set) < 2:
        return json.dumps({
            "status": "declined",
            "reason": "not_complementary",
            "verdict": verdict,
            "message": (
                "The datasets are not genuinely complementary (no 2+ that share "
                "a system and a join axis), so a parallel cross-analysis with "
                "fusion was NOT run. Consider analyzing them independently via "
                "delegate_to_analysis, or one with the other as a plain "
                "auxiliary. See the verdict for redundant/unrelated groupings."
            ),
        }, indent=2, default=str)

    # --- confirmation ---
    proceed, reason = _confirm_fanout(orch, verdict, fanout_set, by_id)
    if not proceed:
        return json.dumps({"status": "declined", "reason": reason,
                           "verdict": verdict,
                           "fanout_set": fanout_set}, indent=2, default=str)

    # --- preallocate ledger slots (sequential, under lock — no concurrent append) ---
    run_branches = [by_id[i] for i in fanout_set]
    with orch._fanout_lock:
        entries = []
        group_id = f"fanout_{len(orch._delegation_ledger) + 1}"
        for b in run_branches:
            entry = orch._open_delegation(
                "analysis", _mesh_task(b, []), b.get("context"), None, b["label"])
            entry["parallel_group"] = group_id
            entry["fanout"] = True
            # Carry the input path/metadata so a later fuse_delegations can
            # recognize this set as already gated (or re-gate a mixed set).
            entry["data_path"] = b.get("data_path")
            entry["metadata"] = b.get("metadata")
            entry["pattern"] = b.get("pattern")
            # The gate's join axis is the fusion stage's merge key (#296):
            # without stamping it here it is lost after run_fanout returns.
            entry["join_axis"] = verdict.get("join_axis")
            entries.append(entry)

    # --- run concurrently; each branch sees all others (full mesh) ---
    print(f"  🔀 Launching {len(run_branches)} parallel analysis branches "
          f"(group {group_id}, full-mesh aux)...")

    def _companions_for(i):
        # A companion must be LOADABLE as an auxiliary operand. For a branch
        # with a resolved file set, its shared directory is not (the aux
        # loaders reject extension-less paths) — hand over a representative
        # file of the series instead (Fix 3).
        comps = []
        for j in range(len(run_branches)):
            if j == i:
                continue
            b = run_branches[j]
            c = {"label": f"companion_{_slug(b['label'])}"}
            if b.get("files"):
                c["data_path"] = b["files"][0]
                c["note"] = (f"one representative file of a "
                             f"{len(b['files'])}-file series "
                             f"('{b.get('pattern')}' in {b['data_path']})")
            else:
                c["data_path"] = b["data_path"]
            comps.append(c)
        return comps

    max_workers = min(len(run_branches), FANOUT_MAX_WORKERS)
    n_total = len(run_branches)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_label = {}
        for i in range(n_total):
            fut = pool.submit(_run_one_branch, orch, run_branches[i],
                              _companions_for(i), entries[i])
            fut_label[fut] = run_branches[i]["label"]
        # Wait with a periodic heartbeat so the user can see the parallel run is
        # alive (a slow branch can otherwise look like a hang). Each branch is
        # announced as it finishes; the rest get a "still running" tick.
        # Plain text only — the branch child orchestrators stream their own
        # progress to the same stdout from worker threads, and this output is
        # often captured (UI / log) where ANSI escapes and \r are not honored.
        # So no in-place rewrite: poll often (prompt completion announcements)
        # but print a "still running" tick at most once per _FANOUT_HEARTBEAT_S,
        # and treat a completion as its own liveness signal (resets the timer).
        pending = set(fut_label)
        start = time.monotonic()
        since_tick = 0.0
        while pending:
            t0 = time.monotonic()
            done, pending = wait(pending, timeout=_FANOUT_POLL_S)
            since_tick += time.monotonic() - t0
            for f in done:
                f.result()  # _run_one_branch never raises; just surfaces oddities
                print(f"  ✅ analysis branch finished: {fut_label[f]}  "
                      f"({n_total - len(pending)}/{n_total} done)")
                since_tick = 0.0  # a completion already shows the run is alive
            if pending and since_tick >= _FANOUT_HEARTBEAT_S:
                since_tick = 0.0
                elapsed = int(time.monotonic() - start)
                print(f"  ⏳ {len(pending)} of {n_total} parallel analyses still "
                      f"running ... (~{elapsed}s elapsed)")

    def _productive(e):
        # A branch that reports success but yields neither findings nor files
        # did no usable work (e.g. codegen aborted with no sandbox). Mirrors the
        # 'empty_but_successful' guard in _summarize_delegation_result.
        return (e.get("status") == "success"
                and (bool(e.get("key_findings")) or bool(e.get("files_produced"))))

    results = [{
        "delegation_index": e["index"],
        "label": e["label"],
        "status": e.get("status"),
        "produced_output": _productive(e),
        "key_findings": e.get("key_findings", []),
        "files_produced": e.get("files_produced", []),
    } for e in entries]
    productive = [r for r in results if r["produced_output"]]
    # A branch is "degraded" if it produced no usable output — whether it
    # hard-errored or reported success with empty findings/files (e.g. codegen
    # could not run). Either way the meta must not treat it as a completed
    # analysis or fuse it.
    degraded = [r for r in results if not r["produced_output"]]

    out = {
        "status": "success",
        "parallel_group": group_id,
        "join_axis": verdict.get("join_axis"),
        "branches_run": len(results),
        "branches_with_output": len(productive),
        "results": results,
        "next_step": (
            "Call fuse_delegations with delegation_indices="
            f"{[r['delegation_index'] for r in productive]} to reconcile these "
            "complementary findings into one cross-dataset interpretation."
            if len(productive) >= 2 else
            "Fewer than two branches produced usable output — report what ran "
            "to the user; do NOT fuse empty branches into a synthesis."
        ),
    }
    if degraded:
        n_err = sum(1 for r in degraded if r["status"] != "success")
        out["warning"] = (
            f"{len(degraded)} branch(es) produced no usable output "
            f"({n_err} errored, {len(degraded) - n_err} succeeded-but-empty, "
            "e.g. analysis code could not execute). Do not treat these as "
            "completed analyses or fuse them; report the gap to the user."
        )
    return json.dumps(out, indent=2, default=str)


# ======================================================================
# Fusion
# ======================================================================

_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_FUSION_FIG_MAX_DIM = 1536   # enough to read spatial structure; keeps payload small


def _branch_key_figure(entry: dict) -> Optional[str]:
    """Pick one representative figure from a branch's produced files.

    Prefers known representative names (segmentation overlay, NMF/PCA summary
    grid, fit-review plot); falls back to the first image. Returns a path or None.
    """
    imgs = [str(f) for f in (entry.get("files_produced") or [])
            if Path(str(f)).suffix.lower() in _FIGURE_EXTS and Path(str(f)).exists()]
    if not imgs:
        return None
    for pat in ("summary_grid", "visualization", "overlay", "review", "fit", "map"):
        for f in imgs:
            if pat in Path(f).name.lower():
                return f
    return imgs[0]


def _load_figure_part(path: str) -> Optional[dict]:
    """Load an image, downscale for the fusion call, return a {mime_type,data} part."""
    try:
        from PIL import Image
        im = Image.open(path)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if max(im.size) > _FUSION_FIG_MAX_DIM:
            im.thumbnail((_FUSION_FIG_MAX_DIM, _FUSION_FIG_MAX_DIM))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return {"mime_type": "image/png", "data": buf.getvalue()}
    except Exception:  # noqa: BLE001 - a bad figure must not break fusion
        return None


# ----------------------------------------------------------------------
# Numerics bundle (issue #296 phase a). Each branch already writes its
# numerical result to disk (features.csv keyed on the control variable,
# trend_analysis / fitting_parameters in analysis_results.json) and the
# ledger carries the paths — but fusion historically read only prose.
# These helpers surface a COMPACT schema preview of those numbers into the
# fusion prompt (never the full tables), so the synthesis can ground its
# cross-dataset statements in computed quantities. Text fusion stays
# authoritative; a branch with no readable numerics degrades to
# text+figures, never blocks.
# ----------------------------------------------------------------------

_NUMERICS_MAX_TABLES = 3          # feature tables previewed per branch
_NUMERICS_MAX_COLS_LISTED = 300   # column names listed per table
_NUMERICS_TREND_MAX_CHARS = 1500  # branch trend_analysis JSON cap in the prompt
_NUMERIC_RESULT_NAMES = ("analysis_results.json", "series_fit_results.json")


def _match_join_column(columns: List[str], join_axis: Optional[str]) -> Optional[str]:
    """Best-effort match of the gate's join-axis phrase to a table column.

    Prefix matching on word tokens, longest join-axis token first, so
    'sample temperature' finds 'temperature_C' and also an abbreviated
    'temp_C', without an incidental substring hit (the 'per' inside
    'temperature' must not select 'per_band_residual'). Returns the column
    name or None — a miss is reported in the preview, never fatal
    (LLM-assisted resolution is a later phase)."""
    if not join_axis:
        return None
    tokens = sorted((t for t in re.split(r"[^a-z0-9]+", str(join_axis).lower())
                     if len(t) >= 3), key=len, reverse=True)
    parts_by_col = {c: [p for p in re.split(r"[^a-z0-9]+", str(c).lower())
                        if len(p) >= 3] for c in columns}
    for tok in tokens:
        for col, parts in parts_by_col.items():
            if any(p.startswith(tok)
                   or (len(p) >= 4 and tok.startswith(p)) for p in parts):
                return col
    return None


def _feature_table_preview(path: str, join_axis: Optional[str]) -> dict:
    """Compact schema preview of one feature table: shape, column names, the
    join-axis column's range — NOT the data. Exception: a one-row table (a
    single-measurement branch) inlines its values, since that row IS the
    branch's scalar result and is the join anchor fusion needs."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = [str(c) for c in df.columns]
    prev: Dict[str, Any] = {"path": str(path), "n_rows": int(len(df)),
                            "n_cols": int(df.shape[1])}
    prev["columns"] = cols[:_NUMERICS_MAX_COLS_LISTED]
    if len(cols) > _NUMERICS_MAX_COLS_LISTED:
        prev["columns_truncated"] = len(cols) - _NUMERICS_MAX_COLS_LISTED
    if join_axis:
        jcol = _match_join_column(cols, join_axis)
        prev["join_axis_column"] = None
        if jcol is not None:
            info: Dict[str, Any] = {"name": jcol}
            try:
                vals = pd.to_numeric(df[jcol], errors="coerce").dropna()
                if len(vals):
                    info.update({"min": float(vals.min()),
                                 "max": float(vals.max()),
                                 "n_points": int(len(vals))})
            except Exception:  # noqa: BLE001 - range is a nicety, name suffices
                pass
            prev["join_axis_column"] = info
    if len(df) == 1:
        prev["values"] = {str(k): v for k, v in df.iloc[0].to_dict().items()
                          if not (isinstance(v, float) and v != v)}  # drop NaN
    return prev


def _branch_numerics(entry: dict, join_axis: Optional[str]) -> Optional[dict]:
    """Locate a branch's on-disk numerical results and reduce them to a
    compact, prompt-ready dict. Returns None when the branch has no readable
    numerics — fusion then proceeds on text+figures for that branch (#296
    guardrail: degrade, never block)."""
    files = [str(f) for f in (entry.get("files_produced") or [])]
    tables = [str(t) for t in (entry.get("feature_tables") or [])
              if Path(str(t)).exists()]
    if not tables:
        tables = [f for f in files
                  if Path(f).name == "features.csv" and Path(f).exists()]

    out: Dict[str, Any] = {}
    previews = []
    for t in tables[:_NUMERICS_MAX_TABLES]:
        try:
            previews.append(_feature_table_preview(t, join_axis))
        except Exception as e:  # noqa: BLE001 - a bad table must not break fusion
            logger.warning(f"fusion numerics: could not preview {t}: {e}")
    if previews:
        out["feature_tables"] = previews
    if len(tables) > _NUMERICS_MAX_TABLES:
        out["tables_omitted"] = len(tables) - _NUMERICS_MAX_TABLES

    # The branch's own trend analysis names WHAT it tracked across the series
    # — the signal fusion needs to pick the physically meaningful columns out
    # of a wide table. Single-measurement branches carry scalar
    # fitting_parameters/fit_quality instead (used only when no table loaded).
    candidates = [Path(t).parent / "analysis_results.json" for t in tables]
    candidates += [Path(f) for f in files
                   if Path(f).name == "analysis_results.json"]
    for cand in candidates:
        if "trend_analysis" in out or not cand.exists():
            continue
        try:
            with open(cand, "r", errors="replace") as fh:
                data = json.load(fh)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"fusion numerics: could not read {cand}: {e}")
            continue
        trend = data.get("trend_analysis")
        if isinstance(trend, dict) and trend and not trend.get("skipped"):
            out["trend_analysis"] = trend
        if not previews:
            for key in ("fitting_parameters", "fit_quality"):
                val = data.get(key)
                if val and key not in out:
                    out[key] = val

    # Paths of the full numeric artifacts, for auditability (and for the
    # phase-b codegen, which loads the tables itself).
    numeric_files = tables + [f for f in files
                              if Path(f).name in _NUMERIC_RESULT_NAMES]
    if out and numeric_files:
        out["numeric_files"] = list(dict.fromkeys(numeric_files))
    return out or None


def _numerics_prompt_block(num: dict) -> str:
    """Render one branch's numerics dict into fusion-prompt text."""
    lines = ["Numerical results on disk (schema preview — full tables NOT "
             "loaded):"]
    for p in num.get("feature_tables", []):
        lines.append(f"- feature table: {p['path']} — {p['n_rows']} rows x "
                     f"{p['n_cols']} cols")
        if "join_axis_column" in p:
            jc = p["join_axis_column"]
            if jc is None:
                lines.append("  join-axis column: none matched the gate's "
                             "join axis")
            elif "min" in jc:
                lines.append(f"  join-axis column: {jc['name']} (range "
                             f"{jc['min']:g} to {jc['max']:g}, "
                             f"{jc['n_points']} points)")
            else:
                lines.append(f"  join-axis column: {jc['name']}")
        cols = ", ".join(p["columns"])
        if p.get("columns_truncated"):
            cols += f", ... (+{p['columns_truncated']} more)"
        lines.append(f"  columns: {cols}")
        if p.get("values"):
            lines.append("  single-row values: "
                         + json.dumps(p["values"], default=str))
    if num.get("tables_omitted"):
        lines.append(f"- ({num['tables_omitted']} further feature table(s) "
                     "not previewed)")
    if num.get("trend_analysis") is not None:
        t = json.dumps(num["trend_analysis"], default=str)
        if len(t) > _NUMERICS_TREND_MAX_CHARS:
            t = t[:_NUMERICS_TREND_MAX_CHARS] + "...(truncated)"
        lines.append(f"- branch's own trend analysis: {t}")
    for key, label in (("fitting_parameters", "fitting parameters"),
                       ("fit_quality", "fit quality")):
        if num.get(key) is not None:
            lines.append(f"- {label}: " + json.dumps(num[key], default=str))
    if num.get("numeric_files"):
        lines.append("- full numeric artifacts (paths, for audit): "
                     + ", ".join(num["numeric_files"]))
    return "\n".join(lines)


def _write_fusion_html(out_dir: Path, fused: dict, figures: list) -> Optional[Path]:
    """Write a self-contained HTML fusion report (narrative + claims + the
    per-dataset figures inline as base64). ``figures`` is a list of
    ``(label, png_bytes)``. Returns the path, or None on failure."""
    import html as _html
    import base64
    try:
        def _claim_li(c):
            score = c.get("novelty_score")
            badge = (f" <span class='novbadge'>novelty {_html.escape(str(score))}/5</span>"
                     if score not in (None, "") else "")
            expl = c.get("novelty_explanation")
            expl_html = (f"<div class='imp'><b>Novelty:</b> {_html.escape(str(expl))}</div>"
                         if expl else "")
            return (f"<li><b>{_html.escape(str(c.get('claim', '')))}</b>{badge}"
                    f"<div class='imp'>{_html.escape(str(c.get('scientific_impact', '')))}</div>"
                    f"{expl_html}</li>")
        _claims = [c for c in (fused.get("scientific_claims") or []) if isinstance(c, dict)]
        claims_html = "".join(_claim_li(c) for c in _claims) or "<li>(none)</li>"
        _claims_hdr = ("Synthesized claims"
                       + ("  <small>(with literature-novelty score per claim)</small>"
                          if any(c.get("novelty_score") not in (None, "") for c in _claims)
                          else ""))
        caveats = [str(c) for c in (fused.get("caveats") or []) if str(c).strip()]
        caveats_html = (
            "<div class='card'><h2>Caveats &amp; limitations</h2><ul class='cav'>"
            + "".join(f"<li>{_html.escape(c)}</li>" for c in caveats)
            + "</ul></div>"
        ) if caveats else ""
        figs_html = "".join(
            f"<div class='fig'><h3>{_html.escape(str(lbl))}</h3>"
            f"<img src='data:image/png;base64,{base64.b64encode(b).decode()}'></div>"
            for lbl, b in figures
        ) or "<p>(no figures available)</p>"
        focus_html = (f"<div class='focus'><b>Focus:</b> {_html.escape(str(fused.get('focus')))}</div>"
                      if fused.get("focus") else "")
        # Styling follows the house report look (see planning_agents/
        # html_generator.py): slate page, white shadowed cards, blue accent.
        doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cross-dataset fusion</title><style>
 :root{{--primary:#2563eb;--bg:#f8fafc;--card-bg:#ffffff}}
 html{{background:var(--bg)}}
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   max-width:1000px;margin:0 auto;padding:40px 18px;color:#334155;background:var(--bg);line-height:1.6}}
 header{{background:var(--card-bg);padding:30px;border-radius:12px;
   box-shadow:0 1px 3px rgba(0,0,0,0.1);margin-bottom:40px;border-bottom:4px solid var(--primary)}}
 h1{{margin:0;color:#1e293b;font-size:1.7em}}
 .meta{{color:#64748b;font-size:0.9em;margin-top:10px}}
 .focus{{background:#eff6ff;border-left:4px solid var(--primary);padding:15px;
   margin-top:20px;color:#1e40af}}
 .card{{background:var(--card-bg);border-radius:12px;margin-bottom:40px;
   box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);overflow:hidden}}
 h2{{margin:0 0 20px;padding:15px 25px;background:#f1f5f9;border-bottom:1px solid #e2e8f0;
   font-size:1.2em;color:#475569}}
 .card > *:not(h2){{margin-left:25px;margin-right:25px}}
 .card > *:last-child{{padding-bottom:25px;margin-bottom:0}}
 .narr{{white-space:pre-wrap}}
 .imp{{color:#64748b;font-size:.9em;margin:2px 0 10px}}
 .fig{{margin:14px 25px}} .fig h3{{color:#475569}}
 .fig img{{max-width:100%;border:1px solid #e2e8f0;border-radius:8px}}
 ol{{padding-left:45px}} li{{margin-bottom:8px}}
 .cav{{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
   padding:10px 16px 10px 32px}}
 .cav li{{color:#92400e}}
 .novbadge{{background:#dbeafe;color:#1e40af;border-radius:20px;
   padding:2px 10px;font-size:.75em;font-weight:bold;white-space:nowrap}}
 h2 small{{color:#94a3b8;font-weight:normal;font-size:.7em}}
</style></head><body>
<header>
<h1>🔀 Cross-dataset fusion</h1>
<div class="meta"><b>Datasets:</b> {_html.escape(", ".join(str(l) for l in (fused.get("labels") or [])))}</div>
{focus_html}
</header>
<div class="card"><h2>Reconciled interpretation</h2>
<div class="narr">{_html.escape(str(fused.get("detailed_analysis", "")))}</div></div>
<div class="card"><h2>{_claims_hdr}</h2><ol>{claims_html}</ol></div>
{caveats_html}
<div class="card"><h2>Source figures (one per dataset)</h2>{figs_html}</div>
</body></html>"""
        path = out_dir / "fusion_report.html"
        path.write_text(doc, encoding="utf-8")
        return path
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not write fusion HTML report: {e}")
        return None


def _assess_fusion_novelty(orch, claims: list):
    """Literature-novelty scoring on the FUSED cross-dataset claims (the joint-
    analysis equivalent of per-analysis `assess_novelty`). Reuses the same
    OwlLiteratureAgent + NoveltyScorer path. Returns a list of scored entries,
    or None when no literature backend (futurehouse_api_key) is configured or
    nothing scored. Best-effort — never raises."""
    key = getattr(orch, "futurehouse_api_key", None)
    if not key or not claims:
        return None
    try:
        from ..lit_agents import OwlLiteratureAgent, NoveltyScorer
        owl = OwlLiteratureAgent(api_key=key, max_wait_time=600)
        scorer = NoveltyScorer(api_key=orch.api_key, model_name=orch.model_name,
                               base_url=orch.base_url)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"fusion novelty: could not init literature agents: {e}")
        return None
    scored = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        q = c.get("has_anyone_question")
        if not q:
            continue
        try:
            res = owl.query_literature(q)
            if res.get("status") != "success":
                continue
            sc = scorer.score_novelty(q, res.get("formatted_answer", ""))
            scored.append({
                "claim": c.get("claim"),
                "question": q,
                "novelty_score": sc.get("novelty_score", 0),
                "novelty_explanation": sc.get("explanation"),
            })
        except Exception as e:  # noqa: BLE001 - one bad claim must not break fusion
            logger.warning(f"fusion novelty: claim {i + 1} failed: {e}")
            continue
    return scored or None


def fuse_delegations(orch, indices: List[int], focus: Optional[str] = None) -> str:
    """Reconcile finished branch findings into one cross-dataset narrative.

    Reuses the HOLISTIC multi-modal synthesis template with the
    anti-spurious-correlation guard so "no correlation found" is a valid
    outcome. Reads each branch's findings from its ledger entry (summary +
    key_findings), plus a compact schema preview of the branch's on-disk
    numerical results (#296 phase a — see ``_branch_numerics``). Records
    itself as a ``mode="fusion"`` ledger entry.
    """
    from ..exp_agents.instruct import HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS

    ledger = orch._delegation_ledger
    by_index = {e["index"]: e for e in ledger}
    try:
        idxs = sorted({int(i) for i in (indices or [])})
    except (TypeError, ValueError):
        return json.dumps({"status": "error",
                           "message": "delegation_indices must be integers."})
    entries = [by_index[i] for i in idxs if i in by_index]
    ok = [e for e in entries if e.get("status") == "success"
          and (e.get("key_findings") or (e.get("summary") or "").strip())]
    if len(ok) < 2:
        return json.dumps({
            "status": "error",
            "message": ("Need >= 2 successful delegations with findings to fuse. "
                        f"Got {len(ok)} usable of {len(idxs)} requested."),
        })

    # Honesty guard (review #293): fusion has no complementarity gate of its
    # own — only run_fanout gates before launching branches. Branches of a
    # single fan-out group were verified complementary at launch; ANY other
    # fused set (notably two direct delegate_to_analysis runs on unrelated
    # data) was NOT, and could manufacture a spurious "they agree" conclusion.
    # Recognize already-gated sets; for the rest, re-run the gate when dataset
    # paths are available, else mark the fusion ungated so the synthesis prompt
    # and the report state it explicitly. We never hard-refuse (a caller may
    # deliberately want a caveated comparison) — we just refuse to do it
    # silently.
    _groups = {e.get("parallel_group") for e in ok}
    gated = (all(e.get("fanout") for e in ok)
             and len(_groups) == 1 and None not in _groups)
    # The gate's join axis is the merge key for the numerics previews below —
    # stamped on fan-out branch entries at launch; a re-gated ad-hoc set gets
    # it from the fresh verdict instead.
    join_axis = next((e.get("join_axis") for e in ok if e.get("join_axis")), None)
    ungated_warning = None
    if not gated:
        _paths = [e.get("data_path") for e in ok]
        if all(_paths) and len(set(_paths)) >= 2:
            _verdict = assess_complementarity(
                orch, [{"path": e.get("data_path"), "metadata": e.get("metadata")}
                       for e in ok])
            _v = (_verdict.get("verdict") or "").lower()
            gated = _v == "complementary"
            join_axis = join_axis or _verdict.get("join_axis")
            if not gated:
                ungated_warning = (
                    f"These datasets were assessed NOT complementary (verdict: {_v}; "
                    f"{_verdict.get('rationale') or 'no shared system/join axis'}). "
                    "Any apparent agreement across them is likely spurious.")
        else:
            ungated_warning = (
                "This fusion was NOT complementarity-gated and the datasets could "
                "not be verified as measuring the same system. Treat any "
                "cross-dataset agreement as unverified and possibly spurious.")

    blocks = []
    branch_numerics: Dict[str, Any] = {}   # label -> numerics dict (audit trail)
    for e in ok:
        findings = e.get("key_findings") or []
        findings_str = "\n".join(f"- {k}" for k in findings) if findings else "- (none)"
        label = e.get("label") or f"delegation {e['index']}"
        block = (
            f"### Dataset: {label} "
            f"(delegation #{e['index']})\n"
            f"Summary:\n{e.get('summary', '') or '(none)'}\n\n"
            f"Key findings:\n{findings_str}"
        )
        num = _branch_numerics(e, join_axis)
        if num:
            branch_numerics[label] = num
            block += "\n\n" + _numerics_prompt_block(num)
        blocks.append(block)

    # One representative figure per branch — attached to the (multimodal) fusion
    # call so spatial correlations can be verified from the actual plots, not
    # only the text. Also embedded in the HTML report. Best-effort: a missing or
    # bad figure just drops that branch's image.
    figures = []          # (label, png_bytes) for the HTML report
    image_parts = []      # interleaved label-strings + image dicts for the LLM
    for e in ok:
        fpath = _branch_key_figure(e)
        part = _load_figure_part(fpath) if fpath else None
        if part:
            label = e.get("label") or f"delegation {e['index']}"
            figures.append((label, part["data"]))
            image_parts.append(f"\n[Figure — {label}]:")
            image_parts.append(part)

    prompt = (
        HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS
        + (f"\n\n⚠️ UNGATED FUSION — {ungated_warning} Do NOT claim the datasets "
           "agree or correlate unless the evidence is overwhelming and explicit; "
           "default to reporting them as independent observations and say plainly "
           "that complementarity was not verified.\n"
           if ungated_warning else "")
        + (f"\n\nFUSION FOCUS (weight your synthesis toward this): {focus}\n"
           if focus else "")
        + ("\n\nFIGURES: one representative figure per dataset is attached after "
           "this text, labeled by dataset. Use them to verify spatial/visual "
           "correlations DIRECTLY rather than relying on the text descriptions "
           "alone.\n" if image_parts else "")
        + (("\n\nNUMERICS: dataset blocks below include a schema preview of the "
            "branch's on-disk numerical result tables (shape, column names, the "
            "join-axis column's range) and the branch's own trend analysis. "
            "Ground cross-dataset statements in these computed quantities, and "
            "quote ONLY numbers that appear in a preview or a finding — the "
            "full tables were NOT loaded, so never invent values beyond them."
            + (f" Shared join axis from the complementarity gate: {join_axis}."
               if join_axis else "")
            + "\n") if branch_numerics else "")
        + "\n\n--- PER-DATASET FINDINGS TO RECONCILE ---\n\n"
        + "\n\n".join(blocks)
    )
    parsed = _llm_json(orch, prompt, extra_parts=image_parts or None)
    if parsed is None and image_parts:
        # Multimodal call failed — fall back to a text-only fusion.
        logger.warning("fusion with figures failed; retrying text-only")
        parsed = _llm_json(orch, prompt)
    if not parsed or "detailed_analysis" not in parsed:
        return json.dumps({"status": "error",
                           "message": "Fusion synthesis did not return a usable result."})

    # Joint-analysis novelty: assess the FUSED claims once (the branches were
    # told to skip per-branch novelty). No-op without a literature backend.
    fused_claims = parsed.get("scientific_claims") or []
    novelty = None
    if getattr(orch, "futurehouse_api_key", None) and fused_claims:
        print(f"  🔍 Assessing novelty of {len(fused_claims)} fused "
              "cross-dataset claim(s)...")
        novelty = _assess_fusion_novelty(orch, fused_claims)
        # Attach each score to ITS claim (matched by the literature question), so
        # novelty enriches the claims inline rather than as a duplicate list.
        if novelty:
            _by_q = {n.get("question"): n for n in novelty if isinstance(n, dict)}
            for c in fused_claims:
                if isinstance(c, dict):
                    _n = _by_q.get(c.get("has_anyone_question"))
                    if _n:
                        c["novelty_score"] = _n.get("novelty_score")
                        c["novelty_explanation"] = _n.get("novelty_explanation")

    # Persist the fused report + record a fusion ledger entry.
    from datetime import datetime
    with orch._fanout_lock:
        fusion_n = sum(1 for e in ledger if e.get("mode") == "fusion") + 1
    out_dir = orch.fusion_dir / f"{fusion_n:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fusion_report.json"
    fused = {
        "fused_from": [e["index"] for e in ok],
        "labels": [e.get("label") for e in ok],
        "focus": focus,
        "complementarity_gated": gated,
        "complementarity_warning": ungated_warning,
        "join_axis": join_axis,
        "branch_numerics": branch_numerics or None,
        "detailed_analysis": (
            (f"⚠️ UNGATED FUSION — {ungated_warning}\n\n" if ungated_warning else "")
            + parsed.get("detailed_analysis", "")),
        "scientific_claims": parsed.get("scientific_claims", []),
        "caveats": (([ungated_warning] if ungated_warning else [])
                    + (parsed.get("caveats", []) or [])),
        "novelty": novelty,
    }
    try:
        with open(report_path, "w") as fh:
            json.dump(fused, fh, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not write fusion report: {e}")

    # Human-facing HTML report: narrative + claims + the per-dataset figures.
    html_path = _write_fusion_html(out_dir, fused, figures)
    produced = [str(report_path)] + ([str(html_path)] if html_path else [])

    with orch._fanout_lock:
        orch._delegation_ledger.append({
            "index": len(orch._delegation_ledger) + 1,
            "timestamp": datetime.now().isoformat(),
            "mode": "fusion",
            "task": f"Fuse delegations {[e['index'] for e in ok]}",
            "label": "cross-dataset fusion",
            "context_from": [e["index"] for e in ok],
            "status": "success",
            "summary": fused["detailed_analysis"],
            "key_findings": [c.get("claim", "") for c in parsed.get("scientific_claims", [])
                             if isinstance(c, dict)],
            "files_produced": produced,
            "warnings": ([ungated_warning] if ungated_warning else []),
            "error": None,
        })

    return json.dumps({
        "status": "success",
        "fused_from": [e["index"] for e in ok],
        "complementarity_gated": gated,
        "complementarity_warning": ungated_warning,
        "join_axis": join_axis,
        "numerics_branches": len(branch_numerics),
        "figures_used": len(figures),
        "detailed_analysis": fused["detailed_analysis"],
        "scientific_claims": fused["scientific_claims"],
        "caveats": fused["caveats"],
        "novelty": novelty,
        "report_path": str(report_path),
        "report_html_path": str(html_path) if html_path else None,
    }, indent=2, default=str)
