"""Parallel multi-dataset analysis (fan-out) + complementarity gating + fusion.

See CLAUDE.md "The meta agent". This is the meta's **fan-out primitive**: run
several analysis branches concurrently over GENUINELY COMPLEMENTARY datasets,
then fuse their findings into one cross-dataset narrative. Branches run
INDEPENDENTLY by default — independence is what makes fusion's agreement
claims mean anything — with two deliberate, recorded exceptions decided by
the gate's join type and per-branch opt-in: a CO-REGISTERED set gets the
operand mesh (pixel-level joint math fusion cannot redo post-hoc), and a
branch may opt into STEERING (a companion's change-point hint under the
additive-only guardrail). Both spend the branch's independence and are
stamped ``informed_by`` so fusion discounts the agreement.

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

from ...hitl import request_human_feedback

logger = logging.getLogger("meta_agent.fanout")

# Concurrency + sizing. The complementary SET (post-gate) is what these bound,
# not the raw input: the gate prunes first, so a 6-upload request with one
# complementary pair runs a 2-way mesh, not a 6-way one.
FANOUT_MAX_WORKERS = int(os.environ.get("SCILINK_FANOUT_MAX_WORKERS", "4"))
"""Peak concurrent branches (rate-limit ceiling). Overridable via the
SCILINK_FANOUT_MAX_WORKERS env var: two concurrent large-datacube branches
(each holding float64 working copies plus a process-pool of fitters) can sum
past physical RAM and get the whole meta process memory-killed — observed
live twice on a 704x704x260 pair. Set to 1 to serialize branches on
memory-constrained machines."""

# Memory-aware co-scheduling guard: don't LAUNCH a branch whose estimated
# working set, together with the branches already running, exceeds the
# machine's available memory — hold it until a slot frees. Estimates are
# coarse (input bytes x a float64-plus-copies factor, + a per-pool
# constant when the data is big enough to invite parallel fitting), and the
# guard only DELAYS branches, never drops them, so a bad estimate costs
# wall-clock, not results.
_BRANCH_MEM_FACTOR = 6.0          # input bytes -> peak working-set multiple
_BRANCH_POOL_OVERHEAD = 4e9       # loky pool cost for large per-pixel fits
_BRANCH_MEM_FLOOR = 5e8           # assume at least this per branch
_BRANCH_MEM_MARGIN = 1.5e9        # keep this much headroom for the OS

import threading as _threading
_mem_cv = _threading.Condition()
_mem_running: dict = {}           # admitted branch key -> estimated bytes


def _branch_mem_estimate(branch: dict) -> float:
    """Coarse peak-working-set estimate for a branch, from its input size."""
    try:
        p = Path(str(branch.get("data_path") or ""))
        nbytes = 0
        if p.is_file():
            nbytes = p.stat().st_size
        elif p.is_dir():
            pat = branch.get("pattern") or "*"
            nbytes = sum(f.stat().st_size for f in p.glob(pat) if f.is_file())
        est = max(nbytes * _BRANCH_MEM_FACTOR, _BRANCH_MEM_FLOOR)
        if nbytes > 1e8:   # big enough to invite a parallel per-pixel fit
            est += _BRANCH_POOL_OVERHEAD
        return est
    except Exception:  # noqa: BLE001 - an estimate must never break a branch
        return _BRANCH_MEM_FLOOR


def _available_memory() -> Optional[float]:
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def _admit_branch(key: str, est: float, label: str) -> None:
    """Hold a branch until its estimated working set fits in available
    memory alongside the branches already running. Progress guarantee: a
    branch is ALWAYS admitted when nothing else is running, so the guard
    can only delay work, never deadlock or drop it."""
    with _mem_cv:
        held_logged = False
        while True:
            avail = _available_memory()
            if (not _mem_running or avail is None
                    or avail >= est + _BRANCH_MEM_MARGIN):
                _mem_running[key] = est
                return
            if not held_logged:
                print(f"  ⏸  holding branch '{label}' for memory headroom "
                      f"(needs ~{est / 1e9:.1f} GB, available "
                      f"{avail / 1e9:.1f} GB; {len(_mem_running)} branch(es) "
                      "running)")
                held_logged = True
            _mem_cv.wait(timeout=30)


def _release_branch(key: str) -> None:
    with _mem_cv:
        _mem_running.pop(key, None)
        _mem_cv.notify_all()
_FANOUT_POLL_S = 5              # how often to check for finished branches
_FANOUT_HEARTBEAT_S = 60        # min gap between "still running" ticks
FANOUT_SOFT_CAP = 5             # warn / confirm beyond this many fused branches
FANOUT_HARD_CAP = 8             # refuse beyond this — cost/quality cliff
# Per-branch wall-clock budget (#358). A stuck branch (e.g. an analysis
# retry loop that keeps failing QC) must not hold the whole fan-out: on
# expiry the branch is recorded degraded and ABANDONED (its subprocesses are
# killed; the thread itself cannot be killed safely and may finish late,
# which is recorded for audit without changing the verdict). <= 0 disables.
FANOUT_BRANCH_TIME_BUDGET_S = 3600.0
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
is mutually complementary on all three criteria and shares ONE join relation \
(>= 2 members to be worth running in parallel). The join relation may be the \
SHARED SAMPLE itself (criterion 3): for a multi-modality study of one sample \
in one campaign, prefer the LARGEST mutually-complementary subset over the \
tightest pair — a same-sample modality is not excluded because its observable \
differs (that difference is what makes it complementary), only when criterion \
1 or 3 actually fails for it. Cluster exact-duplicate / \
same-information datasets in `redundant_clusters`. List datasets that belong \
to a different system or have no join axis in `unrelated`.

Be conservative: if you are not confident the datasets share a system and a \
join axis, prefer `uncertain` over `complementary`. A wrong `complementary` \
call produces a fabricated cross-dataset claim, which is worse than declining.

Classify the JOIN TYPE of the fanout_set — it decides how the branches are \
wired (`join_type`):
- "co_registered" — the datasets share one coordinate grid POINT-FOR-POINT \
(same scan/session/field-of-view, or an explicit registration step is \
stated). Pixel/point-level joint math between them is valid. Do NOT infer \
this from two images merely being "of the same sample" — different \
instruments are essentially never pixel-aligned.
- "shared_parameter_axis" — series over the same control variable \
(temperature, time, energy, ...): reconciliation happens between each \
branch's REDUCED trends, after analysis.
- "shared_sample" — same specimen but no shared axis or grid \
(bulk-vs-local, complementary observables): reconciliation compares \
derived quantities.

Respond in valid JSON with EXACTLY these keys:
{
  "verdict": "complementary" | "partially_complementary" | "redundant" | "unrelated" | "uncertain",
  "confidence": <float 0..1>,
  "rationale": "<one or two sentences: what the datasets are and why this verdict>",
  "join_axis": "<the shared axis the fanout_set reconciles on, or null>",
  "join_type": "co_registered" | "shared_parameter_axis" | "shared_sample" | null,
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

def _operand_mesh(verdict: dict) -> bool:
    """Whether the branches see each other as auxiliary operands.

    The gate's join type decides (#296 follow-up to #293): only a
    CO-REGISTERED set gets the operand mesh — pixel/point-level joint math
    is the one case fusion cannot reconstruct post-hoc from reduced trends.
    Shared-axis and bulk-vs-local sets run INDEPENDENT branches: fusion does
    the join there, and independence is what makes its agreement claims mean
    anything (don't spend it by accident). A missing/unknown join_type
    defaults to independent."""
    return (verdict.get("join_type") or "").strip().lower() == "co_registered"


def _confirm_fanout(orch, verdict: dict, fanout_set: List[str],
                    branches_by_id: Dict[str, dict],
                    harmonize: bool = False) -> tuple:
    """Decide whether to fire the fan-out. Returns (proceed: bool, reason: str).

    AUTOPILOT (human attached): show the verdict + the exact plan and ask the
    user to confirm. AUTONOMOUS (no human): the verdict is the gate — proceed
    only on a confident 'complementary' read within the soft cap.
    """
    n = len(fanout_set)
    mesh = _operand_mesh(verdict)
    n_aux = n * (n - 1) if mesh else 0  # operand mesh: each sees the other n-1

    if n > FANOUT_HARD_CAP:
        return False, (f"Complementary set has {n} datasets (> hard cap "
                       f"{FANOUT_HARD_CAP}); refuse to fan out. Run them in "
                       "smaller complementary groups.")

    if not orch._enable_human_feedback:
        # AUTONOMOUS: verdict-gated, conservative. 'partially_complementary'
        # is accepted alongside 'complementary': it is the verdict on the
        # INPUT set, while the gate's fanout_set is already PRUNED to the
        # mutually-complementary subset it vouches for — refusing it would
        # make partition-then-prune impossible without a human (found live
        # on a 4-modality study pruned to a coherent pair).
        if harmonize:
            # Harmonized replay is an explicit caller declaration of a
            # same-technique series comparison; the same-subject prune has
            # already run, so a "redundant" verdict is the EXPECTED reading
            # of the set, not a reason to decline. Caps still apply.
            if n > FANOUT_SOFT_CAP:
                return False, (f"Autonomous mode declines a {n}-way "
                               f"harmonized run (> soft cap "
                               f"{FANOUT_SOFT_CAP}); too expensive to fire "
                               "unattended.")
            return True, "autonomous: harmonized series declared by caller"
        v = (verdict.get("verdict") or "").lower()
        conf = float(verdict.get("confidence") or 0.0)
        if (v not in ("complementary", "partially_complementary")
                or conf < AUTONOMOUS_CONFIDENCE_THRESHOLD):
            return False, (f"Autonomous mode declines fan-out: verdict='{v}' "
                           f"confidence={conf:.2f} (needs 'complementary' or "
                           "a confidently-pruned 'partially_complementary' "
                           f">= {AUTONOMOUS_CONFIDENCE_THRESHOLD}). "
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
        f"  Join axis               : {verdict.get('join_axis')} "
        f"(join type: {verdict.get('join_type') or 'unspecified'})",
        f"  Rationale               : {verdict.get('rationale')}",
        "",
        (f"  Will run {n} branches concurrently as an OPERAND MESH "
         f"(co-registered set; ~{n_aux} auxiliary loads — results become "
         "jointly computed, flagged to fusion):" if mesh else
         f"  Will run {n} INDEPENDENT branches concurrently (no companion "
         "operands — fusion reconciles their reduced results):"),
    ]
    for bid in fanout_set:
        b = branches_by_id.get(bid, {})
        name = Path(b.get("data_path") or bid).name
        lines.append(f"    • {b.get('label') or _slug(name)}  ({name})")
    if verdict.get("redundant_clusters"):
        lines.append(f"  Pruned as redundant     : {verdict['redundant_clusters']}")
    if verdict.get("unrelated"):
        lines.append(f"  Pruned as unrelated     : {verdict['unrelated']}")
    steered = [branches_by_id[bid].get("label") for bid in fanout_set
               if branches_by_id.get(bid, {}).get("steer")]
    if steered:
        lines.append(f"  Steering opt-in         : {steered} — receives "
                     "companion change-point hints (spends independence; "
                     "fusion will discount the agreement)")
    if n > FANOUT_SOFT_CAP:
        lines.append(f"  ⚠️  {n}-way mesh exceeds the soft cap ({FANOUT_SOFT_CAP}) "
                     "— this is expensive.")
    if _branch_hitl_enabled(orch):
        lines.append("  Branches will PAUSE for approvals (served one at a "
                     "time, labeled per branch).")
    else:
        lines.append("  Branches run AUTONOMOUSLY (no per-branch approval pauses).")
    lines.append("=" * 78)
    print("\n".join(lines))

    try:
        ans = request_human_feedback(
            "\n🤔 Launch this parallel analysis? [y/N]: ",
            kind="confirm",
            options=["y", "n"],
            default="n",
            origin={"stage": "fanout_confirm"},
        ).strip().lower()
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

def _branch_hitl_enabled(orch) -> bool:
    """Per-branch human-in-the-loop is OPT-IN (default off).

    When ``orch.fanout_branch_hitl`` is truthy AND the meta has a human
    attached, branches keep the meta's autonomy level and their prompts are
    queued to the coordinator instead of being suppressed. Default-off
    preserves the historical contract: branches run AUTONOMOUS.
    """
    return bool(getattr(orch, "fanout_branch_hitl", False)) \
        and orch._enable_human_feedback


class _BranchChannel:
    """Tags a branch's requests with its label, then parks them on the
    shared queue for the coordinator to serve."""

    def __init__(self, queue_channel, label: str) -> None:
        self._qch = queue_channel
        self._label = label

    def ask(self, req):
        req.origin.setdefault("branch_label", self._label)
        return self._qch.ask(req)


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


def _steering_block(steering: List[dict]) -> List[str]:
    """Render a branch's steering payloads (#296 phase d) with the
    ADDITIVE-ONLY guardrail. Steering may only add hypotheses or effort —
    never subtract scope or lower an acceptance bar — because a range
    restriction silently converts into a conclusion (it removes the branch's
    ability to be falsified outside the window)."""
    lines = ["", "",
             "STEERING (explicit opt-in; ADDITIVE-ONLY): a cheap unsupervised "
             "reduction of each companion series below locates where THAT "
             "series changes along the shared control variable. Each is a "
             "NOMINATED HYPOTHESIS for your analysis, nothing more:"]
    for s in steering:
        extra = []
        if s.get("flags", {}).get("shift_dominated"):
            extra.append("shift-dominated (location valid; its loadings are "
                         "not species)")
        if s.get("flags", {}).get("intensity_drift"):
            extra.append("component 1 tracks overall intensity")
        note = f"; {', '.join(extra)}" if extra else ""
        fig = (f"; score curve: {s['score_curve_path']}"
               if s.get("score_curve_path") else "")
        lines.append(
            f"  - companion '{s.get('label')}': sharpest change near "
            f"control ≈ {s.get('change_point'):g} "
            f"(sharpness {s.get('change_sharpness')}, PC1 "
            f"{s.get('variance_explained', [0])[0]:.0%} of variance"
            f"{note}{fig})")
    lines += [
        "RULES (non-negotiable): you may TEST for corresponding features "
        "near the indicated value(s) and spend extra refinement effort "
        "there — but you must still analyze the FULL range/series. Never "
        "restrict a fit window or crop to the indicated region; never relax "
        "an acceptance threshold for a companion-suggested feature (it must "
        "clear your normal bar unaided); agreement with a companion is an "
        "observation to report, never a target to fit toward. Report what "
        "YOUR data shows — if it disagrees with the companion's indication, "
        "report the disagreement plainly; that is a valid, valuable outcome."]
    return lines


def _mesh_task(branch: dict, companions: List[dict]) -> str:
    """Compose a branch's self-contained task with its full-mesh companions.

    Companions are named as auxiliary datasets with distinct labels so the
    specialist passes them through ``run_analysis``'s ``auxiliary_data`` /
    ``auxiliary_label`` — the existing operand path — and the codegen may use
    a shape-aligned companion numerically (correlate / mask / normalize).
    A steered branch (#296 phase d) additionally gets each series
    companion's change-point hint under the additive-only guardrail.
    """
    task = branch["task"].rstrip()
    # This is ONE branch of a joint multi-dataset analysis — novelty is assessed
    # ONCE on the fused cross-dataset interpretation, not per branch.
    block = ["", "",
             "NOTE — this is ONE branch of a JOINT multi-dataset analysis: do NOT "
             "run novelty / literature assessment (assess_novelty) on this branch's "
             "findings. Novelty is evaluated once on the FUSED cross-dataset "
             "interpretation afterward. Complete the analysis itself normally."]
    primary = _branch_primary_path(branch)
    block += ["", "",
              f"PRIMARY dataset for THIS analysis: {primary} — pass it "
              "VERBATIM as run_analysis's `data_path`."
              + (" The companion(s) below are AUXILIARY ONLY; do NOT "
                 "analyze a companion as the primary." if companions else "")]
    if branch.get("pattern") and branch.get("files"):
        block += [f"That path is a FILE PATTERN selecting this branch's "
                  f"{len(branch['files'])} file(s) out of a directory that "
                  "also holds other datasets — pass the pattern itself, do "
                  "NOT replace it with the parent directory (that would pull "
                  "in the other datasets)."]
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
    if branch.get("_steering"):
        block += _steering_block(branch["_steering"])
    if branch.get("figure_style"):
        block += ["",
                  "FIGURE PRESENTATION (user preference — apply to every "
                  "figure this branch produces): "
                  + str(branch["figure_style"])
                  + " Forward this instruction verbatim via run_analysis's "
                  "`hints` parameter so it reaches the plotting code."]
    if not block:
        return task
    return task + "\n".join(block)


def _run_one_branch(orch, branch: dict, companions: List[dict],
                    entry: dict, queue_channel=None,
                    branch_autonomy=None) -> None:
    """Execute a single fan-out branch into its preallocated ledger slot.

    Each worker touches ONLY its own ``entry`` dict, so there is no shared
    mutation across threads. Never raises — failures are captured into the
    ledger slot like ``_delegate`` does.

    Records its start time and thread id on the entry so ``run_fanout``'s
    wait loop can enforce the branch wall-clock budget (#358): an abandoned
    branch has ``timed_out`` set on its entry, in which case a late
    completion must NOT clobber the timeout verdict (fusion already ran
    without it) — the late outcome is recorded under ``late_result`` for
    audit instead.
    """
    import threading
    index = entry["index"]
    slug = _slug(branch.get("label") or Path(branch["data_path"]).stem)
    base_dir = orch.fanout_dir / f"{index:02d}_{slug}"
    mem_key = f"{index}:{slug}"
    _admit_branch(mem_key, _branch_mem_estimate(branch),
                  branch.get("label") or slug)
    entry["_started_at"] = time.monotonic()
    entry["_branch_tid"] = threading.get_ident()
    if queue_channel is not None:
        from ...hitl import set_thread_channel
        set_thread_channel(_BranchChannel(
            queue_channel, branch.get("label") or slug))
    try:
        try:
            from ..exp_agents.analysis_orchestrator import AnalysisMode
            child = _make_ephemeral_analysis_child(orch, base_dir)
            result = child.run_task(
                _mesh_task(branch, companions),
                context=branch.get("context"),
                autonomy=(branch_autonomy if branch_autonomy is not None
                          else AnalysisMode.AUTONOMOUS),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"fan-out branch {index} failed: {e}")
            result = {"status": "error", "error": str(e), "summary": "",
                      "key_findings": [], "files_produced": [],
                      "suggested_followups": [], "warnings": []}
    finally:
        if queue_channel is not None:
            from ...hitl import set_thread_channel
            set_thread_channel(None)
        _release_branch(mem_key)
    if entry.get("timed_out"):
        logger.warning(
            f"fan-out branch {index} finished AFTER its wall-clock budget "
            f"expired (late status: {result.get('status')}); the timeout "
            "verdict stands — late outcome recorded for audit only.")
        entry["late_result"] = {
            "status": result.get("status"),
            "summary_head": (result.get("summary") or "")[:300],
            "n_files": len(result.get("files_produced") or [])}
        return
    orch._close_delegation(entry, result)


def run_fanout(orch, branches: List[dict],
               branch_time_budget_s: Optional[float] = None,
               figure_style: Optional[str] = None,
               harmonize: bool = False) -> str:
    """Gate → confirm → run branches concurrently. Returns JSON.

    `branches` is a list of ``{"data_path", "task", "label", "metadata"?,
    "context"?, "pattern"?, "steer"?}``. ``pattern`` is a filename glob
    selecting the branch's own files when ``data_path`` is a directory
    holding several datasets (issue #326 Fix 3). The complementarity gate
    prunes to the complementary subset; only that subset runs. Branches run
    independently unless the gate classifies the set co-registered (operand
    mesh) — see ``_operand_mesh`` — and/or a branch opts into steering.

    ``branch_time_budget_s`` caps each branch's wall clock (#358; default
    ``FANOUT_BRANCH_TIME_BUDGET_S``, <= 0 disables): an overdue branch is
    recorded degraded and abandoned so the fan-out completes and fusion runs
    over the productive branches.

    ``harmonize=True`` (same-technique sibling datasets): the FIRST branch
    runs alone as the pipeline DONOR; the rest then replay its approved
    analysis script verbatim (locked reuse), so cross-branch magnitudes come
    from ONE frozen pipeline. Followers' ledger entries are stamped
    ``harmonized_with`` and fusion is told the magnitudes are
    method-comparable. Falls back loudly to independent branches when the
    donor yields no approved script. The complementarity gate changes role
    in this mode: a same-technique series is by construction "redundant"
    under the gate's rubric — which is the point — so only the same-subject
    criterion is enforced (datasets the gate calls unrelated are pruned);
    the non-redundancy criterion is waived by the caller's declaration.
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
            "steer": bool(b.get("steer")),
            # User figure-presentation preference: forwarded verbatim into
            # every branch task (_mesh_task) and stashed on the orchestrator
            # for the later fusion codegen. None -> branches byte-identical
            # to pre-feature behavior.
            "figure_style": (str(figure_style) if figure_style else None),
        })
    # Stash for fuse_delegations (a separate tool call): fusion codegen
    # applies the same presentation preference to fusion_figure.png.
    orch._fanout_figure_style = str(figure_style) if figure_style else None
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
    if harmonize:
        # Harmonized replay deliberately INVERTS the gate's non-redundancy
        # criterion: a same-technique per-condition/series set — which the
        # gate rightly calls "redundant ... a per-condition series to be
        # compared" — is exactly this mode's use case. Keep criterion 1
        # (same subject): prune only what the gate calls unrelated; a
        # redundant cluster IS the series.
        _unrelated = set(verdict.get("unrelated") or [])
        fanout_set = [i for i in by_id if i not in _unrelated]
        if _unrelated:
            print(f"  🚪 Harmonized gate: pruned {len(_unrelated)} "
                  f"unrelated dataset(s); keeping the same-subject series.")
    else:
        fanout_set = [i for i in (verdict.get("fanout_set") or [])
                      if i in by_id]

    if len(fanout_set) < 2:
        return json.dumps({
            "status": "declined",
            "reason": ("not_same_subject" if harmonize
                       else "not_complementary"),
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
    proceed, reason = _confirm_fanout(orch, verdict, fanout_set, by_id,
                                      harmonize=harmonize)
    if not proceed:
        return json.dumps({"status": "declined", "reason": reason,
                           "verdict": verdict,
                           "fanout_set": fanout_set}, indent=2, default=str)

    run_branches = [by_id[i] for i in fanout_set]

    # --- steering payloads (#296 phase d) — explicit opt-in per branch ---
    # A steered branch receives each SERIES companion's change-point hint
    # (cheap unsupervised reduction, computed here at launch time) under the
    # additive-only guardrail. Steering SPENDS the branch's independence:
    # it is stamped as informed_by on the ledger entry so fusion discounts
    # the agreement. A failed reduction just skips that payload.
    for i, b in enumerate(run_branches):
        if not b.get("steer"):
            continue
        from ...skills._shared.series_reduction import reduce_series
        payloads = []
        for j, c in enumerate(run_branches):
            if j == i or not c.get("files"):
                continue
            sdir = (orch.fanout_dir / "steering"
                    / f"{_slug(b['label'])}_from_{_slug(c['label'])}")
            red = reduce_series(c["files"], out_dir=str(sdir),
                                label=c["label"])
            if red.get("status") == "success":
                payloads.append(red)
            else:
                logger.warning(
                    f"fan-out steering: reduction of '{c['label']}' failed "
                    f"({red.get('error')}); no hint passed to '{b['label']}'")
        if payloads:
            b["_steering"] = payloads
            print(f"  🧭 Steering '{b['label']}' with change-point hint(s) "
                  f"from: {[p['label'] for p in payloads]} (independence "
                  "spent — fusion will be told)")

    # --- mesh policy: the gate's join type decides the wiring ---
    mesh = _operand_mesh(verdict)

    def _companions_for(i):
        # A companion must be LOADABLE as an auxiliary operand. For a branch
        # with a resolved file set, its shared directory is not (the aux
        # loaders reject extension-less paths) — hand over a representative
        # file of the series instead (Fix 3).
        comps = []
        for j in range(len(run_branches)):
            if j == i:
                continue
            c = run_branches[j]
            comp = {"label": f"companion_{_slug(c['label'])}"}
            if c.get("files"):
                comp["data_path"] = c["files"][0]
                comp["note"] = (f"one representative file of a "
                                f"{len(c['files'])}-file series "
                                f"('{c.get('pattern')}' in {c['data_path']})")
            else:
                comp["data_path"] = c["data_path"]
            comps.append(comp)
        return comps

    branch_companions = [_companions_for(i) if mesh else []
                         for i in range(len(run_branches))]

    # --- preallocate ledger slots (sequential, under lock — no concurrent append) ---
    with orch._fanout_lock:
        entries = []
        group_id = f"fanout_{len(orch._delegation_ledger) + 1}"
        for i, b in enumerate(run_branches):
            # The ledger records the ACTUAL task the branch receives —
            # companions included — so the audit trail shows what informed it.
            entry = orch._open_delegation(
                "analysis", _mesh_task(b, branch_companions[i]),
                b.get("context"), None, b["label"])
            entry["parallel_group"] = group_id
            entry["fanout"] = True
            # Independence provenance: ANY companion contact with a branch's
            # numbers — operand mesh or steering — spends independence and is
            # recorded mechanically so fusion can discount the agreement.
            informed, via = [], []
            if mesh and len(run_branches) > 1:
                informed += [c["label"] for c in run_branches
                             if c is not b]
                via.append("co_registered_operands")
            for p in (b.get("_steering") or []):
                if p["label"] not in informed:
                    informed.append(p["label"])
                if "steering" not in via:
                    via.append("steering")
            if informed:
                entry["informed_by"] = informed
                entry["informed_via"] = "+".join(via)
            # Carry the input path/metadata so a later fuse_delegations can
            # recognize this set as already gated (or re-gate a mixed set).
            entry["data_path"] = b.get("data_path")
            entry["metadata"] = b.get("metadata")
            entry["pattern"] = b.get("pattern")
            # The gate's join axis is the fusion stage's merge key (#296):
            # without stamping it here it is lost after run_fanout returns.
            entry["join_axis"] = verdict.get("join_axis")
            entries.append(entry)

    # --- harmonized pipeline replay (donor-first) ---
    # The FIRST branch is the pipeline donor: it runs alone under the normal
    # analysis path; its approved dynamic-analysis script(s) are then
    # replayed VERBATIM by every follower (via the hyperspectral agent's
    # locked-reuse surface), so cross-branch magnitudes are measured by ONE
    # frozen pipeline instead of N independently generated ones — the
    # methodological confound fusion otherwise has to discount. A donor that
    # yields no approved replayable script falls back loudly to today's
    # independent branches (never worse).
    follower_start = 0
    harmonized_info = None
    if harmonize and len(run_branches) >= 2:
        donor = run_branches[0]
        donor_label = donor.get("label") or "donor"
        print(f"  🧬 Harmonized mode: running pipeline donor '{donor_label}' "
              "first (followers will replay its approved script)...")
        _run_one_branch(orch, donor, branch_companions[0], entries[0],
                        None, None)
        follower_start = 1
        reuse_dir = None
        if entries[0].get("status") == "success":
            _donor_dir = (orch.fanout_dir
                          / f"{entries[0]['index']:02d}_{_slug(donor_label)}")
            _cands = sorted(_donor_dir.rglob("dynamic_analysis_records.json"),
                            key=lambda p: p.stat().st_mtime)
            for _c in reversed(_cands):
                try:
                    _recs = json.loads(_c.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - keep looking
                    continue
                if any(isinstance(r, dict) and r.get("script")
                       and r.get("task_success") for r in _recs):
                    reuse_dir = _c.parent
                    break
        if reuse_dir is None:
            msg = (f"harmonize requested but donor '{donor_label}' produced "
                   "no approved replayable script — followers run "
                   "independently (unharmonized fallback).")
            print(f"  ⚠️  {msg}")
            logger.warning(f"fan-out: {msg}")
            harmonized_info = {"donor": donor_label, "status": "fallback",
                               "reason": "no approved donor script"}
        else:
            print(f"  🧬 Donor script approved — followers will replay "
                  f"{reuse_dir.name} verbatim.")
            harm_block = (
                "\n\nHARMONIZED PIPELINE REPLAY (mandatory): a sibling "
                f"dataset (donor '{donor_label}') was already analyzed and "
                "its approved analysis script must be REPLAYED VERBATIM on "
                "this dataset so the results are method-comparable. When "
                "calling run_analysis, pass EXACTLY: prior_analysis_paths="
                f"[\"{reuse_dir}\"] and reuse_locked_script=true. Do NOT "
                "run a fresh analysis plan or alter the script.")
            for _i in range(1, len(run_branches)):
                run_branches[_i]["task"] = run_branches[_i]["task"] + harm_block
                entries[_i]["task"] = _mesh_task(run_branches[_i],
                                                 branch_companions[_i])
                entries[_i]["harmonized_with"] = donor_label
                entries[_i]["harmonized_donor_index"] = entries[0]["index"]
            harmonized_info = {
                "donor": donor_label, "status": "harmonized",
                "reuse_dir": str(reuse_dir),
                "followers": [b.get("label") for b in run_branches[1:]],
            }

    # --- run concurrently; wiring per the mesh policy ---
    print(f"  🔀 Launching {len(run_branches) - follower_start} parallel "
          f"analysis branches "
          f"(group {group_id}, "
          f"{'operand mesh — co-registered set' if mesh else 'independent branches'})...")

    max_workers = min(len(run_branches), FANOUT_MAX_WORKERS)
    n_total = len(run_branches)
    budget = (FANOUT_BRANCH_TIME_BUDGET_S if branch_time_budget_s is None
              else float(branch_time_budget_s))
    # No context manager: an abandoned (timed-out) branch thread cannot be
    # killed, and `with` would join it — the pool is shut down without
    # waiting instead (#358).
    # Per-branch HITL (opt-in): branches keep the meta's autonomy and their
    # prompts are parked on a queue the coordinator serves serially below.
    queue_channel = None
    branch_autonomy = None
    if _branch_hitl_enabled(orch):
        from ...hitl import QueueChannel
        from ..exp_agents.analysis_orchestrator import AnalysisMode
        queue_channel = QueueChannel()
        branch_autonomy = AnalysisMode[orch.meta_mode.name]

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        fut_label, fut_entry = {}, {}
        for i in range(follower_start, n_total):
            fut = pool.submit(_run_one_branch, orch, run_branches[i],
                              branch_companions[i], entries[i],
                              queue_channel, branch_autonomy)
            fut_label[fut] = run_branches[i]["label"]
            fut_entry[fut] = entries[i]
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
            if queue_channel is not None:
                # Serve queued branch prompts through the coordinator's own
                # channel (console prompt / UI modal), one at a time.
                if queue_channel.serve_pending():
                    since_tick = 0.0  # a served prompt shows the run is alive
            for f in done:
                f.result()  # _run_one_branch never raises; just surfaces oddities
                print(f"  ✅ analysis branch finished: {fut_label[f]}  "
                      f"({n_total - len(pending)}/{n_total} done)")
                since_tick = 0.0  # a completion already shows the run is alive
            # Branch wall-clock budget (#358): abandon a branch that has run
            # past its deadline — record it degraded (the existing
            # degraded-branch machinery excludes it from fusion), kill its
            # registered subprocesses, and stop waiting for its thread.
            if budget > 0:
                overdue = [f for f in pending
                           if fut_entry[f].get("_started_at") is not None
                           and fut_entry[f].get("status") == "running"
                           and (time.monotonic() - fut_entry[f]["_started_at"]
                                > budget)]
                for f in overdue:
                    e = fut_entry[f]
                    e["timed_out"] = True
                    print(f"  ⏱️  analysis branch '{fut_label[f]}' exceeded "
                          f"its wall-clock budget ({int(budget)}s) — "
                          "abandoning it (degraded, excluded from fusion).")
                    logger.warning(
                        f"fan-out branch {e['index']} ('{fut_label[f]}') "
                        f"abandoned after {int(budget)}s wall-clock budget")
                    tid = e.get("_branch_tid")
                    if tid:
                        try:
                            from ...executors import kill_subprocesses_for_thread
                            kill_subprocesses_for_thread(tid)
                        except Exception:  # noqa: BLE001 - best-effort cleanup
                            pass
                    orch._close_delegation(e, {
                        "status": "error",
                        "error": (f"branch wall-clock budget exceeded "
                                  f"({int(budget)}s); branch abandoned"),
                        "summary": "", "key_findings": [],
                        "files_produced": [], "suggested_followups": [],
                        "warnings": [f"abandoned after {int(budget)}s "
                                     "wall-clock budget (#358)"],
                    })
                    pending.discard(f)
            if pending and since_tick >= _FANOUT_HEARTBEAT_S:
                since_tick = 0.0
                elapsed = int(time.monotonic() - start)
                print(f"  ⏳ {len(pending)} of {n_total} parallel analyses still "
                      f"running ... (~{elapsed}s elapsed)")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

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
        "join_type": verdict.get("join_type"),
        "mesh": "operand" if mesh else "independent",
        **({"harmonized": harmonized_info} if harmonized_info else {}),
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
        n_timeout = sum(1 for e in entries if e.get("timed_out"))
        out["warning"] = (
            f"{len(degraded)} branch(es) produced no usable output "
            f"({n_err} errored"
            + (f", of which {n_timeout} abandoned on the "
               f"branch wall-clock budget" if n_timeout else "")
            + f", {len(degraded) - n_err} succeeded-but-empty, "
            "e.g. analysis code could not execute). Do not treat these as "
            "completed analyses or fuse them; report the gap to the user."
        )
        if n_timeout:
            out["branches_timed_out"] = n_timeout
    return json.dumps(out, indent=2, default=str)


# ======================================================================
# Fusion
# ======================================================================

_FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_FUSION_FIG_MAX_DIM = 1536   # enough to read spatial structure; keeps payload small


def _branch_key_figure(entry: dict) -> Optional[str]:
    """Pick one representative figure from a branch's produced files.

    Priority order encodes evidence SCALE: a series branch's story is its
    cross-series trend (parameter_trends.png), not any single unit's fit —
    fusion reconciles trends, so the trend plot is what both the report
    reader and the fusion LLM should see. Single-measurement branches
    produce no trend figure and fall through to the representative
    per-unit names (summary grid, overlay, fit plot); last resort is the
    first image. Returns a path or None.
    """
    imgs = [str(f) for f in (entry.get("files_produced") or [])
            if Path(str(f)).suffix.lower() in _FIGURE_EXTS and Path(str(f)).exists()]
    if not imgs:
        return None
    for pat in ("trend", "summary_grid", "visualization", "overlay",
                "review", "fit", "map"):
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


# ----------------------------------------------------------------------
# Computed reconciliation (issue #296 phase b). Behind the complementarity
# gate and the sandbox gate, the fusion stage generates ONE reconciliation
# script over the branches' result tables, executes it, and persists the
# script + its numerical output next to the report — so the fused numbers
# trace to code, not to prose. Method selection lives in the baseline
# prompt (no hardcoded estimator); a failure degrades to text+figure
# fusion with a warning, never blocks.
# ----------------------------------------------------------------------

_FUSION_CODEGEN_ATTEMPTS = 3
_FUSION_SCRIPT_TIMEOUT_S = 300

FUSION_CODEGEN_INSTRUCTIONS = """You are a measurement scientist writing ONE Python script that COMPUTES the \
quantitative reconciliation of several analysis branches' findings. Each \
branch has already reduced its raw data to result tables on disk (paths and \
schema previews below). Your script loads those tables and computes the \
reconciliation, so the fused claims trace to computed numbers instead of \
prose.

Choose the method the data actually supports — inspect the previews first:
- correspondence / axis alignment — for a shared 1-D join axis: load each \
branch's physically meaningful trend (prefer the columns the branch's own \
trend analysis names; derived scalars over raw per-peak parameters), align \
on the join axis, locate each technique's transition (breakpoint / sigmoid \
midpoint / peak / onset), and report the offsets between techniques.
- derived quantity — compute a physical quantity that needs several branches.
- cross-correlation — only where trends are dense and sampled on comparable grids.
- weighted estimate / consistency test — ONLY where the tables carry real \
per-parameter uncertainties (e.g. *_err columns). NEVER weight by, or \
fabricate, an uncertainty that does not trace to a real branch value; when \
sigma is unavailable, set "sigma_available": false and use unweighted methods.
- qualitative consistency — the honest fallback when no computation is \
supportable.

Join topology: the join axis may be a table column (shared parameter axis), \
a scalar-vs-scalar comparison (same sample, no axis), or a shared 2-D grid. \
Compute a pixel-level join ONLY if the inputs establish co-registration; \
otherwise degrade to region statistics or a scalar comparison and record \
that in "notes". "No correlation found" is a valid, valuable computed result.

Marker reliability is part of the contract: a marker whose fit is degenerate \
(error comparable to the axis span), or that contradicts the branch's own \
trend analysis, must NOT be propagated into aggregate or headline quantities \
(means, spreads, agreement verdicts) — substitute the branch's own reported \
value for that technique, or exclude the technique from the aggregates, and \
record the substitution or exclusion in "notes".

SCRIPT CONTRACT (mandatory):
- Standard scientific libraries only (numpy, pandas, scipy, matplotlib with \
matplotlib.use('Agg')). No network access. Load inputs ONLY from the \
absolute paths given below.
- Write ./fusion_numerics.json (current working directory): {"method": ..., \
"sigma_available": true|false, "quantities": {<computed values, keys named \
with units>}, "notes": [<what was skipped or degraded, and why>]}. Every \
number in it must be computed by this script. Plain floats only; write \
NaN/inf as null.
- Strongly preferred: write ./fusion_figure.png. Design the figure YOURSELF \
for what these particular results need — there is no fixed template, and the \
right form (overlay, difference plot, timeline, small multiples, ...) follows \
from the join type and the finding. The only requirements: it shows the \
RECONCILIATION itself (the cross-branch relationship, not a re-plot of one \
branch); every plotted number comes from this script's computation; \
uncertainties are drawn when available; the SCALE is chosen so the finding \
is actually visible (an effect small against the absolute values needs a \
difference/zoomed view, not full-range axes); and known confounds/flags are \
annotated on the figure. A reader should be able to judge the headline \
claim from the figure alone.
- Print a short (<= 20 lines) plain-text summary to stdout.
- Be robust: a missing column or unreadable file must degrade (recorded in \
"notes"), not crash.

Respond in valid JSON with EXACTLY these keys (no markdown fences inside \
values):
{
  "method": "correspondence" | "derived_quantity" | "cross_correlation" | "weighted_estimate" | "qualitative",
  "rationale": "<2-3 sentences: why this method fits these inputs>",
  "script": "<the complete Python script, as a single JSON string>"
}
"""


FUSION_VERIFICATION_INSTRUCTIONS = """You are a skeptical measurement scientist AUDITING a computed cross-dataset \
reconciliation. A generated script was executed over several analysis \
branches' result tables; its numerical output (and overlay figure, when \
attached) follows, together with the executed script and the same per-branch \
inputs the script was given. Judge whether the COMPUTATION is scientifically \
sound — a clean exit with an implausible number is a failure.

Check, in order of importance:
1. CROSS-CHECK against the branches' own trend analyses: each branch's inputs \
name what it tracked and where its transitions lie. A computed marker that \
disagrees substantially with the branch's own value needs a reason — and if \
the script's choice of column / order parameter explains the disagreement, \
that is a flaw in the script, not a discovery.
2. DEGENERATE or PATHOLOGICAL fits: transition widths far below the sampling \
interval, values pinned to a grid point or a range edge, sigmoids fit to \
non-monotonic or spike-contaminated series.
3. ORDER-PARAMETER CHOICE (use the figure): is each plotted trend actually \
transition-shaped for its technique, or noisy/spiked such that the located \
marker is an artifact? Prefer the quantity the branch's own trend analysis \
tracked.
4. SIGMA HONESTY: "sigma_available": true is valid only if real per-parameter \
uncertainties (e.g. *_err columns) were actually used; never accept \
fabricated weights.
5. METHOD: is the chosen method appropriate for the join topology and the \
sampling?
6. FIGURE INTEGRITY (when the overlay figure is attached): the figure is a \
deliverable, not decoration — an EMPTY or broken panel (axes with no data \
drawn, a legend referencing absent curves, a panel contradicting the \
computed numbers) is a script defect: verdict "refine" naming the panel. \
Judge what is actually rendered, not what the script intended to render.

Do NOT ask for refinement over cosmetic points — numerical nitpicks that do \
not change the scientific conclusion are "accept" with the issue noted. Ask \
for refinement when a reported quantity is misleading or an artifact.

Respond in valid JSON with EXACTLY these keys:
{
  "verdict": "accept" | "refine",
  "issues": ["<each concrete problem found; empty list if none>"],
  "refinement_instructions": "<if refine: precise instructions for the next script attempt (which column / order parameter / fix); else empty string>"
}
"""


def _verify_fusion_numerics(orch, candidate: dict, script: str,
                            inputs: str) -> dict:
    """Audit the computed reconciliation (#296 phase c): the fusion LLM
    inspects the produced numbers, the script, and the overlay figure, and
    returns {"verdict": "accept"|"refine", "issues": [...],
    "refinement_instructions": ...}. On an unusable LLM reply, returns
    verdict "unavailable" (recorded, never blocks)."""
    parts = None
    if candidate.get("figure_path"):
        part = _load_figure_part(candidate["figure_path"])
        if part:
            parts = ["\n[Figure — the computed reconciliation overlay]:", part]
    prompt = (
        FUSION_VERIFICATION_INSTRUCTIONS
        + "\n\n--- COMPUTED OUTPUT (fusion_numerics.json) ---\n"
        + json.dumps(candidate.get("results"), indent=2, default=str)[:6000]
        + "\n\n--- SCRIPT STDOUT ---\n" + (candidate.get("stdout") or "")[:1500]
        + "\n\n--- THE EXECUTED SCRIPT ---\n" + (script or "")[:6000]
        + "\n\n--- THE PER-BRANCH INPUTS THE SCRIPT RECEIVED ---"
        + inputs
    )
    parsed = _llm_json(orch, prompt, extra_parts=parts)
    if not parsed or parsed.get("verdict") not in ("accept", "refine"):
        return {"verdict": "unavailable", "issues": [],
                "refinement_instructions": ""}
    return {"verdict": parsed["verdict"],
            "issues": [str(i) for i in (parsed.get("issues") or [])],
            "refinement_instructions":
                str(parsed.get("refinement_instructions") or "")}


def _fusion_codegen_inputs(ok: List[dict], branch_numerics: Dict[str, Any],
                           join_axis: Optional[str],
                           focus: Optional[str]) -> str:
    """Assemble the codegen prompt's input block: per-branch numerics (table
    paths + schema previews + the branch's own trend analysis) plus the join
    axis and any fusion focus."""
    per_branch = []
    for e in ok:
        label = e.get("label") or f"delegation {e['index']}"
        num = branch_numerics.get(label)
        if not num:
            continue
        entry = {"dataset": label,
                 "branch_summary": (e.get("summary") or "")[:600],
                 **num}
        if e.get("informed_by"):
            entry["informed_by"] = e["informed_by"]
            via = e.get("informed_via") or "steering"
            notes = []
            if "steering" in via:
                notes.append("steered at launch by the listed companion(s): "
                             "agreement near the hinted value is partly by "
                             "construction")
            if "co_registered_operands" in via:
                notes.append("received the listed companion(s) as "
                             "co-registered operands: overlapping results "
                             "may be jointly computed, not independent")
            entry["independence_note"] = "; ".join(notes)
        per_branch.append(entry)
    return (f"\n\n--- JOIN AXIS (from the complementarity gate) ---\n"
            f"{join_axis or 'not stated'}\n"
            + (f"\n--- FUSION FOCUS ---\n{focus}\n" if focus else "")
            + "\n--- PER-BRANCH NUMERICS (tables on disk; load from these "
              "paths) ---\n"
            + json.dumps(per_branch, indent=2, default=str))


def _run_fusion_codegen(orch, ok: List[dict], branch_numerics: Dict[str, Any],
                        join_axis: Optional[str], focus: Optional[str],
                        out_dir: Path) -> dict:
    """Generate -> execute -> verify -> persist the reconciliation (#296 b+c).

    After a successful execution the computed output is AUDITED
    (``_verify_fusion_numerics``): an "accept" returns it; a "refine" feeds
    the audit's instructions into the next generation attempt. If the
    attempt budget runs out with only refine-flagged results, the best
    executed result is returned WITH its unresolved issues recorded (an
    audited-but-flagged number beats a silent one). Returns a dict with
    ``status`` ('success' | 'skipped' | 'failed'), the persisted artifact
    paths, the parsed fusion_numerics.json under ``results``, the
    ``verification`` verdict, and a ``warning`` on anything non-clean.
    Never raises."""
    from ...executors import ScriptExecutor, require_sandbox_approval

    if not require_sandbox_approval(
            context="Cross-dataset fusion (computed reconciliation)"):
        return {"status": "skipped", "attempts": 0,
                "warning": ("sandbox approval unavailable — computed "
                            "reconciliation skipped; fusion is text+figures "
                            "only. Set UNSAFE_EXECUTION_OK=true or run in "
                            "Docker/VM/Colab to enable it.")}

    inputs = _fusion_codegen_inputs(ok, branch_numerics, join_axis, focus)
    executor = ScriptExecutor(timeout=_FUSION_SCRIPT_TIMEOUT_S)
    script_path = out_dir / "fusion_reconciliation.py"
    numerics_path = out_dir / "fusion_numerics.json"
    feedback = ""
    err = "no usable script generated"
    best = None          # last executed-but-refine-flagged candidate
    best_script = None
    for attempt in range(1, _FUSION_CODEGEN_ATTEMPTS + 1):
        # User figure-presentation preference from the fan-out (if any) —
        # applies to fusion_figure.png too. Empty when unset (byte-identical
        # prompt to pre-feature behavior).
        _style = getattr(orch, "_fanout_figure_style", None)
        _style_block = (
            ("\nFIGURE PRESENTATION (user preference — apply to "
             "fusion_figure.png): " + _style + "\n") if _style else "")
        parsed = _llm_json(orch, FUSION_CODEGEN_INSTRUCTIONS + _style_block
                           + inputs + feedback)
        script = (parsed or {}).get("script")
        if not script or not isinstance(script, str):
            err = "reply carried no usable 'script' string"
            feedback = ("\n\n--- PREVIOUS ATTEMPT FAILED ---\n" + err
                        + ". Return the complete JSON again.")
            continue
        try:
            script_path.write_text(script, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"fusion codegen: could not persist script: {e}")
        res = executor.execute_script(script, working_dir=str(out_dir))
        if res.get("status") == "success" and numerics_path.exists():
            try:
                with open(numerics_path, "r", errors="replace") as fh:
                    results = json.load(fh)
            except Exception as e:  # noqa: BLE001
                results = None
                err = f"fusion_numerics.json is not valid JSON: {e}"
            if results is not None:
                fig = out_dir / "fusion_figure.png"
                candidate = {"status": "success",
                             "method": parsed.get("method"),
                             "rationale": parsed.get("rationale"),
                             "script_path": str(script_path),
                             "numerics_path": str(numerics_path),
                             "figure_path": str(fig) if fig.exists() else None,
                             "results": results,
                             "stdout": (res.get("stdout") or "")[-2000:],
                             "attempts": attempt, "warning": None}
                verification = _verify_fusion_numerics(orch, candidate,
                                                       script, inputs)
                candidate["verification"] = verification
                if verification["verdict"] != "refine":
                    return candidate
                best, best_script = candidate, script
                issues = "; ".join(verification["issues"])[:1500]
                err = f"verification flagged the computation: {issues}"
                logger.warning(f"fusion codegen attempt {attempt}: {err[:400]}")
                feedback = (
                    "\n\n--- PREVIOUS ATTEMPT EXECUTED, BUT ITS OUTPUT FAILED "
                    "A SCIENTIFIC AUDIT ---\nIssues: " + issues
                    + "\nInstructions: "
                    + verification["refinement_instructions"][:1500]
                    + "\nWrite an improved script and return the complete "
                      "JSON again.")
                continue
        elif res.get("status") == "success":
            err = "the script ran but did not write ./fusion_numerics.json"
        else:
            err = res.get("message") or "execution failed"
        logger.warning(f"fusion codegen attempt {attempt} failed: {err[:400]}")
        feedback = ("\n\n--- PREVIOUS ATTEMPT FAILED ---\n" + err[:2000]
                    + "\nFix the script and return the complete JSON again.")
    if best is not None:
        # A later, worse attempt may have overwritten the persisted artifacts
        # — restore the returned candidate's script + numerics so the audit
        # trail matches what is reported. (The figure cannot be restored the
        # same way; it is best-effort.)
        try:
            script_path.write_text(best_script, encoding="utf-8")
            with open(numerics_path, "w", encoding="utf-8") as fh:
                json.dump(best["results"], fh, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"fusion codegen: could not restore artifacts: {e}")
        issues = "; ".join(best["verification"]["issues"])[:400]
        best["warning"] = ("computed reconciliation kept with UNRESOLVED "
                           f"verification issues: {issues}")
        return best
    return {"status": "failed", "attempts": _FUSION_CODEGEN_ATTEMPTS,
            "script_path": str(script_path) if script_path.exists() else None,
            "warning": (f"computed reconciliation failed after "
                        f"{_FUSION_CODEGEN_ATTEMPTS} attempts "
                        f"({err[:200]}); fusion degraded to text+figures.")}


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
        # Computed reconciliation card (#296 phase c): the executed script's
        # quantities, the audit verdict, and the audit-trail paths.
        comp = fused.get("computed_reconciliation") or {}
        comp_html = ""
        if comp and comp.get("status") == "success":
            res = comp.get("results") or {}

            def _fmt_qty(v):
                return (f"{v:.6g}" if isinstance(v, float)
                        else _html.escape(str(v)))
            qrows = "".join(
                f"<tr><td>{_html.escape(str(k))}</td>"
                f"<td class='num'>{_fmt_qty(v)}</td></tr>"
                for k, v in (res.get("quantities") or {}).items()
            ) or "<tr><td colspan='2'>(none)</td></tr>"
            notes_html = "".join(f"<li>{_html.escape(str(n))}</li>"
                                 for n in (res.get("notes") or []))
            ver = comp.get("verification") or {}
            vbadge = {"accept": "<span class='vok'>audit: accepted</span>",
                      "refine": "<span class='vbad'>audit: unresolved issues</span>"
                      }.get(ver.get("verdict"),
                            "<span class='vna'>audit: unavailable</span>")
            issues_html = ("<ul class='cav'>" + "".join(
                f"<li>{_html.escape(str(i))}</li>"
                for i in ver.get("issues") or []) + "</ul>"
            ) if ver.get("issues") else ""
            comp_html = (
                "<div class='card'><h2>Computed reconciliation "
                f"<small>method: {_html.escape(str(comp.get('method')))} · "
                f"σ available: {_html.escape(str(res.get('sigma_available')))}"
                f"</small> {vbadge}</h2>"
                f"<table class='qt'><tr><th>quantity</th><th>value</th></tr>"
                f"{qrows}</table>"
                + (f"<ul>{notes_html}</ul>" if notes_html else "")
                + issues_html
                + "<div class='paths'>script: "
                + _html.escape(str(comp.get('script_path')))
                + "<br>numerics: "
                + _html.escape(str(comp.get('numerics_path'))) + "</div>"
                "</div>")
        elif comp:
            comp_html = (
                "<div class='card'><h2>Computed reconciliation</h2>"
                "<div class='imp'>Not computed — "
                + _html.escape(str(comp.get("warning") or comp.get("status")))
                + "</div></div>")
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
 .qt{{border-collapse:collapse;margin-bottom:14px}}
 .qt td,.qt th{{border:1px solid #e2e8f0;padding:4px 12px;text-align:left}}
 .qt .num{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
 .vok,.vbad,.vna{{border-radius:20px;padding:2px 10px;font-size:.6em;
   font-weight:bold;white-space:nowrap;vertical-align:middle}}
 .vok{{background:#dcfce7;color:#166534}}
 .vbad{{background:#fee2e2;color:#991b1b}}
 .vna{{background:#e2e8f0;color:#475569}}
 .paths{{color:#94a3b8;font-size:.8em;margin-top:8px;
   font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body>
<header>
<h1>🔀 Cross-dataset fusion</h1>
<div class="meta"><b>Datasets:</b> {_html.escape(", ".join(str(l) for l in (fused.get("labels") or [])))}</div>
{focus_html}
</header>
<div class="card"><h2>Reconciled interpretation</h2>
<div class="narr">{_html.escape(str(fused.get("detailed_analysis", "")))}</div></div>
{comp_html}
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
    numerical results (#296 phase a — see ``_branch_numerics``). When the
    fusion is gated and >= 2 branches carry numerics, also generates and
    EXECUTES a reconciliation script over those tables (#296 phase b — see
    ``_run_fusion_codegen``), grounding the synthesis in computed quantities;
    the script and its outputs are persisted next to the report. Records
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
        if all(_paths):
            # The re-gate descriptors must carry the same identity signals
            # the launch gate gets (#326): id/task/label separate two
            # same-directory pattern branches — path-only descriptors
            # collapse them and read the set as redundant (found live on an
            # incremental fusion over a shared upload directory).
            _verdict = assess_complementarity(
                orch, [{"path": e.get("data_path"),
                        "metadata": e.get("metadata"),
                        "id": (f"{e.get('label') or 'delegation'}"
                               f"#{e['index']}"),
                        "task": (e.get("task") or "")[:400],
                        "label": e.get("label"),
                        "files": _resolve_branch_files(
                            e.get("data_path"), e.get("pattern"))}
                       for e in ok])
            _v = (_verdict.get("verdict") or "").lower()
            # A partial verdict gates this fusion only if the gate's pruned
            # set covers EVERY entry being fused (the verdict describes the
            # input; fanout_set is what it vouches for).
            _ids = {f"{e.get('label') or 'delegation'}#{e['index']}"
                    for e in ok}
            gated = (_v == "complementary"
                     or (_v == "partially_complementary"
                         and _ids <= set(_verdict.get("fanout_set") or [])))
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

    # Independence provenance (#296 phase d + mesh policy): a branch whose
    # numbers were touched by a companion — steering hint OR co-registered
    # operand — is not a fully independent observation of that companion.
    # The record is mechanical (never left to the LLM): stamped at launch,
    # surfaced in the synthesis prompt, written into the report caveats.
    informed = {(e.get("label") or f"delegation {e['index']}"): e["informed_by"]
                for e in ok if e.get("informed_by")}
    informed_via = {(e.get("label") or f"delegation {e['index']}"):
                    (e.get("informed_via") or "steering")
                    for e in ok if e.get("informed_by")}
    independence_caveats = []
    for lbl, srcs in informed.items():
        via = informed_via.get(lbl, "steering")
        if "steering" in via:
            independence_caveats.append(
                f"Branch '{lbl}' was steered at launch by a change-point "
                f"hint from {srcs}; its agreement with those companion(s) "
                "near the hinted value is partly by construction and must "
                "not be counted as independent corroboration.")
        if "co_registered_operands" in via:
            independence_caveats.append(
                f"Branch '{lbl}' received {srcs} as co-registered numerical "
                "operand(s); overlapping results may be jointly computed — "
                "treat cross-branch agreement there as one joint "
                "measurement, not as two independent confirmations.")
        if "fusion_feedback" in via:
            independence_caveats.append(
                f"Branch '{lbl}' was re-analyzed with feedback from a prior "
                f"fusion of {srcs}; it has effectively seen its companions' "
                "findings, so its agreement with them is partly by "
                "construction and must not be counted as independent "
                "corroboration.")

    # Harmonized replay — METHOD coupling, the opposite of an independence
    # spend: branches that replayed the donor's approved script verbatim are
    # measured by ONE frozen pipeline, so their magnitudes are comparable
    # across datasets and differences reflect the data, not pipeline
    # variance. No branch saw another's results, so observational
    # independence is intact.
    harmonized = {(e.get("label") or f"delegation {e['index']}"):
                  e.get("harmonized_with")
                  for e in ok if e.get("harmonized_with")}
    harmonized_note = ""
    if harmonized:
        _pairs = "; ".join(f"'{lbl}' ← donor '{d}'"
                           for lbl, d in harmonized.items())
        harmonized_note = (
            "\n\nHARMONIZED BRANCHES (single frozen pipeline): " + _pairs
            + ". These branches REPLAYED the donor branch's approved "
            "analysis script verbatim, so their extracted magnitudes ARE "
            "method-comparable across datasets — treat differences between "
            "them as properties of the data, never as analysis-pipeline "
            "variance. This couples METHOD only, not findings: no branch "
            "saw another's results, so cross-branch agreement still counts "
            "as independent observation.\n")

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

    # Allocate the fusion output dir up front: the computed reconciliation
    # (#296 phase b) persists its script + artifacts there before the
    # synthesis call runs.
    with orch._fanout_lock:
        fusion_n = sum(1 for e in ledger if e.get("mode") == "fusion") + 1
        # Collision guard: the ledger count alone is not a safe allocator —
        # a restored session whose checkpoint predates a fusion entry (or any
        # counter drift) would re-allocate an existing dir and silently
        # OVERWRITE the earlier fusion_report.html/json (found live on a
        # restore_checkpoint=True second fan-out). Advance past whatever
        # already exists on disk.
        while (orch.fusion_dir / f"{fusion_n:02d}").exists():
            fusion_n += 1
    out_dir = orch.fusion_dir / f"{fusion_n:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Computed reconciliation (#296 phase b): generate + execute + persist a
    # reconciliation script over the branch tables. Only behind the
    # complementarity gate — an ungated fusion must not compute a correlation
    # — and only when >= 2 branches actually carry numerics. Degrades to
    # text+figure fusion (with a recorded warning) on any failure.
    computed = None
    if len(branch_numerics) >= 2:
        if gated:
            print("  🧮 Computing quantitative reconciliation "
                  f"({len(branch_numerics)} branches with numerics)...")
            computed = _run_fusion_codegen(orch, ok, branch_numerics,
                                           join_axis, focus, out_dir)
        else:
            computed = {"status": "skipped", "attempts": 0,
                        "warning": ("ungated fusion — computed reconciliation "
                                    "not run (an unverified dataset pairing "
                                    "must not have a correlation computed "
                                    "over it).")}

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
    if computed and computed.get("figure_path"):
        part = _load_figure_part(computed["figure_path"])
        if part:
            figures.append(("computed reconciliation", part["data"]))
            image_parts.append("\n[Figure — computed reconciliation overlay]:")
            image_parts.append(part)

    # Verification verdict (#296 phase c) → one line for the synthesis: an
    # accepted audit adds confidence; unresolved issues must be weighed.
    _audit_note = ""
    if computed and computed.get("status") == "success":
        _v = computed.get("verification") or {}
        _iss = "; ".join(_v.get("issues") or [])[:1200]
        if _v.get("verdict") == "refine":
            _audit_note = ("\n⚠️ A scientific audit of this computation "
                           "flagged UNRESOLVED issues — weigh the flagged "
                           "quantities against the branches' own trend values "
                           f"and prefer the latter where they conflict: {_iss}\n")
        elif _v.get("verdict") == "accept":
            _audit_note = ("\nA scientific audit ACCEPTED this computation"
                           + (f" (with notes: {_iss})" if _iss else "") + ".\n")

    prompt = (
        HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS
        + (f"\n\n⚠️ UNGATED FUSION — {ungated_warning} Do NOT claim the datasets "
           "agree or correlate unless the evidence is overwhelming and explicit; "
           "default to reporting them as independent observations and say plainly "
           "that complementarity was not verified.\n"
           if ungated_warning else "")
        + (f"\n\nFUSION FOCUS (weight your synthesis toward this): {focus}\n"
           if focus else "")
        + harmonized_note
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
        + ((f"\n\nCOMPUTED RECONCILIATION (method: {computed.get('method')}): "
            "a reconciliation script was generated and EXECUTED over the "
            "branch tables; its output follows. These quantities are "
            "COMPUTED, not transcribed — ground the cross-dataset synthesis "
            "in them first, and where they conflict with numbers recalled in "
            "prose, prefer the computed values. If a computed value looks "
            "physically implausible, say so rather than adopting it.\n"
            + json.dumps(computed.get("results"), indent=2, default=str)[:6000]
            + _audit_note
            + f"\n(script persisted at {computed.get('script_path')}; its "
              "stdout summary:\n"
            + (computed.get("stdout") or "")[:1500] + ")\n")
           if computed and computed.get("status") == "success" else "")
        + ((f"\n\n⚠️ COMPUTED RECONCILIATION UNAVAILABLE — "
            f"{computed.get('warning')} Reconcile from the findings, "
            "previews, and figures only; do not present any cross-dataset "
            "number as computed.\n")
           if computed and computed.get("status") != "success" else "")
        + ((f"\n\nINDEPENDENCE PROVENANCE: these branches are NOT fully "
            "independent of the listed companions "
            f"(mode per branch: {json.dumps(informed_via)}): "
            f"{json.dumps(informed)}. A STEERED branch saw its companion's "
            "change-point hint — where its finding coincides with that "
            "companion near the hinted value, the agreement is partly by "
            "construction: discount it and say so. A branch that received "
            "CO-REGISTERED OPERANDS may have computed results jointly with "
            "them — treat agreement there as one joint measurement, not two "
            "independent confirmations. A branch re-analyzed with FUSION "
            "FEEDBACK has effectively seen ALL its companions' findings — "
            "the same discount applies. Branch pairs NOT listed here are "
            "independent, and their agreement carries full weight.\n")
           if informed else "")
        + ("\n\nBRANCH RE-ANALYSIS: if some branch's OWN analysis appears "
           "flawed in a way a re-analysis could fix (wrong model order, a "
           "poorly chosen order parameter, a companion-indicated feature it "
           "never tested), add a JSON key `branch_reanalysis`: a list of "
           "{\"label\", \"reason\", \"suggestion\"} objects naming the "
           "dataset label, the flaw, and a concrete re-analysis instruction. "
           "Use it ONLY for branch-level flaws (not for issues in this "
           "fusion's own computation), and omit the key when no re-analysis "
           "is warranted.\n")
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

    # Branch re-analysis suggestions (structured, so a caller/meta can act):
    # rendered as followups that carry the provenance instruction — a re-run
    # citing this fusion in context_from gets informed_via=fusion_feedback
    # stamped and the next fusion discounts it.
    reanalysis = [r for r in (parsed.get("branch_reanalysis") or [])
                  if isinstance(r, dict) and r.get("label")]
    reanalysis_followups = [
        (f"Re-analyze '{r.get('label')}': {str(r.get('reason', '')).strip()} — "
         f"{str(r.get('suggestion', '')).strip()} (When re-delegating, cite "
         "this fusion's delegation index in context_from so the independence "
         "provenance is stamped; the guidance is additive-only.)")
        for r in reanalysis]

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
    report_path = out_dir / "fusion_report.json"
    fused = {
        "fused_from": [e["index"] for e in ok],
        "labels": [e.get("label") for e in ok],
        "focus": focus,
        "complementarity_gated": gated,
        "complementarity_warning": ungated_warning,
        "join_axis": join_axis,
        "branch_numerics": branch_numerics or None,
        "computed_reconciliation": computed,
        "independence": informed or None,
        "harmonized_branches": harmonized or None,
        "branch_reanalysis": reanalysis or None,
        "detailed_analysis": (
            (f"⚠️ UNGATED FUSION — {ungated_warning}\n\n" if ungated_warning else "")
            + parsed.get("detailed_analysis", "")),
        "scientific_claims": parsed.get("scientific_claims", []),
        "caveats": (([ungated_warning] if ungated_warning else [])
                    + ([computed["warning"]]
                       if computed and computed.get("warning") else [])
                    + independence_caveats
                    + (parsed.get("caveats", []) or [])),
        "novelty": novelty,
    }
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(fused, fh, indent=2, default=str)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not write fusion report: {e}")

    # Human-facing HTML report: narrative + claims + the per-dataset figures.
    html_path = _write_fusion_html(out_dir, fused, figures)
    produced = [str(report_path)] + ([str(html_path)] if html_path else [])
    if computed:
        produced += [p for p in (computed.get("script_path"),
                                 computed.get("numerics_path"),
                                 computed.get("figure_path")) if p]

    with orch._fanout_lock:
        orch._delegation_ledger.append({
            "index": len(orch._delegation_ledger) + 1,
            "timestamp": datetime.now().isoformat(),
            "mode": "fusion",
            "task": f"Fuse delegations {[e['index'] for e in ok]}",
            "label": "cross-dataset fusion",
            # The fused branch labels: _open_delegation reads these to stamp
            # informed_via=fusion_feedback on a re-analysis that cites this
            # fusion in context_from.
            "labels": [e.get("label") for e in ok],
            "context_from": [e["index"] for e in ok],
            "status": "success",
            "summary": fused["detailed_analysis"],
            "key_findings": [c.get("claim", "") for c in parsed.get("scientific_claims", [])
                             if isinstance(c, dict)],
            "files_produced": produced,
            "suggested_followups": reanalysis_followups,
            "warnings": ([ungated_warning] if ungated_warning else []),
            "error": None,
        })
    # Persist the fusion entry the way _close_delegation persists analysis
    # entries: without this, a resumed session restores a ledger with no
    # mode="fusion" entries — losing fusion history and resetting the
    # fusion-dir counter (the overwrite the collision guard above defends
    # against).
    try:
        orch._auto_checkpoint()
    except Exception as e:  # noqa: BLE001 - checkpointing must not fail the fusion
        logger.warning(f"could not checkpoint after fusion: {e}")

    return json.dumps({
        "status": "success",
        "fused_from": [e["index"] for e in ok],
        "complementarity_gated": gated,
        "complementarity_warning": ungated_warning,
        "join_axis": join_axis,
        "numerics_branches": len(branch_numerics),
        "computed_reconciliation": computed,
        "independence": informed or None,
        "harmonized_branches": harmonized or None,
        "branch_reanalysis": reanalysis or None,
        "suggested_followups": reanalysis_followups,
        "figures_used": len(figures),
        "detailed_analysis": fused["detailed_analysis"],
        "scientific_claims": fused["scientific_claims"],
        "caveats": fused["caveats"],
        "novelty": novelty,
        "report_path": str(report_path),
        "report_html_path": str(html_path) if html_path else None,
    }, indent=2, default=str)
