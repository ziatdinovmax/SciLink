"""Shared per-item codegen-QC engine (issue #327, phase 4 — Layer 1).

One engine drives the per-item quality-control flow that was previously
duplicated as near-copies in the curve-fitting and image-analysis
controllers (``_fit_with_quality_control`` / ``_execute_and_verify``):

    reuse fast-path -> initial attempt -> [anchor] verification loop with
    adaptive constraint annealing -> post-loop accept / human / judge /
    best-available fallback

The engine owns only the mechanics that were verbatim-identical in both
copies: the loop shell, the no-config-change escalation, the refinement-
error tagging, stale-visualization cleanup, config/annealing-level state
sync, the escalate-into-hot script-drop detection, ``_produced_at_level``
stamping, the for/else final-verify dispatch, and the post-loop config
restore.

Everything modality-specific stays on the host controller as ``qc_*``
hook methods whose bodies were MOVED (not rewritten) from the two
original drivers. The deliberate CF/IA asymmetries live entirely inside
those hooks and are unchanged by construction — see the asymmetry ledger
in ``analysis_qc_unification_plan.md`` §2–§5 (curve: Option B
``best_ever_rejected``, retroactive physics promotion, rate-based
escalation, verifier bypass, ``adjust_threshold`` human action; image:
verdict-assigned scores, pre-accept patience+floor annealing,
DUMP_ITER_VIZ, CO_PILOT accept approval, judge-on-verify-error).

Host contract (duck-typed): the controller provides ``logger``,
``max_verification_iterations``, ``_CONSTRAINT_ANNEALING_SCHEDULE`` and
the ``qc_*`` hooks referenced in ``run_item``/``_verification_loop``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Bank minimal-edit adaptation (Phase B), shared across the codegen
# twins. Fires only on matches clearing the retrieval floor (0.45, hard
# filtered on fingerprint kind + axis/pixel metadata) — calibrated live
# on the curve corpus: the canonical adapt case scores 0.495-0.522
# across runs, so stricter thresholds never fired. One attempt, capped
# by the fall-through ladder. $SCILINK_BANK_EDIT_ADAPT_SCORE raises it;
# $SCILINK_BANK_EDIT_ADAPT=0 disables the mode.
BANK_EDIT_ADAPT_MIN_SCORE = 0.45


# Model-family vocabulary for the cross-kind guard: each word maps to the
# KIND of signal a model describes. Two descriptions that share no kind
# describe different physics (a peak profile vs a ring-down vs a step),
# and an "adaptation" between them is a rewrite of the model, not a
# re-seeding of it. Swapping one peak profile for another (EMG → Voigt)
# stays within a kind. Baseline words are not in the vocabulary, and a
# description with no family word never triggers the guard.
_MODEL_FAMILY_WORDS = (
    ("voigt", "peak"), ("lorentz", "peak"), ("gauss", "peak"),
    ("exponentially-modified", "peak"), ("exponentially modified", "peak"),
    ("emg", "peak"), ("doniach", "peak"), ("fano", "peak"),
    ("lognormal", "peak"), ("log-normal", "peak"), ("pearson", "peak"),
    ("damped", "oscillation"), ("cosine", "oscillation"),
    ("sinusoid", "oscillation"), ("sine", "oscillation"),
    ("oscillat", "oscillation"), ("ring-down", "oscillation"),
    ("ringdown", "oscillation"),
    ("exponential decay", "decay"), ("exp decay", "decay"),
    ("stretched", "decay"), ("bi-exponential", "decay"),
    ("biexponential", "decay"), ("relaxation", "decay"),
    ("sigmoid", "step"), ("logistic", "step"), ("error function", "step"),
    ("erf", "step"), ("step", "step"),
    ("power law", "powerlaw"), ("power-law", "powerlaw"),
    ("polynomial", "polynomial"),
)


def model_family_tokens(text) -> set:
    """The signal kinds a free-text model description names."""
    t = str(text or "").lower()
    return {kind for needle, kind in _MODEL_FAMILY_WORDS if needle in t}


def model_families_disjoint(a, b) -> bool:
    """True when both descriptions name a signal kind and share none."""
    fa, fb = model_family_tokens(a), model_family_tokens(b)
    return bool(fa) and bool(fb) and not (fa & fb)


def relax_snippet_edits(text: str, edits: list) -> tuple:
    """Re-anchor edits whose ``old_text`` is not in ``text`` because of
    leading whitespace only.

    Live: the adapter returned a script's top-level ``params.add(...)``
    lines with a four-space indent four times out of four, and the exact
    applier rejected every batch. Each such edit is matched by its lines
    with leading whitespace ignored; a UNIQUE block match rewrites
    ``old_text`` to the script's actual text and re-indents ``new_text``
    by the same shift. Ambiguous or absent blocks are left for the applier
    to refuse. Returns ``(edits, n_relaxed)``; never raises.
    """
    lines = text.splitlines()
    stripped = [l.lstrip() for l in lines]
    out, n_relaxed = [], 0
    for e in edits:
        if not isinstance(e, dict):
            out.append(e)
            continue
        old = e.get("old_text") or ""
        if not old or old in text:
            out.append(e)
            continue
        want = old.splitlines()
        want_s = [l.lstrip() for l in want]
        if not want_s or not want_s[0]:
            out.append(e)
            continue
        hits = [i for i in range(len(lines) - len(want) + 1)
                if stripped[i:i + len(want)] == want_s]
        if len(hits) != 1:
            out.append(e)
            continue
        i = hits[0]
        actual = "\n".join(lines[i:i + len(want)])
        # Shift new_text's indentation the way old_text's was shifted.
        model_indent = len(want[0]) - len(want_s[0])
        real_indent = len(lines[i]) - len(stripped[i])
        new = e.get("new_text") or ""
        fixed_new = []
        for l in new.splitlines():
            lead = len(l) - len(l.lstrip())
            if l.strip():
                lead = max(0, lead - model_indent) + real_indent
                fixed_new.append(" " * lead + l.lstrip())
            else:
                fixed_new.append(l)
        out.append({**e, "old_text": actual, "new_text": "\n".join(fixed_new)})
        n_relaxed += 1
    return out, n_relaxed


def try_bank_edit_adapt(host, ctx, *, domain: str, script_kind: str,
                        output_contract: str, config_key: str,
                        data_context: str, run_fn):
    """Adapt a strongly matching banked script via an LLM edit LIST,
    applied mechanically — instead of whole-script re-emission.

    Full-regeneration "adapt" erodes exactly the provenance that made
    the script worth banking; a minimal-edit adaptation keeps the proven
    logic byte-recognizable and (on clean acceptance) accumulates
    cross-session success evidence on the SAME bank record. The result
    enters the UNCHANGED verification loop — execution and the quality
    gate stay the arbiters. Every fall-through is logged with its reason
    (no silent fallbacks); returning None resumes the exemplar-
    generation path exactly.

    `host` supplies model / generation_config / logger; `run_fn(script)`
    executes the adapted script through the modality's single-item
    runner and returns its result dict.
    """
    import json as _json
    import os as _os

    state = ctx.state
    exemplar = state.get("_bank_exemplar")
    if not exemplar:
        return None
    if (_os.environ.get("SCILINK_BANK_EDIT_ADAPT", "").strip().lower()
            in ("0", "false", "off", "no")):
        return None
    try:
        min_score = float(_os.environ.get(
            "SCILINK_BANK_EDIT_ADAPT_SCORE", BANK_EDIT_ADAPT_MIN_SCORE))
    except ValueError:
        min_score = BANK_EDIT_ADAPT_MIN_SCORE
    score = float(exemplar.get("score") or 0)
    if score < min_score:
        host.logger.info(
            f"   🏦 Edit-adapt skipped: match score {score} < "
            f"{min_score} — exemplar-guided generation instead.")
        return None
    rec = exemplar.get("record") or {}
    banked = (rec.get("working_script") or "").strip()
    if not banked:
        return None
    try:
        from .instruct import BANK_EDIT_ADAPT_INSTRUCTIONS
        from scilink.skills._shared._graduation import parse_json_response
        from scilink.utils.file_edit import apply_snippet_edits
        prompt = BANK_EDIT_ADAPT_INSTRUCTIONS.format(
            script_kind=script_kind,
            banked_script=banked,
            locked_config=_json.dumps(state.get(config_key) or {},
                                      indent=2, default=str),
            data_context=data_context,
            output_contract=output_contract,
        )
        # Attempt provenance for the assist log, from the moment the call is
        # spent: an adaptation that never applies or never runs still cost
        # an LLM call, and that waste has to be visible.
        ctx.bank_adapt_attempt = {"n_edits": None, "applied": False,
                                  "executed": False}
        raw = host.model.generate_content(
            prompt, generation_config=host.generation_config)
        raw = raw.text if hasattr(raw, "text") else str(raw)
        try:
            parsed = parse_json_response(raw)
        except ValueError:
            retry = ("Respond with ONLY a single JSON object — no prose "
                     "before or after it.\n\n" + prompt)
            raw = host.model.generate_content(
                retry, generation_config=host.generation_config)
            raw = raw.text if hasattr(raw, "text") else str(raw)
            parsed = parse_json_response(raw)
        edits = parsed.get("edits")
        if not isinstance(edits, list):
            raise ValueError("no edits list in the adaptation reply")
        # The adapter is asked to say when THIS data needs a different model
        # family than the proven script implements. That is not an
        # adaptation (live: a peak script "adapted" into a damped cosine in
        # eight edits, then credited as a peak-fit success) — fall through
        # to normal generation and leave the record's evidence alone.
        kept = parsed.get("model_family_kept")
        if kept is False or (isinstance(kept, str)
                             and kept.strip().lower() in ("false", "no")):
            ctx.bank_adapt_attempt["model_mismatch"] = True
            raise ValueError("model family mismatch: this dataset needs a "
                             "different model than the proven script "
                             f"({str(parsed.get('rationale'))[:100]})")
        if edits:
            edits, n_relaxed = relax_snippet_edits(banked, edits)
            if n_relaxed:
                ctx.bank_adapt_attempt["n_relaxed"] = n_relaxed
            res = apply_snippet_edits(banked, edits)
            if res["status"] != "success":
                # One corrected attempt with the applier's own message: the
                # model sees which edit failed and why, against the same
                # script.
                fix = ("Your previous edit list did not apply: "
                       f"{res['message']}\nReturn a corrected edit list. "
                       "Copy each old_text VERBATIM from the proven script "
                       "above, including its exact leading whitespace "
                       "(top-level lines have none).\n\n" + prompt)
                raw = host.model.generate_content(
                    fix, generation_config=host.generation_config)
                raw = raw.text if hasattr(raw, "text") else str(raw)
                parsed = parse_json_response(raw)
                edits = parsed.get("edits")
                if not isinstance(edits, list) or not edits:
                    raise ValueError(f"edits do not apply: {res['message']}")
                edits, n_relaxed2 = relax_snippet_edits(banked, edits)
                if n_relaxed2:
                    ctx.bank_adapt_attempt["n_relaxed"] = (
                        ctx.bank_adapt_attempt.get("n_relaxed") or 0) + n_relaxed2
                ctx.bank_adapt_attempt["retried"] = True
                res = apply_snippet_edits(banked, edits)
                if res["status"] != "success":
                    raise ValueError(f"edits do not apply: {res['message']}")
            adapted_script, n_edits = res["text"], res["n_edits"]
        else:
            adapted_script, n_edits = banked, 0
        host.logger.info(
            f"   🏦 ✏️  Bank edit-adapt: record {rec.get('id')} "
            f"(score {score}), {n_edits} edit(s) — "
            f"{str(parsed.get('rationale'))[:120]}")
        ctx.bank_adapt_attempt.update(n_edits=n_edits, applied=True)
        result = run_fn(adapted_script)
        ctx.bank_adapt_attempt["executed"] = bool(result.get("success"))
        if not result.get("success"):
            host.logger.info(
                "   🏦 ↩️  Edit-adapted script did not execute cleanly — "
                "falling back to exemplar-guided generation.")
            return None
        ctx.initial_label = (f"bank edit-adapt of {rec.get('id')} "
                             f"({n_edits} edit(s))")
        result["bank_edit_adapt"] = {
            "id": rec.get("id"), "score": score, "n_edits": n_edits,
            "edits": list(edits), "rationale": parsed.get("rationale"),
        }
        return result
    except Exception as e:  # noqa: BLE001 - never worse than today
        _att = getattr(ctx, "bank_adapt_attempt", None)
        if isinstance(_att, dict):
            _att["fell_through"] = str(e)[:160]
        host.logger.info(
            f"   🏦 ↩️  Edit-adapt fell through ({e}) — falling back to "
            "exemplar-guided generation.")
        return None


def record_bank_assist(host, ctx, res, *, domain: str,
                       seconds: Optional[float] = None,
                       anchor_only: bool = True) -> Optional[dict]:
    """Attach a ``bank_assist`` block to a finished QC-loop item and append
    it to the bank's assist log.

    One block per item that went through the codegen QC loop (``anchor_only``
    = the curve / image rule that only anchors do; hyperspectral targets all
    do) — including
    ``mode="none"`` (nothing in the bank matched), which is the baseline the
    assisted runs are compared against. Locked-script reuse is not a bank
    event and is skipped. ``survived`` says whether an edit-adapted script is
    still the accepted one: a verification-loop refit drops the
    ``bank_edit_adapt`` provenance, and with it the claim that the bank
    helped. Bookkeeping only — never raises, never changes ``res``'s verdict.
    """
    try:
        if (not isinstance(res, dict) or res.get("reuse_validity")
                or res.get("locked_replay")):
            return None
        # Curve / image run the verification loop on anchors only; a
        # non-anchor item's zero iterations would corrupt the baseline.
        if anchor_only and not getattr(ctx, "is_anchor", True):
            return None
        match = getattr(ctx, "bank_exemplar", None) or {}
        attempt = getattr(ctx, "bank_adapt_attempt", None) or {}
        rec = match.get("record") or {}
        if attempt:
            mode = "edit_adapt"
        elif rec:
            mode = "exemplar"
        else:
            mode = "none"
        qh = res.get("quality_history") or {}
        block = {
            "domain": domain,
            "mode": mode,
            "record_id": rec.get("id"),
            "score": match.get("score"),
            "fingerprint_score": match.get("fingerprint_score"),
            "iterations": len(qh.get("verification_iterations") or []),
            "approved": bool(qh.get("approved")),
            "item": getattr(ctx, "item_name", None),
        }
        if mode == "edit_adapt":
            block["n_edits"] = attempt.get("n_edits")
            block["applied"] = bool(attempt.get("applied"))
            block["executed"] = bool(attempt.get("executed"))
            block["survived"] = bool(res.get("bank_edit_adapt"))
            if attempt.get("fell_through"):
                block["fell_through"] = attempt["fell_through"]
            for k in ("n_relaxed", "retried", "model_mismatch", "model_changed"):
                if attempt.get(k):
                    block[k] = attempt[k]
        if seconds is not None:
            block["seconds"] = round(float(seconds), 2)
        res["bank_assist"] = block
        from scilink.skills._shared import _script_bank
        # The bank must learn bad news too: an adaptation that executed and
        # was then replaced, or did not execute, cost a call and delivered
        # nothing — that is evidence about the SCRIPT. An edit list that
        # never applied, or an adapter that declined the data as a different
        # model family, says nothing about the script (live: four
        # non-applying edit lists archived a good record as "never
        # succeeds"), so those are logged but not charged.
        if (mode == "edit_adapt" and not block["survived"] and rec.get("id")
                and block["applied"]):
            _script_bank.record_failure(
                domain, rec["id"],
                "edit_adapt_" + ("replaced" if block["executed"]
                                 else "not_executed"),
                session=Path(str(getattr(host, "output_dir", "") or "")).name or None)
        _script_bank.log_assist(
            {**block, "session": Path(
                str((ctx.state or {}).get("output_dir")
                    or getattr(host, "output_dir", "") or "")).name or None})
        return block
    except Exception:  # noqa: BLE001 - bookkeeping must never fail a run
        return None


def bump_bank_adapt_success(host, res, *, domain: str, ctx=None) -> None:
    """CLEAN acceptance of an edit-adapted script accumulates proven-N
    evidence on the SAME bank record. A verification-loop refit replaces
    the result and drops the provenance, so a rejected adaptation never
    bumps — and a kept-but-flagged poor fit (quality_warning) must not
    either (live: an NMR adaptation executed fine at R²=0.42 and would
    have counted as "proven"). Proven-N feeds the graduation signal;
    only unflagged survivals are evidence."""
    try:
        bea = res.get("bank_edit_adapt") if isinstance(res, dict) else None
        if (bea and bea.get("id") and res.get("success")
                and not res.get("quality_warning")):
            from scilink.skills._shared import _script_bank
            # Cross-kind guard (belt to the adapter's own braces): a result
            # whose model family shares nothing with the record's is a
            # rewrite, and its success is evidence for a different method.
            rec = ((getattr(ctx, "bank_exemplar", None) or {}).get("record")
                   or {}) if ctx is not None else {}
            rec_model = (rec.get("technique_signals") or {}).get("model_type")
            if model_families_disjoint(rec_model, res.get("model_type")):
                bea["model_changed"] = True
                attempt = getattr(ctx, "bank_adapt_attempt", None)
                if isinstance(attempt, dict):
                    attempt["model_changed"] = True
                host.logger.info(
                    f"   🏦 ⚠️  Bank record {bea['id']}: adapted script fits a "
                    "different model family "
                    f"({str(rec_model)[:40]!r} → {str(res.get('model_type'))[:40]!r}); "
                    "no cross-session credit.")
                return
            # Evidence = the NEW data's digest (independent of the data the
            # record was banked on), plus the session.
            _script_bank.record_success(
                domain, bea["id"],
                session=Path(str(getattr(host, "output_dir", "") or "")).name or None,
                fingerprint=getattr(ctx, "bank_query_fingerprint", None),
                # An adaptation with edits means the ADAPTED script passed,
                # not the banked one: evidence that the record is a good
                # starting point, not that it runs unchanged. An adaptation
                # with ZERO edits is the banked script itself, accepted under
                # LLM verification — verbatim evidence earned under review,
                # which is also the only way a script born under a
                # reduced-depth profile can become eligible for unreviewed
                # reuse.
                adapted=bool(bea.get("n_edits")))
            host.logger.info(
                f"   🏦 📈 Bank record {bea['id']}: cross-session success "
                "recorded (edit-adapted script survived QC).")
    except Exception:  # noqa: BLE001 - bookkeeping must never fail a run
        pass
def apply_reuse_script_edits(state: dict, reuse_script, reuse_source,
                             logger=None):
    """Apply the caller's surgical ``script_edits`` to a reused script.

    Shared across the #172 reuse twins (curve fitting, image analysis):
    the prior script runs byte-identical EXCEPT the requested edits, so
    consecutive runs stay comparable one variable at a time. Edits were
    validated at ``analyze()`` entry against this same prior script, so a
    failure here means the prior run changed on disk mid-run — refuse
    loudly rather than silently run the UNEDITED script the caller asked
    to change.
    """
    edits = state.get("script_edits") or []
    if not (reuse_script and edits):
        return reuse_script, reuse_source
    from scilink.utils.file_edit import apply_snippet_edits
    res = apply_snippet_edits(reuse_script, edits)
    if res["status"] != "success":
        raise RuntimeError(
            f"script_edits no longer apply to the prior script "
            f"({res['message']}) — the prior run changed on disk after "
            "validation. Nothing was run.")
    if logger:
        logger.info(f"   ✏️  Applied {res['n_edits']} surgical edit(s) to "
                    f"the reused script (execution will verify)")
    return res["text"], f"{reuse_source} + {res['n_edits']} edit(s)"


def attach_script_edit_provenance(ctx, reuse_result: dict) -> None:
    """Record the exact old/new pairs on a reuse result, so the delta
    between consecutive runs is auditable from saved artifacts alone."""
    if ctx.state.get("script_edits"):
        reuse_result["script_edits_applied"] = list(
            ctx.state["script_edits"])
        rv = reuse_result.get("reuse_validity")
        if isinstance(rv, dict):
            rv["script_edits"] = len(ctx.state["script_edits"])


@dataclass(frozen=True)
class QCEngineSpec:
    """Modality constants consumed by the shared engine shell.

    config_key      -- the state key holding the locked plan/config
                       ("locked_fitting_config" / "locked_analysis_config").
                       None = the modality has no locked-config plumbing
                       (hyperspectral): the shell skips the unchanged-config
                       escalation, the state config/annealing-level writes,
                       and the post-loop config restore.
    refine_anchor   -- which result's script anchors a (non-hot) refit
                       prompt: "best" (curve refines the high-water script),
                       "current" (image refines the latest script), or
                       "none" (hyperspectral regenerates from the prompt;
                       structural freedom lives in the retry-feedback text).
                       Deliberate asymmetry, preserved.
    refit_fail_msg  -- log line when a refit attempt fails.
    """

    config_key: Optional[str]
    refine_anchor: str  # "best" | "current" | "none"
    refit_fail_msg: str


class QCItemContext:
    """Mutable per-item state shared between the engine shell and hooks.

    Generic fields mirror the locals of the original drivers (curve's
    ``best_r2`` -> ``best_score``); modality-specific loop state
    (``best_ever_rejected``, ``accept_gate``, ``verification_attempts``,
    ...) is attached by the host's ``qc_setup`` / ``qc_loop_setup`` hooks.
    """

    def __init__(self, *, state: dict, data: Any, data_path: str,
                 item_name: str, item_idx: int,
                 is_regime_anchor: bool = False,
                 reuse_script: Optional[str] = None,
                 reuse_source: Optional[str] = None):
        self.state = state
        self.data = data
        self.data_path = data_path
        self.item_name = item_name
        self.item_idx = item_idx
        self.is_regime_anchor = is_regime_anchor
        self.reuse_script = reuse_script
        self.reuse_source = reuse_source

        # Anchor = first item overall OR first in a regime; gets full QC
        self.is_anchor = item_idx == 0 or is_regime_anchor

        # Set by the engine when the run's time budget ran out mid-loop.
        self.budget_expired: bool = False
        # Set by the engine when a reduced-depth profile's iteration cap was
        # reached with the verifier still rejecting.
        self.capped: bool = False

        self.all_attempts: list = []
        self.verification_history: list = []
        self.best_result: Optional[dict] = None
        self.best_score: float = -1.0
        self.best_config: dict = {}
        self.current_result: Optional[dict] = None
        self.current_score: float = -1.0
        self.approved = False
        self.initial_label: Any = None
        self.initial_result: Optional[dict] = None

        # Annealing state (populated by the engine at loop entry)
        self.n_levels: int = 0
        self.start_level: int = 0
        self.annealing_level: int = 0
        self.previous_annealing_level: int = 0
        self.stall_count: int = 0
        self.prev_best_score: float = -1.0
        self.iteration: int = 0
        # A host's reason to stop the loop early with the best result so
        # far (#568: a prescribed fix that will not take) — checked after
        # each refinement, logged, then the final-verify path runs as for
        # the wall-clock budget.
        self.stop_reason: Optional[str] = None


class CodegenQCEngine:
    """Layer-1 per-item QC engine: shared shell + host-owned hooks."""

    def __init__(self, host, spec: QCEngineSpec):
        self.host = host
        self.spec = spec

    def run_item(self, ctx: QCItemContext) -> dict:
        host = self.host
        ctx.n_levels = len(host._CONSTRAINT_ANNEALING_SCHEDULE)
        host.qc_setup(ctx)

        # --- #172: locked-script reuse fast path ---
        if ctx.reuse_script and ctx.is_anchor:
            reuse_out = host.qc_try_reuse(ctx)
            if reuse_out is not None:
                return reuse_out

        # --- Initial attempt (annealing schedule starts here; default T=0) ---
        # A re-run may start the schedule HIGHER (e.g. hot) via
        # `_starting_annealing_level`, so it does not repeat early constraint
        # stages a prior run already found inadequate. Default 0 = unchanged.
        ctx.start_level = max(0, min(int(ctx.state.get("_starting_annealing_level") or 0),
                                     ctx.n_levels - 1))
        if self.spec.config_key is not None:
            ctx.state["_annealing_level"] = ctx.start_level

        result = host.qc_run_initial(ctx)
        if not result["success"]:
            host.qc_record_initial_failure(ctx, result)
            # A first fit that produced NOTHING leaves no best attempt for any
            # later stage to return. A host may spend one bounded recovery here
            # (curve: one re-plan told why the first plan could not be fitted).
            recover = getattr(host, "qc_recover_initial_failure", None)
            recovered = recover(ctx, result) if recover is not None else None
            if recovered is not None and recovered.get("success"):
                result = recovered
        ctx.initial_result = result

        if result["success"]:
            host.qc_record_initial(ctx, result)

            if host.qc_verification_bypass(ctx):
                # Host accepted the initial result without verification
                # (curve's max_verification_iterations<=0 fast/in-situ path;
                # the hook sets ctx.approved itself).
                pass
            elif ctx.is_anchor:
                if not ctx.best_result or not ctx.best_result.get("success"):
                    host.qc_log_skip_verification(ctx)
                else:
                    self._verification_loop(ctx)
                    # Restore config to match best result after verification loop
                    if self.spec.config_key is not None:
                        ctx.state[self.spec.config_key] = ctx.best_config

            post = host.qc_post_verification(ctx)
            if post is not None:
                return post

        # --- Human feedback / judge / best-available fallback ---
        return host.qc_fallback(ctx)

    def _verification_loop(self, ctx: QCItemContext) -> None:
        host, spec = self.host, self.spec
        logger = host.logger

        # Adaptive annealing state. max(start-1,0): starting AT hot still
        # registers the transition into hot (fires fresh generation); start
        # at 0 restores the original `= 0`.
        ctx.annealing_level = ctx.start_level
        ctx.previous_annealing_level = max(ctx.start_level - 1, 0)
        ctx.stall_count = 0

        # current_* tracks the latest refit (what the verifier diagnoses
        # next); best_* is the high-water mark used as the refinement anchor
        # and final return value.
        ctx.current_result = ctx.best_result
        ctx.current_score = ctx.best_score
        ctx.prev_best_score = ctx.best_score

        host.qc_loop_setup(ctx)

        # Cumulative wall-clock budget for the whole verification loop
        # (#358). OPT-IN via a host attribute — hosts that don't set it get
        # byte-identical behavior. Attempt counts alone don't bound time:
        # each iteration regenerates + re-executes code, so N cheap-but-
        # failing attempts can run for hours without a ceiling.
        import time as _time
        _budget = getattr(host, "qc_time_budget_s", None)
        _loop_t0 = _time.monotonic()
        # The RUN's deadline (QCProfile.time_budget_s, stamped into the state
        # by the agent) is separate from the host's own loop budget above and
        # stricter about what it spends once it is gone: no final verify, no
        # judge — the best result so far is returned as-is, and the host
        # marks it unverified (ctx.budget_expired).
        _run_deadline = (ctx.state or {}).get("_run_deadline")

        max_iters = host.max_verification_iterations
        for verification_iter in range(max_iters):
            ctx.iteration = verification_iter
            if _run_deadline is not None and _time.monotonic() >= _run_deadline:
                logger.warning(
                    f"   ⏱️  Run time budget spent after {verification_iter} "
                    "verification iteration(s) — returning the best result "
                    "so far, unverified (no further LLM calls).")
                ctx.budget_expired = True
                return
            if _budget and _time.monotonic() - _loop_t0 > _budget:
                logger.warning(
                    f"   Verification loop wall-clock budget exceeded "
                    f"({int(_budget)}s) after {verification_iter} "
                    "iteration(s) — stopping with the best result so far.")
                host.qc_final_verify(ctx)
                return
            # Hosts whose verification runs INSIDE the attempt (hyperspectral)
            # can override this banner — printed here, between a finished
            # attempt and the next one, "Verification k/N" reads as misplaced
            # for them. Returning None suppresses it (the host logs its own
            # line at the semantically right spot); hosts without the hook
            # keep the default text byte-identically.
            _banner = getattr(host, "qc_iteration_banner", None)
            if _banner is not None:
                _msg = _banner(ctx, verification_iter + 1, max_iters)
                if _msg:
                    logger.info(f"   {_msg}")
            else:
                logger.info(
                    f"   Verification {verification_iter + 1}/{max_iters} "
                    f"(annealing level {ctx.annealing_level})..."
                )

            verification = host.qc_verify(ctx)
            if verification is None:
                host.qc_on_verify_none(ctx)
                return

            # Score extraction / promotion / history append (+ modality
            # extras: curve's Option-B + physics promotion, image's
            # pre-accept patience/floor annealing).
            host.qc_assess(ctx, verification)

            if host.qc_check_accept(ctx, verification):
                ctx.approved = True
                return

            # Reduced-depth profiles (purpose-scoped verification): the last
            # allowed verification has just rejected. Refining and refitting
            # once more would produce an attempt nobody may verify within the
            # cap — observed live, that attempt then cost a final verify AND a
            # judge (four LLM calls, ~170 s) only for the best fit to be
            # accepted on the deterministic gate anyway. Stop here; the host
            # returns the best attempt, flagged (ctx.capped).
            if (verification_iter == max_iters - 1
                    and (ctx.state or {}).get("_verification_mode") == "purpose"):
                logger.info(
                    "   Iteration cap reached with the verifier still rejecting "
                    "— returning the best attempt so far, flagged (reduced-depth "
                    "profile: no further refit, final verify or judge).")
                ctx.capped = True
                return

            # Apply the verifier's recommended fixes.
            refined_config = host.qc_refine(ctx, verification)

            # If the refinement LLM call failed (transient API error), tag
            # the history so the next verifier knows the fix was never applied.
            refinement_error = refined_config.pop("_refinement_error", None)
            if refinement_error:
                ctx.verification_history[-1]["refinement_error"] = refinement_error

            if ctx.stop_reason:
                logger.warning(f"   Stopping verification: {ctx.stop_reason}")
                host.qc_final_verify(ctx)
                return

            if (spec.config_key is not None
                    and refined_config == ctx.state.get(spec.config_key, {})):
                # No changes at current temperature — escalate to give the
                # LLM more freedom before giving up.
                _cur_level = ctx.annealing_level
                ctx.annealing_level = min(ctx.annealing_level + 1, ctx.n_levels - 1)
                if ctx.annealing_level == _cur_level:
                    logger.info("   No config changes at max annealing level, stopping verification")
                    return
                logger.info(f"   No config changes suggested, escalating to annealing level {ctx.annealing_level}")
                continue

            # Clean up old visualization (but not the best result's viz —
            # best_result and current_result share the same path when
            # current was just promoted).
            old_viz_path = ctx.current_result.get("visualization_path")
            if (old_viz_path
                    and Path(old_viz_path).exists()
                    and ctx.current_result is not ctx.best_result):
                try:
                    os.remove(old_viz_path)
                except Exception:
                    pass

            if spec.config_key is not None:
                ctx.state[spec.config_key] = refined_config

                # Sync skill strictness with adaptive annealing level
                ctx.state["_annealing_level"] = ctx.annealing_level

            # Drop the script anchor on the single iteration where the
            # annealing level escalates INTO the hot level, so the code
            # generator can restructure freely without anchor bias from
            # the structure that prompted the escalation.
            _hot = ctx.n_levels - 1
            just_escalated_to_hot = (
                ctx.annealing_level >= _hot
                and ctx.previous_annealing_level < _hot
            )
            if spec.refine_anchor == "best":
                _anchor_result = ctx.best_result
            elif spec.refine_anchor == "current":
                _anchor_result = ctx.current_result
            else:  # "none"
                _anchor_result = None
            refine_from = (
                None if just_escalated_to_hot
                else (_anchor_result or {}).get("script")
            )

            refit_result = host.qc_refit(ctx, verification, refine_from,
                                         just_escalated_to_hot)
            # Stamp the annealing level this refit was generated at, so a
            # downstream consumer can tell whether the WINNING result came
            # from a hot (fresh-generation) regeneration vs. the original
            # plan — used by T=2 auto-distillation to decide a fit is a
            # "novel pipeline". Travels with the result dict through every
            # promotion / judge path.
            if isinstance(refit_result, dict):
                refit_result["_produced_at_level"] = ctx.annealing_level
            # Record the level this refit ran at, so the next iteration can
            # detect the escalation into hot. Updated only on an actual
            # refit (not the no-config-change escalate-and-continue branch).
            ctx.previous_annealing_level = ctx.annealing_level

            if refit_result["success"]:
                host.qc_after_refit(ctx, refit_result, verification)
            else:
                logger.warning(spec.refit_fail_msg)
                return
        else:
            # Loop exhausted without approval — one final pass to rate the
            # latest state.
            host.qc_final_verify(ctx)
