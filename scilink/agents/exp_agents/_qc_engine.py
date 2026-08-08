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
        if edits:
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
        result = run_fn(adapted_script)
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
        host.logger.info(
            f"   🏦 ↩️  Edit-adapt fell through ({e}) — falling back to "
            "exemplar-guided generation.")
        return None


def bump_bank_adapt_success(host, res, *, domain: str) -> None:
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
            _script_bank.record_success(domain, bea["id"])
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
        else:
            host.qc_record_initial_failure(ctx, result)

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

        max_iters = host.max_verification_iterations
        for verification_iter in range(max_iters):
            ctx.iteration = verification_iter
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

            # Apply the verifier's recommended fixes.
            refined_config = host.qc_refine(ctx, verification)

            # If the refinement LLM call failed (transient API error), tag
            # the history so the next verifier knows the fix was never applied.
            refinement_error = refined_config.pop("_refinement_error", None)
            if refinement_error:
                ctx.verification_history[-1]["refinement_error"] = refinement_error

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
