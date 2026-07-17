"""Engine-neutral supervised-execution (refinement) loop.

A single deterministic loop that runs a simulation to convergence: execute
the inputs, assess the finished run with the post-run critic, apply the
critic's proposed fixes, and re-run — repeating until the result is
acceptable, a policy stops it, or a cycle budget is exhausted. The loop is
scale- and engine-agnostic: it instantiates the same shape for periodic DFT,
classical MD, and ML-potential MD by swapping the executor, the critic's
active skill, and the policy.

The skeleton is source and deterministic, which is what makes a refinement
run reproducible for benchmarking. The judgments *inside* the loop are
delegated:

* **Execution** is an :class:`Executor` — "materialize these inputs, run
  them, hand back an output directory." The run command and binary name
  arrive as data, so no engine name appears here.
* **Assessment** is the post-run critic (``RunCritic.assess``), grounded in
  the active skill's ``interpretation`` section. Its verdict consolidates
  both the engine-error question (``run_status``) and the physical-quality
  question (``verdict``) into one call.
* **Continuation and interaction** are a :class:`RefinementPolicy`. The
  three autonomy levels are three policies, so "different autonomy → different
  feedback" falls out of policy selection rather than branching in the loop.

The loop reads a normalized list of :class:`Phase` objects, so a single-stage
DFT run and a multi-stage MD run (optimization → equilibration → production)
flow through the same code; the quality check fires per phase.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Verdict ordering
# ──────────────────────────────────────────────────────────────────────────

# Post-run verdicts, best to worst. The loop and the built-in policies reason
# about the ordinal rank rather than any engine-specific number, so no
# universal quality metric is required (DFT cares about forces/energy, MD
# about energy drift; both map onto this ordinal scale via the critic).
_VERDICT_RANK = {"good": 3, "warning": 2, "poor": 1, "needs_fixes": 0}


def _verdict_rank(verdict: Optional[Dict[str, Any]]) -> int:
    """Return the ordinal rank of a critic verdict (higher is better)."""
    if not verdict:
        return 0
    return _VERDICT_RANK.get(verdict.get("verdict", "needs_fixes"), 0)


def _is_acceptable(verdict: Optional[Dict[str, Any]]) -> bool:
    """Whether a verdict is good enough to stop refining a phase."""
    return _verdict_rank(verdict) >= _VERDICT_RANK["warning"]


class CycleDecision(Enum):
    """Continuation decision after a finished run."""

    STOP = "stop"
    REFINE = "refine"


# ──────────────────────────────────────────────────────────────────────────
# Phase
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Phase:
    """One executable stage of a simulation.

    A single-stage calculation (e.g. a DFT relaxation) is one phase; a staged
    MD run is several (optimization → equilibration → production). All fields
    are engine-neutral: ``input_files`` and ``run_command`` are produced by
    the foundation agent / skill bundle, never assembled here.

    Attributes:
        name: Short phase label, used for logging and the result record.
        input_files: Mapping of filename to file contents to materialize in
            the run directory before execution.
        run_command: The command the executor runs in the run directory.
        run_dir: Directory the phase executes in (created if absent).
    """

    name: str
    input_files: Dict[str, str]
    run_command: str
    run_dir: str
    entry_file: str = ""   # the deck filename in input_files (for the dry-run gate)


@dataclass
class Stage:
    """One stage of a simulation campaign: a group of phases run together.

    A campaign is an ordered list of stages. A stage is one of three shapes,
    all engine-neutral — what a fan-out's members differ by (temperature,
    restraint center, …) lives in each member's ``input_files``, authored by
    the generator, never here:

    * **sequential step** (``parallel=False``): its phases run in order and
      chain through restart files (e.g. optimization → equilibration →
      production). Typically one phase per stage.
    * **parallel fan-out** (``parallel=True``): independent member phases that
      differ only in their inputs — a temperature sweep, or the windows of an
      umbrella-sampling run. Members are refined independently; one member's
      failure does not abort its siblings.
    * **combine** (``kind="combine"``): a single post-processing phase that
      consumes a prior fan-out's outputs (e.g. assembling a free-energy
      profile). Run once and judged, never iterated.

    Attributes:
        name: Short stage label for logging and the result record.
        phases: The phases this stage runs.
        parallel: Whether the phases are independent (fan-out) rather than a
            chained sequence.
        kind: ``"run"`` for a normal stage, ``"combine"`` for a run-once
            post-processing stage.
        min_success: For a fan-out, the minimum number of members that must
            succeed for the stage to succeed. ``None`` requires all of them.
    """

    name: str
    phases: List[Phase]
    parallel: bool = False
    kind: str = "run"
    min_success: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────
# Critic protocol (duck-typed against RunCritic.assess)
# ──────────────────────────────────────────────────────────────────────────

class _RunCriticLike(Protocol):
    """Structural type for the post-run critic the loop consumes.

    ``RunCritic`` (``critics.py``) satisfies this; tests pass a scripted
    fake. Returning a separate protocol keeps the loop importable without
    pulling in the LLM-backed critic stack.
    """

    def assess(
        self,
        output_dir: str,
        research_goal: str,
        skill: Optional[str] = ...,
        domain: Optional[str] = ...,
        fixes_mode: str = ...,
        input_files: Optional[Dict[str, str]] = ...,
        check_observables: bool = ...,
    ) -> Dict[str, Any]:
        ...


# ──────────────────────────────────────────────────────────────────────────
# Executor contract
# ──────────────────────────────────────────────────────────────────────────

class Executor(ABC):
    """Run a set of inputs and hand back an output directory.

    The one new abstraction the refinement loop needs. Implementations cover
    where a run happens — local subprocess, container, or HPC scheduler —
    without the loop knowing which. An executor never decides whether a run
    *succeeded* by parsing engine logs; that judgment belongs to the post-run
    critic. The executor only reports how the process exited and where its
    output landed.
    """

    @abstractmethod
    def run(
        self, input_files: Dict[str, str], run_command: str, run_dir: str
    ) -> Dict[str, Any]:
        """Materialize ``input_files`` in ``run_dir``, run ``run_command``.

        Args:
            input_files: Mapping of filename to contents to write before the
                run.
            run_command: Command to execute in ``run_dir``.
            run_dir: Working directory for the run (created if absent).

        Returns:
            A dict with at least ``status`` (``"completed"`` if the process
            ran to exit, regardless of exit code; ``"error"`` if it could not
            be launched), ``output_dir``, and ``returncode``.
        """


class LocalExecutor(Executor):
    """Run inputs as a local subprocess in the run directory.

    Writes each input file, runs the command, and persists ``stdout``,
    ``stderr``, and ``returncode`` to files in the run directory so the
    post-run critic's snapshot can read them. It does not interpret the
    output — a non-zero exit code is reported, not judged.

    Attributes:
        timeout: Per-run wall-clock limit in seconds.
    """

    # Files the executor writes alongside the run so the snapshot can surface
    # them to the critic. Engine-neutral names.
    STDOUT_FILE = "run_stdout.log"
    STDERR_FILE = "run_stderr.log"
    RETURNCODE_FILE = "run_returncode.txt"

    def __init__(self, timeout: int = 3600):
        self.timeout = timeout

    def run(
        self, input_files: Dict[str, str], run_command: str, run_dir: str
    ) -> Dict[str, Any]:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        for name, contents in (input_files or {}).items():
            (run_path / name).write_text(contents)

        logger.info("LocalExecutor: running %r in %s", run_command, run_dir)
        try:
            proc = subprocess.run(
                run_command,
                shell=True,
                cwd=str(run_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            (run_path / self.STDERR_FILE).write_text(
                f"Timed out after {self.timeout}s\n{e}"
            )
            (run_path / self.RETURNCODE_FILE).write_text("timeout")
            return {
                "status": "error",
                "output_dir": str(run_path),
                "returncode": None,
                "error": f"Run timed out after {self.timeout}s",
            }
        except (OSError, FileNotFoundError) as e:
            (run_path / self.STDERR_FILE).write_text(str(e))
            (run_path / self.RETURNCODE_FILE).write_text("launch_error")
            return {
                "status": "error",
                "output_dir": str(run_path),
                "returncode": None,
                "error": f"Could not launch run command: {e}",
            }

        (run_path / self.STDOUT_FILE).write_text(proc.stdout or "")
        (run_path / self.STDERR_FILE).write_text(proc.stderr or "")
        (run_path / self.RETURNCODE_FILE).write_text(str(proc.returncode))
        return {
            "status": "completed",
            "output_dir": str(run_path),
            "returncode": proc.returncode,
        }


# ──────────────────────────────────────────────────────────────────────────
# Refinement context
# ──────────────────────────────────────────────────────────────────────────

# Signature of the human-feedback handle the policies may call. Given a
# prompt and a context payload, it returns the human's response string. The
# orchestrator injects its existing prompt mechanism; headless callers pass
# None and the policies proceed without consulting a human.
InteractFn = Callable[[str, Dict[str, Any]], str]


@dataclass
class RefinementContext:
    """Shared state threaded through a refinement run.

    Carries the routing identity (scale/engine/skill/domain), the research
    goal the critic judges against, the autonomy level, the cycle budget, the
    running history, and the optional human-feedback handle. The policy — not
    the loop — decides whether to call :attr:`interact`, so autonomy lives in
    one place and reuses the caller's existing interaction machinery.

    Attributes:
        research_goal: What the user is trying to compute; passed to the
            critic on every assessment.
        scale: Simulation scale (e.g. ``"molecular_dynamics"``).
        engine: Engine within the scale (e.g. ``"lammps"``).
        skill: Skill bundle name for the critic (defaults to ``engine``).
        domain: Skill subdirectory (defaults to ``scale``).
        autonomy: Autonomy level label (``"co-pilot"`` / ``"autopilot"`` /
            ``"autonomous"``).
        max_cycles: Maximum refine cycles per phase.
        cycle: Refine cycles spent on the current phase (loop-managed).
        coverage_votes: Independent coverage checks to majority-vote in the
            pre-run gate (1 = single check; >1 damps the stochastic decision).
        history: Per-cycle records appended by :meth:`record`.
        interact: Optional human-feedback handle; ``None`` for headless runs.
    """

    research_goal: str
    scale: Optional[str] = None
    engine: Optional[str] = None
    skill: Optional[str] = None
    domain: Optional[str] = None
    autonomy: str = "autonomous"
    max_cycles: int = 3
    cycle: int = 0
    coverage_votes: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    interact: Optional[InteractFn] = None

    def record(
        self,
        phase: Phase,
        result: Dict[str, Any],
        verdict: Dict[str, Any],
    ) -> None:
        """Append a record of one finished cycle to :attr:`history`."""
        self.history.append({
            "phase": phase.name,
            "cycle": self.cycle,
            "run_status": verdict.get("run_status"),
            "verdict": verdict.get("verdict"),
            "returncode": result.get("returncode"),
            "had_fixes": bool(verdict.get("suggested_fixes")),
        })


# ──────────────────────────────────────────────────────────────────────────
# Policies — the autonomy levels
# ──────────────────────────────────────────────────────────────────────────

class RefinementPolicy(ABC):
    """The two decisions a refinement loop delegates to autonomy.

    A refinement run makes exactly two kinds of decision: whether to accept a
    proposed set of inputs (the pre-run inputs and each proposed fix share
    this gate), and whether a finished run should stop or refine. Each
    autonomy level is one concrete policy.
    """

    @abstractmethod
    def approve_change(
        self,
        proposed_inputs: Optional[Dict[str, str]],
        verdict: Optional[Dict[str, Any]],
        ctx: RefinementContext,
    ) -> Optional[Dict[str, str]]:
        """Gate a proposed input set.

        Used at the pre-run gate (``verdict`` from the input validator) and
        after each fix (``verdict`` from the run critic). Returns the inputs
        to run (possibly edited), or ``None`` to abort/stop.
        """

    @abstractmethod
    def after_run(
        self,
        result: Dict[str, Any],
        verdict: Dict[str, Any],
        ctx: RefinementContext,
    ) -> CycleDecision:
        """Given a finished run and its verdict, stop or refine."""


class AutonomousPolicy(RefinementPolicy):
    """Run end to end without consulting a human.

    Refines toward a ``good`` verdict, applying the critic's proposed fixes
    as-is, and never calls ``interact``. The decision to keep refining is
    driven by the critic, not by a fixed verdict threshold: any verdict below
    ``good`` is refined *only* when the critic proposed an actionable fix (its
    ``suggested_fixes`` is its "re-running with this would help" signal) and
    the loop is still making progress. So a fixable warning gets a correction
    attempt, while a benign warning the critic offers no fix for — or a
    stalled loop — stops. Bounded by fixes-available, the stall check, and the
    cycle budget.
    """

    def approve_change(self, proposed_inputs, verdict, ctx):
        return proposed_inputs

    def after_run(self, result, verdict, ctx):
        # Already good — nothing left to improve.
        if _verdict_rank(verdict) >= _VERDICT_RANK["good"]:
            return CycleDecision.STOP
        # Below good: refine only if the critic flagged something actionable.
        if not verdict.get("suggested_fixes"):
            return CycleDecision.STOP
        # ...and only while the verdict is still improving.
        if _stalled(ctx):
            return CycleDecision.STOP
        return CycleDecision.REFINE


class AutopilotPolicy(RefinementPolicy):
    """Lead with reasonable defaults, surfacing only risky changes.

    Applies fixes automatically, but if a human-feedback handle is present it
    surfaces low-confidence pre-run changes for a one-line confirmation.
    Refines while the result is below acceptable and improving; stops when it
    converges or stalls.
    """

    def approve_change(self, proposed_inputs, verdict, ctx):
        if proposed_inputs is None:
            return None
        # Surface only genuinely risky pre-run changes, and only when a human
        # is attached. A "fails" pre-run verdict is the risky case.
        if (
            ctx.interact is not None
            and verdict is not None
            and verdict.get("validation_status") == "fails"
        ):
            answer = ctx.interact(
                "Pre-run validation flagged the inputs as failing. Apply the "
                "proposed changes and run anyway?",
                {"verdict": verdict, "proposed_inputs": proposed_inputs},
            )
            if _is_negative(answer):
                return None
        return proposed_inputs

    def after_run(self, result, verdict, ctx):
        if _is_acceptable(verdict):
            return CycleDecision.STOP
        if not verdict.get("suggested_fixes"):
            return CycleDecision.STOP
        # Stop if the last cycle did not improve the verdict (stalled).
        if _stalled(ctx):
            return CycleDecision.STOP
        return CycleDecision.REFINE


class CoPilotPolicy(RefinementPolicy):
    """Human leads; the loop asks before each input change and continuation.

    Requires a human-feedback handle to be useful. Without one (``interact``
    is ``None``) it degrades to applying fixes and stopping as soon as the
    result is acceptable, so a co-pilot policy never blocks a headless run.
    """

    def approve_change(self, proposed_inputs, verdict, ctx):
        if proposed_inputs is None or ctx.interact is None:
            return proposed_inputs
        answer = ctx.interact(
            "Apply these proposed input changes and run?",
            {"verdict": verdict, "proposed_inputs": proposed_inputs},
        )
        return None if _is_negative(answer) else proposed_inputs

    def after_run(self, result, verdict, ctx):
        if ctx.interact is None:
            return (
                CycleDecision.STOP
                if _is_acceptable(verdict) or not verdict.get("suggested_fixes")
                else CycleDecision.REFINE
            )
        if not verdict.get("suggested_fixes"):
            return CycleDecision.STOP
        answer = ctx.interact(
            f"Run finished with verdict={verdict.get('verdict')!r}. "
            "Refine and re-run? (yes to refine, no to stop)",
            {"verdict": verdict, "result": result},
        )
        return CycleDecision.REFINE if _is_affirmative(answer) else CycleDecision.STOP


# Autonomy-label → policy. The loop reads the session's existing autonomy
# level and uses the matching built-in policy rather than reinventing
# autonomy. Labels match SimulationMode values.
_POLICIES = {
    "co-pilot": CoPilotPolicy,
    "autopilot": AutopilotPolicy,
    "autonomous": AutonomousPolicy,
}


def policy_for(autonomy: str) -> RefinementPolicy:
    """Return the built-in policy for an autonomy label (default autonomous)."""
    return _POLICIES.get((autonomy or "").strip().lower(), AutonomousPolicy)()


def _is_negative(answer: Optional[str]) -> bool:
    return (answer or "").strip().lower() in {"n", "no", "abort", "stop", "cancel"}


def _is_affirmative(answer: Optional[str]) -> bool:
    return (answer or "").strip().lower() in {"y", "yes", "ok", "go", "refine"}


def _stalled(ctx: RefinementContext) -> bool:
    """Whether the last two cycles of the current phase did not improve."""
    if len(ctx.history) < 2:
        return False
    last, prev = ctx.history[-1], ctx.history[-2]
    if last.get("phase") != prev.get("phase"):
        return False
    rank = _VERDICT_RANK.get
    return rank(last.get("verdict"), 0) <= rank(prev.get("verdict"), 0)


# ──────────────────────────────────────────────────────────────────────────
# The loop
# ──────────────────────────────────────────────────────────────────────────

def _refine_phase(
    phase: Phase,
    executor: Executor,
    run_critic: _RunCriticLike,
    policy: RefinementPolicy,
    ctx: RefinementContext,
) -> Dict[str, Any]:
    """Run one phase to convergence: run → assess → fix → re-run.

    The per-phase primitive shared by sequential steps and fan-out members:
    runs the executor, assesses the output, and — when the policy says to
    refine — applies the critic's whole-file fixes and re-runs, up to the
    context's cycle budget. The pre-run gate is *not* applied here; the caller
    gates the campaign's first phase once. Appends one history entry per cycle.

    Returns:
        A phase record: ``phase``, ``status`` (``"success"``, ``"stopped"``,
        ``"aborted"``, or ``"exhausted"``), ``cycles``, ``verdict``, and
        ``run_status``.
    """
    inputs = phase.input_files
    ctx.cycle = 0
    last_verdict: Dict[str, Any] = {}
    phase_status = "failed"

    while ctx.cycle < ctx.max_cycles:
        result = executor.run(inputs, phase.run_command, phase.run_dir)

        if result.get("status") == "error":
            # Could not launch — still assess (the critic reads the persisted
            # stderr) so a fix can be proposed.
            logger.warning(
                "Executor could not launch phase %s: %s",
                phase.name, result.get("error"),
            )

        verdict = run_critic.assess(
            output_dir=result.get("output_dir", phase.run_dir),
            research_goal=ctx.research_goal,
            skill=ctx.skill,
            domain=ctx.domain,
            input_files=inputs,
        )
        last_verdict = verdict
        ctx.record(phase, result, verdict)

        decision = policy.after_run(result, verdict, ctx)
        if decision == CycleDecision.STOP:
            phase_status = "success" if _is_acceptable(verdict) else "stopped"
            break

        fixes = verdict.get("suggested_fixes")
        if not fixes:
            phase_status = "success" if _is_acceptable(verdict) else "stopped"
            break

        approved = policy.approve_change(fixes, verdict, ctx)
        if approved is None:
            phase_status = "aborted"
            break
        inputs = approved
        ctx.cycle += 1
    else:
        # Budget exhausted without an acceptable verdict.
        phase_status = "exhausted"

    return {
        "phase": phase.name,
        "status": phase_status,
        "cycles": ctx.cycle + 1,
        "verdict": last_verdict.get("verdict"),
        "run_status": last_verdict.get("run_status"),
        "failure_class": last_verdict.get("failure_class"),
    }


def _run_once_phase(
    phase: Phase,
    executor: Executor,
    run_critic: _RunCriticLike,
    ctx: RefinementContext,
) -> Dict[str, Any]:
    """Run a phase exactly once and assess it, with no refine loop.

    Used for a combine stage: it consumes a prior fan-out's outputs (e.g.
    assembling a free-energy profile from umbrella windows), so re-running it
    with a fix is not the right recovery — it is judged once, not iterated.
    """
    result = executor.run(phase.input_files, phase.run_command, phase.run_dir)
    if result.get("status") == "error":
        logger.warning(
            "Executor could not launch combine phase %s: %s",
            phase.name, result.get("error"),
        )
    verdict = run_critic.assess(
        output_dir=result.get("output_dir", phase.run_dir),
        research_goal=ctx.research_goal,
        skill=ctx.skill,
        domain=ctx.domain,
        input_files=phase.input_files,
    )
    ctx.record(phase, result, verdict)
    return {
        "phase": phase.name,
        "status": "success" if _is_acceptable(verdict) else "stopped",
        "cycles": 1,
        "verdict": verdict.get("verdict"),
        "run_status": verdict.get("run_status"),
    }


_MAX_DRYRUN_CYCLES = 5


def _resolve_skill_callable(skill: Optional[str], domain: Optional[str],
                            fn_name: str):
    """Return a named callable from the active skill's module, or None.

    Engine-neutral: imports the skill bundle by convention
    (``scilink.skills.{domain}.{skill}.{skill}``) and looks up ``fn_name``.
    No engine names appear here — an engine that does not provide the callable
    simply yields None.
    """
    if not (skill and domain):
        return None
    import importlib
    try:
        mod = importlib.import_module(f"scilink.skills.{domain}.{skill}.{skill}")
    except Exception:
        return None
    fn = getattr(mod, fn_name, None)
    return fn if callable(fn) else None


def _stage_dry_dir(run_dir: str, dry_dir: str, entry: str) -> None:
    """Recreate ``dry_dir`` with the run's dependency files linked from
    ``run_dir`` — everything except the deck itself and subdirs.

    Symlinks the dependencies rather than copying them: the dry run only reads
    the inputs (the deck is written fresh into ``dry_dir``), so re-copying large
    inputs — e.g. a serialized force-field system — every gate cycle is wasted
    I/O. Engine-neutral: it never inspects filenames, so it makes no assumption
    about which files an engine reads. Falls back to a copy if the filesystem
    refuses the link."""
    if os.path.isdir(dry_dir):
        shutil.rmtree(dry_dir)
    os.makedirs(dry_dir, exist_ok=True)
    for name in os.listdir(run_dir):
        src = os.path.abspath(os.path.join(run_dir, name))
        if name == entry or not os.path.isfile(src):
            continue
        dst = os.path.join(dry_dir, name)
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _dry_run_gate(
    phase: Phase,
    executor: Executor,
    run_critic: _RunCriticLike,
    ctx: RefinementContext,
    max_cycles: int = _MAX_DRYRUN_CYCLES,
) -> Optional[Dict[str, Any]]:
    """Cheap setup-validation pre-flight before the expensive production run.

    Runs a trimmed "dry-run" twin of the deck (engine setup only — see the
    skill's ``prepare_dry_run``) and, while it fails to even start, fixes the
    setup on the REAL deck until it does. This catches syntax/setup errors
    (style/coeff mismatches, command ordering, kspace-vs-triclinic) in seconds
    on the node, instead of one-per-expensive-run.

    Engine-neutral: the active skill supplies ``prepare_dry_run``; an engine
    without it (e.g. VASP, whose single SCF step is not cheap) skips the gate.
    **Fail-open** — any internal error returns without touching the deck, so the
    gate never blocks a run it could not validate.

    The twin runs only to PRODUCE the engine's authoritative error; the critic
    assesses and fixes against the REAL deck (placed in the judged directory), so
    ``suggested_fixes`` patch the full deck — preserving run lengths and fixing
    only setup — rather than the trimmed twin.

    Returns a record dict, or None when the gate is skipped.
    """
    prepare = _resolve_skill_callable(ctx.skill, ctx.domain, "prepare_dry_run")
    if prepare is None:
        return None
    entry = phase.entry_file
    if not entry or entry not in (phase.input_files or {}):
        return None
    try:
        return _run_dry_run_gate(phase, executor, run_critic, ctx, prepare,
                                 entry, max_cycles)
    except Exception as e:  # fail-open: never block a run we cannot validate
        logger.warning("Dry-run gate skipped (internal error): %s", e)
        return {"status": "skipped", "reason": f"gate error: {e}"}


def _run_dry_run_gate(phase, executor, run_critic, ctx, prepare, entry,
                      max_cycles) -> Dict[str, Any]:
    dry_dir = os.path.join(phase.run_dir, "_dryrun")
    history: List[Dict[str, Any]] = []
    for cycle in range(max_cycles):
        real_deck = phase.input_files[entry]
        twin = prepare(real_deck)
        # Stage deps (e.g. the data file) and run the trimmed twin to surface the
        # engine's setup error.
        _stage_dry_dir(phase.run_dir, dry_dir, entry)
        result = executor.run({entry: twin}, phase.run_command, dry_dir)
        out_dir = result.get("output_dir", dry_dir)
        # Put the REAL deck where the critic judges, so a fix patches the full
        # deck (run lengths preserved), not the run-0 twin.
        with open(os.path.join(out_dir, entry), "w") as fh:
            fh.write(real_deck)
        # Coverage is a static (goal, deck) check independent of setup success:
        # the trimmed twin never exercises observables, so a deck that starts
        # cleanly but omits a required output still fails the gate and is fixed.
        # The coverage decision is stochastic; ctx.coverage_votes>1 majority-votes
        # it to damp the sampling variance (single call when <=1).
        verdict = majority_coverage(
            run_critic, getattr(ctx, "coverage_votes", 1),
            output_dir=out_dir, research_goal=ctx.research_goal,
            skill=ctx.skill, domain=ctx.domain,
            input_files={entry: real_deck},
        )
        run_status = verdict.get("run_status")
        missing = verdict.get("missing_observables") or []
        history.append({"cycle": cycle, "run_status": run_status,
                        "missing_observables": missing,
                        "coverage_vote": verdict.get("coverage_vote")})
        if run_status == "succeeded" and not missing:
            return {"status": "passed", "cycles": cycle + 1, "history": history}
        deck_fix = _select_deck_fix(verdict.get("suggested_fixes"), entry, real_deck)
        if not deck_fix:
            return {"status": "unfixed", "cycles": cycle + 1, "history": history}
        phase.input_files[entry] = deck_fix
    return {"status": "exhausted", "cycles": max_cycles, "history": history}


def majority_coverage(run_critic, n_votes: int, **assess_kwargs):
    """Aggregate the observable-coverage BLOCKING decision over repeated calls.

    The coverage judgment is stochastic — a single call catches the clear cases
    reliably but waffles on borderline ones. This runs
    ``run_critic.assess(check_observables=True, **assess_kwargs)`` ``n_votes``
    times and blocks only on a strict majority, damping the sampling variance
    (it cannot correct a systematic bias — a wrongly-confident majority stays
    wrong). ``n_votes <= 1`` is a single pass-through (no behavior change).

    Returns one verdict dict: a representative blocking vote when the majority
    blocks (``missing_observables`` / ``suggested_fixes`` intact), else a
    non-blocking vote with ``missing_observables`` cleared. A ``coverage_vote``
    summary ``{n_votes, n_blocked, agreement, split}`` is added — ``split`` marks
    disagreement, the case a human-in-the-loop policy should escalate.
    """
    assess_kwargs.setdefault("check_observables", True)
    n = max(1, int(n_votes))
    verdicts = [run_critic.assess(**assess_kwargs) for _ in range(n)]
    blocked = [bool(v.get("missing_observables")) for v in verdicts]
    n_blocked = sum(blocked)
    if n_blocked * 2 > n:                       # strict majority blocks
        rep = dict(next(v for v, b in zip(verdicts, blocked) if b))
    else:
        rep = dict(next((v for v, b in zip(verdicts, blocked) if not b),
                        verdicts[0]))
        rep["missing_observables"] = []
    rep["coverage_vote"] = {
        "n_votes": n, "n_blocked": n_blocked,
        "agreement": max(n_blocked, n - n_blocked) / n,
        "split": 0 < n_blocked < n,
    }
    return rep


def _select_deck_fix(fixes, entry: str, real_deck: str):
    """Pick the corrected entry deck from a critic's ``suggested_fixes``.

    Prefers a fix keyed by the entry filename. As a backstop for a critic
    that keyed a full rewrite under a near-miss name (e.g. ``run.lammps`` ->
    ``run.lammps_header_patch``), accepts a mis-keyed fix only when it is a
    COMPLETE file — at least as long as the current deck. A short header or
    patch fragment is rejected (applying it would drop the run commands),
    leaving the gate to report ``unfixed``. Length is the only engine-neutral
    signal available at this layer.
    """
    if not isinstance(fixes, dict):
        return None
    direct = fixes.get(entry)
    if direct:
        return direct
    candidates = [
        v for k, v in fixes.items()
        if k != entry and isinstance(v, str) and len(v) >= len(real_deck)
    ]
    return max(candidates, key=len) if candidates else None


def run_campaign(
    stages: List[Stage],
    executor: Executor,
    run_critic: _RunCriticLike,
    policy: RefinementPolicy,
    ctx: RefinementContext,
    pre_run_verdict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive a staged MD campaign of sequential and parallel phases.

    Generalizes the single-chain refinement loop to the shapes a real MD study
    needs: sequential steps (optimization → equilibration → production, chained
    through restart files), a parallel fan-out (independent runs differing only
    in their inputs — a temperature sweep or umbrella-sampling windows), and an
    optional combine step that post-processes a fan-out's outputs. Each phase
    is still refined by the same run → assess → fix loop; fan-out members are
    refined independently, so one member's failure does not abort its siblings.

    Stages run in order; a stage that does not succeed stops the campaign
    (later stages depend on its output). Whether a fan-out's members actually
    run concurrently is the executor's concern — the campaign only expresses
    that they are independent. The pre-run gate runs once on the first phase's
    inputs, reusing ``pre_run_verdict`` so no second validation call is made.

    Args:
        stages: Ordered stages to execute.
        executor: How to run a phase's inputs.
        run_critic: Post-run critic exposing ``assess(...)``.
        policy: Autonomy policy gating changes and continuation.
        ctx: Shared refinement state (goal, routing, budget, history).
        pre_run_verdict: Optional input-validator report for the pre-run gate.

    Returns:
        A dict with ``status`` (``"success"``, ``"aborted"``, or ``"failed"``),
        ``stages`` (per-stage status + member records), ``phases`` (every phase
        record, flattened in execution order — back-compatible with the
        single-chain result), and the full ``history``.
    """
    runnable = [s for s in stages if s.phases]
    if not runnable:
        return {"status": "failed", "error": "no phases to run",
                "stages": [], "phases": [], "history": ctx.history}

    # ── Pre-run gate (1): accept the first phase's inputs before any run ──
    first_phase = runnable[0].phases[0]
    gated = policy.approve_change(first_phase.input_files, pre_run_verdict, ctx)
    if gated is None:
        return {"status": "aborted", "reason": "pre-run inputs rejected",
                "stages": [], "phases": [], "history": ctx.history}
    first_phase.input_files = gated

    # ── Pre-run gate (2): cheap engine dry-run — converge the deck to a clean
    #    setup before the expensive run. Engine-neutral (skipped when the active
    #    skill provides no prepare_dry_run) and fail-open. Validates the first
    #    phase's setup, which staged phases share.
    dry_run_record = _dry_run_gate(first_phase, executor, run_critic, ctx)

    flat: List[Dict[str, Any]] = []
    stage_records: List[Dict[str, Any]] = []
    overall = "success"

    for stage in runnable:
        if stage.kind == "combine":
            rec = _run_once_phase(stage.phases[0], executor, run_critic, ctx)
            flat.append(rec)
            status = "success" if rec["status"] == "success" else "failed"
            stage_records.append({
                "name": stage.name, "parallel": False, "kind": "combine",
                "status": status, "members": [rec],
            })

        elif not stage.parallel:
            members: List[Dict[str, Any]] = []
            status = "success"
            for ph in stage.phases:
                rec = _refine_phase(ph, executor, run_critic, policy, ctx)
                flat.append(rec)
                members.append(rec)
                if rec["status"] != "success":
                    status = ("aborted" if rec["status"] == "aborted"
                              else "failed")
                    break
            stage_records.append({
                "name": stage.name, "parallel": False, "kind": "run",
                "status": status, "members": members,
            })

        else:
            # Fan-out: every member is refined independently. A failing member
            # does not abort its siblings — collect them all, then judge the
            # stage against its success quorum.
            members = []
            n_success = 0
            any_aborted = False
            for ph in stage.phases:
                rec = _refine_phase(ph, executor, run_critic, policy, ctx)
                flat.append(rec)
                members.append(rec)
                if rec["status"] == "success":
                    n_success += 1
                elif rec["status"] == "aborted":
                    any_aborted = True
            required = (stage.min_success if stage.min_success is not None
                        else len(stage.phases))
            if n_success >= required:
                status = "success"
            elif any_aborted and n_success == 0:
                status = "aborted"
            else:
                status = "failed"
            stage_records.append({
                "name": stage.name, "parallel": True, "kind": "run",
                "status": status, "members": members,
                "n_success": n_success, "required": required,
            })

        if status != "success":
            overall = "aborted" if status == "aborted" else "failed"
            break

    result = {"status": overall, "stages": stage_records, "phases": flat,
              "history": ctx.history}
    # Surface a structure-caused failure so the caller can regenerate the
    # structure rather than re-editing a deck that cannot fix a broken pack.
    if overall != "success" and any(
        p.get("failure_class") == "structure" for p in flat
    ):
        result["failure_class"] = "structure"
    if dry_run_record is not None:
        result["dry_run"] = dry_run_record
    return result


def run_refinement(
    phases: List[Phase],
    executor: Executor,
    run_critic: _RunCriticLike,
    policy: RefinementPolicy,
    ctx: RefinementContext,
    pre_run_verdict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Drive a single sequential phase chain to convergence.

    A thin wrapper over :func:`run_campaign` for the common single-chain case
    (one phase for DFT, a restart-chained optimization → equilibration →
    production for MD). Equivalent to one sequential stage; the result keeps
    the same ``status`` / ``phases`` / ``history`` shape it has always had.

    Args:
        phases: Ordered phases to execute (one for DFT, several for staged MD).
        executor: How to run a phase's inputs.
        run_critic: Post-run critic exposing ``assess(...)``.
        policy: Autonomy policy gating changes and continuation.
        ctx: Shared refinement state (goal, routing, budget, history).
        pre_run_verdict: Optional input-validator report for the pre-run gate.

    Returns:
        A dict with ``status``, ``phases`` (per-phase final verdict + cycles),
        and the full ``history``.
    """
    stage = Stage(name="run", phases=list(phases), parallel=False)
    return run_campaign([stage], executor, run_critic, policy, ctx,
                        pre_run_verdict)
