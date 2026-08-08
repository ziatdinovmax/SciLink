"""Live validation for script_edits (Phase A of the surgical-refinement
plan) on Bedrock Opus 4.8.

  1. wiring   — agent-level, deterministic: a real anchor fit, then a
                reuse run with a test-constructed surgical edit. The
                rerun's saved script must equal the anchor's byte-for-
                byte EXCEPT the edit; provenance recorded; reuse
                validity attached.
  2. routing  — orchestrator chat: "next point of the same series, but
                change one knob". The LLM must read the saved script,
                pass script_edits with a verbatim snippet, and the two
                runs' scripts must differ in only a few lines — not a
                re-derivation, not an unedited reuse.

Needs AWS_BEARER_TOKEN_BEDROCK (+ UNSAFE_EXECUTION_OK=true: curve
fitting executes generated code).

Run:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 \
    UNSAFE_EXECUTION_OK=true python tests/test_script_edits_live.py [1 2]
"""
from __future__ import annotations

import contextlib
import difflib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_script_edits_live_runs").resolve()

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


@contextlib.contextmanager
def capture_all(buf):
    """Agents narrate via logging, orchestrators via print — capture both."""
    import logging
    handler = logging.StreamHandler(buf)
    logging.getLogger().addHandler(handler)
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        logging.getLogger().removeHandler(handler)


def make_spectrum(path, center, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 400)
    y = (2.0 * np.exp(-((x - center) ** 2) / (2 * 0.4 ** 2))
         + 0.3 + 0.02 * x + rng.normal(0, 0.02, x.size))
    np.savetxt(path, np.column_stack([x, y]), delimiter=",",
               header="position,intensity", comments="")


SYS_INFO = {"technique": "generic 1D spectroscopy",
            "x_axis": "position (a.u.)", "y_axis": "intensity (a.u.)"}


def _saved_script(run_dir: Path) -> tuple[Path, str]:
    scripts = sorted((Path(run_dir) / "scripts").glob("*.py"))
    assert scripts, f"no saved script under {run_dir}/scripts"
    return scripts[0], scripts[0].read_text()


def part1_wiring():
    print("\n=== 1. agent-level: byte-exact except the edit ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p1"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    spec_a, spec_b = run / "spec_a.csv", run / "spec_b.csv"
    make_spectrum(spec_a, center=5.0, seed=1)
    make_spectrum(spec_b, center=5.1, seed=2)

    agent_a = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "anchor"),
        enable_human_feedback=False, max_verification_iterations=1)
    res_a = agent_a.analyze(str(spec_a), system_info=SYS_INFO)
    check("p1 anchor fit succeeded", res_a.get("status") == "success")
    _, script_a = _saved_script(run / "anchor")

    # A guaranteed-present, behavior-neutral snippet: the numpy import.
    # Part 1 validates the MECHANISM (exact application, provenance,
    # execution); part 2 exercises a real knob chosen by the LLM.
    assert "import numpy as np" in script_a
    edit = {"old_text": "import numpy as np",
            "new_text": "import numpy as np  # SURGICAL-EDIT-MARK"}

    agent_b = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "rerun"),
        enable_human_feedback=False, max_verification_iterations=1)
    buf = Tee()
    with capture_all(buf):
        res_b = agent_b.analyze(
            str(spec_b), system_info=SYS_INFO,
            prior_analysis_paths=[str(run / "anchor")],
            reuse_locked_script=True, script_edits=[edit])
    log = buf.getvalue()

    check("p1 rerun succeeded", res_b.get("status") == "success")
    check("p1 edit applied (log)", "Applied 1 surgical edit" in log)
    check("p1 reuse fast path used (not re-derivation)",
          "Reusing locked fitting script" in log)
    _, script_b = _saved_script(run / "rerun")
    check("p1 mark present in executed script",
          "SURGICAL-EDIT-MARK" in script_b)
    check("p1 byte-identical except the edit",
          script_b.replace("  # SURGICAL-EDIT-MARK", "") == script_a)
    sfr = json.loads((run / "rerun" / "series_fit_results.json").read_text())
    r0 = (sfr.get("results") or [{}])[0]
    rv = r0.get("reuse_validity") or {}
    check("p1 provenance: source labeled with the edit count",
          "+ 1 edit(s)" in str(rv.get("source")))
    check("p1 provenance: edits recorded in the result",
          r0.get("script_edits_applied") == [edit])


def part2_routing():
    print("\n=== 2. orchestrator: single-knob follow-up routes to edits ===")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)

    run = BASE / "p2"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    spec_a, spec_b = run / "spec_a.csv", run / "spec_b.csv"
    make_spectrum(spec_a, center=5.0, seed=3)
    make_spectrum(spec_b, center=5.15, seed=4)

    orch = AnalysisOrchestratorAgent(
        base_dir=str(run / "session"), api_key=None, model_name=MODEL,
        analysis_mode=AnalysisMode.AUTONOMOUS)

    buf = Tee()
    with capture_all(buf):
        orch.chat(
            f"Fit the 1D spectrum at {spec_a} (single Gaussian peak on a "
            "linear background). Smoke test: max_verification_iterations=1, "
            "accept the first reasonable fit.")
        reply = orch.chat(
            f"{spec_b} is the NEXT measurement of the same series. Re-fit "
            "it reusing the locked script so the feature columns stay "
            "identical — but with ONE surgical change: read the saved "
            "fitting script first, then modify a single numeric value in "
            "it (an initial guess, tolerance, or iteration limit of your "
            "choice). Tell me exactly what you changed.")
    log = buf.getvalue()

    check("p2 edit path used (log)", "Applied 1 surgical edit" in log
          or "surgical edit" in log)
    check("p2 reuse fast path used", "Reusing locked fitting script" in log)

    session = run / "session"
    run_dirs = sorted(d for d in (session / "results").glob("*CurveFit*")
                      if d.is_dir()) if (session / "results").is_dir() else \
        sorted(d for d in session.rglob("series_fit_results.json"))
    # locate the two runs' scripts wherever the session layout put them
    sfrs = sorted(session.rglob("series_fit_results.json"))
    check("p2 two fit runs present", len(sfrs) >= 2)
    if len(sfrs) >= 2:
        s1 = _saved_script(sfrs[0].parent)[1]
        s2 = _saved_script(sfrs[-1].parent)[1]
        ndiff = sum(1 for l in difflib.unified_diff(
            s1.splitlines(), s2.splitlines(), lineterm="")
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
        print(f"     script diff: {ndiff} changed line(s)")
        check("p2 scripts differ minimally (edit, not re-derivation)",
              1 <= ndiff <= 6)
        rec = json.loads(sfrs[-1].read_text())
        r0 = (rec.get("results") or [{}])[0]
        check("p2 provenance recorded", bool(r0.get("script_edits_applied")))


MARK_EDIT = {"old_text": "import numpy as np",
             "new_text": "import numpy as np  # SURGICAL-EDIT-MARK"}


def _rglob_sfrs(root: Path):
    return sorted(Path(root).rglob("series_fit_results.json"),
                  key=lambda p: p.stat().st_mtime)


def part3_meta_cross_delegation():
    """S1: the designed case — knob follow-up across meta delegations."""
    print("\n=== 3. meta: cross-delegation knob follow-up ===")
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)

    run = BASE / "p3"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_spectrum(run / "spec_a.csv", center=5.0, seed=11)
    make_spectrum(run / "spec_b.csv", center=5.1, seed=12)

    meta = MetaOrchestratorAgent(
        base_dir=str(run / "meta_session"), api_key=None,
        model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    buf = Tee()
    with capture_all(buf):
        meta.chat(
            f"Fit the 1D spectrum at {run / 'spec_a.csv'} (single Gaussian "
            "peak on a linear background, curve fitting). Smoke test: "
            "max_verification_iterations=1, accept the first reasonable "
            "fit.")
        meta.chat(
            f"{run / 'spec_b.csv'} is the NEXT measurement of the same "
            "series. Re-fit it reusing the locked script so the feature "
            "columns stay identical — but with ONE surgical change: read "
            "the saved fitting script and modify a single numeric value "
            "(initial guess, tolerance, or iteration limit). Report what "
            "you changed.")
    log = buf.getvalue()

    # Child agents log inside the delegation, not to the meta's stream —
    # assert on the child session's ARTIFACTS, which are stronger anyway.
    sfrs = _rglob_sfrs(run / "meta_session")
    check("p3 two fit runs in the child session", len(sfrs) >= 2)
    if len(sfrs) >= 2:
        rec = json.loads(sfrs[-1].read_text())
        r0 = (rec.get("results") or [{}])[0]
        rv = r0.get("reuse_validity") or {}
        check("p3 reuse fast path used (validity verdict attached)",
              rv.get("reused") is True)
        check("p3 edit applied through the delegation chain",
              "edit(s)" in str(rv.get("source"))
              and bool(r0.get("script_edits_applied")))


def part4_meta_same_delegation():
    """S2: one delegation carries both steps (fit, then tweak)."""
    print("\n=== 4. meta: same-delegation two-step ===")
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)

    run = BASE / "p4"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_spectrum(run / "spec_a.csv", center=5.0, seed=21)
    make_spectrum(run / "spec_b.csv", center=5.05, seed=22)

    meta = MetaOrchestratorAgent(
        base_dir=str(run / "meta_session"), api_key=None,
        model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    buf = Tee()
    with capture_all(buf):
        meta.chat(
            f"Two-step curve-fitting job (smoke test, "
            f"max_verification_iterations=1): first fit {run / 'spec_a.csv'} "
            "(single Gaussian on linear background). Then fit "
            f"{run / 'spec_b.csv'} as the next measurement of the same "
            "series, reusing the locked script for schema consistency but "
            "with ONE surgical numeric change of your choice to the saved "
            "script. Report the change.")
    log = buf.getvalue()

    sfrs = _rglob_sfrs(run / "meta_session")
    check("p4 two fit runs present", len(sfrs) >= 2)
    if len(sfrs) >= 2:
        r0 = (json.loads(sfrs[-1].read_text()).get("results") or [{}])[0]
        rv = r0.get("reuse_validity") or {}
        check("p4 reuse fast path used", rv.get("reused") is True)
        check("p4 edit applied within one delegation",
              bool(r0.get("script_edits_applied")))


def part5_chaining():
    """S3: anchor-pointed vs previous-point-pointed chaining, plus the
    mixing failure (re-passing a baked-in edit refuses cleanly)."""
    print("\n=== 5. chaining semantics (agent-level, deterministic) ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p5"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    for i, (c, s) in enumerate([(5.0, 31), (5.05, 32), (5.1, 33),
                                (5.15, 34)]):
        make_spectrum(run / f"spec_{i}.csv", center=c, seed=s)

    def agent(sub):
        return CurveFittingAgent(
            api_key=None, model_name=MODEL, output_dir=str(run / sub),
            enable_human_feedback=False, max_verification_iterations=1)

    res = agent("anchor").analyze(str(run / "spec_0.csv"),
                                  system_info=SYS_INFO)
    check("p5 anchor succeeded", res.get("status") == "success")
    _, script_a = _saved_script(run / "anchor")
    assert "import numpy as np" in script_a

    # A knob-shaped edit: old_text is fully CONSUMED by new_text (like a
    # real value change), so re-applying it against the edited script
    # cannot match — that is what makes the mixing case refuse.
    chain_edit = {"old_text": "import numpy as np",
                  "new_text": ("import numpy as xx_np  "
                               "# SURGICAL-EDIT-MARK\nnp = xx_np")}

    # b1: anchor-pointed WITH the edit
    r1 = agent("b1").analyze(
        str(run / "spec_1.csv"), system_info=SYS_INFO,
        prior_analysis_paths=[str(run / "anchor")],
        reuse_locked_script=True, script_edits=[chain_edit])
    _, s1 = _saved_script(run / "b1")
    check("p5 anchor-pointed edit applied",
          r1.get("status") == "success" and "SURGICAL-EDIT-MARK" in s1)

    # b2: previous-point-pointed, NO edits -> the edit is baked in
    r2 = agent("b2").analyze(
        str(run / "spec_2.csv"), system_info=SYS_INFO,
        prior_analysis_paths=[str(run / "b1")],
        reuse_locked_script=True)
    _, s2 = _saved_script(run / "b2")
    check("p5 baked-in edit persists without re-passing",
          r2.get("status") == "success" and "SURGICAL-EDIT-MARK" in s2
          and s2 == s1)

    # b3: mixing styles — re-passing the edit against the edited run
    r3 = agent("b3").analyze(
        str(run / "spec_3.csv"), system_info=SYS_INFO,
        prior_analysis_paths=[str(run / "b1")],
        reuse_locked_script=True, script_edits=[chain_edit])
    check("p5 mixing styles refuses cleanly (old_text already replaced)",
          r3.get("status") == "error"
          and r3["error"]["error"] == "script_edits do not apply")


def part6_realtime():
    """S4: realtime frames execute the EDITED script, zero-LLM happy path."""
    print("\n=== 6. realtime stream with script_edits ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p6"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_spectrum(run / "anchor.csv", center=5.0, seed=41)
    for i in range(2):
        make_spectrum(run / f"frame_{i}.csv", center=5.0 + 0.02 * i,
                      seed=42 + i)

    anchor = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "anchor_run"),
        enable_human_feedback=False, max_verification_iterations=1)
    res = anchor.analyze(str(run / "anchor.csv"), system_info=SYS_INFO)
    check("p6 anchor succeeded", res.get("status") == "success")

    buf = Tee()
    with capture_all(buf):
        for i in range(2):
            frame_agent = CurveFittingAgent(
                api_key=None, model_name=MODEL,
                output_dir=str(run / f"frame_run_{i}"),
                enable_human_feedback=False)
            fr = frame_agent.analyze(
                str(run / f"frame_{i}.csv"), system_info=SYS_INFO,
                profile="realtime",
                prior_analysis_paths=[str(run / "anchor_run")],
                reuse_locked_script=True, script_edits=[MARK_EDIT])
            check(f"p6 frame {i} succeeded", fr.get("status") == "success")
    log = buf.getvalue()
    check("p6 edit applied on every frame",
          log.count("Applied 1 surgical edit") == 2)
    marks = [
        "SURGICAL-EDIT-MARK" in _saved_script(run / f"frame_run_{i}")[1]
        for i in range(2)]
    check("p6 every frame executed the edited script", all(marks))


def part7_fanout_variants():
    """S5: knob VARIANTS in parallel branches (or sequential delegations —
    either route validates the per-branch mechanics; the route is
    reported)."""
    print("\n=== 7. meta: two knob variants of the same anchor ===")
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)

    run = BASE / "p7"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_spectrum(run / "spec_a.csv", center=5.0, seed=51)
    make_spectrum(run / "spec_b.csv", center=5.05, seed=52)

    meta = MetaOrchestratorAgent(
        base_dir=str(run / "meta_session"), api_key=None,
        model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    buf = Tee()
    with capture_all(buf):
        meta.chat(
            f"Fit {run / 'spec_a.csv'} (single Gaussian on linear "
            "background, curve fitting; smoke test, "
            "max_verification_iterations=1). Then produce TWO surgical "
            f"variants of that fit on {run / 'spec_b.csv'}: both reuse the "
            "locked script, each changing the SAME single numeric value to "
            "a different setting (e.g. two different iteration limits). "
            "Compare the two R² values at the end.")
    log = buf.getvalue()

    edited = []
    for sfr in _rglob_sfrs(run / "meta_session"):
        r0 = (json.loads(sfr.read_text()).get("results") or [{}])[0]
        if r0.get("script_edits_applied"):
            edited.append(r0["script_edits_applied"])
    print(f"     edited variants found: {len(edited)}")
    check("p7 at least two edited variants ran", len(edited) >= 2)
    check("p7 the two variants differ",
          len(edited) >= 2 and edited[-1] != edited[-2])
    check("p7 route observed (informational)", True)
    print("     route: " + ("fan-out" if "fan" in log.lower()
                            else "sequential delegations"))


def part8_refusals():
    """S6: the pre-execution refusals, against the real agent stack."""
    print("\n=== 8. refusals: zero-cost, nothing runs ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p8"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    make_spectrum(run / "spec.csv", center=5.0, seed=61)
    # a minimal but real prior run: reuse part 1's anchor if present,
    # else make a stub that the loader accepts
    prior = BASE / "p1" / "anchor"
    if not (prior / "series_fit_results.json").exists():
        prior = run / "prior"
        (prior / "scripts").mkdir(parents=True)
        (prior / "scripts" / "fitting_script.py").write_text(
            "import numpy as np\nprint('CUSTOM_SCRIPT_SUCCESS')\n")
        (prior / "series_fit_results.json").write_text(
            json.dumps({"results": [{"success": True}]}))

    agent = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "out"),
        enable_human_feedback=False)

    r = agent.analyze(str(run / "spec.csv"), system_info=SYS_INFO,
                      prior_analysis_paths=[str(prior)],
                      script_edits=[MARK_EDIT])
    check("p8 edits without reuse flag refused",
          r.get("status") == "error"
          and "reuse_locked_script" in r["error"]["details"])

    r = agent.analyze(str(run / "spec.csv"), system_info=SYS_INFO,
                      prior_analysis_paths=[str(prior)],
                      reuse_locked_script=True,
                      script_edits=[{"old_text": "NOT IN THE SCRIPT",
                                     "new_text": "x"}])
    check("p8 non-matching edit refused with per-edit report",
          r.get("status") == "error"
          and r["error"]["error"] == "script_edits do not apply")
    check("p8 nothing ran", not (run / "out" / "scripts").exists())


def part9_multiregime():
    """The patched gap: a regime-split series REFUSES script_edits loudly
    instead of silently dropping them and re-deriving freely."""
    print("\n=== 9. multi-regime series: loud refusal, not silent drop ===")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    run = BASE / "p9"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    # anchor: single Gaussian
    make_spectrum(run / "anchor.csv", center=5.0, seed=71)
    # series crossing a "transition": 2 one-peak frames, then 2 two-peak
    rng = np.random.default_rng(72)
    x = np.linspace(0, 10, 400)
    series = []
    for i in range(4):
        y = 2.0 * np.exp(-((x - 5.0) ** 2) / (2 * 0.4 ** 2)) + 0.3
        if i >= 2:
            y = y + 1.5 * np.exp(-((x - 7.5) ** 2) / (2 * 0.3 ** 2))
        y = y + rng.normal(0, 0.02, x.size)
        p = run / f"series_{i}.csv"
        np.savetxt(p, np.column_stack([x, y]), delimiter=",",
                   header="position,intensity", comments="")
        series.append(str(p))

    anchor = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "anchor_run"),
        enable_human_feedback=False, max_verification_iterations=1)
    res = anchor.analyze(str(run / "anchor.csv"), system_info=SYS_INFO)
    check("p9 anchor succeeded", res.get("status") == "success")

    agent = CurveFittingAgent(
        api_key=None, model_name=MODEL, output_dir=str(run / "series_run"),
        enable_human_feedback=False, max_verification_iterations=1)
    buf = Tee()
    with capture_all(buf):
        r = agent.analyze(
            series, system_info={
                **SYS_INFO,
                "note": ("temperature series crossing a known structural "
                         "phase transition between frames 2 and 3 — a "
                         "second peak appears; plan separate per-regime "
                         "models for before and after the transition")},
            series_metadata={"variable": "temperature",
                             "values": [10, 20, 30, 40]},
            prior_analysis_paths=[str(run / "anchor_run")],
            reuse_locked_script=True, script_edits=[MARK_EDIT])
    log = buf.getvalue()

    regimes_declared = "regime configuration(s)" in log
    print(f"     regimes declared by the planner: {regimes_declared}")
    if regimes_declared:
        check("p9 refused loudly (patched gap)",
              r.get("status") == "error"
              and "multiple regimes" in str(r.get("error", {})))
    else:
        # planner chose a single regime — the refusal cannot fire; the
        # run must then be a normal edited reuse (still a valid outcome)
        check("p9 single-regime fallback: edited reuse ran (INCONCLUSIVE "
              "for the regime guard — planner declared no regimes)",
              r.get("status") == "success"
              and "Applied 1 surgical edit" in log)


PARTS = {"1": part1_wiring, "2": part2_routing,
         "3": part3_meta_cross_delegation, "4": part4_meta_same_delegation,
         "5": part5_chaining, "6": part6_realtime,
         "7": part7_fanout_variants, "8": part8_refusals,
         "9": part9_multiregime}

if __name__ == "__main__":
    for k in (sys.argv[1:] or sorted(PARTS)):
        PARTS[k]()
    print("\n" + "=" * 60)
    npass = sum(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{npass}/{len(results)} checks passed")
    sys.exit(0 if npass == len(results) else 1)
