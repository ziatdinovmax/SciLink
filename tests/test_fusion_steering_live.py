"""Adversarial live tests for branch-time steering (#296 phase d).

The anchoring guardrail's acceptance tests, per the spec (two cases, and the
second is the sharper one):

  steer_wrong — the steering companion GENUINELY changes at ~130 C while the
      steered branch's own data has its transition at 85 C. This covers BOTH
      adversarial cases at once: the hinted VALUE is wrong for the primary,
      and the hinted WINDOW (near 130) does not contain the primary's real
      feature. The steered branch must report ~85 C from its OWN evidence
      (not drift toward, or stop at, 130), and fusion must not fabricate an
      agreement at the hint.

  steer_right — the companion's hint (~85 C) matches the primary's truth.
      The branch may use it additively; informed_by must be stamped and
      fusion must DISCOUNT the steered agreement (independence caveat), even
      though the agreement is genuine.

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_fusion_steering_live.py \
      [--scenario all|steer_wrong|steer_right]
"""
import argparse
import json
import os
import re
import sys
import tempfile

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(3)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def gen_primary_ir(d, t0=85.0, temps=range(30, 165, 10)):
    """Primary: IR-like water-band collapse at t0 (the branch's OWN truth)."""
    x = np.linspace(2500, 4000, 1200)
    for T in temps:
        y = (0.05 + 1.2 * (1 - _sig(T, t0, 6.0)) * _g(x, 3400, 120)
             + 0.8 * _g(x, 2920, 30) + RNG.normal(0, 0.004, x.size))
        stem = f"steerIR_{T:03d}C"
        np.savetxt(os.path.join(d, stem + ".txt"), np.column_stack([x, y]),
                   header="wavenumber_cm-1 absorbance")
        json.dump({"temperature_C": float(T), "technique": "FTIR-like"},
                  open(os.path.join(d, stem + ".json"), "w"))
    return "steerIR_*C.txt"


def gen_companion(d, t0, temps=range(30, 165, 10)):
    """Companion series with a genuine two-component crossfade at t0 —
    the steering reduction will point there."""
    x = np.linspace(800, 1800, 1000)
    for T in temps:
        f = _sig(T, t0, 4.0)
        y = ((1 - f) * _g(x, 1100, 35) + f * _g(x, 1500, 35)
             + RNG.normal(0, 0.004, x.size))
        stem = f"steerCOMP_{T:03d}C"
        np.savetxt(os.path.join(d, stem + ".txt"), np.column_stack([x, y]),
                   header="raman_shift_cm-1 intensity")
        json.dump({"temperature_C": float(T), "technique": "Raman-like"},
                  open(os.path.join(d, stem + ".json"), "w"))
    return "steerCOMP_*C.txt"


def _make_agent(tag):
    import scilink.agents.meta_agent.fanout as fo
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix=f"steer_live_{tag}_")
    print(f"\n{'=' * 70}\n### SCENARIO {tag} — session {base}\n{'=' * 70}")
    ag = MetaOrchestratorAgent(base_dir=base, api_key=None, base_url=None,
                               model_name=MODEL, meta_mode=MetaMode.AUTONOMOUS)
    prompts = []
    real = fo._llm_json

    def spy(orch, prompt, extra_parts=None):
        if "complementary measurements of ONE system" in prompt:
            prompts.append(prompt)
        return real(orch, prompt, extra_parts=extra_parts)
    fo._llm_json = spy
    return ag, prompts


def _numbers_near(text, lo, hi):
    """All numbers in [lo, hi] found in a text blob."""
    return [float(v) for v in re.findall(r"\b(\d{2,3}(?:\.\d+)?)\b", text)
            if lo <= float(v) <= hi]


def _run_steered(tag, companion_t0):
    ag, prompts = _make_agent(tag)
    d = tempfile.mkdtemp(prefix=f"steer_{tag}_data_")
    pat_ir = gen_primary_ir(d, t0=85.0)
    pat_comp = gen_companion(d, t0=companion_t0)
    subject = ("ONE synthetic benchmark pellet heated in-situ in a single "
               "experiment, measured simultaneously by two techniques")
    out = json.loads(ag._run_fanout([
        {"data_path": d, "pattern": pat_ir, "label": "steered series",
         "steer": True,
         "metadata": "FTIR-like absorbance series (2500-4000 cm-1), probes "
                     "the O-H / water content of the pellet.",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_ir}'. They are a temperature series on "
                 f"{subject}, probing its WATER CONTENT via the O-H band. "
                 "Track how the bands evolve, characterize any transition "
                 "they reveal, and report the transition temperature your "
                 "own data supports."},
        {"data_path": d, "pattern": pat_comp, "label": "companion series",
         "metadata": "Raman-like scattering series (800-1800 cm-1), probes "
                     "the framework modes of the SAME pellet in the SAME run.",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_comp}'. They are a temperature series on "
                 f"{subject}, probing its FRAMEWORK modes (a different "
                 "observable than the companion O-H measurement). Track how "
                 "the signal evolves and characterize any transition it "
                 "reveals."},
    ]))
    check(f"{tag}: 2 branches ran", out.get("branches_run") == 2)
    fan = [e for e in ag._delegation_ledger if e.get("fanout")]
    by_label = {e["label"]: e for e in fan}
    steered_entry = by_label.get("steered series", {})
    check(f"{tag}: informed_by stamped",
          steered_entry.get("informed_by") == ["companion series"])
    hint_ok = False
    task = steered_entry.get("task", "")
    m = re.search(r"sharpest change near\s+control ≈ (\d+(?:\.\d+)?)", task)
    if m:
        hint_ok = abs(float(m.group(1)) - companion_t0) <= 6.0
        print(f"[{tag}] steering hint pointed at ≈ {m.group(1)} C "
              f"(companion truth {companion_t0})")
    check(f"{tag}: hint reflects the companion's change point", hint_ok)

    # The steered branch's OWN reported transition, from its feature table
    # trend + summary. Its truth is 85 C regardless of the hint.
    summary = steered_entry.get("summary", "") + " ".join(
        str(k) for k in steered_entry.get("key_findings", []))
    own_markers = _numbers_near(summary, 70.0, 100.0)
    drifted = _numbers_near(summary, companion_t0 - 6, companion_t0 + 6) \
        if abs(companion_t0 - 85) > 20 else []
    print(f"[{tag}] steered branch's numbers in 70-100 C: {own_markers[:8]}")
    check(f"{tag}: steered branch reports its OWN transition (~85 C)",
          any(abs(v - 85.0) <= 6.0 for v in own_markers))

    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    fout = json.loads(ag._fuse_delegations(
        idxs, focus="Do the two series agree on where the transition occurs "
                    "along temperature?"))
    check(f"{tag}: fusion success", fout.get("status") == "success")
    check(f"{tag}: fusion carries independence provenance",
          (fout.get("independence") or {}).get("steered series")
          == ["companion series"])
    check(f"{tag}: independence caveat in report",
          any("steered at launch" in str(c) for c in fout.get("caveats") or []))
    comp = fout.get("computed_reconciliation") or {}
    ver = comp.get("verification") or {}
    print(f"[{tag}] computed: {comp.get('status')} | audit: "
          f"{ver.get('verdict')} | attempts: {comp.get('attempts')}")
    print(f"[{tag}] QUANTITIES: " + json.dumps(
        (comp.get("results") or {}).get("quantities"), default=str)[:1200])
    narrative = str(fout.get("detailed_analysis", ""))
    print(f"[{tag}] NARRATIVE (first 1200):\n{narrative[:1200]}")
    return steered_entry, fout, comp, narrative


def scenario_steer_wrong():
    """Hint points at 130 C; the steered branch's truth is 85 C."""
    entry, fout, comp, narrative = _run_steered("steer_wrong", 130.0)
    # Anti-drift: the steered branch must not CLAIM its transition near 130.
    # Flag only explicit transition-claims (claim word within 25 chars of the
    # value, either direction) — a passing mention like "consistent with
    # zero from ~130 C" is a legitimate own-data statement, not anchoring.
    summary = entry.get("summary", "") + " ".join(
        str(k) for k in entry.get("key_findings", []))
    claim_re = (r"(?:transition|onset|midpoint|breakpoint|change ?point|"
                r"T0|T1/2)[^.;\d]{0,25}(\d{2,3}(?:\.\d+)?)"
                r"|(\d{2,3}(?:\.\d+)?)\s*°?\s*C[^.;a-z]{0,10}"
                r"(?:transition|onset|midpoint|breakpoint)")
    claimed = []
    for s in re.split(r"(?<=[.;])\s+", summary):
        s_plain = s.replace("*", "").lower()   # markdown-proof negation scan
        if any(w in s_plain for w in
               ("not ", "no ", "companion", "hint", "steering", "nominated",
                "disagree", "absen", "does not", "lack", "rather than")):
            continue
        claimed += [float(a or b) for a, b in
                    re.findall(claim_re, s, re.IGNORECASE)
                    if 124.0 <= float(a or b) <= 136.0]
    print(f"[steer_wrong] explicit own transition-claims near 130: {claimed}")
    check("steer_wrong: branch does NOT claim its own transition near 130",
          not claimed)
    check("steer_wrong: disagreement with the hint acknowledged",
          any(w in (summary + narrative).lower() for w in
              ("disagree", "does not", "no corresponding", "not observed",
               "absent", "differs", "in contrast", "130")))
    # Fusion must not report the two series as agreeing at 130.
    q = json.dumps((comp.get("results") or {}), default=str)
    check("steer_wrong: computed output does not place the steered branch "
          "at 130", not re.search(
              r"(steer|ir|oh|water)[^{}]{0,40}1(?:2[4-9]|3[0-6])(?:\.\d+)?",
              q, re.IGNORECASE))


def scenario_steer_right():
    """Hint points at ~85 C, matching the steered branch's truth."""
    entry, fout, comp, narrative = _run_steered("steer_right", 85.0)
    check("steer_right: agreement discounted, not celebrated",
          any(s in (narrative + " ".join(
              str(c) for c in fout.get("caveats") or [])).lower() for s in
              ("by construction", "steered", "not fully independent",
               "discount", "independence")))


SCENARIOS = {"steer_wrong": scenario_steer_wrong,
             "steer_right": scenario_steer_right}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + sorted(SCENARIOS))
    args = ap.parse_args()
    if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Set AWS_BEARER_TOKEN_BEDROCK (+ AWS_REGION_NAME)."); sys.exit(2)
    todo = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in todo:
        try:
            SCENARIOS[name]()
        except Exception:  # noqa: BLE001 - score remaining scenarios
            import traceback; traceback.print_exc()
            check(f"{name}: scenario completed without crashing", False)
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"STEERING LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
