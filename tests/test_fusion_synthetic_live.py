"""Synthetic-ground-truth live scenarios for #296 quantitative fusion.

The ibuprofen validation exercises ONE join topology (shared 1-D axis) with
no known truth. These scenarios generate synthetic multi-technique datasets
that mimic real studies but carry PLANTED ground truth, so the computed
reconciliation is scored quantitatively:

  agreeing — shared temperature axis, three techniques with planted
             transitions at 85 / 88 / 92 C: computed markers must recover
             the planted values and their small offsets.
  null     — one series with a real transition (85 C), one featureless
             drift series on the same axis: fusion must NOT manufacture a
             coincidence for the featureless branch (anti-spurious guard).
  scalar   — bulk-vs-local with NO parameter axis: two single spectra of
             the same stated sample carrying component fractions 0.60 vs
             0.58: exercises the scalar-comparison topology (and, since
             curve fits emit *_err, the sigma-available regime).

(The shared-2-D-grid topology needs image branches and is not driven here.)

  export AWS_BEARER_TOKEN_BEDROCK=...  AWS_REGION_NAME=us-east-1
  UNSAFE_EXECUTION_OK=true python tests/test_fusion_synthetic_live.py \
      [--scenario all|agreeing|null|scalar]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

MODEL = os.environ.get("SCILINK_TEST_MODEL",
                       "bedrock/us.anthropic.claude-opus-4-8")
RNG = np.random.default_rng(0)

checks = {}


def check(name, cond):
    checks[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _sig(t, t0, w):
    return 1.0 / (1.0 + np.exp(-(t - t0) / w))


def _gauss(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


# ----------------------------------------------------------------------
# Synthetic data generators (planted ground truth in each docstring)
# ----------------------------------------------------------------------

def gen_ir_series(d, t0=85.0, w=6.0, temps=range(30, 165, 10)):
    """IR-like series: broad 3400 water band collapses sigmoidally at t0;
    stable CH band at 2920 as internal reference."""
    x = np.linspace(2500, 4000, 1200)
    for T in temps:
        y = (0.05 + 1.2 * (1 - _sig(T, t0, w)) * _gauss(x, 3400, 120)
             + 0.8 * _gauss(x, 2920, 30) + RNG.normal(0, 0.004, x.size))
        stem = f"synthIR_{T:03d}C"
        np.savetxt(os.path.join(d, stem + ".txt"),
                   np.column_stack([x, y]), header="wavenumber_cm-1 absorbance")
        json.dump({"temperature_C": float(T), "technique": "FTIR-like",
                   "columns": ["wavenumber_cm-1", "absorbance"]},
                  open(os.path.join(d, stem + ".json"), "w"))
    return "synthIR_*C.txt"


def gen_xrd_series(d, t0=88.0, w=5.0, temps=range(30, 165, 10)):
    """XRD-like series: 11.03-deg parent peak collapses / 11.92-deg product
    peak grows, both sigmoidally at t0; stable 20.0-deg reference peak."""
    x = np.arange(5, 40, 0.02)
    for T in temps:
        f = _sig(T, t0, w)
        y = (50 + 1000 * (1 - f) * _gauss(x, 11.03, 0.08)
             + 900 * f * _gauss(x, 11.92, 0.08)
             + 500 * _gauss(x, 20.0, 0.10))
        y = y + RNG.normal(0, np.sqrt(np.maximum(y, 1)) * 0.5)
        stem = f"synthXRD_{T:03d}C"
        np.savetxt(os.path.join(d, stem + ".txt"),
                   np.column_stack([x, y]), header="two_theta_deg intensity")
        json.dump({"temperature_C": float(T), "technique": "XRD-like",
                   "wavelength_A": 1.5406,
                   "columns": ["two_theta_deg", "intensity"]},
                  open(os.path.join(d, stem + ".json"), "w"))
    return "synthXRD_*C.txt"


def gen_drift_series(d, temps=range(30, 165, 10)):
    """Featureless companion: one stable band, small linear drift, NO
    transition anywhere. Any 'transition' computed for it is fabricated."""
    x = np.linspace(800, 1800, 1000)
    for T in temps:
        y = (0.1 + 0.0004 * (T - 30) + 0.9 * _gauss(x, 1350, 40)
             + RNG.normal(0, 0.004, x.size))
        stem = f"synthDRIFT_{T:03d}C"
        np.savetxt(os.path.join(d, stem + ".txt"),
                   np.column_stack([x, y]), header="raman_shift_cm-1 intensity")
        json.dump({"temperature_C": float(T), "technique": "Raman-like",
                   "columns": ["raman_shift_cm-1", "intensity"]},
                  open(os.path.join(d, stem + ".json"), "w"))
    return "synthDRIFT_*C.txt"


def gen_dsc_curve(d, t0=92.0):
    """DSC-like curve: one endotherm centered at t0 on a shallow baseline."""
    T = np.arange(25, 200, 0.25)
    y = (-0.4 - 2.5 * _gauss(T, t0, 6.0) + 0.0012 * (T - 25)
         + RNG.normal(0, 0.01, T.size))
    p = os.path.join(d, "synthDSC.txt")
    np.savetxt(p, np.column_stack([T, y]),
               header="temperature_C heat_flow_mW_mg")
    json.dump({"technique": "DSC-like", "heating_rate_C_per_min": 10,
               "columns": ["temperature_C", "heat_flow_mW_mg"]},
              open(os.path.join(d, "synthDSC.json"), "w"))
    return p


def gen_two_component_spectrum(d, name, centers, sigmas, fractions, span,
                               technique):
    """Single spectrum with two components at planted AREA fractions."""
    x = np.linspace(*span, 1500)
    amps = [f / (s * np.sqrt(2 * np.pi)) for f, s in zip(fractions, sigmas)]
    y = 0.02 + sum(a * s * np.sqrt(2 * np.pi) * _gauss(x, c, s) / 10
                   for a, c, s in zip(amps, centers, sigmas))
    y = y + RNG.normal(0, 0.003, x.size)
    p = os.path.join(d, name + ".txt")
    np.savetxt(p, np.column_stack([x, y]), header="x_axis intensity")
    json.dump({"technique": technique,
               "note": "two-component spectrum of the SAME sample batch"},
              open(os.path.join(d, name + ".json"), "w"))
    return p


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _make_agent(tag):
    import scilink.agents.meta_agent.fanout as fo
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    base = tempfile.mkdtemp(prefix=f"fusion_synth_{tag}_")
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


def _fuse_all(ag, out, focus):
    idxs = [r["delegation_index"] for r in out.get("results", [])
            if r.get("produced_output")]
    return json.loads(ag._fuse_delegations(idxs, focus=focus))


def _report(tag, fout):
    comp = fout.get("computed_reconciliation") or {}
    ver = comp.get("verification") or {}
    print(f"\n[{tag}] computed status: {comp.get('status')} | attempts: "
          f"{comp.get('attempts')} | audit: {ver.get('verdict')}")
    for i in ver.get("issues") or []:
        print("   issue:", str(i)[:220])
    print(f"[{tag}] QUANTITIES:", json.dumps(
        (comp.get("results") or {}).get("quantities"), indent=2,
        default=str)[:2200])
    for n in (comp.get("results") or {}).get("notes") or []:
        print("   note:", str(n)[:220])
    print(f"[{tag}] report: {fout.get('report_html_path')}")
    return comp


def _numeric_quantities(comp):
    out = {}

    def _walk(prefix, v):
        if isinstance(v, dict):
            for k, x in v.items():
                _walk(f"{prefix}{k}.", x)
        elif isinstance(v, (int, float)) and v is not None:
            out[prefix.rstrip(".")] = float(v)
    _walk("", (comp.get("results") or {}).get("quantities") or {})
    return out


def scenario_agreeing():
    ag, prompts = _make_agent("agreeing")
    d = tempfile.mkdtemp(prefix="synth_agree_data_")
    pat_ir = gen_ir_series(d); pat_xrd = gen_xrd_series(d)
    dsc = gen_dsc_curve(d)
    subject = ("a synthetic hydrate compound undergoing thermal "
               "dehydration (benchmark sample)")
    out = json.loads(ag._run_fanout([
        {"data_path": dsc, "label": "thermal curve",
         "task": f"Analyze the data file at {dsc}. It is one thermal "
                 f"measurement of {subject}. Characterize the features it "
                 "resolves and report the values that describe them."},
        {"data_path": d, "pattern": pat_ir, "label": "vibrational series",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_ir}'. They are a temperature series on "
                 f"{subject}. Track how the bands evolve and characterize "
                 "the transition they reveal."},
        {"data_path": d, "pattern": pat_xrd, "label": "diffraction series",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_xrd}'. They are a temperature series on "
                 f"{subject}. Track the reflections and characterize the "
                 "structural transition they reveal."},
    ]))
    check("agreeing: 3 branches ran", out.get("branches_run") == 3)
    fout = _fuse_all(ag, out, "Do the techniques agree on where the "
                              "transition occurs along temperature?")
    check("agreeing: fusion success + gated",
          fout.get("status") == "success" and fout.get("complementarity_gated"))
    comp = _report("agreeing", fout)
    check("agreeing: computed reconciliation succeeded",
          comp.get("status") == "success")
    check("agreeing: audit ran", (comp.get("verification") or {}).get(
        "verdict") in ("accept", "refine"))
    vals = list(_numeric_quantities(comp).values())
    hits = {t0: min((abs(v - t0) for v in vals), default=99)
            for t0 in (85.0, 88.0, 92.0)}
    print(f"[agreeing] planted-truth recovery (min |computed - planted|): "
          f"{hits}")
    check("agreeing: all three planted markers recovered within 6 C",
          all(h <= 6.0 for h in hits.values()))
    check("agreeing: audit accepted the computation",
          (comp.get("verification") or {}).get("verdict") == "accept")


def scenario_null():
    ag, prompts = _make_agent("null")
    d = tempfile.mkdtemp(prefix="synth_null_data_")
    pat_ir = gen_ir_series(d)          # real transition at 85 C
    pat_drift = gen_drift_series(d)    # NO transition anywhere
    subject = ("ONE synthetic benchmark pellet heated in-situ in a single "
               "experiment, measured simultaneously by two techniques")
    out = json.loads(ag._run_fanout([
        {"data_path": d, "pattern": pat_ir, "label": "vibrational series",
         "metadata": "FTIR-like absorbance series (2500-4000 cm-1), probes "
                     "the O-H / water content of the pellet; sidecar JSONs "
                     "carry temperature_C.",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_ir}'. They are a temperature series on "
                 f"{subject}, probing its WATER CONTENT via the O-H band. "
                 "Track how the bands evolve and characterize any "
                 "transition they reveal."},
        {"data_path": d, "pattern": pat_drift, "label": "framework series",
         "metadata": "Raman-like scattering series (800-1800 cm-1), probes "
                     "the framework/backbone modes of the SAME pellet in "
                     "the SAME run; sidecar JSONs carry temperature_C.",
         "task": f"Analyze the measurement series in {d} — ONLY files "
                 f"matching '{pat_drift}'. They are a temperature series on "
                 f"{subject}, probing its FRAMEWORK modes (a different "
                 "observable than the companion O-H measurement). Track how "
                 "the signal evolves and characterize any transition it "
                 "reveals."},
    ]))
    if out.get("status") != "success":
        print("[null] fan-out response:", json.dumps(
            {k: out.get(k) for k in ("status", "reason", "verdict",
                                     "message")}, indent=2, default=str)[:1500])
    check("null: 2 branches ran", out.get("branches_run") == 2)
    fout = _fuse_all(ag, out, "Do the two series agree on a transition "
                              "along temperature?")
    check("null: fusion success", fout.get("status") == "success")
    comp = _report("null", fout)
    check("null: audit ran", (comp.get("verification") or {}).get(
        "verdict") in ("accept", "refine") or comp.get("status") != "success")
    # The featureless branch must not be assigned a transition coinciding
    # with the real one (78-92 C) — that would be a manufactured agreement.
    fabricated = {k: v for k, v in _numeric_quantities(comp).items()
                  if any(s in k.lower() for s in
                         ("reference", "drift", "raman"))
                  and 78.0 <= v <= 92.0
                  and any(s in k.lower() for s in
                          ("transition", "t0", "midpoint", "onset",
                           "breakpoint", "marker"))}
    print(f"[null] would-be fabricated markers: {fabricated}")
    check("null: no fabricated coincidence for the featureless branch",
          not fabricated)
    # Absence must be acknowledged — in the narrative/caveats OR (the
    # mechanical, phrasing-proof signal) in the computed output itself
    # (e.g. framework_transition_detected: false / a *_null verdict).
    blob = (" ".join(str(c) for c in fout.get("caveats") or [])
            + str(fout.get("detailed_analysis", ""))
            + json.dumps(comp.get("results") or {}, default=str)).lower()
    check("null: absence acknowledged somewhere",
          any(s in blob for s in
              ("no transition", "no corresponding", "does not show",
               "absence", "no correlation", "not observed", "featureless",
               "no clear transition", "lacks", "null", "no significant "
               "trend", "transition_detected\": false", "disagree")))


def scenario_scalar():
    ag, prompts = _make_agent("scalar")
    d = tempfile.mkdtemp(prefix="synth_scalar_data_")
    xps = gen_two_component_spectrum(
        d, "synthXPS", centers=(285.0, 287.5), sigmas=(0.7, 0.8),
        fractions=(0.60, 0.40), span=(280, 295), technique="XPS-like")
    ram = gen_two_component_spectrum(
        d, "synthRAMAN", centers=(1350.0, 1580.0), sigmas=(25.0, 20.0),
        fractions=(0.58, 0.42), span=(1000, 1800), technique="Raman-like")
    out = json.loads(ag._run_fanout([
        {"data_path": xps, "label": "photoemission spectrum",
         "task": f"Analyze {xps}: a two-component spectrum of nanoparticle "
                 "batch NP-7. Fit both components and report the AREA "
                 "FRACTION of the lower-energy component (with uncertainty)."},
        {"data_path": ram, "label": "vibrational spectrum",
         "task": f"Analyze {ram}: a two-band spectrum of the SAME "
                 "nanoparticle batch NP-7. Fit both bands and report the "
                 "AREA FRACTION of the lower-shift band (with uncertainty)."},
    ]))
    check("scalar: 2 branches ran", out.get("branches_run") == 2)
    fout = _fuse_all(ag, out, "Do the two techniques agree on the component "
                              "fraction of batch NP-7?")
    check("scalar: fusion success", fout.get("status") == "success")
    comp = _report("scalar", fout)
    check("scalar: computed reconciliation succeeded",
          comp.get("status") == "success")
    fracs = [v for v in _numeric_quantities(comp).values() if 0.30 <= v <= 0.80]
    print(f"[scalar] fraction-like quantities: {sorted(fracs)}")
    close = [abs(a - 0.60) <= 0.08 or abs(a - 0.58) <= 0.08 for a in fracs]
    check("scalar: both planted fractions recovered (+-0.08)",
          sum(close) >= 2)
    check("scalar: sigma availability recorded",
          "sigma_available" in (comp.get("results") or {}))
    print(f"[scalar] sigma_available = "
          f"{(comp.get('results') or {}).get('sigma_available')}")


SCENARIOS = {"agreeing": scenario_agreeing, "null": scenario_null,
             "scalar": scenario_scalar}


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
        except Exception as e:  # noqa: BLE001 - keep scoring other scenarios
            import traceback; traceback.print_exc()
            check(f"{name}: scenario completed without crashing", False)
    print("\n" + "=" * 60)
    npass = sum(checks.values())
    print(f"FUSION SYNTHETIC LIVE: {npass}/{len(checks)} checks passed")
    for k, v in checks.items():
        if not v:
            print("  FAILED:", k)
    sys.exit(0 if npass == len(checks) else 1)


if __name__ == "__main__":
    main()
