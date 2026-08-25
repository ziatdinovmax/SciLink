"""Comprehensive LIVE test suite for agent-side multi-skill auto-selection (#256).

Validates, against a real LLM, the full feature set built on
``feature-agent-multiskill-autoselect``:

  * technique-aware single-select + the #251 no-substitution rule
  * per-domain ``exclusive`` policy (curve = one technique; image/hsi compose)
  * multi-skill composition (prose + codegen), end-to-end with execution
  * ``skill_hint`` authority (orchestrator suggests, agent confirms/overrides/rejects)
  * custom-skill visibility / auto-select (fix #1) + foreign-modality filtering
  * orchestrator deferral vs explicit ``skill`` vs ``skill_hint``

Uses REAL example data (grains image, EELS spectrum + datacube) and SYNTHETIC
data (XPS/EPR/XRD/Raman/generic spectra; atomic-STEM / AFM-grain / control
images) generated in-process.

Run (Bedrock; agent/orchestrator tiers execute generated code):

  UNSAFE_EXECUTION_OK=true \\
  AWS_BEARER_TOKEN_BEDROCK=<key> AWS_REGION_NAME=us-east-1 \\
    python tests/test_multiskill_autoselect_live.py --tier all --model bedrock/us.anthropic.claude-opus-4-8

  --tier selector      fast, no code execution (default)
  --tier agent         full analyze() runs (slow, executes code)
  --tier orchestrator  orchestrator routing decisions (slow)
  --tier all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(REPO, "examples")
MODEL_DEFAULT = "bedrock/us.anthropic.claude-opus-4-8"

# ───────────────────────────── result harness ─────────────────────────────
_RESULTS = []  # (tier, label, status, detail)  status ∈ {PASS, FAIL, SOFT, INFO}


def record(tier, label, status, detail=""):
    _RESULTS.append((tier, label, status, detail))
    mark = {"PASS": "✅", "FAIL": "❌", "SOFT": "🟡", "INFO": "ℹ️"}[status]
    print(f"  {mark} [{tier}] {label}: {detail}")


def check(tier, label, ok, got, *, soft=False):
    record(tier, label, ("PASS" if ok else ("SOFT" if soft else "FAIL")),
           f"got {got!r}")


# ───────────────────────────── synthetic data ─────────────────────────────
def synth_xps():
    x = np.linspace(280, 295, 400)
    bg = 0.4 * (x - 280) / 15
    y = bg + 3.0 * np.exp(-0.5 * ((x - 284.8) / 0.5) ** 2) \
          + 1.4 * np.exp(-0.5 * ((x - 286.3) / 0.7) ** 2)
    y += 0.02 * np.random.RandomState(1).randn(x.size)
    return x, y, {"technique": "XPS", "detail": "C 1s region", "x_units": "eV (binding energy)"}


def synth_epr():
    x = np.linspace(300, 360, 1024)  # mT
    def deriv_gauss(c, w, A):
        z = (x - c) / w
        return A * (-z) * np.exp(-0.5 * z ** 2)
    y = deriv_gauss(322, 1.2, 1.0) + deriv_gauss(345, 1.5, 0.8)  # axial-ish
    y += 0.01 * np.random.RandomState(2).randn(x.size)
    return x, y, {"technique": "EPR", "detail": "X-band CW first-derivative", "x_units": "mT"}


def synth_xrd():
    x = np.linspace(20, 80, 1200)  # 2theta
    y = 0.05 * np.ones_like(x)
    for c, A, w in [(28.4, 1.0, 0.15), (32.9, 0.6, 0.16), (47.2, 0.5, 0.2), (56.1, 0.4, 0.22)]:
        y += A / (1 + ((x - c) / w) ** 2)
    y += 0.01 * np.random.RandomState(3).randn(x.size)
    return x, y, {"technique": "XRD", "detail": "powder diffraction", "x_units": "2theta deg"}


def synth_raman():
    x = np.linspace(100, 900, 700)  # cm^-1
    y = 0.1 + 1.0 / (1 + ((x - 305) / 12) ** 2) + 0.7 / (1 + ((x - 515) / 18) ** 2) \
          + 0.5 / (1 + ((x - 720) / 20) ** 2)
    y += 0.01 * np.random.RandomState(4).randn(x.size)
    return x, y, {"technique": "Raman", "detail": "BaTiO3 vibrational bands", "x_units": "cm^-1"}


def synth_decay():
    x = np.linspace(0, 10, 300)
    y = 2.0 * np.exp(-x / 2.5) + 0.3 + 0.02 * np.random.RandomState(5).randn(x.size)
    return x, y, {"technique": "generic sensor voltage vs time", "detail": "exponential decay, no spectroscopy"}


def synth_atomic_image(n=256, spacing=16):
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), float)
    for cy in range(spacing // 2, n, spacing):
        for cx in range(spacing // 2, n, spacing):
            img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
    img = (img / img.max() * 255).astype(np.uint8)
    return img, {"microscopy_type": "STEM HAADF", "technique": "atomic-resolution Z-contrast",
                 "detail": "zone-axis crystalline lattice, atomic columns"}


def synth_afm_grains(n=512, n_seeds=45, seed=0):
    rng = np.random.RandomState(seed)
    pts = rng.rand(n_seeds, 2) * n
    heights = rng.rand(n_seeds)
    yy, xx = np.mgrid[0:n, 0:n]
    d = (xx[..., None] - pts[:, 1]) ** 2 + (yy[..., None] - pts[:, 0]) ** 2
    idx = d.argmin(-1)
    img = heights[idx]
    try:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, 3)
    except Exception:
        pass
    # dark grain boundaries where the nearest-seed label changes
    b = np.zeros((n, n), bool)
    b[:, 1:] |= idx[:, 1:] != idx[:, :-1]
    b[1:, :] |= idx[1:, :] != idx[:-1, :]
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    img[b] = 0.0
    img = (img * 255).astype(np.uint8)
    return img, {"microscopy_type": "AFM", "technique": "topography (height)",
                 "detail": "densely packed touching grains, intensity is height in nm",
                 "x_units": "nm"}


def synth_control_image(n=256):
    yy, xx = np.mgrid[0:n, 0:n]
    img = ((xx / n) * 255).astype(np.uint8)  # plain gradient
    return img, {"detail": "featureless intensity gradient, no recognizable structure"}


# ───────────────────────────── context builders ───────────────────────────
def _plot_bytes(x, y):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, y, lw=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def curve_ctx(x, y, meta):
    stats = {"n_points": int(x.size),
             "x_range": [float(x.min()), float(x.max())],
             "y_range": [float(y.min()), float(y.max())]}
    return [f"Metadata: {meta}", f"Data statistics: {stats}",
            {"mime_type": "image/jpeg", "data": _plot_bytes(x, y)}]


def image_ctx(img, meta):
    from scilink.skills._shared.image_analysis_tools import image_to_thumbnail_bytes
    return [f"Metadata: {meta}", {"mime_type": "image/jpeg", "data": image_to_thumbnail_bytes(img)}]


def meta_ctx(meta):
    return [f"Metadata: {meta}"]


# ───────────────────────────── model + parse ──────────────────────────────
def make_model(model_name):
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    return LiteLLMGenerativeModel(model=model_name, api_key=None)


def parse_fn(resp):
    t = (getattr(resp, "text", "") or "").strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    return (json.loads(m.group(0)) if m else {}), None


# ════════════════════════════ TIER 1: selector ════════════════════════════
def tier_selector(model_name):
    from scilink.skills._shared._skill_selector import select_relevant_skills, build_skill_catalog
    m = make_model(model_name)

    def sel(domain, ctx, **kw):
        return select_relevant_skills(model=m, parse_fn=parse_fn, domain=domain,
                                      context_parts=ctx, **kw)

    T = "selector"
    print("\n── TIER 1: selector (technique-aware, exclusive, hint, custom) ──")

    # catalog sanity
    _, cf = build_skill_catalog("curve_fitting")
    check(T, "catalog/curve has epr,xps,xrd", {"epr", "xps", "xrd_profile"} <= cf, sorted(cf))

    # CURVE (exclusive) — synthetic spectra by technique
    x, y, mta = synth_xps()
    check(T, "curve XPS -> xps", sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True) == ["xps"], "xps")
    x, y, mta = synth_epr()
    check(T, "curve EPR -> epr", sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True) == ["epr"], "epr")
    x, y, mta = synth_xrd()
    check(T, "curve XRD -> xrd_profile", sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True) == ["xrd_profile"], "xrd_profile")
    x, y, mta = synth_raman()
    r = sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True)
    check(T, "curve RAMAN -> [] (no substitution #251)", r == [], r)  # no raman skill; must NOT pick xrd/xps
    x, y, mta = synth_decay()
    r = sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True)
    check(T, "curve generic-decay -> []", r == [], r)

    # IMAGE (composable)
    img = np.load(os.path.join(EX, "polycrystalline_grains_demo/image.npy"))
    meta = json.load(open(os.path.join(EX, "polycrystalline_grains_demo/image.json")))
    r = sel("image_analysis", image_ctx(img, meta))
    check(T, "image REAL grains -> overlapping_objects", "overlapping_objects" in r, r)
    aimg, ameta = synth_atomic_image()
    r = sel("image_analysis", image_ctx(aimg, ameta))
    check(T, "image synth atomic-STEM -> atomic_stem", "atomic_stem" in r, r, soft=True)
    gimg, gmeta = synth_afm_grains()
    r = sel("image_analysis", image_ctx(gimg, gmeta))
    check(T, "image synth AFM-grains -> overlapping_objects (+afm multi)", "overlapping_objects" in r, r, soft=True)
    check(T, "image synth AFM-grains multi-skill fires", len(r) >= 2, r, soft=True)
    cimg, cmeta = synth_control_image()
    r = sel("image_analysis", image_ctx(cimg, cmeta))
    check(T, "image control-gradient -> []", r == [], r, soft=True)

    # HYPERSPECTRAL (metadata-only selector — weakest signal path; single skill)
    em = json.load(open(os.path.join(EX, "eels_plasmons_demo/datacube.json")))
    r = sel("hyperspectral", meta_ctx(em))
    check(T, "hsi REAL EELS datacube -> eels", "eels" in r, r, soft=True)
    em2 = json.load(open(os.path.join(EX, "eels_identification_demo/spectrum.json")))
    r = sel("hyperspectral", meta_ctx(em2))
    check(T, "hsi REAL EELS spectrum -> eels", "eels" in r, r)
    r = sel("hyperspectral", meta_ctx({"technique": "optical photoluminescence, no EELS"}))
    check(T, "hsi non-EELS -> []", r == [], r, soft=True)

    # skill_hint authority
    x, y, mta = synth_xps()
    xps_ctx = curve_ctx(x, y, mta)
    check(T, "hint confirm (xps/xps)", sel("curve_fitting", xps_ctx, exclusive=True, hint="xps") == ["xps"], "xps")
    check(T, "hint OVERRIDE (epr hint, xps data)", sel("curve_fitting", xps_ctx, exclusive=True, hint="epr") == ["xps"], "xps")
    x, y, mta = synth_decay()
    check(T, "hint REJECT (xps hint, generic data)", sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True, hint="xps") == [], "[]")
    r = sel("image_analysis", image_ctx(img, meta), hint="atomic_stem")
    check(T, "hint OVERRIDE image (atomic_stem hint, grains)", "overlapping_objects" in r and "atomic_stem" not in r, r, soft=True)

    # custom skills (fix #1)
    d = tempfile.mkdtemp()
    raman_md = os.path.join(d, "raman_custom.md")
    open(raman_md, "w").write(
        "---\ndescription: Raman peak fitting (Lorentzian/Voigt bands, baseline)\n"
        "technique: Raman, Raman spectroscopy\n---\n## overview\nRaman band fitting.\n"
        "## planning\nFit bands with pseudo-Voigt; subtract polynomial baseline.\n")
    foreign_md = os.path.join(d, "img_foreign.md")
    open(foreign_md, "w").write("---\ndomain: image_analysis\n---\n## overview\nimage thing.\n")
    CUST = {"raman_custom": raman_md, "img_foreign": foreign_md}

    x, y, mta = synth_raman()
    rctx = curve_ctx(x, y, mta)
    check(T, "custom raman REGISTERED + Raman data -> raman_custom (fix #1)",
          sel("curve_fitting", rctx, exclusive=True, custom_skills=CUST) == ["raman_custom"], "raman_custom")
    check(T, "custom raman NOT registered + Raman -> [] (was the gap)",
          sel("curve_fitting", rctx, exclusive=True) == [], "[]")
    x, y, mta = synth_xps()
    check(T, "custom raman registered + XPS data -> xps (ignore irrelevant custom)",
          sel("curve_fitting", curve_ctx(x, y, mta), exclusive=True, custom_skills=CUST) == ["xps"], "xps")
    _, names = build_skill_catalog("curve_fitting", custom_skills=CUST)
    check(T, "custom foreign-modality filtered from catalog", "img_foreign" not in names and "raman_custom" in names, sorted(n for n in names if "custom" in n or "foreign" in n))


# ════════════════════════════ TIER 2: agent e2e ═══════════════════════════
def _write_dataset(dirpath, name, arr, meta):
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name + ".npy")
    np.save(p, arr)
    json.dump(meta, open(os.path.join(dirpath, name + ".json"), "w"))
    return p


def tier_agent(model_name):
    T = "agent-e2e"
    print("\n── TIER 2: agent end-to-end (executes generated code) ──")
    work = tempfile.mkdtemp(prefix="multiskill_e2e_")

    # 2a. IMAGE two-skill composition on a SYNTHETIC AFM-grains image where BOTH
    # skills genuinely apply: afm (flatten / physical-unit discipline) +
    # overlapping_objects (segment touching grains). Saved as a file with a full
    # AFM metadata sidecar so afm's physical-unit rule is satisfiable.
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    gimg, _ = synth_afm_grains()
    afm_meta = {
        "microscopy_type": "AFM",
        "technique": "topography (height channel)",
        "data_range_minimum": 0.0,
        "data_range_maximum": 45.0,
        "data_range_units": "nm",
        "spatial_info": {"field_of_view_x": 2.0, "field_of_view_y": 2.0,
                         "field_of_view_units": "um"},
        "detail": "densely packed touching grains; intensity is height in nm",
    }
    afm_path = _write_dataset(os.path.join(work, "afm_grains"), "afm_grains", gimg, afm_meta)
    try:
        agent = ImageAnalysisAgent(model_name=model_name, enable_human_feedback=False,
                                   output_dir=os.path.join(work, "img_twoskill"),
                                   max_verification_iterations=1)
        res = agent.analyze(afm_path, system_info=afm_meta, skill=["afm", "overlapping_objects"])
        check(T, "image two-skill (afm+overlapping) executes", res.get("status") == "success", res.get("status"))
    except Exception as e:
        record(T, "image two-skill (afm+overlapping)", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")

    img = np.load(os.path.join(EX, "polycrystalline_grains_demo/image.npy"))
    meta = json.load(open(os.path.join(EX, "polycrystalline_grains_demo/image.json")))

    # 2b. IMAGE auto-select (no skill) on grains -> overlapping_objects
    try:
        agent = ImageAnalysisAgent(model_name=model_name, enable_human_feedback=False,
                                   output_dir=os.path.join(work, "img_auto"),
                                   max_verification_iterations=1)
        res = agent.analyze(img, system_info=meta)
        check(T, "image auto-select executes", res.get("status") == "success", res.get("status"))
    except Exception as e:
        record(T, "image auto-select", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")

    # 2c. CURVE auto-select on synthetic XPS -> xps fit
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    x, y, mta = synth_xps()
    xps_path = _write_dataset(os.path.join(work, "xps_data"), "xps", np.vstack([x, y]), mta)
    try:
        agent = CurveFittingAgent(model_name=model_name, enable_human_feedback=False,
                                  output_dir=os.path.join(work, "curve_auto"),
                                  max_verification_iterations=1)
        res = agent.analyze(xps_path, system_info=mta)
        check(T, "curve auto-select (synth XPS) executes", res.get("status") == "success", res.get("status"))
    except Exception as e:
        record(T, "curve auto-select", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")

    # 2d. CUSTOM-skill end-to-end: register a raman skill, fit synth Raman
    x, y, mta = synth_raman()
    raman_path = _write_dataset(os.path.join(work, "raman_data"), "raman", np.vstack([x, y]), mta)
    d = tempfile.mkdtemp()
    raman_md = os.path.join(d, "raman_custom.md")
    open(raman_md, "w").write(
        "---\ndescription: Raman peak fitting (Lorentzian/pseudo-Voigt bands)\n"
        "technique: Raman, Raman spectroscopy\n---\n## overview\nRaman band fitting.\n"
        "## planning\nFit each band with a pseudo-Voigt on a polynomial baseline.\n"
        "## implementation\nUse scipy.optimize.curve_fit with a sum of pseudo-Voigt "
        "profiles plus a low-order polynomial baseline; report center, FWHM, amplitude.\n")
    try:
        agent = CurveFittingAgent(model_name=model_name, enable_human_feedback=False,
                                  output_dir=os.path.join(work, "curve_custom"),
                                  max_verification_iterations=1)
        res = agent.analyze(raman_path, system_info=mta, custom_skills={"raman_custom": raman_md})
        check(T, "custom raman skill drives a real fit (fix #1 e2e)", res.get("status") == "success", res.get("status"))
    except Exception as e:
        record(T, "custom raman e2e", "FAIL", f"{type(e).__name__}: {str(e)[:160]}")

    print(f"  (e2e artifacts under {work})")


# ════════════════════════════ TIER 3: orchestrator ════════════════════════
def tier_orchestrator(model_name):
    T = "orch"
    print("\n── TIER 3: orchestrator routing (defer / skill / skill_hint) ──")
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode)

    work = tempfile.mkdtemp(prefix="multiskill_orch_")
    img = os.path.join(EX, "polycrystalline_grains_demo/image.npy")
    x, y, mta = synth_xps()
    # Explicit-technique copy (metadata declares XPS) for the "explicit" case.
    xps_path = _write_dataset(os.path.join(work, "xps"), "xps", np.vstack([x, y]), mta)
    # NEUTRAL-metadata copy (no declared technique) so the orchestrator has
    # nothing to pick from and must DEFER to the agent (which sees the plot).
    neutral_path = _write_dataset(os.path.join(work, "neutral"), "neutral",
                                  np.vstack([x, y]),
                                  {"detail": "1D spectrum; fit the peaks"})

    class _Stop(Exception):
        pass

    def capture(message, turns=2):
        agent = AnalysisOrchestratorAgent(base_dir=tempfile.mkdtemp(prefix="orch_"),
                                          model_name=model_name,
                                          analysis_mode=AnalysisMode.AUTONOMOUS)
        cap = {"skill": "__nocall__", "hint": "__nocall__"}
        real = agent.tools.execute_tool

        def wrap(tool_name, **kw):
            if tool_name == "run_analysis":
                cap["skill"] = kw.get("skill", None)
                cap["hint"] = kw.get("skill_hint", None)
                raise _Stop()
            return real(tool_name, **kw)
        agent.tools.execute_tool = wrap
        msg = message
        for _ in range(turns):
            try:
                agent.chat(msg)
            except _Stop:
                break
            except Exception as e:
                record(T, f"chat ({message[:30]})", "INFO", f"{type(e).__name__}: {str(e)[:80]}")
                break
            msg = "Proceed with the analysis now. No literature search needed."
        return cap

    # image, no skill named -> defer (skill None) OR an autonomous pick (benign, has preview)
    c = capture(f"Analyze this image and characterize the grains: {img}")
    check(T, "image no-skill: defers (skill None) or autonomous-correct",
          c["skill"] in (None, "overlapping_objects", ["overlapping_objects"]), c["skill"], soft=True)
    # image, explicit skill -> passes it
    c = capture(f"Analyze this image using the overlapping_objects skill: {img}")
    check(T, "image explicit -> overlapping_objects",
          c["skill"] == "overlapping_objects" or c["skill"] == ["overlapping_objects"], c["skill"])
    # curve, no skill, NEUTRAL metadata -> defers (orchestrator has nothing to
    # pick from; the agent's selector sees the plot and decides).
    c = capture(f"Fit the peaks in this spectrum: {neutral_path}")
    check(T, "curve no-skill (neutral meta) -> defers (None)", c["skill"] is None, c["skill"])
    # curve, explicit technique -> passes xps
    c = capture(f"This is an XPS spectrum (C 1s). Fit it: {xps_path}")
    check(T, "curve explicit XPS -> xps",
          c["skill"] == "xps" or c["skill"] == ["xps"], c["skill"])


# ───────────────────────────── runner ─────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["selector", "agent", "orchestrator", "all"], default="selector")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()

    print(f"=== MULTI-SKILL LIVE SUITE  model={args.model}  tier={args.tier} ===")
    if args.tier in ("selector", "all"):
        tier_selector(args.model)
    if args.tier in ("agent", "all"):
        tier_agent(args.model)
    if args.tier in ("orchestrator", "all"):
        tier_orchestrator(args.model)

    # summary
    n_pass = sum(1 for *_, s, _ in _RESULTS if s == "PASS")
    n_fail = sum(1 for *_, s, _ in _RESULTS if s == "FAIL")
    n_soft = sum(1 for *_, s, _ in _RESULTS if s == "SOFT")
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass} pass, {n_fail} FAIL, {n_soft} soft/stochastic")
    if n_fail:
        print("FAILURES:")
        for tier, label, s, detail in _RESULTS:
            if s == "FAIL":
                print(f"  ❌ [{tier}] {label}: {detail}")
    print("=" * 60)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
