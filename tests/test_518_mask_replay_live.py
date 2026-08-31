"""Live validation for #518 on Bedrock Opus 4.8: a harmonized locked replay
must reproduce a mask-scoped donor's SCOPING, not only its arithmetic.

Synthetic sibling XRF-style cubes (160x160x96): a decaying continuum
everywhere plus a small disk-shaped emitter inclusion carrying a Mn-Ka-like
Gaussian at 5.9 keV. The inclusion sits at a DIFFERENT location in every
sibling and its amplitude ramps with the (synthetic) sintering temperature —
so the donor's mask geometry cannot transfer; only the persisted recipe
(endmember + half-max rule) can.

  1. donor    — fresh analysis of cube A steered to fit_scope=component_mask;
                the approved record must persist fit_mask_recipe.
  2. follower — locked replay on cube B: the mask is re-derived from the
                recipe on B's own data (B's inclusion is elsewhere), record
                stays clean and carries the recipe forward.
  3. legacy   — replay on B against a STRIPPED copy of the donor records
                (recipe removed, pre-#518 shape): loud DEGRADED HARMONIZATION
                on the record and on the analyze() response.
  4. fanout   — meta-level delegate_to_analyses(harmonize=True) over A/B/C:
                both followers replay with reproduced scoping; fusion runs.

Run (creds auto-loaded from ~/.scilink/credentials.env):
    UNSAFE_EXECUTION_OK=true python tests/test_518_mask_replay_live.py [1 2 3 4]
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Script mode puts tests/ (not the repo root) on sys.path, so without this an
# editable install of another checkout would shadow the tree under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BASE = Path("tests/_518_mask_replay_live_runs").resolve()
H, W, E = 160, 160, 96
AXIS = np.linspace(1.0, 10.5, E)          # keV
PEAK_KEV = 5.9
BLOBS = {"A": (40, 40), "B": (115, 90), "C": (80, 125)}   # (row, col)
AMPS = {"A": 0.7, "B": 1.0, "C": 1.3}
TEMPS = {"A": 600, "B": 700, "C": 800}
BLOB_R = 10

results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


class Tee(io.StringIO):
    def write(self, s):
        sys.__stdout__.write(s)
        return super().write(s)


@contextlib.contextmanager
def capture_all(buf):
    root = logging.getLogger()
    prev_level = root.level
    h = logging.StreamHandler(buf)
    root.addHandler(h)
    root.setLevel(logging.INFO)
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        root.removeHandler(h)
        root.setLevel(prev_level)


def make_cube(key, seed):
    rng = np.random.default_rng(seed)
    cont = 40.0 * np.exp(-AXIS / 3.0)                       # shared continuum
    scale = 1.0 + 0.1 * rng.standard_normal((H, W, 1))      # spatial mottle
    cube = cont[None, None, :] * scale
    yy, xx = np.mgrid[0:H, 0:W]
    r0, c0 = BLOBS[key]
    disk = ((yy - r0) ** 2 + (xx - c0) ** 2) <= BLOB_R ** 2
    peak = AMPS[key] * 60.0 * np.exp(-0.5 * ((AXIS - PEAK_KEV) / 0.15) ** 2)
    cube[disk] += peak[None, :]
    cube += rng.normal(0, 0.6, cube.shape)
    return np.clip(cube, 0, None).astype(np.float32), disk


def cube_path(key):
    return BASE / "data" / f"coupon_{TEMPS[key]}C.npy"


def write_data():
    (BASE / "data").mkdir(parents=True, exist_ok=True)
    for i, key in enumerate("ABC"):
        cube, _ = make_cube(key, seed=100 + i)
        np.save(cube_path(key), cube)
        meta = dict(si_for(key))
        cube_path(key).with_suffix(".json").write_text(json.dumps(meta))


def si_for(key):
    return {
        "experiment_type": "Spectroscopy",
        "experiment": {
            "technique": "XRF spectrum imaging (SEM-EDS style datacube)",
            "details": (f"160x160 px, 96 channels 1.0-10.5 keV. Mn-bearing "
                        f"inclusion in a light matrix; coupon sintered at "
                        f"{TEMPS[key]} C (sibling series 600/700/800 C)."),
        },
        "sample": {"material": "Mn-doped ceramic coupon",
                   "description": f"sintered at {TEMPS[key]} C; one small "
                   "Mn-rich inclusion in an otherwise uniform matrix."},
        "energy_range": {"start": 1.0, "end": 10.5, "units": "keV"},
    }


OBJECTIVE = (
    "Quantify the Mn Ka emission (5.90 keV) of the Mn-rich inclusion: run the "
    "unsupervised decomposition first to locate the inclusion component, then "
    "fit a Gaussian + linear background per pixel over 5.0-6.8 keV and return "
    "Amplitude_Map and Center_Map. The inclusion occupies only a small part "
    "of the frame, so scope the per-pixel fit to the high-abundance region "
    "of the inclusion component (set fit_scope=component_mask with that "
    "component's number on the custom_code target) instead of fitting the "
    "full frame."
)


def agent(out):
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    return HyperspectralAnalysisAgent(
        api_key=None, model_name=MODEL, output_dir=str(out),
        enable_human_feedback=False, max_verification_iterations=1)


def load_records(run_dir):
    out = []
    for f in sorted(Path(run_dir).rglob("dynamic_analysis_records.json")):
        try:
            out.extend(r for r in json.loads(f.read_text())
                       if isinstance(r, dict))
        except Exception:
            continue
    return out


def records_dir(run_dir):
    for f in sorted(Path(run_dir).rglob("dynamic_analysis_records.json")):
        return f.parent
    return None


def run_analysis(name, key, **kw):
    out = BASE / name
    if out.exists():
        shutil.rmtree(out)
    buf = Tee()
    with capture_all(buf):
        res = agent(out).analyze(
            data=str(cube_path(key)), system_info=si_for(key),
            objective=OBJECTIVE, **kw)
    return res, buf.getvalue(), out


def part1_donor():
    print("\n=== 1. Donor (cube A, 600 C): fresh masked analysis ===")
    res, log, out = run_analysis("donor_A", "A")
    if "Fit mask from Component" not in log:
        print("     (donor retry — planner skipped the component mask)")
        res, log, out = run_analysis("donor_A_retry", "A")
    check("p1 donor completed", res.get("status") in ("success", "partial"))
    check("p1 fit mask was built", "Fit mask from Component" in log)
    recs = [r for r in load_records(out) if r.get("task_success")]
    recipes = [r.get("fit_mask_recipe") for r in recs if r.get("fit_mask_recipe")]
    check("p1 approved record persists fit_mask_recipe", bool(recipes))
    if recipes:
        rec = recipes[0]
        check("p1 recipe spectrum has cube channel count",
              len(rec.get("component_spectrum") or []) == E)
        spec = np.asarray(rec.get("component_spectrum") or [0])
        check("p1 recipe endmember peaks near Mn Ka",
              abs(AXIS[int(np.argmax(spec))] - PEAK_KEV) < 0.8)
        check("p1 recipe mask fraction plausible",
              0 < rec.get("mask_fraction", 0) < 0.5)
    results["_donor_dir"] = str(records_dir(out) or "")
    return out


def part2_follower():
    print("\n=== 2. Follower (cube B, 700 C): locked replay, mask re-derived ===")
    donor_dir = results.get("_donor_dir") or ""
    if not donor_dir:
        for cand in ("donor_A", "donor_A_retry"):
            d = records_dir(BASE / cand)
            if d and any(r.get("fit_mask_recipe")
                         for r in load_records(d)):
                donor_dir = str(d)
        results["_donor_dir"] = donor_dir
    check("p2 donor records available", bool(donor_dir))
    if not donor_dir:
        return
    res, log, out = run_analysis("follower_B", "B",
                                 prior_analysis_paths=[donor_dir],
                                 reuse_locked_script=True)
    check("p2 replay plan (no planning LLM)",
          "Locked-script replay plan" in log)
    check("p2 scoping reproduced on B", "Replay scoping reproduced" in log)
    check("p2 no degraded warning", "DEGRADED HARMONIZATION" not in log)
    recs = load_records(out)
    replayed = [r for r in recs if r.get("locked_replay")]
    check("p2 replay record exists", bool(replayed))
    if replayed:
        r = replayed[0]
        check("p2 task_success on follower", r.get("task_success") is True)
        check("p2 replay verbatim", r.get("replay_verbatim") is True)
        check("p2 recipe carried forward", bool(r.get("fit_mask_recipe")))
        check("p2 record not scope-degraded",
              not r.get("replay_scope_degraded"))
    sr = res.get("script_reuse") or {}
    check("p2 response script_reuse verbatim", sr.get("verbatim") is True)
    check("p2 response not scope-degraded", not sr.get("scope_degraded"))

    # Deterministic localization proof (same code path the live run took):
    # the donor recipe must mask B's OWN inclusion, which sits elsewhere.
    from scilink.agents.exp_agents.controllers import (
        hyperspectral_controllers as hc)
    recipe = next((r["fit_mask_recipe"] for r in load_records(donor_dir)
                   if r.get("fit_mask_recipe")), None)
    cube_b = np.load(cube_path("B"))
    mask = hc._rebuild_fit_mask_from_recipe(
        cube_b.astype(float), recipe, (H, W),
        logging.getLogger("p2.mask"))
    check("p2 recipe re-derives a mask on B", mask is not None)
    if mask is not None:
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        r0, c0 = BLOBS["B"]
        check("p2 mask centred on B's inclusion (not A's)",
              abs(cy - r0) < BLOB_R and abs(cx - c0) < BLOB_R)
        check("p2 mask is a small footprint", mask.mean() < 0.25)


def part3_legacy_degraded():
    print("\n=== 3. Legacy donor (recipe stripped): loud degraded flag ===")
    donor_dir = results.get("_donor_dir") or ""
    check("p3 donor records available", bool(donor_dir))
    if not donor_dir:
        return
    legacy = BASE / "legacy_donor"
    if legacy.exists():
        shutil.rmtree(legacy)
    legacy.mkdir(parents=True)
    recs = json.loads(
        (Path(donor_dir) / "dynamic_analysis_records.json").read_text())
    for r in recs:
        r.pop("fit_mask_recipe", None)
    (legacy / "dynamic_analysis_records.json").write_text(
        json.dumps(recs, default=str))

    res, log, out = run_analysis("follower_B_legacy", "B",
                                 prior_analysis_paths=[str(legacy)],
                                 reuse_locked_script=True)
    check("p3 degraded warning in log", "DEGRADED HARMONIZATION" in log)
    replayed = [r for r in load_records(out) if r.get("locked_replay")]
    check("p3 record flagged scope-degraded",
          any("NOT reproduced" in str(r.get("replay_scope_degraded"))
              for r in replayed))
    sr = res.get("script_reuse") or {}
    check("p3 response script_reuse.scope_degraded", sr.get("scope_degraded") is True)
    check("p3 response warnings carry the flag",
          any("DEGRADED HARMONIZATION" in w
              for w in (res.get("warnings") or [])))
    outcome = [(r.get("task_success"), r.get("salvaged")) for r in replayed]
    print(f"     full-frame replay outcome (task_success, salvaged): {outcome}")


def part4_fanout():
    print("\n=== 4. Meta fan-out harmonize=True over A/B/C ===")
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode)
    session = BASE / "meta_session"
    if session.exists():
        shutil.rmtree(session)
    ag = MetaOrchestratorAgent(
        base_dir=str(session), api_key=None, model_name=MODEL,
        meta_mode=MetaMode.AUTONOMOUS)
    branches = [{
        "data_path": str(cube_path(k)),
        "metadata": str(cube_path(k).with_suffix(".json")),
        "label": f"coupon {TEMPS[k]} C",
        "task": (f"Analyze the XRF datacube {cube_path(k)} of the Mn-doped "
                 f"coupon sintered at {TEMPS[k]} C (sibling series "
                 f"600/700/800 C). {OBJECTIVE} Complete this as ONE "
                 "run_analysis call."),
    } for k in "ABC"]
    buf = Tee()
    with capture_all(buf):
        out = json.loads(ag._run_fanout(branches, harmonize=True))
    log = buf.getvalue()
    (BASE / "fanout_result.json").write_text(
        json.dumps(out, indent=2, default=str))
    check("p4 fanout succeeded", out.get("status") == "success")
    check("p4 harmonized mode engaged",
          (out.get("harmonized") or {}).get("status") == "harmonized")
    check("p4 followers reproduced scoping",
          log.count("Replay scoping reproduced") >= 2)
    check("p4 no degraded harmonization", "DEGRADED HARMONIZATION" not in log)
    fan_recs = load_records(session)
    replayed = [r for r in fan_recs if r.get("locked_replay")]
    check("p4 two follower replay records", len(replayed) >= 2)
    # The #518 invariant: every replay is verbatim WITH the donor's scoping
    # reproduced (recipe carried, no degraded flag). The per-map QC verdict
    # on each follower is the judge's independent (stochastic) call on that
    # cube — reported, not asserted, beyond one clean success.
    check("p4 replays verbatim with scoping reproduced",
          len(replayed) >= 2 and all(
              r.get("replay_verbatim") and r.get("fit_mask_recipe")
              and not r.get("replay_scope_degraded") for r in replayed))
    check("p4 at least one follower passed QC cleanly",
          any(r.get("task_success") for r in replayed))
    print("     follower QC outcomes (task_success, salvaged): "
          + str([(r.get("task_success"), r.get("salvaged"))
                 for r in replayed]))
    idx = [r["delegation_index"] for r in out.get("results", [])
           if r.get("produced_output")]
    if len(idx) >= 2:
        buf2 = Tee()
        with capture_all(buf2):
            fout = json.loads(ag._fuse_delegations(
                idx, focus=("Does the inclusion's Mn Ka amplitude ramp with "
                            "sintering temperature across the harmonized "
                            "600/700/800 C siblings?")))
        (BASE / "fusion_result.json").write_text(
            json.dumps(fout, indent=2, default=str))
        check("p4 fusion completed", fout.get("status") == "success")
        check("p4 fusion sees harmonized branches",
              bool(fout.get("harmonized_branches")))


if __name__ == "__main__":
    from scilink.mcp_server import _load_shared_credentials
    _load_shared_credentials()
    parts = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    BASE.mkdir(parents=True, exist_ok=True)
    write_data()
    import time
    t0 = time.time()
    for p in parts:
        {1: part1_donor, 2: part2_follower,
         3: part3_legacy_degraded, 4: part4_fanout}[p]()
    fails = [k for k, v in results.items()
             if not k.startswith("_") and not v]
    print(f"\n=== {sum(1 for k, v in results.items() if v and not k.startswith('_'))}"
          f"/{sum(1 for k in results if not k.startswith('_'))} checks passed "
          f"({time.time()-t0:.0f}s) ===")
    if fails:
        print("FAILED:", fails)
    sys.exit(1 if fails else 0)
