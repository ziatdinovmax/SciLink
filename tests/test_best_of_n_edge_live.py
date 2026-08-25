#!/usr/bin/env python3
"""Extensive live edge-case suite for best-of-N v2.

Scenarios (sequential; each independent):
  G. Multi-regime series + escalation: per-regime anchors escalate
     independently; per-anchor table/dirs consistent with each judge's
     escalated flag; non-anchors untouched.
  H. Hard image (faint overlapping blobs, heavy noise): exercises the
     genuine escalate branch + judge over genuinely-different candidates.
  I. Clamp stress: n_candidates=10 -> clamps to 8; 8 dirs; judge handles 8
     visualizations under the image budget.
  J. reuse_locked_script bypass: prior run reused with n_candidates=3 ->
     NO fan-out, reuse_validity present.
  K. Aux + escalation combined: serial attempt 0 then parallel remainder
     with atomic aux staging; temp file loads cleanly afterwards.
  L. Curve 2-regime series (decay -> Gaussian peak) with opt-in n=2:
     per-regime curve anchors fan out; non-anchors reuse locked script.
  M. Curve r2_threshold=0.999 + escalation: fast-accept margin cap branch.
  N. Join-approval live override + 'more': digit override flips the winner
     (human_override recorded); 'more' after a fast-accept runs the rest.
  O. Meta-agent delegation: default-path delegation inherits escalation via
     the child orchestrator's tool body.

Needs a vendor key (AWS_BEARER_TOKEN_BEDROCK + AWS_REGION_NAME) and
UNSAFE_EXECUTION_OK=true.

Usage: python tests/test_best_of_n_edge_live.py [model_name] [only=G,H,...]
"""
import builtins
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scilink import auth

GRAINS = str(
    Path(__file__).resolve().parents[1]
    / "examples/polycrystalline_grains_demo/image.npy"
)
SYS_INFO = "A microscopy image of a polycrystalline material."
OBJECTIVE = "Segment the grains and report grain size statistics."

RESULTS = []


def scenario(tag):
    def deco(fn):
        fn._tag = tag
        return fn
    return deco


def _image_agent(model_name, api_key, out, **kw):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    kw.setdefault("enable_human_feedback", False)
    return ImageAnalysisAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        use_literature=False, **kw,
    )


def _curve_agent(model_name, api_key, out, **kw):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    kw.setdefault("enable_human_feedback", False)
    return CurveFittingAgent(
        api_key=api_key, model_name=model_name, output_dir=out,
        use_literature=False, **kw,
    )


def _logs():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    logging.getLogger().addHandler(h)
    return buf, h


def _cand_dirs(out, unit_dir):
    cdir = Path(out) / unit_dir / "_candidates"
    return sorted(p.name for p in cdir.glob("cand_*")) if cdir.is_dir() else []


def _consistent(escalated, dirs, table, n):
    if escalated is True:
        return len(dirs) == n and len(table) == n
    if escalated is False:
        return dirs == ["cand_00"] and len(table) == 1
    return False


def _make_two_regime_images(tmp: Path) -> list:
    rng = np.random.default_rng(0)
    grains = np.load(GRAINS).astype(np.float64)

    def save(idx, arr):
        a = (arr - arr.min()) / max(np.ptp(arr), 1e-9) * 255
        p = tmp / f"frame_{idx}.npy"
        np.save(p, a.astype(np.uint8))
        return str(p)

    paths = [save(0, grains + rng.normal(0, 4, grains.shape)),
             save(1, np.roll(grains, (37, -22), axis=(0, 1))
                  + rng.normal(0, 4, grains.shape))]
    yy, xx = np.mgrid[0:512, 0:512]
    for idx in (2, 3):
        img = rng.normal(20, 4, (512, 512))
        for _ in range(35):
            cy, cx = rng.uniform(20, 492, 2)
            r = rng.uniform(6, 16)
            img += 200 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                                  / (2 * r ** 2)))
        paths.append(save(idx, img))
    return paths


# ---------------------------------------------------------------------------

@scenario("G")
def g_multiregime_escalation(model_name, api_key):
    tmp = Path(tempfile.mkdtemp(prefix="edge_g_in_"))
    paths = _make_two_regime_images(tmp)
    out = Path(tempfile.mkdtemp(prefix="edge_g_"))
    agent = _image_agent(model_name, api_key, str(out))
    res = agent.analyze(
        data=paths,
        system_info=("SEM image series during in-situ annealing; grains "
                     "transform into isolated particles partway through."),
        objective=("Characterize each morphological regime appropriately: "
                   "grain statistics, then particle statistics."),
        series_metadata={"variable": "temperature",
                         "values": [300, 400, 500, 600], "unit": "K"},
        n_candidates=3, candidate_escalation=True,
    )
    tables = res.get("anchor_candidates") or {}
    print(f"  status: {res.get('status')!r}; anchor tables: {len(tables)} "
          f"(indices {sorted(tables)})")
    ok = res.get("status") == "success" and len(tables) >= 1
    anchor_idx = {int(k) for k in tables}
    for idx in range(len(paths)):
        dirs = _cand_dirs(out, f"image_{idx:04d}")
        if idx in anchor_idx:
            t = tables[idx] if idx in tables else tables[str(idx)]
            esc = (t.get("judge") or {}).get("escalated")
            cons = _consistent(esc, dirs, t.get("candidates") or [], 3)
            print(f"  anchor {idx}: escalated={esc}, dirs={dirs}, "
                  f"consistent={cons}")
            ok = ok and cons
        else:
            print(f"  non-anchor {idx}: dirs={dirs}")
            ok = ok and not dirs
    print(f"  output: {out}")
    return ok


@scenario("H")
def h_hard_image_escalates(model_name, api_key):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:512, 0:512]
    img = rng.normal(100, 12, (512, 512))
    for _ in range(60):  # faint, overlapping blobs
        cy, cx = rng.uniform(10, 502, 2)
        r = rng.uniform(8, 28)
        img += 25 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
    img = (img - img.min()) / np.ptp(img) * 255
    p = Path(tempfile.mkdtemp(prefix="edge_h_in_")) / "hard.npy"
    np.save(p, img.astype(np.uint8))

    out = Path(tempfile.mkdtemp(prefix="edge_h_"))
    agent = _image_agent(model_name, api_key, str(out))
    res = agent.analyze(
        data=str(p),
        system_info="Low-contrast micrograph of overlapping faint particles "
                    "with heavy detector noise.",
        objective="Detect ALL particles and report their size distribution.",
        n_candidates=3, candidate_escalation=True,
    )
    judge = res.get("anchor_judge") or {}
    table = res.get("anchor_candidates") or []
    dirs = _cand_dirs(out, "image_0000")
    esc = judge.get("escalated")
    print(f"  status: {res.get('status')!r}; escalated={esc}; dirs={dirs}; "
          f"table={len(table)}")
    if esc is True:
        scores = [c["score"] for c in table]
        print(f"  scores: {scores}; judge: {judge.get('reasoning','')[:140]!r}")
    print(f"  output: {out}")
    return res.get("status") == "success" and _consistent(esc, dirs, table, 3)


@scenario("I")
def i_clamp_stress(model_name, api_key):
    out = Path(tempfile.mkdtemp(prefix="edge_i_"))
    agent = _image_agent(model_name, api_key, str(out))
    t0 = time.monotonic()
    res = agent.analyze(data=GRAINS, system_info=SYS_INFO,
                        objective=OBJECTIVE, n_candidates=10)  # clamps to 8
    dt = time.monotonic() - t0
    dirs = _cand_dirs(out, "image_0000")
    table = res.get("anchor_candidates") or []
    sel = [c for c in table if c.get("selected")]
    print(f"  status: {res.get('status')!r}; dirs={len(dirs)}; "
          f"table={len(table)}; selected={len(sel)}; wall={dt:.0f}s")
    print(f"  judge: {(res.get('anchor_judge') or {}).get('reasoning','')[:140]!r}")
    print(f"  output: {out}")
    return (res.get("status") == "success" and len(dirs) == 8
            and len(table) == 8 and len(sel) == 1)


@scenario("J")
def j_reuse_bypasses_fanout(model_name, api_key):
    out1 = Path(tempfile.mkdtemp(prefix="edge_j_prior_"))
    agent1 = _image_agent(model_name, api_key, str(out1))
    res1 = agent1.analyze(data=GRAINS, system_info=SYS_INFO,
                          objective=OBJECTIVE)  # plain n=1 prior run
    if res1.get("status") != "success":
        print("  prior run failed; cannot test reuse")
        return False

    out2 = Path(tempfile.mkdtemp(prefix="edge_j_reuse_"))
    agent2 = _image_agent(model_name, api_key, str(out2))
    buf, h = _logs()
    try:
        res2 = agent2.analyze(
            data=GRAINS, system_info=SYS_INFO, objective=OBJECTIVE,
            prior_analysis_paths=[str(out1)], reuse_locked_script=True,
            n_candidates=3,
        )
    finally:
        logging.getLogger().removeHandler(h)
    log = buf.getvalue()
    dirs = _cand_dirs(out2, "image_0000")
    rv = res2.get("reuse_validity") or {}
    no_fanout_log = "Best-of-3" not in log
    print(f"  status: {res2.get('status')!r}; reuse verdict: "
          f"{rv.get('verdict')!r}; cand dirs: {dirs}; "
          f"no fan-out log: {no_fanout_log}")
    print(f"  output: {out2}")
    return (res2.get("status") == "success" and not dirs
            and bool(rv) and no_fanout_log
            and "anchor_candidates" not in res2)


@scenario("K")
def k_aux_plus_escalation(model_name, api_key):
    img = np.load(GRAINS)
    aux = (img.astype(np.float64) * 0.5 + 10).astype(np.uint8)
    aux_path = Path(tempfile.mkdtemp(prefix="edge_k_in_")) / "ref.npy"
    np.save(aux_path, aux)
    out = Path(tempfile.mkdtemp(prefix="edge_k_"))
    agent = _image_agent(model_name, api_key, str(out))
    buf, h = _logs()
    try:
        res = agent.analyze(
            data=GRAINS, system_info=SYS_INFO, objective=OBJECTIVE,
            auxiliary_data=str(aux_path),
            auxiliary_label="co-registered reference channel",
            n_candidates=3, candidate_escalation=True,
        )
    finally:
        logging.getLogger().removeHandler(h)
    log = buf.getvalue()
    judge = res.get("anchor_judge") or {}
    table = res.get("anchor_candidates") or []
    dirs = _cand_dirs(out, "image_0000")
    esc = judge.get("escalated")
    # The staged aux temp file (process CWD) must load cleanly (not torn).
    temps = list(Path(os.getcwd()).glob("temp_auxiliary_*.npy"))
    torn = False
    for t in temps:
        try:
            np.load(t)
        except Exception:
            torn = True
    print(f"  status: {res.get('status')!r}; escalated={esc}; dirs={dirs}; "
          f"temp aux files: {len(temps)} (torn: {torn}); "
          f"disable log absent: {'best-of-N disabled' not in log}")
    print(f"  output: {out}")
    return (res.get("status") == "success" and not torn
            and "best-of-N disabled" not in log
            and _consistent(esc, dirs, table, 3))


@scenario("L")
def l_curve_two_regime_series(model_name, api_key):
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp(prefix="edge_l_in_"))
    x = np.linspace(0, 50, 400)
    paths = []
    for i in (0, 1):  # regime 1: exponential decay
        y = (800 - 60 * i) * np.exp(-x / (4.0 + 0.3 * i)) + 30
        p = tmp / f"spec_{i}.npy"
        np.save(p, np.column_stack([x, y + rng.normal(0, 6, x.shape)]))
        paths.append(str(p))
    for i in (2, 3):  # regime 2: Gaussian peak on baseline
        y = 30 + (500 + 40 * i) * np.exp(-((x - 25) ** 2) / (2 * 3.5 ** 2))
        p = tmp / f"spec_{i}.npy"
        np.save(p, np.column_stack([x, y + rng.normal(0, 6, x.shape)]))
        paths.append(str(p))

    out = Path(tempfile.mkdtemp(prefix="edge_l_"))
    agent = _curve_agent(model_name, api_key, str(out))
    res = agent.analyze(
        data=paths,
        system_info=("Spectroscopy series during a phase transition: "
                     "exponential decay traces transform into a single "
                     "Gaussian emission peak partway through the series."),
        objective=("Fit each regime with the appropriate model: decay "
                   "lifetimes, then peak position/width."),
        series_metadata={"variable": "temperature",
                         "values": [10, 50, 200, 300], "unit": "K"},
        n_candidates=2,
    )
    tables = res.get("anchor_candidates") or {}
    print(f"  status: {res.get('status')!r}; anchor tables: {len(tables)} "
          f"(indices {sorted(tables)})")
    ok = res.get("status") == "success" and len(tables) >= 1
    anchor_idx = {int(k) for k in tables}
    for idx in range(len(paths)):
        dirs = _cand_dirs(out, f"spectrum_{idx:04d}")
        if idx in anchor_idx:
            t = tables.get(idx) or tables.get(str(idx)) or {}
            cands = t.get("candidates") or []
            sel = [c for c in cands if c.get("selected")]
            print(f"  anchor {idx}: dirs={dirs}, table={len(cands)}, "
                  f"selected={len(sel)}")
            ok = ok and len(dirs) == 2 and len(cands) == 2 and len(sel) == 1
        else:
            print(f"  non-anchor {idx}: dirs={dirs}")
            ok = ok and not dirs
    print(f"  output: {out}")
    return ok


@scenario("M")
def m_curve_threshold_cap(model_name, api_key):
    rng = np.random.default_rng(7)
    x = np.linspace(0, 50, 400)
    y = 800 * np.exp(-x / 4.0) + 250 * np.exp(-x / 18.0) + 30
    y = y + rng.normal(0, 8, x.shape)
    p = Path(tempfile.mkdtemp(prefix="edge_m_in_")) / "decay.npy"
    np.save(p, np.column_stack([x, y]))
    out = Path(tempfile.mkdtemp(prefix="edge_m_"))
    agent = _curve_agent(model_name, api_key, str(out))
    # fast_thr = 0.999 + min(0.02, 0.0005) = 0.9995 — noisy data sits below,
    # so escalation must trigger (exercises the margin-cap branch).
    res = agent.analyze(
        data=str(p),
        system_info="Photoluminescence decay trace: time (ns) vs intensity.",
        objective="Fit the decay and extract lifetimes.",
        n_candidates=2, candidate_escalation=True, r2_threshold=0.999,
    )
    judge = res.get("anchor_judge") or {}
    table = res.get("anchor_candidates") or []
    dirs = _cand_dirs(out, "spectrum_0000")
    esc = judge.get("escalated")
    scores = [round(c.get("score", 0), 5) for c in table]
    print(f"  status: {res.get('status')!r}; escalated={esc}; "
          f"R²s={scores}; dirs={dirs}")
    print(f"  output: {out}")
    # With threshold 0.999 the noisy data can't fast-accept; expect escalation
    # AND consistency. (If the fit magically exceeds 0.9995 in <=2 iters,
    # fast-accept is also self-consistent — accept either, require coherence.)
    return (res.get("status") == "success"
            and _consistent(esc, dirs, table, 2))


@scenario("N")
def n_join_approval_override_and_more(model_name, api_key):
    real_input = builtins.input

    # Part 1: digit override on a fixed n=2 run.
    out = Path(tempfile.mkdtemp(prefix="edge_n1_"))
    agent = _image_agent(model_name, api_key, str(out),
                         enable_human_feedback=True)
    state = {"winner": None, "overridden": None}

    def responder(prompt=""):
        # The input() prompt names the judge's pick ("Enter = accept
        # candidate N"); deliberately pick the OTHER candidate.
        m = re.search(r"accept candidate (\d+)", prompt)
        if m and state["overridden"] is None:
            picked = int(m.group(1))
            state["winner"] = picked
            state["overridden"] = 1 - picked  # n=2 -> the other one
            return str(state["overridden"])
        return ""

    builtins.input = responder
    try:
        res = agent.analyze(data=GRAINS, system_info=SYS_INFO,
                            objective=OBJECTIVE, n_candidates=2)
    finally:
        builtins.input = real_input
    table = res.get("anchor_candidates") or []
    sel = [c for c in table if c.get("selected")]
    judge = res.get("anchor_judge") or {}
    ok1 = (res.get("status") == "success" and len(sel) == 1
           and state["overridden"] is not None
           and sel[0]["attempt"] == state["overridden"]
           and judge.get("human_override") is True)
    print(f"  part 1 (override): judge pick={state['winner']}, "
          f"override={state['overridden']}, selected={sel[0]['attempt'] if sel else None}, "
          f"human_override={judge.get('human_override')} -> "
          f"{'PASS' if ok1 else 'CHECK'}  ({out})")

    # Part 2: 'more' after an escalation fast-accept.
    out2 = Path(tempfile.mkdtemp(prefix="edge_n2_"))
    agent2 = _image_agent(model_name, api_key, str(out2),
                          enable_human_feedback=True)
    answered_more = {"done": False}

    def responder2(prompt=""):
        if "remaining candidates" in prompt or (
                "Your choice" in prompt and not answered_more["done"]):
            answered_more["done"] = True
            return "more"
        return ""

    builtins.input = responder2
    try:
        res2 = agent2.analyze(data=GRAINS, system_info=SYS_INFO,
                              objective=OBJECTIVE, n_candidates=3,
                              candidate_escalation=True)
    finally:
        builtins.input = real_input
    judge2 = res2.get("anchor_judge") or {}
    table2 = res2.get("anchor_candidates") or []
    dirs2 = _cand_dirs(out2, "image_0000")
    if judge2.get("escalated") and answered_more["done"] and len(table2) == 3:
        ok2 = res2.get("status") == "success" and len(dirs2) == 3
        note = "'more' ran the remaining candidates"
    elif judge2.get("escalated") and not answered_more["done"]:
        # attempt 0 was weak -> escalated automatically; 'more' never offered.
        ok2 = res2.get("status") == "success" and len(dirs2) == 3
        note = "first attempt weak; escalated automatically ('more' not offered)"
    else:
        ok2 = False
        note = "unexpected shape"
    print(f"  part 2 ('more'): escalated={judge2.get('escalated')}, "
          f"table={len(table2)}, dirs={len(dirs2)} ({note}) -> "
          f"{'PASS' if ok2 else 'CHECK'}  ({out2})")
    return ok1 and ok2


@scenario("O")
def o_meta_delegation(model_name, api_key):
    import shutil as _sh
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent, MetaMode,
    )
    session = Path(tempfile.mkdtemp(prefix="edge_o_meta_"))
    data_dir = session / "uploads"
    data_dir.mkdir()
    img = data_dir / "image.npy"
    _sh.copy2(GRAINS, img)
    meta_file = data_dir / "metadata.json"
    meta_file.write_text(json.dumps({
        "sample": "polycrystalline thin film", "technique": "SEM",
    }))
    meta = MetaOrchestratorAgent(
        base_dir=str(session), api_key=api_key, model_name=model_name,
        meta_mode=MetaMode.AUTONOMOUS,
    )
    buf, h = _logs()
    try:
        reply = meta.chat(
            f"Analyze the microscopy image at {img} (metadata file: "
            f"{meta_file}). Segment the grains and report size statistics."
        )
    finally:
        logging.getLogger().removeHandler(h)
    log = buf.getvalue()
    esc_log = "Best-of-3 (escalation)" in log
    # Child orchestrator's stored record should carry the table.
    table_found = False
    child = getattr(meta, "_children", {}).get("analysis") or \
        getattr(meta, "analysis_child", None)
    if child is not None and getattr(child, "analysis_results", None):
        full = (child.analysis_results[-1] or {}).get("full_result") or {}
        table_found = bool(full.get("anchor_candidates"))
    print(f"  reply head: {str(reply)[:120]!r}")
    print(f"  escalation log via child: {esc_log}; child table: {table_found}")
    print(f"  session: {session}")
    return esc_log and table_found


# ---------------------------------------------------------------------------

def main() -> int:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key for", model_name)
        return 2
    only = None
    for a in sys.argv[2:]:
        if a.startswith("only="):
            only = set(a[5:].upper().split(","))

    scenarios = [g_multiregime_escalation, h_hard_image_escalates,
                 i_clamp_stress, j_reuse_bypasses_fanout,
                 k_aux_plus_escalation, l_curve_two_regime_series,
                 m_curve_threshold_cap, n_join_approval_override_and_more,
                 o_meta_delegation]
    for fn in scenarios:
        tag = fn._tag
        if only and tag not in only:
            continue
        print(f"\n=== {tag}: {fn.__name__} ===")
        t0 = time.monotonic()
        try:
            ok = fn(model_name, api_key)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok = False
        RESULTS.append((tag, ok, time.monotonic() - t0))
        print(f"  => {tag}: {'PASS' if ok else 'CHECK'} "
              f"({RESULTS[-1][2]:.0f}s)")

    print("\n" + "=" * 50)
    for tag, ok, dt in RESULTS:
        print(f"  {tag}: {'PASS' if ok else 'CHECK'} ({dt:.0f}s)")
    all_ok = all(ok for _, ok, _ in RESULTS) and RESULTS
    print("RESULT:", "PASS — all edge scenarios" if all_ok
          else "CHECK — see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
