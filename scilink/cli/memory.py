#!/usr/bin/env python3
"""
scilink memory - Manage SciLink's persistent memory

The persistent store (``~/.scilink/graduated_skills``, relocatable via
``SCILINK_HOME``) holds graduated and auto-distilled skills. It survives a
``pip`` upgrade and is auto-discovered by the skill loader.

Auto-distilled skills (from successful T=2 "hot annealing" curve fits) are
written **provisional**: discoverable and explicitly usable, but kept out of
the auto-routing menu until you review and promote them.

Skills (graduated_skills) subcommands:
  list      List persisted skills (use --provisional-only to triage)
  show      Print a skill's markdown
  promote   Clear a skill's provisional flag so it routes normally
  demote    Set a promoted skill back to provisional (out of auto-routing)
  prune     Delete a skill bundle
  fork      Copy a BUILT-IN skill into the persistent store (shadows it;
            makes it upgradable) — copy-on-write for shipped skills
  diff-builtin  Diff a forked skill against its shipped original (for
            contributing improvements back as a package PR)

Staged T=2 solutions (distill_staging) subcommands:
  staged       List staged raw T=2 solutions, grouped by technique
  upgrade      Merge a staged solution INTO an existing skill (--into <domain>/<name>)
  consolidate  Distill all staged solutions of a technique into a NEW skill
  prune-staged Delete staged solution(s)

Script bank (script_bank — episodic memory of successful scripts) subcommands:
  bank         List banked scripts with cross-session usage stats
               (★ proven = graduation candidates)
  bank-show    Print a bank record including its script
  bank-promote Send a proven record into distill staging — it then flows
               through the same review-gated upgrade/consolidate path
  bank-prune   Delete a bank record

`upgrade`/`consolidate` call an LLM; configure with --model / --base-url / --api-key.
"""

import argparse
import os
import sys

from scilink.skills._shared._memory import (
    list_memory,
    show_memory,
    promote_memory,
    prune_memory,
)
from scilink.skills._shared import _staging


def _split_ref(ref: str):
    """Parse a 'domain/name' reference. Returns (domain, name)."""
    if "/" not in ref:
        raise SystemExit(
            f"❌ Expected '<domain>/<name>' (e.g. 'curve_fitting/auto_voigt_ab12cd34'), got: {ref}"
        )
    domain, name = ref.split("/", 1)
    return domain, name


def _make_llm_call(args):
    """Build an llm_call callable from the standard --model/--base-url/--api-key.

    Mirrors how the other CLI commands construct a model (LiteLLMGenerativeModel).
    """
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    model = LiteLLMGenerativeModel(
        model=getattr(args, "model", None) or "claude-opus-4-6",
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
    )

    def _call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)

    return _call


def _add_model_args(p):
    p.add_argument("--model", default="claude-opus-4-6",
                   help="Model for the distillation LLM (default: claude-opus-4-6)")
    p.add_argument("--base-url", dest="base_url", default=None,
                   help="Custom API base URL (e.g. internal proxy)")
    p.add_argument("--api-key", dest="api_key", default=None,
                   help="API key (else taken from the conventional vendor env var)")


def _cmd_memory_toggle(args) -> int:
    """`scilink memory enable|disable|status` — the persistent-memory master switch."""
    from scilink.skills import loader
    if args.action == "status":
        on = loader.memory_enabled()
        env = os.environ.get("SCILINK_MEMORY", "").strip()
        src = f"env SCILINK_MEMORY={env!r}" if env else f"config ({loader._config_path()})"
        print(f"Persistent memory: {'ON' if on else 'OFF'}   [{src}]")
        return 0
    enabled = (args.action == "enable")
    p = loader.set_memory_enabled(enabled)
    print(f"Persistent memory {'ENABLED' if enabled else 'DISABLED'}  (saved to {p})")
    print("  → staging + graduated-skill loading are now ACTIVE."
          if enabled else
          "  → inert: nothing is staged and graduated skills are not loaded into runs.")
    if os.environ.get("SCILINK_MEMORY", "").strip():
        print("  ⚠️  SCILINK_MEMORY env var is set and overrides this saved setting.")
    return 0


def _cmd_list(args) -> int:
    from scilink.skills import loader
    print(f"[persistent memory: {'ON' if loader.memory_enabled() else 'OFF — inert; `scilink memory enable` to use'}]\n")
    provisional = None
    if args.provisional_only:
        provisional = True
    elif args.promoted_only:
        provisional = False
    rows = list_memory(domain=args.domain, provisional=provisional)
    if not rows:
        print("No skills in persistent memory.")
        return 0
    for r in rows:
        tag = "  [provisional]" if r["provisional"] else ""
        _m = _staging.metric_label(r)
        r2 = f"  {_m}" if _m else ""
        prov = f"  ({r['provenance']})" if r.get("provenance") else ""
        print(f"{r['domain']}/{r['name']}{tag}{r2}{prov}")
        if r.get("description"):
            print(f"    {r['description']}")
    print(f"\n{len(rows)} skill(s).")
    return 0


def _cmd_show(args) -> int:
    domain, name = _split_ref(args.ref)
    try:
        print(show_memory(domain, name))
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    return 0


def _cmd_promote(args) -> int:
    _warn_memory_off()
    domain, name = _split_ref(args.ref)
    try:
        res = promote_memory(domain, name, to_domain=args.to_domain)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    print(f"✅ Promoted {domain}/{name} → {res['domain']}/{name} (now auto-routable).")
    print(f"   {res['path']}")
    return 0


def _cmd_demote(args) -> int:
    from scilink.skills._shared._memory import demote_memory
    domain, name = _split_ref(args.ref)
    try:
        demote_memory(domain, name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    print(f"🟡 Demoted {domain}/{name} back to provisional — out of the "
          f"auto-routing menu (still explicitly loadable; re-`promote` after review).")
    return 0


def _cmd_fork(args) -> int:
    """`scilink memory fork` — copy a built-in skill into the persistent store."""
    _warn_memory_off()  # a fork only SHADOWS while persistent memory is on
    from scilink.skills._shared._memory import fork_builtin
    domain, name = _split_ref(args.ref)
    try:
        out = fork_builtin(domain, name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    if out.get("status") != "success":
        print(f"❌ {out.get('message')}")
        return 1
    print(f"🍴 Forked built-in {domain}/{name} → {out['path']}")
    print("   The fork SHADOWS the shipped skill immediately (loader precedence)"
          " and is now a valid `upgrade --into` target.")
    if out.get("has_sibling_tools"):
        print(f"   Note: the built-in bundle also ships tools "
              f"({', '.join(out['sibling_tools'])}) — those stay with the "
              f"package and keep registering for this skill; only the "
              f"markdown was forked.")
    print(f"   Contribute back later: `scilink memory diff-builtin {domain}/{name}`; "
          f"prune the fork once merged upstream.")
    return 0


def _cmd_diff_builtin(args) -> int:
    from scilink.skills._shared._memory import diff_builtin
    domain, name = _split_ref(args.ref)
    try:
        out = diff_builtin(domain, name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    if out["identical"]:
        print(f"Fork of {domain}/{name} is identical to the built-in.")
    else:
        print(out["diff"])
    return 0


def _cmd_prune(args) -> int:
    domain, name = _split_ref(args.ref)
    if not args.yes:
        resp = input(f"Delete {domain}/{name} from persistent memory? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1
    try:
        prune_memory(domain, name)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    print(f"🗑️  Pruned {domain}/{name}.")
    return 0


def _consolidate_n() -> int:
    return _staging.consolidate_min_n()

def _warn_memory_off():
    """Distilling/promoting while memory is off produces skills that won't load
    into runs until it's enabled — warn so the action isn't silently inert."""
    from scilink.skills import loader
    if not loader.memory_enabled():
        print("⚠️  Persistent memory is OFF — this won't affect runs until you "
              "enable it (`scilink memory enable`).")


def _cmd_staged(args) -> int:
    groups = _staging.group_by_technique(args.domain) if args.domain else {}
    if args.domain:
        domains = {args.domain: groups}
    else:
        # all domains
        domains = {}
        for rec in _staging.list_staged():
            domains.setdefault(rec["domain"], {}).setdefault(
                rec.get("technique") or "unlabeled", []).append(rec)
    total = 0
    threshold = _consolidate_n()
    any_ready = False
    for dom, by_tech in sorted(domains.items()):
        for tech, recs in sorted(by_tech.items()):
            total += len(recs)
            if len(recs) >= threshold:
                any_ready = True
                status = " — ready to consolidate"
            else:
                status = f" — accumulating {len(recs)}/{threshold} for a new skill"
            print(f"{dom}/{tech}: {len(recs)} staged{status}")
            for r in recs:
                metric = r.get("r_squared") or r.get("quality_score")
                mtxt = f"  metric={metric}" if metric is not None else ""
                prov = _staging.PROVENANCE_LABELS.get(r.get("provenance", "t2_solution"),
                                                      r.get("provenance", ""))
                print(f"    · id={r['id']}  [{prov}]  session={r.get('session','?')}{mtxt}")
    if not total:
        print("No staged solutions.")
    else:
        # New-skill consolidation accumulates first (>= N of a technique). Upgrading an
        # existing skill from a single solution is always available.
        tip = "`upgrade <domain>/<id> --into <domain>/<name>` (enrich an existing skill)"
        if any_ready:
            tip += " or `consolidate <domain>/<technique>` (techniques marked ready)"
        print(f"\n{total} staged solution(s). {tip}.")
    return 0


def _cmd_upgrade(args) -> int:
    _warn_memory_off()
    import difflib
    from scilink.agents.exp_agents.instruct import (
        KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
    )
    domain, sid = _split_ref(args.ref)  # staged ref = <domain>/<id>
    tdomain, tname = _split_ref(args.into)

    # 1) propose (build the merged skill without writing it)
    prop = _staging.propose_skill_upgrade(
        domain, [sid], target_domain=tdomain, target_name=tname,
        llm_call=_make_llm_call(args),
        fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
        update_template=SKILL_UPDATE_INSTRUCTIONS,
    )
    if prop.get("status") != "success":
        print(f"❌ {prop.get('message', prop)}")
        return 1

    for w in prop.get("regression_warnings") or []:
        print(f"⚠️  ADDITIVITY: {w}")

    # 2) show the diff for review (upgrade mutates an existing skill in place)
    if not args.yes:
        diff = difflib.unified_diff(
            prop["existing_content"].splitlines(),
            prop["proposed_content"].splitlines(),
            fromfile=f"{tdomain}/{tname} (current)",
            tofile=f"{tdomain}/{tname} (after upgrade)", lineterm="")
        print(f"\nProposed upgrade of {tdomain}/{tname} from staged {sid}:\n")
        printed = False
        for line in diff:
            printed = True
            if line.startswith("+") and not line.startswith("+++"):
                print(f"\033[32m{line}\033[0m")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"\033[31m{line}\033[0m")
            else:
                print(line)
        if not printed:
            print("(no textual change)")
        resp = input("\nApply this upgrade? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted — nothing written, staged solution kept.")
            return 1

    # 3) apply the approved content (backs up the current file)
    res = _staging.apply_skill_upgrade(
        domain, prop["staged_ids"], target_domain=tdomain, target_name=tname,
        proposed_content=prop["proposed_content"],
    )
    if res.get("status") != "success":
        print(f"❌ {res.get('message', res)}")
        return 1
    print(f"✅ Upgraded {tdomain}/{tname} from staged {sid}.")
    print(f"   {res.get('skill_path')}")
    print(f"   backup: {res.get('backup_path')}")
    return 0


def _cmd_consolidate(args) -> int:
    _warn_memory_off()
    from scilink.agents.exp_agents.instruct import (
        T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
    )
    domain, technique = _split_ref(args.ref)  # <domain>/<technique>
    res = _staging.consolidate_technique(
        domain, technique,
        llm_call=_make_llm_call(args),
        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
        update_template=SKILL_UPDATE_INSTRUCTIONS,
    )
    if res.get("status") != "success":
        print(f"❌ {res.get('message', res)}")
        return 1
    print(f"✅ Consolidated {res.get('n_examples')} staged solution(s) → "
          f"{domain}/auto_{technique} (provisional).")
    print(f"   {res.get('skill_path')}")
    return 0


def _cmd_prune_staged(args) -> int:
    domain, sid = _split_ref(args.ref)
    if not args.yes:
        resp = input(f"Delete staged solution {domain}/{sid}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1
    n = _staging.remove_staged(domain, [sid])
    print(f"🗑️  Removed {n} staged solution(s).")
    return 0 if n else 1


def _cmd_bank(args) -> int:
    """`scilink memory bank` — list script-bank records (episodic memory)."""
    from scilink.skills._shared import _script_bank
    rows = _script_bank.bank_summary(args.domain)
    if args.proven_only:
        rows = [r for r in rows if r["proven"]]
    if not rows:
        print("No banked scripts." + (" (proven-only filter)" if args.proven_only else ""))
        return 0
    threshold = _script_bank.proven_n()
    by_domain = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    n_proven = 0
    for dom, recs in sorted(by_domain.items()):
        print(f"{dom}: {len(recs)} banked script(s)")
        for r in recs:
            metric = r["metric"]
            mtxt = (f"  {metric['name']}={metric['value']}"
                    if isinstance(metric, dict) and metric.get("value") is not None else "")
            marks = ""
            if r["proven"]:
                marks += "  ★ proven"
                n_proven += 1
            if r["promoted_to_staging"]:
                marks += f"  ✓ promoted (staged {r['promoted_to_staging']})"
            print(f"    · id={r['id']}  successes={r['n_successes']} "
                  f"retrievals={r['n_retrievals']}{mtxt}{marks}")
            print(f"      {r['label'][:100]}")
    print(f"\n{len(rows)} record(s); ★ = succeeded in ≥{threshold} sessions "
          f"(a graduation candidate). Inspect with `bank-show <domain>/<id>`; "
          f"send a proven one into the review-gated skill path with "
          f"`bank-promote <domain>/<id>`.")
    if n_proven and not args.proven_only:
        print(f"{n_proven} proven record(s) — see them alone with `bank --proven-only`.")
    return 0


def _cmd_bank_show(args) -> int:
    from scilink.skills._shared import _script_bank
    domain, rid = _split_ref(args.ref)
    rec = _script_bank.get_record(domain, rid)
    if rec is None:
        print(f"❌ No bank record {domain}/{rid}.")
        return 1
    import json as _json
    script = rec.pop("working_script", "")
    print(_json.dumps(rec, indent=2, default=str))
    print("\n--- working_script " + "-" * 50)
    print(script)
    return 0


def _cmd_bank_promote(args) -> int:
    """Send bank record(s) into distill staging (the graduation path).

    Several refs promote as a GROUP under one shared technique label, so
    `consolidate` later reviews the variants together."""
    _warn_memory_off()
    from scilink.skills._shared import _script_bank
    refs = [_split_ref(r) for r in args.refs]
    domain = refs[0][0]
    if any(d != domain for d, _ in refs):
        print("❌ Group promotion takes records from ONE domain.")
        return 1
    if len(refs) == 1:
        out = _script_bank.promote_to_staging(domain, refs[0][1],
                                              technique=args.technique)
        if out.get("status") != "success":
            print(f"❌ {out.get('message')}")
            return 1
        print(f"🧠 Promoted bank record {refs[0][1]} → staged {out['staged_id']} "
              f"[{out['technique']}].")
    else:
        out = _script_bank.promote_group_to_staging(
            domain, [rid for _, rid in refs], technique=args.technique)
        if out.get("status") != "success":
            print(f"❌ {out.get('message')}")
            return 1
        print(f"🧠 Promoted {len(out['staged_ids'])} record(s) as one technique "
              f"[{out['technique']}] → staged {', '.join(out['staged_ids'])}.")
        for s in out.get("skipped", []):
            print(f"   (skipped {s['id']}: {s['reason']})")
        if out.get("ready_to_consolidate"):
            print(f"   ✅ {out['n_staged_total']} staged for this technique — "
                  f"ready: `scilink memory consolidate {domain}/{out['technique']}`")
        else:
            print(f"   Accumulating {out['n_staged_total']}/"
                  f"{_staging.consolidate_min_n()} for consolidation "
                  f"(`consolidate` can force it now).")
    print("   Review with `scilink memory staged`; bank records are kept "
          "(marked ✓ promoted).")
    return 0


def _cmd_bank_groups(args) -> int:
    """`scilink memory bank-groups` — same-system variant clusters."""
    from scilink.skills._shared import _script_bank
    domains = ([args.domain] if args.domain else
               sorted({r["domain"] for r in _script_bank.bank_summary()}))
    total = 0
    for dom in domains:
        thr = (args.threshold if args.threshold is not None
               else _script_bank.VARIANT_GROUP_THRESHOLD)
        for g in _script_bank.find_variant_groups(dom, threshold=thr):
            total += 1
            print(f"{dom}: {len(g['ids'])} variants of one system "
                  f"(min pairwise similarity {g['min_similarity']})")
            for r in g["records"]:
                m = r["metric"]
                mtxt = (f"  {m['name']}={m['value']}"
                        if isinstance(m, dict) and m.get("value") is not None else "")
                mark = " ✓ promoted" if r["promoted_to_staging"] else ""
                print(f"    · {r['id']}  successes={r['n_successes']}{mtxt}{mark}")
                print(f"      {r['label'][:90]}")
            refs = " ".join(f"{dom}/{i}" for i in g["ids"])
            print(f"    → promote together: `scilink memory bank-promote {refs} "
                  f"--technique {g['suggested_technique']}`")
    if not total:
        print("No variant groups found (need ≥2 records of the same system "
              "in a domain).")
    return 0


def _cmd_bank_prune(args) -> int:
    from scilink.skills._shared import _script_bank
    domain, rid = _split_ref(args.ref)
    if not args.yes:
        resp = input(f"Delete bank record {domain}/{rid}? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1
    n = _script_bank.remove_records(domain, [rid])
    print(f"🗑️  Removed {n} bank record(s).")
    return 0 if n else 1


def main():
    """Entry point for 'scilink memory'."""
    parser = argparse.ArgumentParser(
        prog="scilink memory",
        description="Manage SciLink's persistent memory (graduated + auto-distilled skills).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action")

    p_en = sub.add_parser("enable", help="Turn persistent memory ON (opt-in)")
    p_en.set_defaults(func=_cmd_memory_toggle)
    p_dis = sub.add_parser("disable", help="Turn persistent memory OFF (the default)")
    p_dis.set_defaults(func=_cmd_memory_toggle)
    p_status = sub.add_parser("status", help="Show whether persistent memory is on or off")
    p_status.set_defaults(func=_cmd_memory_toggle)

    p_list = sub.add_parser("list", help="List persisted skills")
    p_list.add_argument("--domain", help="Restrict to one domain (e.g. curve_fitting)")
    p_list.add_argument("--provisional-only", action="store_true",
                        help="Show only provisional (auto-distilled, unreviewed) skills")
    p_list.add_argument("--promoted-only", action="store_true",
                        help="Show only promoted (non-provisional) skills")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Print a skill's markdown")
    p_show.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_show.set_defaults(func=_cmd_show)

    p_promote = sub.add_parser("promote", help="Clear a skill's provisional flag")
    p_promote.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_promote.add_argument("--to-domain", help="Optionally move the bundle to a curated domain")
    p_promote.set_defaults(func=_cmd_promote)

    p_demote = sub.add_parser(
        "demote", help="Set a skill back to provisional (out of auto-routing)")
    p_demote.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_demote.set_defaults(func=_cmd_demote)

    p_fork = sub.add_parser(
        "fork", help="Copy a built-in skill into the persistent store "
                     "(shadows it; enables upgrade)")
    p_fork.add_argument("ref", help="Built-in skill ref '<domain>/<name>'")
    p_fork.set_defaults(func=_cmd_fork)

    p_db = sub.add_parser(
        "diff-builtin", help="Diff a forked skill against the shipped built-in")
    p_db.add_argument("ref", help="Skill ref '<domain>/<name>'")
    p_db.set_defaults(func=_cmd_diff_builtin)

    p_prune = sub.add_parser("prune", help="Delete a skill bundle")
    p_prune.add_argument("ref", help="Skill reference '<domain>/<name>'")
    p_prune.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_prune.set_defaults(func=_cmd_prune)

    # --- staged T=2 solutions ---
    p_staged = sub.add_parser("staged", help="List staged knowledge (solved-from-scratch solutions, feedback, error fixes) by technique")
    p_staged.add_argument("--domain", help="Restrict to one domain")
    p_staged.set_defaults(func=_cmd_staged)

    p_up = sub.add_parser("upgrade", help="Merge a staged solution INTO an existing skill")
    p_up.add_argument("ref", help="Staged solution ref '<domain>/<id>'")
    p_up.add_argument("--into", required=True, help="Target skill '<domain>/<name>'")
    p_up.add_argument("--yes", action="store_true",
                      help="Apply without showing the diff / prompting")
    _add_model_args(p_up)
    p_up.set_defaults(func=_cmd_upgrade)

    p_con = sub.add_parser("consolidate", help="Distill all staged solutions of a technique into a NEW skill")
    p_con.add_argument("ref", help="Technique ref '<domain>/<technique>'")
    _add_model_args(p_con)
    p_con.set_defaults(func=_cmd_consolidate)

    p_ps = sub.add_parser("prune-staged", help="Delete a staged solution")
    p_ps.add_argument("ref", help="Staged solution ref '<domain>/<id>'")
    p_ps.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_ps.set_defaults(func=_cmd_prune_staged)

    # --- script bank (episodic memory of successful scripts) ---
    p_bank = sub.add_parser(
        "bank", help="List banked scripts (episodic memory) with usage stats")
    p_bank.add_argument("--domain", help="Restrict to one domain")
    p_bank.add_argument("--proven-only", action="store_true",
                        help="Show only records proven across sessions "
                             "(graduation candidates)")
    p_bank.set_defaults(func=_cmd_bank)

    p_bs = sub.add_parser("bank-show", help="Print a bank record incl. its script")
    p_bs.add_argument("ref", help="Bank record ref '<domain>/<id>'")
    p_bs.set_defaults(func=_cmd_bank_show)

    p_bp = sub.add_parser(
        "bank-promote",
        help="Send bank record(s) into distill staging; several refs promote "
             "as a group under one shared technique label")
    p_bp.add_argument("refs", nargs="+",
                      help="Bank record ref(s) '<domain>/<id>' (same domain)")
    p_bp.add_argument("--technique", default=None,
                      help="Staging technique label (default: derived from the "
                           "best record)")
    p_bp.set_defaults(func=_cmd_bank_promote)

    p_bg = sub.add_parser(
        "bank-groups",
        help="Show same-system variant clusters (candidates for group "
             "promotion + consolidation)")
    p_bg.add_argument("--domain", help="Restrict to one domain")
    p_bg.add_argument("--threshold", type=float, default=None,
                      help="Fingerprint-similarity grouping threshold "
                           "(default 0.85)")
    p_bg.set_defaults(func=_cmd_bank_groups)

    p_br = sub.add_parser("bank-prune", help="Delete a bank record")
    p_br.add_argument("ref", help="Bank record ref '<domain>/<id>'")
    p_br.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p_br.set_defaults(func=_cmd_bank_prune)

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
