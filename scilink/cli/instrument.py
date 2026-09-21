#!/usr/bin/env python3
"""scilink instrument — what the instruments on this machine remember.

A live loop opened with ``remember=True`` keeps its verified recipes and a
summary of every run per instrument, under the persistent SciLink home
(``~/.scilink/instruments/<id>/``; ``SCILINK_HOME`` relocates it). A remembered
recipe arms the next run on that instrument with no model call, after it has
been replayed and judged on the new reference.

    scilink instrument list
    scilink instrument show <id>
    scilink instrument forget <id> <recipe id>      # one recipe
    scilink instrument forget <id> --all            # everything about it
"""

import argparse
import json
import sys


def _outputs(meta) -> str:
    return ", ".join(sorted(meta.get("outputs") or {})) or ", ".join((meta.get("reports") or [])[:4]) or "-"


def main():
    from scilink.live import instrument_home as store

    parser = argparse.ArgumentParser(prog="scilink instrument", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="instruments this machine remembers")
    p_show = sub.add_parser("show", help="one instrument's recipes and runs")
    p_show.add_argument("instrument")
    p_show.add_argument("--json", action="store_true", help="print the raw record")
    p_forget = sub.add_parser("forget", help="delete a remembered recipe, or an instrument")
    p_forget.add_argument("instrument")
    p_forget.add_argument("recipe", nargs="?", help="recipe id (see `show`)")
    p_forget.add_argument("--all", action="store_true", help="forget the whole instrument")
    p_forget.add_argument("-y", "--yes", action="store_true", help="do not ask")
    args = parser.parse_args()

    if args.action == "list":
        known = store.known_instruments()
        if not known:
            print(f"No instrument is remembered yet ({store.instruments_root()}).")
            return 0
        for info in known:
            print(f"{info.get('id')}  [{info.get('modality') or 'curve'}]  "
                  f"{info.get('technique') or 'technique not recorded'}")
            print(f"    recipes {info['recipes']}, runs {info['runs']}, last seen {info.get('last_seen')}")
        return 0

    if args.action == "show":
        record = store.remembered(args.instrument)
        if record is None:
            print(f"❌ No instrument '{args.instrument}' is remembered.")
            return 1
        if args.json:
            print(json.dumps(record, indent=2, default=str))
            return 0
        info = record["instrument"]
        print(f"{info.get('id')}  [{info.get('modality') or 'curve'}]  {info.get('technique') or ''}")
        print(f"first seen {info.get('first_seen')}, last seen {info.get('last_seen')}")
        print(f"\nRecipes ({len(record['recipes'])}):")
        for r in record["recipes"]:
            print(f"  {r['recipe_id']}{'  [contested]' if r.get('contested') else ''}  recalled {r.get('uses', 0)}x, last used {r.get('last_used')}, "
                  f"{r.get('size_mb')} MB, from {r.get('source') or 'a reference analysis'}")
            print(f"      sample: {r.get('sample') or '-'}   tracks: {_outputs(r)}")
        print(f"\nRuns ({len(record['runs'])} most recent):")
        for run in record["runs"]:
            print(f"  {run.get('when')}  frames {run.get('frames')} (clean {run.get('clean_frames')}), "
                  f"changes {len(run.get('novelties') or [])}, rebuilds {run.get('reanchors')}, "
                  f"audits {run.get('audits')}")
        return 0

    if args.action == "forget":
        if not args.all and not args.recipe:
            print("❌ Name a recipe id, or pass --all to forget the whole instrument.")
            return 1
        what = f"everything about '{args.instrument}'" if args.all else f"recipe {args.recipe}"
        if not args.yes:
            if input(f"Forget {what}? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted.")
                return 0
        done = (store.forget_instrument(args.instrument) if args.all
                else store.forget_recipe(args.instrument, args.recipe))
        print(f"🗑️ Forgot {what}." if done else f"❌ Nothing to forget: {what} was not found.")
        return 0 if done else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
