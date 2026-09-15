#!/usr/bin/env python3
"""scilink kb — manage named, reusable knowledge bases.

Named KBs live under the persistent SciLink home
(``~/.scilink/knowledge_bases/<name>/``; ``SCILINK_HOME`` relocates them)
and are usable from any launch directory:

    scilink plan --knowledge-dir <name>
    scilink explore --knowledge-dir <name>
    (or in a meta chat: "attach my <name> knowledge base")

Building embeds the documents once (requires the embedding provider's API
key); every later session reuses the persisted index.
"""

import argparse
import json
import os
import sys


def _resolve_embedding_key(embedding_model: str, explicit: str | None) -> str | None:
    """Explicit key wins; otherwise the conventional vendor env var for the
    embedding model's provider (LiteLLM also auto-discovers, but resolving
    here lets us warn early)."""
    if explicit:
        return explicit
    m = embedding_model.lower()
    if "gemini" in m or "google" in m:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if "text-embedding" in m or m.startswith("openai"):
        return os.getenv("OPENAI_API_KEY")
    return None


def main():
    from scilink.knowledge import kb_store

    parser = argparse.ArgumentParser(
        prog="scilink kb",
        description="Manage named, reusable knowledge bases "
                    "(stored under the persistent SciLink home).",
    )
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("list", help="List named knowledge bases")

    p_create = sub.add_parser(
        "create", help="Build a named KB from documents (embeds them once)")
    p_create.add_argument("name", help="KB name (letters, digits, ._-)")
    p_create.add_argument("--from", dest="sources", nargs="+", required=True,
                          metavar="PATH",
                          help="Document files/folders (PDF, text/markdown, "
                               "Excel/CSV, images)")
    p_create.add_argument("--description", default="",
                          help="One-line description (shown to the meta-agent "
                               "when it decides relevance)")
    p_create.add_argument("--embedding-model", default="gemini-embedding-001")
    p_create.add_argument("--embedding-api-key", default=None)
    p_create.add_argument("--base-url", default=None,
                          help="OpenAI-compatible proxy for embeddings")
    p_create.add_argument("--overwrite", action="store_true")

    p_add = sub.add_parser(
        "add", help="Incrementally add documents to a KB (embeds only the "
                    "new ones, with the KB's own embedding model)")
    p_add.add_argument("name")
    p_add.add_argument("--from", dest="sources", nargs="+", required=True,
                       metavar="PATH", help="New document files/folders")
    p_add.add_argument("--embedding-api-key", default=None,
                       help="Key for the KB's embedding provider (default: "
                            "the provider env var for the manifest's model)")
    p_add.add_argument("--base-url", default=None)

    p_show = sub.add_parser("show", help="Print a KB's manifest")
    p_show.add_argument("name")

    p_import = sub.add_parser(
        "import", help="Adopt an existing kb_storage directory as a named KB")
    p_import.add_argument("name")
    p_import.add_argument("--from", dest="from_dir", required=True,
                          metavar="DIR",
                          help="Directory holding default_kb_docs.* index "
                               "files (e.g. ./kb_storage)")
    p_import.add_argument("--embedding-model", default="unknown",
                          help="Model that BUILT the index, if known — "
                               "recorded for compatibility checks")
    p_import.add_argument("--description", default="")
    p_import.add_argument("--overwrite", action="store_true")

    p_rebuild = sub.add_parser(
        "rebuild", help="Re-embed a KB's stored sources (e.g. to switch "
                        "embedding providers)")
    p_rebuild.add_argument("name")
    p_rebuild.add_argument("--embedding-model", required=True)
    p_rebuild.add_argument("--embedding-api-key", default=None)
    p_rebuild.add_argument("--base-url", default=None)

    p_delete = sub.add_parser("delete", help="Delete a named KB")
    p_delete.add_argument("name")
    p_delete.add_argument("--yes", action="store_true",
                          help="Skip the confirmation prompt")

    args = parser.parse_args()
    if not args.action:
        parser.print_help()
        return 0

    try:
        if args.action == "list":
            kbs = kb_store.list_kbs()
            if not kbs:
                print("No named knowledge bases. Create one with "
                      "'scilink kb create <name> --from <docs...>'.")
                return 0
            for m in kbs:
                desc = f" — {m['description']}" if m.get("description") else ""
                print(f"  {m['name']}{desc}")
                print(f"      embedding: {m.get('embedding_model', '?')}, "
                      f"vectors: {m.get('n_vectors', '?')}, "
                      f"origin: {m.get('origin', '?')}")
                srcs = m.get("sources") or []
                if srcs:
                    shown = ", ".join(srcs[:6])
                    more = f", … (+{len(srcs) - 6})" if len(srcs) > 6 else ""
                    print(f"      sources: {shown}{more}")
            return 0

        if args.action == "create":
            key = _resolve_embedding_key(args.embedding_model,
                                         args.embedding_api_key)
            if not key and not args.base_url:
                print(f"⚠️  No API key found for '{args.embedding_model}' — "
                      "set the provider env var or pass --embedding-api-key. "
                      "Trying anyway (LiteLLM may auto-discover)...")
            manifest = kb_store.create_kb(
                args.name, args.sources,
                embedding_model=args.embedding_model,
                api_key=key, base_url=args.base_url,
                description=args.description, overwrite=args.overwrite,
            )
            print(f"✅ KB '{args.name}' created "
                  f"({manifest['n_vectors']} vectors).")
            print(f"   Use it: scilink plan --knowledge-dir {args.name}")
            return 0

        if args.action == "add":
            manifest = kb_store.read_manifest(kb_store.kb_path(args.name)) or {}
            key = _resolve_embedding_key(manifest.get("embedding_model", ""),
                                         args.embedding_api_key)
            manifest = kb_store.add_to_kb(
                args.name, args.sources,
                api_key=key, base_url=args.base_url,
            )
            print(f"✅ KB '{args.name}' updated "
                  f"({manifest['n_vectors']} vectors total).")
            return 0

        if args.action == "show":
            manifest = kb_store.read_manifest(kb_store.kb_path(args.name))
            if manifest is None:
                print(f"❌ KB '{args.name}' not found (or has no manifest).")
                return 1
            print(json.dumps(manifest, indent=2))
            return 0

        if args.action == "import":
            manifest = kb_store.import_kb(
                args.name, args.from_dir,
                embedding_model=args.embedding_model,
                description=args.description, overwrite=args.overwrite,
            )
            print(f"✅ KB '{args.name}' imported.")
            if manifest.get("embedding_model") in (None, "", "unknown"):
                print("   ⚠️ Embedding model unrecorded — sessions will warn "
                      "until you re-import with --embedding-model or rebuild "
                      "from original documents.")
            return 0

        if args.action == "rebuild":
            key = _resolve_embedding_key(args.embedding_model,
                                         args.embedding_api_key)
            manifest = kb_store.rebuild_kb(
                args.name, embedding_model=args.embedding_model,
                api_key=key, base_url=args.base_url,
            )
            print(f"✅ KB '{args.name}' rebuilt with "
                  f"{manifest['embedding_model']} "
                  f"({manifest['n_vectors']} vectors).")
            return 0

        if args.action == "delete":
            if not args.yes:
                resp = input(f"Delete KB '{args.name}'? [y/N] ").strip().lower()
                if resp not in ("y", "yes"):
                    print("Aborted.")
                    return 0
            kb_store.delete_kb(args.name)
            print(f"🗑️ KB '{args.name}' deleted.")
            return 0

    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"❌ {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
