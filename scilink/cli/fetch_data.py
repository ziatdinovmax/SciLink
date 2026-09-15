"""``scilink fetch-xrd-library`` — install the prebuilt XRD fingerprint library.

One-time download of the COD-derived reference library (several hundred MB)
into the persistent per-user store (``~/.scilink/xrd_fingerprints/``), enabling
``search_match_pattern`` — offline fingerprint identification of unknown powder
patterns. Explicit by design: a download this size should be a user decision,
never an implicit side effect of an analysis run.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from scilink.skills.structure_matching.xrd.fingerprint import (
        DEFAULT_LIBRARY_URL, fetch_fingerprint_library, _default_store_path)

    ap = argparse.ArgumentParser(
        prog="scilink fetch-xrd-library",
        description=("Download the prebuilt XRD fingerprint reference library "
                     "(one-time, ~hundreds of MB) for offline powder-pattern "
                     "identification."))
    ap.add_argument("--url", default=None,
                    help=f"artifact URL (default: {DEFAULT_LIBRARY_URL})")
    ap.add_argument("--dest", default=None,
                    help=f"install path (default: {_default_store_path()})")
    ap.add_argument("--sha256", default=None,
                    help="expected checksum (default: the published one)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an already-installed library")
    args = ap.parse_args()

    try:
        info = fetch_fingerprint_library(url=args.url, dest=args.dest,
                                         sha256=args.sha256,
                                         overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"Already installed: {exc}")
        return 0
    except Exception as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1

    print(f"Installed {info['n_entries']} reference entries -> {info['path']}")
    print(f"sha256: {info['sha256']}")
    print("search_match_pattern will now find the library automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
