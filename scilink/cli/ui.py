"""CLI entry point for the SciLink web UI.

Default: the React web UI (served by the FastAPI backend, `scilink-web`).
`--streamlit` launches the classic Streamlit app instead. If the React bundle
is not built (a source checkout without `scripts/build_webui.sh`), it falls
back to Streamlit with a note — release wheels ship the bundle, so a normal
`pip install scilink` gets React with no extra step.
"""

import subprocess
import sys
from pathlib import Path


def _react_bundle_present() -> bool:
    """True if a built React bundle exists (repo `webui/dist` or the wheel's
    `server/static`). Mirrors the resolution in server/app.py."""
    server = Path(__file__).resolve().parent.parent / "server"
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "webui" / "dist",
        server / "static",
    ]
    return any((d / "index.html").is_file() for d in candidates)


def _run_streamlit(argv) -> int:
    """Launch the classic Streamlit app."""
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    app_path = ui_dir / "app.py"
    theme_args = []
    config_path = ui_dir / ".streamlit" / "config.toml"
    if config_path.exists():
        try:
            import tomllib
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            for key, value in config.get("theme", {}).items():
                theme_args.extend([f"--theme.{key}", str(value)])
        except Exception:  # noqa: BLE001
            pass
    return subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         *theme_args, "--", *argv]
    ).returncode


def main() -> int:
    argv = list(sys.argv[1:])

    if "--streamlit" in argv:
        argv = [a for a in argv if a != "--streamlit"]
        return _run_streamlit(argv)

    # React is the default. Fall back to Streamlit when the bundle is not
    # built (source checkout) so the UI still works without Node.
    if not _react_bundle_present():
        print(
            "scilink ui: the React web UI bundle is not built (source "
            "checkout) — launching the classic Streamlit UI instead.\n"
            "  Build the React UI once with:  scripts/build_webui.sh\n"
            "  (release wheels — pip install scilink — ship it already,\n"
            "   and `scilink ui --streamlit` always uses the classic UI).",
            file=sys.stderr,
        )
        return _run_streamlit(argv)

    try:
        from scilink.server.cli import main as web_main
    except ImportError as exc:  # pragma: no cover - web deps are core now
        print(f"scilink ui: web backend unavailable ({exc}); "
              "falling back to Streamlit.", file=sys.stderr)
        return _run_streamlit(argv)
    return web_main(argv)


if __name__ == "__main__":
    sys.exit(main())
