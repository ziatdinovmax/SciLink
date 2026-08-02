"""CLI entry point for launching the SciLink Streamlit UI."""

import subprocess
import sys
from pathlib import Path


def main():
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    app_path = ui_dir / "app.py"

    # Pass theme settings as CLI flags so Streamlit picks them up
    # regardless of working directory.  This lets us run from the
    # user's cwd (so session directories are created there) instead
    # of from the package's ui/ directory.
    theme_args = []
    config_path = ui_dir / ".streamlit" / "config.toml"
    if config_path.exists():
        try:
            import tomllib
            with open(config_path, "rb") as f:
                config = tomllib.load(f)
            for key, value in config.get("theme", {}).items():
                theme_args.extend([f"--theme.{key}", str(value)])
        except Exception:
            pass

    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            *theme_args,
            "--", *sys.argv[1:],
        ],
    )
