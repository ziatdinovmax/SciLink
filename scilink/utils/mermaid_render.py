"""Render Mermaid diagram source to PNG/SVG via mermaid-cli.

Optional-dependency seam, same philosophy as the sim stack's `ase`: the
renderer is Node + headless Chromium, so SciLink never requires it —
callers probe `mermaid_available()` and degrade gracefully (documents
simply ship without a diagram) when the toolchain is absent.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

INSTALL_HINT = (
    "Mermaid rendering needs mermaid-cli: `npm install -g "
    "@mermaid-js/mermaid-cli` (or have `npx` on PATH for on-demand use)."
)


def find_mermaid_cli() -> Optional[List[str]]:
    """Command prefix for mermaid-cli, or None if unavailable.

    Prefers an installed ``mmdc``; falls back to ``npx`` (which fetches
    the package on first use — slower once, then cached).
    """
    if shutil.which("mmdc"):
        return ["mmdc"]
    if shutil.which("npx"):
        return ["npx", "--yes", "@mermaid-js/mermaid-cli"]
    return None


def mermaid_available() -> bool:
    return find_mermaid_cli() is not None


def _clean_error(output: str) -> str:
    """Reduce mmdc's noisy stderr to the lines a retry prompt can use."""
    lines = [ln for ln in (output or "").splitlines()
             if ln.strip() and not ln.strip().startswith("at ")]
    keep = [ln for ln in lines
            if re.search(r"error|expect|line \d|parse|unknown|invalid",
                         ln, re.I)]
    return "\n".join((keep or lines)[-8:]) or "mermaid render failed"


def render_mermaid(code: str, out_path, timeout: int = 240,
                   scale: int = 2,
                   background: str = "white") -> Tuple[bool, str]:
    """Render mermaid source to ``out_path`` (suffix selects the format:
    .png / .svg / .pdf). Returns ``(ok, error_message)`` — the error text
    is cleaned for feeding back into a regeneration prompt.
    """
    cli = find_mermaid_cli()
    if cli is None:
        return False, INSTALL_HINT

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", suffix=".mmd", delete=False) as f:
        f.write(code)
        src = f.name
    try:
        proc = subprocess.run(
            cli + ["-i", src, "-o", str(out_path),
                   "-b", background, "-s", str(scale)],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"mermaid render timed out after {timeout}s"
    except OSError as exc:
        return False, f"mermaid-cli could not run: {exc}"
    finally:
        Path(src).unlink(missing_ok=True)

    if proc.returncode != 0 or not out_path.exists():
        return False, _clean_error((proc.stderr or "") + (proc.stdout or ""))
    return True, ""
