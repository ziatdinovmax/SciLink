"""Every non-Python file a skill bundle needs must be named in the package
data. Live: a container built without MANIFEST.in shipped 149 skill .py
files and zero skill .md files, so no skill existed inside it ("Skill
'atomic_stem' not found"). The patterns are checked statically here so the
gap cannot reopen when a new kind of skill data file appears."""
import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "scilink"


def _package_data_patterns():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[tool.setuptools.package-data]", 1)[1]
    block = block.split("\n[", 1)[0]
    return re.findall(r'"([^"]+)"', block.split("=", 1)[1])


def test_every_skill_data_file_is_covered_by_package_data():
    patterns = _package_data_patterns()
    missing = []
    for f in (PKG / "skills").rglob("*"):
        if not f.is_file() or f.suffix in (".py", ".pyc") or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(PKG).as_posix()
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            missing.append(rel)
    assert not missing, f"skill data files not covered by package-data: {missing[:8]}"
    assert any(p.endswith(".md") and p.startswith("skills/") for p in patterns)


def test_dockerfile_copies_the_manifest():
    assert "MANIFEST.in" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
