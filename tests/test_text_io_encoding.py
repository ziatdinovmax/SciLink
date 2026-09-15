"""UTF-8 text I/O helpers and the literature path that now uses them.

The bug these guard against: `open(path, 'w')` with no `encoding=` writes in
the locale encoding, so on a cp1252 Windows machine persisting a literature
answer containing "→" raised UnicodeEncodeError and discarded a completed
Edison search. The two properties that matter are (1) writes are UTF-8
regardless of locale and (2) nothing changes on a system that was already
UTF-8 — so the byte-identity test below is as load-bearing as the fix.
"""
import ast
import locale
import logging
from pathlib import Path

import pytest

from scilink.utils.text_io import read_text_utf8, write_text_utf8


# The characters that actually broke it live, plus the general suspects.
SCIENTIFIC_TEXT = (
    "# Literature Search Results (hypothesis_context)\n\n"
    "Kr/Xe sieving in CO₂-activated MOFs; ΔH ≥ 30 kJ mol⁻¹; "
    "aperture → 4 Å; −196 °C; μ-pore; α-phase.\n"
)


# ---------------------------------------------------------------- write side

def test_write_is_utf8_regardless_of_locale(tmp_path):
    p = write_text_utf8(tmp_path / "lit.md", SCIENTIFIC_TEXT)
    assert p.read_bytes().decode("utf-8") == SCIENTIFIC_TEXT


def test_write_survives_the_character_that_broke_it(tmp_path):
    # U+2192 at the reported crash, U+2082 at the one before it.
    for ch in ("→", "₂", "−", "≥", "Å"):
        p = write_text_utf8(tmp_path / f"c_{ord(ch)}.md", f"x {ch} y")
        assert ch in p.read_text(encoding="utf-8")


def test_append_extends_rather_than_truncates(tmp_path):
    p = tmp_path / "lit.md"
    write_text_utf8(p, "first ₂\n")
    write_text_utf8(p, "second →\n", append=True)
    assert p.read_text(encoding="utf-8") == "first ₂\nsecond →\n"


def test_append_creates_a_missing_file(tmp_path):
    p = write_text_utf8(tmp_path / "new.md", "only →", append=True)
    assert p.read_text(encoding="utf-8") == "only →"


def test_returns_a_path(tmp_path):
    assert write_text_utf8(str(tmp_path / "x.md"), "t") == tmp_path / "x.md"


# --------------------------------------------- no-op on an already-UTF-8 OS

@pytest.mark.skipif(
    (locale.getpreferredencoding(False) or "").lower().replace("-", "")
    not in ("utf8",),
    reason="byte-identity only claimed where the locale was already UTF-8",
)
@pytest.mark.parametrize("text", [
    "",
    "plain ascii\n",
    SCIENTIFIC_TEXT,
    "trailing newlines\n\n\n",
    "embedded\r\ncrlf\r\n",
    "no trailing newline",
])
def test_byte_identical_to_the_bare_call_it_replaces(tmp_path, text):
    """The fix must not alter working behaviour on Linux/macOS.

    Same bytes, including newline translation — which is why the helper
    leaves `newline` at the platform default instead of pinning it.
    """
    before = tmp_path / "before.md"
    with open(before, "w") as f:  # what the code did prior to the fix
        f.write(text)
    after = write_text_utf8(tmp_path / "after.md", text)
    assert after.read_bytes() == before.read_bytes()


@pytest.mark.skipif(
    (locale.getpreferredencoding(False) or "").lower().replace("-", "")
    not in ("utf8",),
    reason="byte-identity only claimed where the locale was already UTF-8",
)
def test_append_byte_identical_to_the_bare_call(tmp_path):
    before, after = tmp_path / "b.md", tmp_path / "a.md"
    for target, writer in ((before, None), (after, write_text_utf8)):
        for chunk in ("head ₂\n", "tail →\n"):
            if writer is None:
                with open(target, "a") as f:
                    f.write(chunk)
            else:
                writer(target, chunk, append=True)
    assert after.read_bytes() == before.read_bytes()


# ----------------------------------------------------------------- read side

def test_read_roundtrips_what_write_produced(tmp_path):
    p = write_text_utf8(tmp_path / "lit.md", SCIENTIFIC_TEXT)
    assert read_text_utf8(p) == SCIENTIFIC_TEXT


def test_read_recovers_a_legacy_cp1252_file_on_a_cp1252_host(
        tmp_path, monkeypatch, caplog):
    """Artifacts an older SciLink wrote on Windows must still load there.

    An em dash is byte 0x97 in cp1252, which is not valid UTF-8 — so a
    strict read would newly break files that used to open fine. The locale
    rung is what recovers them, simulated here so the Windows behaviour is
    covered from any host.
    """
    real_read_text = Path.read_text

    def locale_is_cp1252(self, encoding=None, *a, **kw):
        return real_read_text(self, encoding=encoding or "cp1252", *a, **kw)

    monkeypatch.setattr(Path, "read_text", locale_is_cp1252)

    p = tmp_path / "legacy.md"
    p.write_bytes("cost — high".encode("cp1252"))
    with caplog.at_level(logging.WARNING):
        text = read_text_utf8(p)
    assert text == "cost — high"  # em dash recovered exactly, not mangled
    assert any("not valid UTF-8" in r.message for r in caplog.records)


def test_read_of_a_legacy_file_never_raises_on_a_utf8_host(tmp_path, caplog):
    """Cross-machine, the encoding is unknowable — degrade, do not crash.

    Bare `read_text()` raised UnicodeDecodeError here, so lossy-with-warning
    is strictly better than what it replaces.
    """
    p = tmp_path / "legacy.md"
    p.write_bytes("cost — high".encode("cp1252"))
    with caplog.at_level(logging.WARNING):
        text = read_text_utf8(p)
    assert "cost" in text and "high" in text
    assert caplog.records


def test_read_degrades_lossily_rather_than_raising(tmp_path, caplog):
    p = tmp_path / "binary.md"
    p.write_bytes(b"start \xff\xfe\xfd end")
    with caplog.at_level(logging.WARNING):
        text = read_text_utf8(p)
    assert "start" in text and "end" in text


def test_read_accepts_str_paths(tmp_path):
    p = write_text_utf8(tmp_path / "lit.md", "text →")
    assert read_text_utf8(str(p)) == "text →"


# --------------------------------------------------- the literature path itself

LITERATURE_IO_MODULES = [
    "scilink/agents/planning_agents/orchestrator_tools.py",
    "scilink/agents/exp_agents/analysis_orchestrator_tools.py",
]


@pytest.mark.parametrize("rel", LITERATURE_IO_MODULES)
def test_no_unencoded_writes_left_in_the_literature_writers(rel):
    """Every markdown write in these modules names its encoding.

    Regression guard: the reported crash was one such call. JSON writes are
    exempt — `json.dump` defaults to ensure_ascii=True, so they are already
    cp1252-safe, and there are many of them.
    """
    src = (Path(__file__).resolve().parents[1] / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            continue
        mode = (node.args[1].value
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                else "r")
        if not any(m in str(mode) for m in ("w", "a", "x")):
            continue
        if any(kw.arg == "encoding" for kw in node.keywords):
            continue
        target = getattr(node.args[0], "id", None) if node.args else None
        if target in ("lit_path", "mol_path"):
            offenders.append((rel, node.lineno, target))
    assert not offenders, f"unencoded literature writes remain: {offenders}"


def test_literature_writers_import_the_helper():
    for rel in LITERATURE_IO_MODULES:
        src = (Path(__file__).resolve().parents[1] / rel).read_text(
            encoding="utf-8")
        assert "from ...utils.text_io import" in src, rel
