"""Process-wide UTF-8: the entry-point guard and the no-unencoded-writes rule.

Round one pinned the encoding on the literature path; the crash simply moved
downstream to the next writer that had never named an encoding. There are
~190 such writes, so the fix has to be categorical rather than per-site:
every text write states UTF-8 (checked here for the whole package), and the
CLI additionally restarts under UTF-8 mode so reads, third-party writers and
anything added later are covered too.
"""
import ast
import pathlib
import sys

import pytest

import importlib

# scilink/cli/__init__.py does `from .main import main`, which rebinds the
# attribute `main` on the package to the FUNCTION — so both
# `from scilink.cli import main` and `import scilink.cli.main as m` hand back
# the function, not the module. Go through importlib to get the module.
cli_main = importlib.import_module("scilink.cli.main")


PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "scilink"


def _unencoded_text_writes():
    """Every text-mode write in the package that does not name an encoding."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - would fail elsewhere first
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if any(kw.arg == "encoding" for kw in node.keywords):
                continue
            rel = path.relative_to(PACKAGE.parent)
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                mode = (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else next((kw.value.value for kw in node.keywords
                               if kw.arg == "mode"
                               and isinstance(kw.value, ast.Constant)), "r")
                )
                mode = str(mode)
                if "b" not in mode and any(c in mode for c in "wax"):
                    offenders.append(f"{rel}:{node.lineno} open({mode!r})")
            elif (isinstance(node.func, ast.Attribute)
                  and node.func.attr == "write_text"):
                offenders.append(f"{rel}:{node.lineno} write_text")
    return offenders


def test_no_text_write_in_the_package_omits_its_encoding():
    """The categorical rule, enforced package-wide.

    A bare `open(p, 'w')` inherits the locale encoding, which is cp1252 on a
    Western-European Windows install — where the scientific characters this
    codebase routinely produces cannot be encoded at all. Name the encoding
    (or use scilink.utils.text_io) and this passes.
    """
    offenders = _unencoded_text_writes()
    assert not offenders, (
        f"{len(offenders)} text write(s) without encoding=; these crash on a "
        f"cp1252 Windows locale:\n  " + "\n  ".join(offenders[:25])
    )


# ------------------------------------------------------------ the CLI guard

class _FakeCompleted:
    def __init__(self, returncode=0):
        self.returncode = returncode


@pytest.fixture
def utf8_probe(monkeypatch):
    """Drive ensure_utf8_mode's inputs and capture whether it re-execs."""
    calls = {}

    def fake_run(cmd, env=None, **kw):
        calls["cmd"] = cmd
        calls["env"] = env
        return _FakeCompleted(0)

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("SCILINK_UTF8_RESTARTED", raising=False)
    return calls


def _pretend_locale(monkeypatch, encoding, utf8_mode=0):
    import locale
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: encoding)
    monkeypatch.setattr(
        sys, "flags",
        type("F", (), {**{f: getattr(sys.flags, f) for f in dir(sys.flags)
                          if not f.startswith("_")},
                       "utf8_mode": utf8_mode})(),
    )


@pytest.mark.parametrize("encoding", ["UTF-8", "utf-8", "utf8"])
def test_noop_when_the_locale_is_already_utf8(monkeypatch, utf8_probe, encoding):
    """Linux and macOS must be untouched — no restart, no output."""
    _pretend_locale(monkeypatch, encoding)
    cli_main.ensure_utf8_mode()
    assert "cmd" not in utf8_probe


def test_noop_when_already_running_in_utf8_mode(monkeypatch, utf8_probe):
    _pretend_locale(monkeypatch, "cp1252", utf8_mode=1)
    cli_main.ensure_utf8_mode()
    assert "cmd" not in utf8_probe


def test_restarts_under_utf8_on_a_cp1252_locale(monkeypatch, utf8_probe):
    _pretend_locale(monkeypatch, "cp1252")
    monkeypatch.setattr(sys, "argv", ["scilink", "plan", "--data-dir", "x"])
    with pytest.raises(SystemExit) as exc:
        cli_main.ensure_utf8_mode()
    assert exc.value.code == 0
    cmd = utf8_probe["cmd"]
    assert cmd[0] == sys.executable
    assert "-X" in cmd and "utf8" in cmd
    assert cmd[-3:] == ["plan", "--data-dir", "x"]  # argv preserved
    assert utf8_probe["env"]["PYTHONUTF8"] == "1"
    assert utf8_probe["env"]["SCILINK_UTF8_RESTARTED"] == "1"
    # `-m` would otherwise make argv[0] the module path, which every
    # subcommand's argparse turns into the program name in its usage line.
    assert utf8_probe["env"]["SCILINK_ARGV0"] == "scilink"


def test_child_exit_code_propagates(monkeypatch, utf8_probe):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted(3))
    _pretend_locale(monkeypatch, "cp1252")
    with pytest.raises(SystemExit) as exc:
        cli_main.ensure_utf8_mode()
    assert exc.value.code == 3


def test_does_not_restart_twice(monkeypatch, utf8_probe, capsys):
    """The re-exec flag is the loop breaker: warn once, then carry on."""
    _pretend_locale(monkeypatch, "cp1252")
    monkeypatch.setenv("SCILINK_UTF8_RESTARTED", "1")
    cli_main.ensure_utf8_mode()  # must return, not exit
    assert "cmd" not in utf8_probe
    captured = capsys.readouterr()
    # stdout is the MCP transport under `scilink serve` — warn on stderr only.
    assert "PYTHONUTF8" in captured.err
    assert captured.out == ""


def test_a_failed_restart_does_not_block_startup(monkeypatch, capsys):
    import subprocess

    def boom(*a, **k):
        raise OSError("no exec for you")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.delenv("SCILINK_UTF8_RESTARTED", raising=False)
    _pretend_locale(monkeypatch, "cp1252")
    cli_main.ensure_utf8_mode()  # returns rather than raising
    captured = capsys.readouterr()
    assert "PYTHONUTF8" in captured.err
    assert captured.out == ""


def test_restored_argv0_keeps_the_program_name_in_usage(monkeypatch):
    """The child restores argv[0] so `--help` still says "scilink"."""
    monkeypatch.setenv("SCILINK_ARGV0", "scilink")
    monkeypatch.setattr(sys, "argv", ["/site-packages/scilink/cli/main.py", "help"])
    _pretend_locale(monkeypatch, "cp1252", utf8_mode=1)  # already restarted
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cli_main.main()
    assert sys.argv[0].startswith("scilink")
