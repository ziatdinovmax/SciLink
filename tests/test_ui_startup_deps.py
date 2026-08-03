"""UI startup must survive an older Streamlit, and Bedrock must be installable.

Two failure modes reported from real installs:

* ``st.html``'s ``unsafe_allow_javascript`` kwarg landed in Streamlit 1.52.0,
  but the pin allowed 1.37. ``inject_theme()`` runs before anything renders on
  every rerun, so the ``TypeError`` took the whole app down on startup.
* litellm's Bedrock path imports boto3/botocore, which nothing in the base
  install pulled in — the failure surfaced as an opaque ``APIConnectionError:
  No module named 'boto3'`` at first call.
"""
import re
import tomllib
from pathlib import Path

import pytest

from scilink.ui import theme

PYPROJECT = tomllib.loads(Path("pyproject.toml").read_text())
DEPS = PYPROJECT["project"]["dependencies"]
THEME_SRC = Path("scilink/ui/theme.py").read_text()


class _FakeSt:
    """Stand-in for the ``streamlit`` module, old or new."""

    __version__ = "1.40.0"

    def __init__(self, supports_kwarg: bool):
        self._supports_kwarg = supports_kwarg
        self.calls = []
        self.session_state = {}

    def html(self, body, **kwargs):
        if kwargs and not self._supports_kwarg:
            raise TypeError(
                "HtmlMixin.html() got an unexpected keyword argument "
                f"'{next(iter(kwargs))}'"
            )
        self.calls.append(("html", body, kwargs))

    def markdown(self, body, **kwargs):
        self.calls.append(("markdown", body, kwargs))


@pytest.fixture
def old_streamlit(monkeypatch):
    fake = _FakeSt(supports_kwarg=False)
    monkeypatch.setattr(theme, "st", fake)
    monkeypatch.setattr(theme, "_JS_DEGRADED_NOTICE_SHOWN", False, raising=False)
    return fake


@pytest.fixture
def new_streamlit(monkeypatch):
    fake = _FakeSt(supports_kwarg=True)
    monkeypatch.setattr(theme, "st", fake)
    return fake


# --- the wrapper -------------------------------------------------------------

def test_inject_js_uses_the_kwarg_when_streamlit_supports_it(new_streamlit):
    theme._inject_js("<script>/* x */</script>")
    assert new_streamlit.calls == [
        ("html", "<script>/* x */</script>", {"unsafe_allow_javascript": True})
    ]


def test_inject_js_retries_without_the_kwarg_on_old_streamlit(old_streamlit):
    theme._inject_js("<script>/* x */</script>")
    # Effect lost, but the DOM slot the theme relies on is still emitted.
    assert old_streamlit.calls == [("html", "<script>/* x */</script>", {})]


def test_inject_js_warns_once_per_process(old_streamlit, capsys):
    for _ in range(3):
        theme._inject_js("<script>/* x */</script>")
    assert capsys.readouterr().err.count("streamlit>=1.52") == 1


def test_inject_js_swallows_any_other_streamlit_failure(monkeypatch):
    """inject_theme() runs on every rerun; cosmetic JS must never kill it."""
    class _Exploding(_FakeSt):
        def html(self, body, **kwargs):
            raise RuntimeError("some future Streamlit API drift")

    monkeypatch.setattr(theme, "st", _Exploding(supports_kwarg=True))
    theme._inject_js("<script>/* x */</script>")  # must not raise


# --- the crash the issue reported --------------------------------------------

@pytest.mark.parametrize("mode", ["dark", "light"])
def test_inject_theme_does_not_crash_on_old_streamlit(old_streamlit, mode):
    old_streamlit.session_state = {"theme_mode": mode}
    theme.inject_theme()  # pre-fix: TypeError, app down on every page load
    assert old_streamlit.calls


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_slot_count_is_the_same_on_old_and_new_streamlit(monkeypatch, mode):
    """The dark branch emits a no-op script purely to keep the layout stable —
    degrading must not drop a slot and shift the page."""
    counts = []
    for supports_kwarg in (True, False):
        fake = _FakeSt(supports_kwarg)
        fake.session_state = {"theme_mode": mode}
        monkeypatch.setattr(theme, "st", fake)
        theme.inject_theme()
        counts.append(len(fake.calls))
    assert counts[0] == counts[1]


def test_no_unwrapped_kwarg_call_sites_remain():
    """A new direct call site would reintroduce the crash."""
    unwrapped = re.findall(r"st\.html\([^)]*unsafe_allow_javascript", THEME_SRC)
    assert len(unwrapped) == 1, "only _inject_js may pass the kwarg"


# --- the pins ----------------------------------------------------------------

def test_streamlit_pin_covers_the_kwarg():
    pin = next(d for d in DEPS if d.startswith("streamlit"))
    major, minor = (int(p) for p in pin.split(">=")[1].split(".")[:2])
    assert (major, minor) >= (1, 52), f"{pin} predates unsafe_allow_javascript"


def test_boto3_is_a_required_dependency():
    """Bedrock is selectable from the base UI, so its SDK ships with it."""
    assert any(d.split(">=")[0].strip() == "boto3" for d in DEPS)


def test_bedrock_is_still_a_live_provider():
    """If Bedrock ever leaves the base install, the boto3 pin can leave too."""
    from scilink.providers import provider_for

    assert provider_for("bedrock/us.anthropic.claude-opus-4-6-v1").name == "bedrock"
