"""Which key a command-line entry point builds its agents with.

One rule (``auth.resolve_llm_key``) for the terminal shell, headless ``-p``
and ``scilink serve``: the key of the model's own vendor, the proxy key only
beside a base URL, none for Bedrock. ``scilink serve`` used to take the first
of SCILINK / GEMINI / OPENAI / ANTHROPIC that was set, whatever the model.
"""
import sys

import pytest

from scilink import auth
from scilink.cli.shell import bootstrap

_KEY_VARS = ("SCILINK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
             "ANTHROPIC_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))      # no real credentials.env
    return monkeypatch


def _all_keys(env):
    env.setenv("SCILINK_API_KEY", "proxy-key")
    env.setenv("GEMINI_API_KEY", "gemini-key")
    env.setenv("OPENAI_API_KEY", "openai-key")
    env.setenv("ANTHROPIC_API_KEY", "anthropic-key")


# ── the rule ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model, key, source", [
    ("claude-opus-5-5", "anthropic-key", "ANTHROPIC_API_KEY"),
    ("anthropic/claude-opus-5-5", "anthropic-key", "ANTHROPIC_API_KEY"),
    ("gpt-6", "openai-key", "OPENAI_API_KEY"),
    ("gemini/gemini-4-argon", "gemini-key", "GEMINI_API_KEY"),
])
def test_each_model_gets_its_own_vendors_key(clean_env, model, key, source):
    _all_keys(clean_env)
    assert auth.resolve_llm_key(model) == (key, source)


def test_proxy_key_only_beside_a_base_url(clean_env):
    _all_keys(clean_env)
    assert auth.resolve_llm_key("claude-opus-5-5", "https://proxy/v1") == \
        ("proxy-key", "SCILINK_API_KEY")
    # with every other key unset the proxy key still never goes to a vendor
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        clean_env.delenv(var)
    assert auth.resolve_llm_key("claude-opus-5-5") == (None, None)


def test_a_vendor_key_never_goes_to_the_proxy(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    assert auth.resolve_llm_key("claude-opus-5-5", "https://proxy/v1") == (None, None)


def test_another_vendors_key_is_not_used(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "gemini-key")
    assert auth.resolve_llm_key("claude-opus-5-5") == (None, None)


def test_bedrock_carries_no_key(clean_env):
    _all_keys(clean_env)
    model = "bedrock/us.anthropic.claude-opus-4-8"
    assert auth.resolve_llm_key(model) == (None, None)
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    assert auth.resolve_llm_key(model) == (None, "AWS environment")
    clean_env.delenv("AWS_BEARER_TOKEN_BEDROCK")
    clean_env.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    assert auth.resolve_llm_key(model) == (None, "AWS environment")


def test_an_explicit_key_wins(clean_env):
    _all_keys(clean_env)
    assert auth.resolve_llm_key("claude-opus-5-5", None, "given") == ("given", "flag")
    assert auth.resolve_llm_key("claude-opus-5-5", "https://proxy/v1", "given") == \
        ("given", "flag")


# ── the shell asks only when the rule finds nothing ──────────────

def _asker(answer=""):
    asked = []

    def ask(prompt="", *, secret=False, default=""):
        asked.append(prompt)
        return answer
    return ask, asked


def test_shell_does_not_ask_when_the_environment_answers(clean_env):
    _all_keys(clean_env)
    ask, asked = _asker()
    creds = bootstrap.resolve_llm_credentials("claude-opus-5-5", None, None, ask)
    assert (creds.api_key, creds.source, asked) == ("anthropic-key", "ANTHROPIC_API_KEY", [])
    creds = bootstrap.resolve_llm_credentials("gpt-6", "https://proxy/v1", None, ask)
    assert (creds.api_key, creds.base_url, creds.source) == \
        ("proxy-key", "https://proxy/v1", "SCILINK_API_KEY")
    assert asked == []


def test_shell_asks_for_what_is_missing(clean_env):
    ask, asked = _asker("typed")
    creds = bootstrap.resolve_llm_credentials("claude-opus-5-5", None, None, ask)
    assert (creds.api_key, creds.source) == ("typed", "prompt") and len(asked) == 1

    ask, asked = _asker("")
    creds = bootstrap.resolve_llm_credentials("claude-opus-5-5", None, None, ask)
    assert (creds.api_key, creds.source) == (None, "auto")
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.resolve_llm_credentials("claude-opus-5-5", "https://proxy/v1", None, ask)

    ask, asked = _asker("bearer")
    creds = bootstrap.resolve_llm_credentials("bedrock/us.anthropic.claude-opus-4-8",
                                              None, None, ask)
    assert (creds.api_key, creds.source) == (None, "prompt")
    import os
    assert os.environ.pop("AWS_BEARER_TOKEN_BEDROCK") == "bearer"


# ── scilink serve ────────────────────────────────────────────────

pytest.importorskip("mcp")
from scilink.cli import serve  # noqa: E402


def test_serve_warns_and_starts_without_a_vendor_key(clean_env, capsys):
    clean_env.setenv("GEMINI_API_KEY", "gemini-key")
    assert serve.resolve_api_key("claude-opus-5-5", None, None) is None
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "gemini-key" not in err


def test_serve_refuses_a_proxy_without_its_key(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    with pytest.raises(ValueError, match="SCILINK_API_KEY"):
        serve.resolve_api_key("claude-opus-5-5", "https://proxy/v1", None)


def _run_serve(monkeypatch, argv):
    """``scilink serve`` up to the transport, with the server faked; returns
    what ``create_server`` was called with."""
    import scilink.mcp_server as ms
    seen = {}

    class _Server:
        def eager_init(self):
            pass

    def fake_create_server(**kwargs):
        seen.update(kwargs)
        return _Server()

    async def fake_run_stdio(server, real_stdout=None):
        return None

    monkeypatch.setattr(ms, "create_server", fake_create_server)
    monkeypatch.setattr(ms, "run_stdio", fake_run_stdio)
    monkeypatch.setattr(sys, "argv", ["scilink serve", *argv])
    monkeypatch.setattr(sys, "stdout", sys.stdout)       # main() redirects it
    code = serve.main()
    return code, seen


def test_serve_passes_the_models_key_not_the_first_one_set(clean_env):
    _all_keys(clean_env)
    code, seen = _run_serve(clean_env, ["--model", "claude-opus-5-5"])
    assert code == 0 and seen["api_key"] == "anthropic-key"
    code, seen = _run_serve(clean_env, ["--model", "gpt-6"])
    assert seen["api_key"] == "openai-key"
    code, seen = _run_serve(clean_env, ["--model", "claude-opus-5-5",
                                        "--base-url", "https://proxy/v1"])
    assert seen["api_key"] == "proxy-key" and seen["base_url"] == "https://proxy/v1"


def test_serve_bedrock_passes_no_key(clean_env):
    _all_keys(clean_env)
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    code, seen = _run_serve(clean_env, ["--model", "bedrock/us.anthropic.claude-opus-4-8"])
    assert code == 0 and seen["api_key"] is None


def test_serve_reads_the_key_from_the_credentials_file(clean_env, tmp_path):
    """The file is loaded BEFORE the key is resolved: it used to be read
    inside create_server, after the environment had already been scanned."""
    d = tmp_path / ".scilink"
    d.mkdir()
    (d / "credentials.env").write_text("ANTHROPIC_API_KEY=from-file\n")
    clean_env.setenv("GEMINI_API_KEY", "gemini-key")
    code, seen = _run_serve(clean_env, ["--model", "claude-opus-5-5"])
    assert code == 0 and seen["api_key"] == "from-file"
    clean_env.delenv("ANTHROPIC_API_KEY", raising=False)   # set by the loader


def test_serve_exits_on_a_proxy_without_its_key(clean_env, capsys):
    code, seen = _run_serve(clean_env, ["--base-url", "https://proxy/v1"])
    assert code == 1 and seen == {}
    assert "SCILINK_API_KEY" in capsys.readouterr().err
