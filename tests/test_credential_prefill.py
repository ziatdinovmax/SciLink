"""
Hermetic tests for environment-variable prefill of the UI credential form.

Two tiers, no Streamlit required (resolve_prefill lives in the Streamlit-free
scilink.ui.config module):

  Tier 1 — auth.py env discovery helpers (find_env_var, find_env_var_for_model,
           get_internal_proxy_base_url) and their precedence.
  Tier 2 — config.resolve_prefill: the field-by-field resolution that the
           sidebar wraps, with emphasis on the proxy-vs-vendor SAFETY GUARD
           (the proxy key must never fill the main field without a base URL).

Env isolation: the clean_env fixture clears every variable these paths read,
and tests opt back in with monkeypatch.setenv.
"""

import pytest

from scilink import auth
from scilink.ui.config import (
    resolve_prefill, reconcile_autofill, resolve_embedding_prefill,
)


_RELEVANT_VARS = [
    "SCILINK_API_KEY", "SCILINK_BASE_URL",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "FUTUREHOUSE_API_KEY", "MP_API_KEY", "MATERIALS_PROJECT_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every credential env var these code paths read."""
    for var in _RELEVANT_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# ─── Tier 1: auth.py env-discovery helpers ─────────────────────────


def test_find_env_var_returns_name_and_value(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    assert auth.find_env_var("openai") == ("OPENAI_API_KEY", "sk-openai")


def test_find_env_var_none_when_unset(clean_env):
    assert auth.find_env_var("openai") is None
    assert auth.find_env_var("materials_project") is None


def test_find_env_var_precedence_first_listed_wins(clean_env):
    """google lists GEMINI_API_KEY before GOOGLE_API_KEY → GEMINI wins."""
    clean_env.setenv("GOOGLE_API_KEY", "g-google")
    assert auth.find_env_var("google") == ("GOOGLE_API_KEY", "g-google")
    clean_env.setenv("GEMINI_API_KEY", "g-gemini")
    assert auth.find_env_var("google") == ("GEMINI_API_KEY", "g-gemini")


@pytest.mark.parametrize(
    "model, env_var, provider_key",
    [
        ("claude-opus-4-6", "ANTHROPIC_API_KEY", "sk-ant"),
        ("gpt-5.4", "OPENAI_API_KEY", "sk-oai"),
        ("gemini-3.1-pro-preview", "GEMINI_API_KEY", "sk-gem"),
        ("openai/gpt-4o", "OPENAI_API_KEY", "sk-oai2"),
    ],
)
def test_find_env_var_for_model_maps_provider(clean_env, model, env_var, provider_key):
    clean_env.setenv(env_var, provider_key)
    assert auth.find_env_var_for_model(model) == (env_var, provider_key)


def test_find_env_var_for_model_unknown_or_empty(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert auth.find_env_var_for_model("") is None
    assert auth.find_env_var_for_model("some-local-llama") is None


def test_get_internal_proxy_base_url(clean_env):
    assert auth.get_internal_proxy_base_url() is None
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy.example/v1")
    assert auth.get_internal_proxy_base_url() == "https://proxy.example/v1"
    assert auth.INTERNAL_PROXY_BASE_URL == "SCILINK_BASE_URL"


# ─── Tier 2: resolve_prefill field resolution ──────────────────────


def test_vendor_key_resolves_for_model(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")
    assert out["base_url"] == ("", None)


def test_vendor_key_matches_chosen_provider_not_others(clean_env):
    """With multiple vendor keys set, only the model's provider key is used."""
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert resolve_prefill("gpt-5.4")["api_key"] == ("sk-oai", "OPENAI_API_KEY")
    assert resolve_prefill("claude-opus-4-6")["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")


def test_proxy_pair_used_when_base_url_in_env(clean_env):
    """SCILINK_API_KEY + SCILINK_BASE_URL → proxy key fills the main field."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy/v1")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("proxy-key", "SCILINK_API_KEY")
    assert out["base_url"] == ("https://proxy/v1", "SCILINK_BASE_URL")


def test_proxy_key_used_when_base_url_already_entered(clean_env):
    """A base URL the user already typed also enables the proxy path."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    out = resolve_prefill("claude-opus-4-6", existing_base_url="https://typed/v1")
    assert out["api_key"] == ("proxy-key", "SCILINK_API_KEY")
    # No SCILINK_BASE_URL env → base_url field itself is not prefilled.
    assert out["base_url"] == ("", None)


def test_provider_match_wins_over_proxy_without_base_url(clean_env):
    """With no base URL, a vendor key matching the model's provider is preferred
    over the proxy key (correct pairing without a manual base URL)."""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    out = resolve_prefill("claude-opus-4-6")  # no base url anywhere
    assert out["api_key"] == ("sk-ant", "ANTHROPIC_API_KEY")


def test_proxy_key_surfaced_without_base_url_when_no_vendor(clean_env):
    """Proxy key set, no base URL, no vendor key → the proxy key IS surfaced so
    the field is populated and the session can start. (The sidebar warns that a
    base URL is still needed; the backend enforces proxy-vs-vendor safety.)"""
    clean_env.setenv("SCILINK_API_KEY", "proxy-key")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("proxy-key", "SCILINK_API_KEY")


def test_no_prefill_when_provider_key_missing(clean_env):
    """A vendor key that does NOT match the selected model's provider must not
    be borrowed: default model is claude, only GOOGLE is set → field stays
    empty rather than prefilling the wrong key."""
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    out = resolve_prefill("claude-opus-4-6")
    assert out["api_key"] == ("", None)
    # The matching model DOES prefill it.
    assert resolve_prefill("gemini-3.1-pro-preview")["api_key"] == ("g-key", "GOOGLE_API_KEY")


def test_service_keys_resolve_independently_of_model(clean_env):
    clean_env.setenv("FUTUREHOUSE_API_KEY", "fh-key")
    clean_env.setenv("MP_API_KEY", "mp-key")
    out = resolve_prefill("gpt-5.4")
    assert out["fh"] == ("fh-key", "FUTUREHOUSE_API_KEY")
    assert out["mp"] == ("mp-key", "MP_API_KEY")


def test_all_fields_empty_when_nothing_set(clean_env):
    out = resolve_prefill("claude-opus-4-6")
    assert out == {
        "api_key": ("", None),
        "base_url": ("", None),
        "fh": ("", None),
        "mp": ("", None),
    }


def test_base_url_prefilled_independently_of_proxy_key(clean_env):
    """SCILINK_BASE_URL set without SCILINK_API_KEY: base_url is still surfaced,
    and the main field falls back to the vendor key."""
    clean_env.setenv("SCILINK_BASE_URL", "https://proxy/v1")
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    out = resolve_prefill("gpt-5.4")
    assert out["base_url"] == ("https://proxy/v1", "SCILINK_BASE_URL")
    assert out["api_key"] == ("sk-oai", "OPENAI_API_KEY")


# ─── reconcile_autofill: dynamic refresh on model/vendor change ────


def test_reconcile_first_seed():
    """Field never existed (None) → adopt the resolved value."""
    assert reconcile_autofill(None, None, "k-anthropic") == ("k-anthropic", "k-anthropic")


def test_reconcile_refreshes_when_unedited():
    """Field still holds the last auto-filled value → switching vendors
    refreshes it to the newly-resolved key. This is the reported bug."""
    assert reconcile_autofill("k-anthropic", "k-anthropic", "k-openai") == ("k-openai", "k-openai")


def test_reconcile_preserves_user_typed_value():
    """A hand-typed value differs from the last auto-fill → never overwritten."""
    assert reconcile_autofill("my-typed-key", "k-anthropic", "k-openai") == ("my-typed-key", "k-anthropic")


def test_reconcile_preserves_cleared_field():
    """A deliberately cleared field (was auto-filled, now empty) stays empty."""
    assert reconcile_autofill("", "k-anthropic", "k-openai") == ("", "k-anthropic")


def test_reconcile_populates_when_provider_gains_key():
    """Field auto-filled empty (no key for prior model) → switching to a model
    whose provider key IS set populates it."""
    assert reconcile_autofill("", "", "k-openai") == ("k-openai", "k-openai")


def test_reconcile_idempotent_when_value_unchanged():
    assert reconcile_autofill("k-anthropic", "k-anthropic", "k-anthropic") == ("k-anthropic", "k-anthropic")


# ─── Embedding model → provider env var ────────────────────────────


def test_infer_provider_recognises_openai_embeddings():
    assert auth.infer_provider("text-embedding-3-small") == "openai"
    assert auth.infer_provider("text-embedding-3-large") == "openai"
    assert auth.infer_provider("text-embedding-ada-002") == "openai"


def test_infer_provider_recognises_google_embeddings():
    assert auth.infer_provider("gemini-embedding-001") == "google"


def test_resolve_embedding_prefill_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert resolve_embedding_prefill("text-embedding-3-small") == ("sk-oai", "OPENAI_API_KEY")


def test_resolve_embedding_prefill_google(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "g-gem")
    assert resolve_embedding_prefill("gemini-embedding-001") == ("g-gem", "GEMINI_API_KEY")


def test_resolve_embedding_prefill_no_match_when_env_missing(clean_env):
    """The matching env var is absent → leave the field empty (no borrowing)."""
    clean_env.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_embedding_prefill("text-embedding-3-small") == ("", None)


def test_resolve_embedding_prefill_no_model_returns_empty(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    assert resolve_embedding_prefill("") == ("", None)
    assert resolve_embedding_prefill(None) == ("", None)


def test_resolve_embedding_prefill_unknown_model_returns_empty(clean_env):
    """A model name whose provider can't be inferred returns empty even when
    vendor env vars are set."""
    clean_env.setenv("OPENAI_API_KEY", "sk-oai")
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert resolve_embedding_prefill("some-local-embedder") == ("", None)


# ─── AWS Bedrock model-name pattern ────────────────────────────────


def test_infer_provider_recognises_bedrock_prefix():
    """bedrock/<vendor>.<model> is the LiteLLM convention for AWS Bedrock."""
    assert auth.infer_provider("bedrock/anthropic.claude-3-sonnet-20240229-v1:0") == "bedrock"
    assert auth.infer_provider("bedrock/amazon.titan-text-express-v1") == "bedrock"
    assert auth.infer_provider("bedrock/meta.llama3-70b-instruct-v1:0") == "bedrock"


def test_find_env_var_bedrock(clean_env):
    """The Bedrock provider maps to AWS_BEARER_TOKEN_BEDROCK in ENV_VARS."""
    assert auth.find_env_var("bedrock") is None
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "aws-token")
    assert auth.find_env_var("bedrock") == ("AWS_BEARER_TOKEN_BEDROCK", "aws-token")


def test_resolve_prefill_bedrock_model_uses_bearer_token(clean_env):
    """Selecting a Bedrock model surfaces AWS_BEARER_TOKEN_BEDROCK into the
    API-key field (the gap Sarah called out: previously the field stayed empty
    for Bedrock users)."""
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "aws-token")
    out = resolve_prefill("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    assert out["api_key"] == ("aws-token", "AWS_BEARER_TOKEN_BEDROCK")


def test_resolve_prefill_bedrock_model_no_token_stays_empty(clean_env):
    """Bedrock model selected, bearer token absent → field stays empty rather
    than borrowing another vendor's key."""
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")  # unrelated key set
    out = resolve_prefill("bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    assert out["api_key"] == ("", None)
