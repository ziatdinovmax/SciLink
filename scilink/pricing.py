"""A dollar estimate for model calls, on the routes whose price SciLink knows.

Those are a provider's own API called directly: Anthropic, OpenAI, Google
Gemini, and Amazon Bedrock (``PRICED_ROUTES``). There the model id names the
model and the route, so its list price is known. A call sent through a base
URL is a proxy or a compatible server that bills by its own rules, whatever
the model is called, and gets no estimate: the caller says when it passed
one (``proxied``), and a base URL the provider's client would pick up from
the environment counts the same (``_custom_base``: the wrapper's own rule,
which compares the URL's host with the provider's). Other routes (Azure
deployments, Vertex, a local server) get none either.

The figure is LiteLLM's. A call made through LiteLLM carries the cost LiteLLM
computed for the model and route it resolved, and the wrapper passes that on.
``estimate_cost`` is the same table read from token counts, for a call
whose response carries no cost. The table is read for the id as it is called: a Bedrock
geo inference profile (``us.``, ``eu.``) has its own entry, priced above the
global profile, and cached input is priced apart from plain input
(``prompt_tokens`` includes what the prompt cache read and wrote).

The table is LiteLLM's too, and it is not pinned here: LiteLLM downloads the
current one when it is imported and falls back to the copy shipped with the
installed package when it cannot (``LITELLM_LOCAL_MODEL_COST_MAP=True`` asks
for that copy outright). A model the table does not hold gets no estimate,
never a guess.

It is an estimate, never the bill: list prices, no discount, commitment or
tax, and only the chat-model calls SciLink's wrappers made (embeddings and
other services are not counted).

The words a surface shows are in ``scilink/ui/vocabulary.py`` (``COST_WORDS``);
LiteLLM is imported on first use.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional


# The routes with a known list price: the provider's own API, called directly.
PRICED_ROUTES = ("anthropic", "openai", "gemini", "bedrock")

def _normalized(model: str) -> str:
    """``model`` with the provider prefix the wrapper would send it under."""
    try:
        from .wrappers.litellm_wrapper import _normalize_model_name
        return _normalize_model_name(model)
    except Exception:
        return model


def route(model) -> Optional[str]:
    """The provider a model id is sent to (``claude-opus-5-5`` → ``anthropic``,
    ``bedrock/us.…`` → ``bedrock``), or None when the id names none."""
    if not isinstance(model, str) or not model:
        return None
    name = _normalized(model)
    if "/" in name:
        return name.split("/", 1)[0].lower()
    # a bare id the wrapper's name rule does not know (``o3``, ``o4-mini``)
    # goes where LiteLLM's table says it does
    try:
        import litellm
        provider = (litellm.model_cost.get(name) or {}).get("litellm_provider")
    except Exception:
        provider = None
    return str(provider).lower() if provider else None


def _custom_base(provider: str) -> Optional[str]:
    """A base URL the provider's client would use although the call passed
    none, the provider's own host counting as none. The rule is the
    wrapper's (``litellm_wrapper._custom_base``), which also decides the
    route of the request, so the price and the route cannot disagree."""
    try:
        from .wrappers.litellm_wrapper import _custom_base as base_of
    except Exception:
        return None
    base = base_of(provider)
    return str(base) if base else None


def priced(model, proxied: bool = False) -> bool:
    """Whether a call to ``model`` gets an estimate: a provider's own API
    (``PRICED_ROUTES``), not sent through a base URL."""
    provider = route(model)
    if proxied or provider not in PRICED_ROUTES:
        return False
    return _custom_base(provider) is None


@lru_cache(maxsize=256)
def _in_table(model: str) -> bool:
    """Whether LiteLLM's table holds ``model``, asked once per model: a model
    it does not hold raises inside LiteLLM on every lookup."""
    try:
        import litellm
        litellm.cost_per_token(model=_normalized(model), prompt_tokens=1, completion_tokens=1)
        return True
    except Exception:
        return False


def estimate_cost(model, prompt_tokens=0, completion_tokens=0,
                  cache_read_tokens=0, cache_write_tokens=0, proxied: bool = False) -> Optional[float]:
    """Dollars for the given tokens, or None when the route or the model has
    no known price. ``prompt_tokens`` includes the cached tokens. Never raises."""
    if not priced(model, proxied) or not _in_table(model):
        return None
    try:
        import litellm
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=_normalized(model),
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            cache_read_input_tokens=int(cache_read_tokens or 0),
            cache_creation_input_tokens=int(cache_write_tokens or 0))
        return float(prompt_cost) + float(completion_cost)
    except Exception:
        return None


def response_cost(model, response, proxied: bool = False) -> Optional[float]:
    """The cost LiteLLM put on ``response`` for a priced route, else None
    (the caller then falls back to ``estimate_cost``)."""
    if not priced(model, proxied):
        return None
    try:
        cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
    except Exception:
        return None
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0:
        return float(cost)
    return None


def amount(cost_usd: float) -> str:
    cost = float(cost_usd or 0.0)
    if 0 < cost < 0.005:
        return "<$0.01"
    return f"${cost:,.2f}"


def cost_text(cost_usd, calls=None, priced_calls=None) -> str:
    """The figure as every surface shows it: ``≈ $1.23``; when only some of
    the calls had a price, ``≈ $1.23 for 4 of 6 calls``; nothing when none
    had one (a total of $0 that covers no call would read as "free")."""
    from .ui.vocabulary import COST_WORDS
    n_priced = int(priced_calls or 0)
    if not n_priced:
        return ""
    text = COST_WORDS["figure"].format(amount=amount(cost_usd))
    n_calls = int(calls or 0)
    if n_calls > n_priced:
        text = COST_WORDS["partial"].format(figure=text, priced=n_priced, calls=n_calls)
    return text
