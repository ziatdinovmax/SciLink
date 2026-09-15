# scilink/providers.py

"""Per-provider UI fields + credential routing.

Streamlit-free on purpose so core code and the UI can both import it.

Each :class:`ProviderSpec` declares two things:

1. **What extra inputs to collect** in the UI (``fields``) for a model that
   belongs to that provider, and what to call the pasted secret (``key_label``).
2. **How the pasted secret + collected field values map onto LiteLLM**
   (``apply`` -> :class:`ProviderAuth`): some providers want the secret as the
   ``api_key`` kwarg (OpenAI/Anthropic/Gemini), some want it in an environment
   variable (Bedrock's bearer token), and some need extra completion kwargs
   (Azure's ``api_version``, Vertex's project/location).

Only ``bedrock`` and the ``direct`` fallback are live today. Azure / Vertex are
documented stubs: wiring them up additionally needs a ``model_kwargs``
passthrough on the LiteLLM wrapper (see ``ProviderAuth.model_kwargs``), which
Bedrock does not require because its token and region ride environment
variables that boto3 reads automatically.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple


# US AWS regions for Bedrock Claude. Scoped to US for now; add other
# geographies (eu-*, ap-*) when cross-region availability is needed.
BEDROCK_REGIONS: Tuple[str, ...] = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
)


@dataclass(frozen=True)
class ProviderField:
    """A provider-specific input rendered in the UI only when that provider is selected."""

    name: str                       # values key + session-state suffix, e.g. "region"
    label: str                      # UI label, e.g. "AWS region"
    kind: str = "select"            # "select" | "text"
    options: Tuple[str, ...] = ()   # choices for kind == "select"
    default: str = ""               # default value / preselected option
    help: str = ""                  # tooltip


@dataclass
class ProviderAuth:
    """How to authenticate a call, produced by :meth:`ProviderSpec.apply`."""

    api_key: Optional[str] = None                       # -> agent constructor / litellm api_key
    env: Dict[str, str] = field(default_factory=dict)   # -> os.environ updates
    model_kwargs: Dict[str, str] = field(default_factory=dict)  # -> litellm.completion (FUTURE)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    matches: Callable[[str], bool]            # model string -> belongs to this provider?
    apply: Callable[..., ProviderAuth]        # (pasted_key, values, base_url) -> ProviderAuth
    key_label: str = "API key"                # what to call the pasted secret in the UI
    fields: Tuple[ProviderField, ...] = ()    # extra UI fields for this provider
    cred_env: Tuple[str, ...] = ()            # env vars signalling ambient credentials
    cred_error: str = "Provide an API key or set the appropriate environment variable."


# --- routing functions -------------------------------------------------------

def _direct_apply(pasted_key, values, base_url):
    """OpenAI / Anthropic / Gemini: the pasted key is litellm's api_key (today's behavior)."""
    return ProviderAuth(api_key=pasted_key)


def _bedrock_apply(pasted_key, values, base_url):
    """Bedrock: token + region ride env vars; api_key MUST be None.

    If api_key is set, litellm uses it instead of the AWS credential chain and
    Bedrock auth fails silently. boto3 reads AWS_BEARER_TOKEN_BEDROCK and
    AWS_REGION_NAME from the environment, so nothing below the sidebar changes.
    """
    return ProviderAuth(
        api_key=None,
        env={
            "AWS_BEARER_TOKEN_BEDROCK": pasted_key or "",
            "AWS_REGION_NAME": (values.get("region") or "us-east-1"),
        },
    )


# --- registry ----------------------------------------------------------------

PROVIDERS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="bedrock",
        matches=lambda m: m.startswith("bedrock/"),
        apply=_bedrock_apply,
        key_label="Bedrock API key",
        fields=(
            ProviderField(
                name="region",
                label="AWS region",
                kind="select",
                options=BEDROCK_REGIONS,
                default="us-east-1",
                help=("Must match the region your AWS console was set to when "
                      "you generated the key (shown top-right of the console, "
                      "e.g. 'N. Virginia' = us-east-1). The key itself does not "
                      "encode the region."),
            ),
        ),
        cred_env=("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID"),
        cred_error=("Paste your Bedrock API key, or set AWS credentials "
                    "(AWS_BEARER_TOKEN_BEDROCK, or AWS_ACCESS_KEY_ID + "
                    "AWS_SECRET_ACCESS_KEY) in the environment."),
    ),

    # ---- FUTURE (stubbed) — each additionally needs the model_kwargs ----
    # ---- passthrough on LiteLLMGenerativeModel (see module docstring). ----
    #
    # def _azure_apply(pasted_key, values, base_url):
    #     # endpoint reuses the existing Base URL field (passed as api_base)
    #     return ProviderAuth(
    #         api_key=pasted_key,
    #         model_kwargs={"api_version": values.get("api_version") or "2024-02-15-preview"},
    #     )
    #
    # ProviderSpec(
    #     name="azure",
    #     matches=lambda m: m.startswith("azure/"),
    #     apply=_azure_apply,
    #     fields=(ProviderField("api_version", "API version", "text",
    #                           default="2024-02-15-preview"),),
    # ),
    #
    # def _vertex_apply(pasted_key, values, base_url):
    #     return ProviderAuth(
    #         api_key=None,
    #         env={"GOOGLE_APPLICATION_CREDENTIALS": pasted_key or ""},
    #         model_kwargs={"vertex_project": values.get("project"),
    #                       "vertex_location": values.get("location") or "us-central1"},
    #     )
    #
    # ProviderSpec(
    #     name="vertex_ai",
    #     matches=lambda m: m.startswith("vertex_ai/"),
    #     apply=_vertex_apply,
    #     key_label="Path to service-account JSON",
    #     fields=(ProviderField("project", "GCP project", "text"),
    #             ProviderField("location", "Location", "text", default="us-central1")),
    # ),
)


_DEFAULT = ProviderSpec(
    name="direct",
    matches=lambda m: True,
    apply=_direct_apply,
    cred_env=("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
    cred_error=("Provide an API key or set an environment variable "
                "(GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)."),
)


def provider_for(model: str) -> ProviderSpec:
    """Return the spec for ``model``'s provider, or the direct-key fallback."""
    model = model or ""
    for spec in PROVIDERS:
        if spec.matches(model):
            return spec
    return _DEFAULT
