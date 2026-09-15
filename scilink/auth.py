# scilink/auth.py

"""
API key management with environment variable auto-discovery.
"""

import os
from typing import Optional, Dict


# Environment variable names per provider.
#
# To register a new provider for credential prefill (three independent rungs —
# populate only the ones the vendor actually needs):
#
#   1. Add an entry here: provider identity (lowercase) -> env var name(s) in
#      preference order. List multiple names when a vendor accepts more than
#      one (see 'google', which honours both GEMINI_API_KEY and GOOGLE_API_KEY).
#      Note that the env-var name is NOT derivable from the provider key in
#      general — 'bedrock' uses AWS_BEARER_TOKEN_BEDROCK, 'materials_project'
#      uses MP_API_KEY — so the explicit mapping is load-bearing, not redundant.
#   2. If LiteLLM names this vendor's models with a "<prefix>/..." form, add
#      the prefix -> provider mapping in `infer_provider`'s prefix_map. The
#      prefix need not equal the provider identity ('gemini' and 'google'
#      prefixes both resolve to provider 'google').
#   3. If unprefixed model names should also map (e.g. 'gpt-5' -> openai,
#      'claude-opus-*' -> anthropic), add a startswith / substring check in
#      `infer_provider`.
ENV_VARS = {
    # LLM Providers
    'google': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    # AWS Bedrock uses a single bearer token; full AWS credential discovery
    # (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY [+ AWS_SESSION_TOKEN]) is
    # handled separately by boto3 and is out of scope for the prefill.
    'bedrock': ['AWS_BEARER_TOKEN_BEDROCK'],

    # Other Services
    'futurehouse': ['FUTUREHOUSE_API_KEY'],
    'materials_project': ['MP_API_KEY', 'MATERIALS_PROJECT_API_KEY'],
}

# Internal proxy key (separate from provider keys)
INTERNAL_PROXY_KEY = 'SCILINK_API_KEY'

# Internal proxy base URL — pairs with INTERNAL_PROXY_KEY so the proxy path
# can be configured entirely from the environment (e.g. container deployments).
INTERNAL_PROXY_BASE_URL = 'SCILINK_BASE_URL'


def infer_provider(model_name: str) -> Optional[str]:
    """
    Infer provider from model name.
    
    Args:
        model_name: Model string (e.g., "gemini/gemini-2.0-flash", "gpt-4o")
    
    Returns:
        Provider name ('google', 'openai', 'anthropic', 'bedrock') or None
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    # Explicit prefix (e.g., "gemini/gemini-2.0-flash", "openai/gpt-4o",
    # "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    if '/' in model_name:
        prefix = model_lower.split('/')[0]
        prefix_map = {
            'gemini': 'google',
            'google': 'google',
            'openai': 'openai',
            'anthropic': 'anthropic',
            'bedrock': 'bedrock',
        }
        if prefix in prefix_map:
            return prefix_map[prefix]
    
    # Infer from model name patterns
    if model_lower.startswith(('gpt-', 'o1-', 'o3-')):
        return 'openai'
    # OpenAI embedding family: text-embedding-ada-002, text-embedding-3-*, ...
    if model_lower.startswith('text-embedding-'):
        return 'openai'
    if model_lower.startswith('claude'):
        return 'anthropic'
    if 'gemini' in model_lower:
        return 'google'

    return None


def get_internal_proxy_key() -> Optional[str]:
    """
    Get API key for internal proxy deployments.

    Returns:
        SCILINK_API_KEY value or None
    """
    return os.getenv(INTERNAL_PROXY_KEY)


def get_internal_proxy_base_url() -> Optional[str]:
    """
    Get the base URL for internal proxy deployments.

    Returns:
        SCILINK_BASE_URL value or None
    """
    return os.getenv(INTERNAL_PROXY_BASE_URL)


def find_env_var(service: str) -> Optional[tuple]:
    """
    Find the first set environment variable for a service.

    Unlike ``get_api_key``, this returns the variable *name* alongside the
    value, so callers (e.g. the UI) can show which env var supplied a key.

    Args:
        service: One of the keys in ``ENV_VARS`` (e.g. 'openai', 'futurehouse').

    Returns:
        ``(env_var_name, value)`` for the first set variable, or None.
    """
    for var_name in ENV_VARS.get(service, []):
        value = os.getenv(var_name)
        if value:
            return var_name, value
    return None


def find_env_var_for_model(model_name: str) -> Optional[tuple]:
    """
    Find the set vendor env var matching a model's inferred provider.

    Returns:
        ``(env_var_name, value)`` or None (no provider inferred, or none set).
    """
    provider = infer_provider(model_name)
    return find_env_var(provider) if provider else None


class APIKeyManager:
    """Simple API key management with environment variable auto-discovery."""
    
    def __init__(self):
        self._keys: Dict[str, str] = {}
    
    def get_key(self, service: str) -> Optional[str]:
        """Get API key for a service."""
        service = service.lower()
        
        # First check if explicitly set
        if service in self._keys:
            return self._keys[service]
        
        # Then check environment variables
        for var_name in ENV_VARS.get(service, []):
            key = os.getenv(var_name)
            if key:
                return key
        
        return None
    
    def get_key_for_model(self, model_name: str) -> Optional[str]:
        """
        Get API key by inferring provider from model name.
        
        For public deployments only. Internal proxy should use get_internal_proxy_key().
        
        Args:
            model_name: Model string (e.g., "gemini/gemini-2.0-flash", "gpt-4o")
        
        Returns:
            API key or None
        """
        provider = infer_provider(model_name)
        if provider:
            return self.get_key(provider)
        return None
    
    def set_key(self, service: str, api_key: str):
        """Set API key for a service."""
        self._keys[service.lower()] = api_key
    
    def clear_key(self, service: str):
        """Clear API key for a service."""
        self._keys.pop(service.lower(), None)
    
    def show_status(self):
        """Show current API key status."""
        print("API Key Status:")
        print("-" * 50)
        
        # Show internal proxy key status
        print("\n🔐 Internal Proxy:")
        proxy_key = get_internal_proxy_key()
        if proxy_key:
            masked = proxy_key[:4] + "..." + proxy_key[-4:] if len(proxy_key) > 12 else "***"
            print(f"  ✓ {'SCILINK_API_KEY':18} {masked:20} (for --base-url)")
        else:
            print(f"  ✗ {'SCILINK_API_KEY':18} {'Not set':20} (for --base-url)")
        
        print("\n🤖 LLM Providers:")
        for service in ['google', 'openai', 'anthropic']:
            self._print_status(service)
        
        print("\n🔧 Other Services:")
        for service in ['futurehouse', 'materials_project']:
            self._print_status(service)
    
    def _print_status(self, service: str):
        """Print status for a single service."""
        key = self.get_key(service)
        if key:
            source = "(configured)" if service in self._keys else f"(${self._find_env(service)})"
            masked = key[:4] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"  ✓ {service:18} {masked:20} {source}")
        else:
            env_var = ENV_VARS.get(service, [''])[0]
            print(f"  ✗ {service:18} {'Not found':20} (set ${env_var})")
    
    def _find_env(self, service: str) -> Optional[str]:
        """Find which env var provided the key."""
        for var_name in ENV_VARS.get(service, []):
            if os.getenv(var_name):
                return var_name
        return None


# Global instance
_api_manager = APIKeyManager()


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service."""
    return _api_manager.get_key(service)


def get_api_key_for_model(model_name: str) -> Optional[str]:
    """Get API key by inferring provider from model name."""
    return _api_manager.get_key_for_model(model_name)


def set_api_key(service: str, api_key: str):
    """Set API key for a service."""
    _api_manager.set_key(service, api_key)


def clear_api_key(service: str):
    """Clear API key for a service."""
    _api_manager.clear_key(service)


def show_api_status():
    """Show current API key status."""
    _api_manager.show_status()


class APIKeyNotFoundError(Exception):
    """Raised when a required API key is not found."""
    
    def __init__(self, service: str, additional_note: Optional[str] = None):
        suggestions = {
            'google': [
                "Set environment variable: export GEMINI_API_KEY='your-key'",
                "Configure in code: scilink.set_api_key('google', 'your-key')",
                "Get your key at: https://aistudio.google.com/apikey"
            ],
            'openai': [
                "Set environment variable: export OPENAI_API_KEY='your-key'",
                "Configure in code: scilink.set_api_key('openai', 'your-key')",
                "Get your key at: https://platform.openai.com/api-keys"
            ],
            'anthropic': [
                "Set environment variable: export ANTHROPIC_API_KEY='your-key'",
                "Configure in code: scilink.set_api_key('anthropic', 'your-key')",
                "Get your key at: https://console.anthropic.com/settings/keys"
            ],
            'futurehouse': [
                "Set environment variable: export FUTUREHOUSE_API_KEY='your-key'",
                "Configure in code: scilink.set_api_key('futurehouse', 'your-key')",
                "Get your key at: https://platform.futurehouse.org/"
            ],
            'materials_project': [
                "Set environment variable: export MP_API_KEY='your-key'",
                "Configure in code: scilink.set_api_key('materials_project', 'your-key')",
                "Get your key at: https://next-gen.materialsproject.org/api"
            ],
            'proxy': [
                "Set environment variable: export SCILINK_API_KEY='your-key'",
                "Or pass api_key parameter directly"
            ]
        }
        
        tips = suggestions.get(service.lower(), [
            f"Set environment variable for '{service}'",
            f"Configure in code: scilink.set_api_key('{service}', 'your-key')"
        ])
        
        msg = f"API key for '{service}' not found.\n\nTry one of these options:\n"
        for tip in tips:
            msg += f"  • {tip}\n"
        if additional_note:
            msg += f"\n{additional_note}\n"

        super().__init__(msg)
        self.service = service


def require_vendor_credentials(model_name: str) -> None:
    """Raise APIKeyNotFoundError if no vendor API key is available for ``model_name``.

    Intended for the direct LiteLLM path (no ``base_url``). Wraps
    ``litellm.validate_environment`` so the resulting message names both the
    expected vendor env var(s) AND -- when ``SCILINK_API_KEY`` is set without
    a ``base_url`` -- explains that ``SCILINK_API_KEY`` is the proxy key, not
    a vendor credential.
    """
    import litellm  # local: keep auth.py lightweight at module load time
    env = litellm.validate_environment(model_name)
    if env["keys_in_environment"]:
        return

    # Bedrock bearer-token escape hatch. LiteLLM honors AWS_BEARER_TOKEN_BEDROCK
    # at call time (sets ``Authorization: Bearer …`` on the Bedrock request), but
    # litellm.validate_environment only checks the traditional
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY path. Recognize the bearer
    # token here so a valid bearer-token Bedrock deployment isn't blocked.
    if model_name.startswith("bedrock/") and os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return

    expected = env["missing_keys"]
    # Best-effort map from a LiteLLM env-var name back to an APIKeyNotFoundError
    # service key (so the tips block is precise). Unknown env vars fall through
    # and the env-var name itself is used as the service (the generic-tip path
    # still names it).
    _env_to_service = {
        "ANTHROPIC_API_KEY": "anthropic",
        "OPENAI_API_KEY": "openai",
        "GOOGLE_API_KEY": "google",
        "GEMINI_API_KEY": "google",
    }
    service = _env_to_service.get(expected[0], expected[0]) if expected else "unknown"

    note = None
    if get_internal_proxy_key():
        note = (
            "SCILINK_API_KEY is currently set, but it's the proxy key — vendors "
            "reject it on the direct LiteLLM path. To use the internal proxy "
            "instead, pass `base_url=` alongside SCILINK_API_KEY."
        )
    raise APIKeyNotFoundError(service, additional_note=note)