# scilink/auth.py

"""
API key management with environment variable auto-discovery.
"""

import os
from typing import Optional, Dict


# Environment variable names per provider
ENV_VARS = {
    # LLM Providers
    'google': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    
    # Other Services
    'futurehouse': ['FUTUREHOUSE_API_KEY'],
    'materials_project': ['MP_API_KEY', 'MATERIALS_PROJECT_API_KEY'],
}

# Internal proxy key (separate from provider keys)
INTERNAL_PROXY_KEY = 'SCILINK_API_KEY'


def infer_provider(model_name: str) -> Optional[str]:
    """
    Infer provider from model name.
    
    Args:
        model_name: Model string (e.g., "gemini/gemini-2.0-flash", "gpt-4o")
    
    Returns:
        Provider name ('google', 'openai', 'anthropic') or None
    """
    if not model_name:
        return None
        
    model_lower = model_name.lower()
    
    # Explicit prefix (e.g., "gemini/gemini-2.0-flash", "openai/gpt-4o")
    if '/' in model_name:
        prefix = model_lower.split('/')[0]
        prefix_map = {
            'gemini': 'google',
            'google': 'google',
            'openai': 'openai',
            'anthropic': 'anthropic',
        }
        if prefix in prefix_map:
            return prefix_map[prefix]
    
    # Infer from model name patterns
    if model_lower.startswith(('gpt-', 'o1-', 'o3-')):
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
    
    def __init__(self, service: str):
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
        
        super().__init__(msg)
        self.service = service