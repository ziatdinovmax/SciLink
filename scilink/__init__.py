from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("scilink")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .auth import set_api_key, show_api_status
from .tracing import enable_tracing, disable_tracing, is_enabled as is_tracing_enabled
import torch  # Load PyTorch's BLAS first to avoid conflicts with faiss

def configure(service: str, api_key: str):
    """Configure API key for a service
    
    Args:
        service: 'google', 'futurehouse', or 'materials_project'
        api_key: The API key
        
    Examples:
        import scilinkllm
        scilinkllm.configure('google', 'your-google-api-key')
        scilinkllm.configure('futurehouse', 'your-futurehouse-key')
    """
    set_api_key(service, api_key)

def configure_from_dict(config: dict):
    """Configure multiple API keys at once
    
    Args:
        config: Dictionary with service names as keys
        
    Example:
        scilinkllm.configure_from_dict({
            'google': 'your-google-key',
            'futurehouse': 'your-futurehouse-key'
        })
    """
    for service, key in config.items():
        set_api_key(service, key)

def show_config():
    """Show current API key configuration status"""
    show_api_status()


__all__ = [
    '__version__',
    'configure',
    'configure_from_dict',
    'show_config',
    'enable_tracing',
    'disable_tracing',
    'is_tracing_enabled',
    'torch'
]