import os as _os
import sys as _sys

# On macOS the torch and faiss-cpu wheels each bundle their own libomp.dylib.
# Importing torch first (below) decides which copy loads first, but both
# still load, and the second aborts the process ("OMP: Error #15") the first
# time faiss runs a search — every dense knowledge-base lookup. Before any
# import, so it is set when either runtime initializes; a value the user set
# is kept. The marker tells ``executors.sandbox_env`` that SciLink set it, so
# a generated script runs as it would without SciLink (a duplicate runtime
# there aborts loudly instead of risking wrong results); child processes
# inherit both and keep that knowledge.
if _sys.platform == "darwin" and "KMP_DUPLICATE_LIB_OK" not in _os.environ:
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _os.environ["_SCILINK_SET_KMP_DUPLICATE_LIB_OK"] = "1"

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