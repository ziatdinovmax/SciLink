"""
Deprecation utilities for backwards-compatible parameter migration.
"""

import warnings
from typing import Optional, Tuple


def normalize_api_key(
    api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    source: str = "Agent"
) -> str:
    """
    Normalize API key parameter, handling deprecated 'google_api_key'.
    
    Args:
        api_key: New parameter name (preferred)
        google_api_key: Deprecated parameter name
        source: Name of the calling class for warning messages
    
    Returns:
        The resolved API key
    
    Raises:
        ValueError: If no API key is provided
    """
    if google_api_key is not None and api_key is not None:
        warnings.warn(
            f"{source}: Both 'api_key' and 'google_api_key' provided. "
            f"Using 'api_key'. 'google_api_key' is deprecated and will be "
            f"removed in v2.0.",
            DeprecationWarning,
            stacklevel=3
        )
        return api_key
    
    if google_api_key is not None:
        warnings.warn(
            f"{source}: 'google_api_key' parameter is deprecated and will be "
            f"removed in v2.0. Use 'api_key' instead.",
            DeprecationWarning,
            stacklevel=3
        )
        return google_api_key
    
    return api_key


def normalize_base_url(
    base_url: Optional[str] = None,
    local_model: Optional[str] = None,
    source: str = "Agent"
) -> Optional[str]:
    """
    Normalize endpoint URL parameter, handling deprecated 'local_model'.
    
    Args:
        base_url: New parameter name (preferred)
        local_model: Deprecated parameter name
        source: Name of the calling class for warning messages
    
    Returns:
        The resolved endpoint URL, or None
    """
    if local_model is not None and base_url is not None:
        warnings.warn(
            f"{source}: Both 'base_url' and 'local_model' provided. "
            f"Using 'base_url'. 'local_model' is deprecated and will be "
            f"removed in v2.0.",
            DeprecationWarning,
            stacklevel=3
        )
        return base_url
    
    if local_model is not None:
        warnings.warn(
            f"{source}: 'local_model' parameter is deprecated and will be "
            f"removed in v2.0. Use 'base_url' instead.",
            DeprecationWarning,
            stacklevel=3
        )
        return local_model
    
    return base_url


def normalize_params(
    api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    local_model: Optional[str] = None,
    source: str = "Agent"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize both API key and base URL parameters.
    
    Returns:
        Tuple of (api_key, base_url)
    """
    resolved_key = normalize_api_key(api_key, google_api_key, source)
    resolved_url = normalize_base_url(base_url, local_model, source)
    return resolved_key, resolved_url