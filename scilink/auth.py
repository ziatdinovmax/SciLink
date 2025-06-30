# scilinkllm/auth.py

import os
from typing import Optional, Dict

class APIKeyManager:
    """Simple API key management with environment variable auto-discovery"""
    
    def __init__(self):
        self._keys: Dict[str, str] = {}
    
    def get_key(self, service: str) -> Optional[str]:
        """Get API key for a service"""
        # First check if explicitly set
        if service in self._keys:
            return self._keys[service]
        
        # Then check environment variables
        env_vars = {
            'google': ['GOOGLE_API_KEY'],
            'futurehouse': ['FUTUREHOUSE_API_KEY'],
            'materials_project': ['MP_API_KEY', 'MATERIALS_PROJECT_API_KEY']
        }
        
        for var_name in env_vars.get(service, []):
            key = os.getenv(var_name)
            if key:
                return key
        
        return None
    
    def set_key(self, service: str, api_key: str):
        """Set API key for a service"""
        self._keys[service] = api_key
    
    def clear_key(self, service: str):
        """Clear API key for a service"""
        self._keys.pop(service, None)
    
    def show_status(self):
        """Show current API key status"""
        services = ['google', 'futurehouse', 'materials_project']
        print("API Key Status:")
        for service in services:
            key = self.get_key(service)
            status = "✓ Found" if key else "✗ Not found"
            source = ""
            if key:
                if service in self._keys:
                    source = "(configured)"
                else:
                    source = "(environment)"
            print(f"  {service}: {status} {source}")

# Global instance
_api_manager = APIKeyManager()

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service"""
    return _api_manager.get_key(service)

def set_api_key(service: str, api_key: str):
    """Set API key for a service"""
    _api_manager.set_key(service, api_key)

def clear_api_key(service: str):
    """Clear API key for a service"""
    _api_manager.clear_key(service)

def show_api_status():
    """Show current API key status"""
    _api_manager.show_status()

class APIKeyNotFoundError(Exception):
    """Raised when a required API key is not found"""
    
    def __init__(self, service: str):
        suggestions = {
            'google': [
                "Set environment variable: export GOOGLE_API_KEY='your-key'",
                "Configure in code: scilinkllm.configure('google', 'your-key')",
                "Get API key at: https://aistudio.google.com/apikey"
            ],
            'futurehouse': [
                "Set environment variable: export FUTUREHOUSE_API_KEY='your-key'",
                "Configure in code: scilinkllm.configure('futurehouse', 'your-key')",
                "Contact FutureHouse for API access"
            ],
            'materials_project': [
                "Set environment variable: export MP_API_KEY='your-key'",
                "Configure in code: scilinkllm.configure('materials_project', 'your-key')",
                "Get API key at: https://next-gen.materialsproject.org/api"
            ]
        }
        
        msg = f"API key for '{service}' not found.\n\nTry one of these options:\n"
        for suggestion in suggestions.get(service, []):
            msg += f"  • {suggestion}\n"
        
        super().__init__(msg)