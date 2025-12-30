"""
Provider Factory
Creates provider instances based on provider code
"""

from typing import Optional
from ..models import Provider
from .yahoo_finance import YahooFinanceProvider
from .polygon import PolygonProvider
from .alpaca import AlpacaProvider


class ProviderFactory:
    """Factory for creating data provider instances"""
    
    PROVIDERS = {
        'YAHOO': YahooFinanceProvider,
        'POLYGON': PolygonProvider,
        'ALPACA': AlpacaProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_code: str) -> Optional[object]:
        """
        Get provider instance based on provider code
        
        Args:
            provider_code: Provider code (e.g., 'YAHOO', 'POLYGON')
        
        Returns:
            Provider instance or None if not found
        """
        provider_class = cls.PROVIDERS.get(provider_code.upper())
        if not provider_class:
            raise ValueError(f"Unknown provider code: {provider_code}")
        
        # For Yahoo Finance, return static class (no initialization needed)
        if provider_code.upper() == 'YAHOO':
            return provider_class
        
        # For other providers, get provider from database and initialize
        try:
            provider = Provider.objects.get(code=provider_code.upper(), is_active=True)
            
            if provider_code.upper() == 'POLYGON':
                if not all([provider.access_key_id, provider.secret_access_key, provider.endpoint_url, provider.bucket_name]):
                    raise ValueError(f"Polygon provider missing required credentials")
                
                # Initialize Polygon provider with credentials (uses class variables)
                provider_class.initialize(
                    access_key_id=provider.access_key_id,
                    secret_access_key=provider.secret_access_key,
                    endpoint_url=provider.endpoint_url,
                    bucket_name=provider.bucket_name
                )
                
                return provider_class
            
            if provider_code.upper() == 'ALPACA':
                if not provider.api_key:
                    raise ValueError(f"Alpaca provider missing required credentials (API key)")
                
                # Alpaca needs API key ID and API secret
                # api_key stores the API Key ID, secret_access_key stores the API Secret
                # If secret_access_key is not set, we'll check if there's a description or notes field
                # For now, secret_access_key should be set for Alpaca provider
                api_secret = provider.secret_access_key
                
                if not api_secret:
                    raise ValueError(f"Alpaca provider missing API secret. Please set secret_access_key field.")
                
                # Initialize Alpaca provider with credentials
                # base_url defaults to paper-api.alpaca.markets if not provided
                provider_class.initialize(
                    api_key=provider.api_key,
                    api_secret=api_secret,
                    base_url=provider.base_url  # Optional, defaults to paper-api
                )
                
                return provider_class
            
            # Add more provider types here as needed
            return provider_class
            
        except Provider.DoesNotExist:
            raise ValueError(f"Provider {provider_code} not found in database")
    
    @classmethod
    def get_provider_instance(cls, provider: Provider):
        """
        Get provider instance from Provider model instance
        
        Args:
            provider: Provider model instance
        
        Returns:
            Provider instance
        """
        return cls.get_provider(provider.code)

