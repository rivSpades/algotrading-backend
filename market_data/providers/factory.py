"""
Provider Factory
Creates provider instances based on provider code
"""

from typing import Optional

from ..models import Provider
from .yahoo_finance import YahooFinanceProvider
from .polygon import PolygonProvider
from .alpaca import AlpacaProvider
from .alpha_vantage import AlphaVantageProvider
from . import credentials as provider_credentials


class ProviderFactory:
    """Factory for creating data provider instances"""
    
    PROVIDERS = {
        'YAHOO': YahooFinanceProvider,
        'POLYGON': PolygonProvider,
        'ALPACA': AlpacaProvider,
        'ALPHA_VANTAGE': AlphaVantageProvider,
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
        code = provider_code.upper()
        provider_class = cls.PROVIDERS.get(code)
        if not provider_class:
            raise ValueError(f"Unknown provider code: {provider_code}")
        
        if code == 'YAHOO':
            return provider_class

        if code == 'POLYGON':
            creds = provider_credentials.get_polygon_credentials()
            if not creds:
                raise ValueError(
                    'Polygon provider missing credentials. '
                    'Set POLYGON_* env vars or Provider DB row.'
                )
            access_key_id, secret_access_key, endpoint_url, bucket_name = creds
            provider_class.initialize(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                endpoint_url=endpoint_url,
                bucket_name=bucket_name,
            )
            return provider_class

        if code == 'ALPACA':
            creds = provider_credentials.get_alpaca_credentials()
            if not creds:
                raise ValueError(
                    'Alpaca provider missing credentials. '
                    'Set ALPACA_API_KEY and ALPACA_API_SECRET in .env '
                    'or Provider DB row.'
                )
            provider_class.initialize(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                base_url=creds.base_url,
            )
            return provider_class

        if code == 'ALPHA_VANTAGE':
            api_key = provider_credentials.get_alpha_vantage_api_key()
            if not api_key:
                raise ValueError(
                    'Alpha Vantage provider missing API key. '
                    'Set ALPHA_VANTAGE_API_KEY in .env or Provider DB row.'
                )
            provider_class.initialize(api_key=api_key)
            return provider_class

        # Ensure catalog row exists for unknown future providers
        Provider.objects.get(code=code, is_active=True)
        return provider_class
    
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
