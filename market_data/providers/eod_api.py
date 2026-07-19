"""
EOD Historical Data API Provider
Handles fetching symbols and market data from EOD Historical Data API
"""

import requests
from typing import List, Dict, Optional
from django.conf import settings


class EODAPIProvider:
    """Provider for EOD Historical Data API"""

    BASE_URL = "https://eodhistoricaldata.com/api"

    @classmethod
    def _api_key(cls) -> str:
        """Resolve the EOD API token from settings/env (never commit a real key)."""
        return getattr(settings, 'EOD_API_KEY', '') or ''

    @classmethod
    def get_exchanges_list(cls) -> List[Dict]:
        """
        Get list of all available exchanges from EOD API
        Returns list of exchange dictionaries
        """
        try:
            url = f"{cls.BASE_URL}/exchanges-list"
            params = {
                'api_token': cls._api_key(),
                'fmt': 'json'
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Validate response is a list
            if not isinstance(data, list):
                print(f"Unexpected response format from EOD API: {type(data)}")
                return []
            
            # Filter out invalid entries and ensure required fields
            valid_exchanges = []
            for exchange in data:
                if isinstance(exchange, dict) and exchange.get('Code'):
                    valid_exchanges.append(exchange)
            
            return valid_exchanges
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching exchanges list: {str(e)}")
            raise
        except Exception as e:
            print(f"Error fetching exchanges list: {str(e)}")
            raise
    
    @classmethod
    def get_exchange_symbols(cls, exchange_code: str) -> List[Dict]:
        """
        Get all symbols for a specific exchange
        Args:
            exchange_code: Exchange code (e.g., 'US', 'NASDAQ', 'NYSE')
        Returns:
            List of symbol dictionaries
        """
        try:
            url = f"{cls.BASE_URL}/exchange-symbol-list/{exchange_code}"
            params = {
                'api_token': cls._api_key(),
                'fmt': 'json'
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching symbols for {exchange_code}: {str(e)}")
            return []

    @classmethod
    def search_symbols(cls, query: str, limit: int = 20) -> List[Dict]:
        """
        Search symbols by ticker or name via EOD API.

        Args:
            query: Search term (ticker or company name)
            limit: Maximum number of results to return

        Returns:
            List of symbol candidate dicts
        """
        try:
            url = f"{cls.BASE_URL}/search/{query}"
            params = {
                'api_token': cls._api_key(),
                'fmt': 'json',
                'limit': limit,
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                return []
            return [item for item in data if isinstance(item, dict) and item.get('Code')]
        except Exception as e:
            print(f"Error searching symbols for {query}: {str(e)}")
            return []

