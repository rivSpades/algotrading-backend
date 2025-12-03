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
    API_KEY = "66ae1118bb93f4.94656892"
    
    @classmethod
    def get_exchanges_list(cls) -> List[Dict]:
        """
        Get list of all available exchanges from EOD API
        Returns list of exchange dictionaries
        """
        try:
            url = f"{cls.BASE_URL}/exchanges-list"
            params = {
                'api_token': cls.API_KEY,
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
                'api_token': cls.API_KEY,
                'fmt': 'json'
            }
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching symbols for {exchange_code}: {str(e)}")
            return []
    
    @classmethod
    def get_multiple_exchange_symbols(cls, exchange_codes: List[str]) -> Dict[str, List[Dict]]:
        """
        Get symbols for multiple exchanges
        Args:
            exchange_codes: List of exchange codes
        Returns:
            Dictionary mapping exchange codes to their symbols
        """
        result = {}
        for exchange_code in exchange_codes:
            symbols = cls.get_exchange_symbols(exchange_code)
            result[exchange_code] = symbols
        return result
    
    @classmethod
    def get_all_exchange_symbols(cls) -> Dict[str, List[Dict]]:
        """
        Get symbols for all available exchanges
        Returns:
            Dictionary mapping exchange codes to their symbols
        """
        exchanges = cls.get_exchanges_list()
        exchange_codes = [ex.get('Code', '') for ex in exchanges if ex.get('Code')]
        return cls.get_multiple_exchange_symbols(exchange_codes)

