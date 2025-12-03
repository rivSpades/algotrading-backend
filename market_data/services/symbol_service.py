"""
Symbol Service
Handles symbol creation and management from EOD API data
"""

from typing import List, Dict, Optional
from django.utils import timezone
from ..models import Symbol, Exchange, Provider
from ..providers.eod_api import EODAPIProvider


class SymbolService:
    """Service for managing symbols"""
    
    @staticmethod
    def get_or_create_eod_provider() -> Provider:
        """Get or create EOD Historical Data provider"""
        provider, created = Provider.objects.get_or_create(
            code='EOD',
            defaults={
                'name': 'EOD Historical Data',
                'api_key': EODAPIProvider.API_KEY,
                'base_url': EODAPIProvider.BASE_URL,
                'is_active': True
            }
        )
        return provider
    
    @staticmethod
    def get_or_create_exchange(exchange_data: Dict) -> Exchange:
        """Get or create exchange from EOD API data"""
        code = exchange_data.get('Code', '')
        name = exchange_data.get('Name', code)
        country = exchange_data.get('Country', '')
        timezone_str = exchange_data.get('Timezone', 'UTC')
        
        exchange, created = Exchange.objects.get_or_create(
            code=code,
            defaults={
                'name': name,
                'country': country,
                'timezone': timezone_str
            }
        )
        return exchange
    
    @staticmethod
    def create_symbol_from_eod_data(symbol_data: Dict, exchange: Exchange, provider: Optional[Provider] = None) -> Optional[Symbol]:
        """
        Create or update symbol from EOD API data
        Provider is optional and should only be set when fetching OHLCV data, not for symbol creation
        """
        try:
            ticker = symbol_data.get('Code', '')
            if not ticker:
                return None
            
            # Determine symbol type
            symbol_type = 'stock'
            symbol_type_from_api = symbol_data.get('Type', '').lower()
            if 'crypto' in ticker.lower() or 'crypto' in exchange.code.lower():
                symbol_type = 'crypto'
            elif 'etf' in symbol_type_from_api:
                symbol_type = 'etf'
            elif 'forex' in symbol_type_from_api:
                symbol_type = 'forex'
            
            name = symbol_data.get('Name', ticker)
            description = symbol_data.get('Description', '')
            
            # Build defaults - provider is optional
            # Symbols are created as disabled by default
            defaults = {
                'exchange': exchange,
                'type': symbol_type,
                'name': name,
                'description': description,
                'status': 'disabled',
                'last_updated': timezone.now()
            }
            
            # Only set provider if provided (for OHLCV data fetching)
            if provider:
                defaults['provider'] = provider
            
            symbol, created = Symbol.objects.update_or_create(
                ticker=ticker,
                defaults=defaults
            )
            return symbol
        except Exception as e:
            print(f"Error creating symbol {symbol_data.get('Code')}: {str(e)}")
            return None
    
    @classmethod
    def import_symbols_from_exchange(cls, exchange_code: str) -> Dict:
        """
        Import symbols from a specific exchange
        Each symbol has its own Exchange field (NYSE, NASDAQ, etc.)
        Returns:
            Dictionary with import statistics
        """
        symbols_data = EODAPIProvider.get_exchange_symbols(exchange_code)
        
        # Get all exchanges list to map symbol exchanges properly
        exchanges_list = EODAPIProvider.get_exchanges_list()
        exchanges_map = {ex.get('Code'): ex for ex in exchanges_list if ex.get('Code')}
        
        created_count = 0
        updated_count = 0
        error_count = 0
        exchange_stats = {}
        
        for symbol_data in symbols_data:
            # Each symbol has its own Exchange field (NYSE, NASDAQ, etc.)
            symbol_exchange_code = symbol_data.get('Exchange', exchange_code)
            
            # Get exchange data from map or create basic data
            if symbol_exchange_code in exchanges_map:
                exchange_data = exchanges_map[symbol_exchange_code]
            else:
                exchange_data = {
                    'Code': symbol_exchange_code,
                    'Name': symbol_exchange_code,
                    'Country': '',
                    'Timezone': 'UTC'
                }
            
            # Get or create exchange with proper data
            exchange = cls.get_or_create_exchange(exchange_data)
            
            # Provider should not be set for symbol creation - only for OHLCV data
            symbol = cls.create_symbol_from_eod_data(symbol_data, exchange, None)
            if symbol:
                if symbol.created_at == symbol.updated_at:
                    created_count += 1
                else:
                    updated_count += 1
                
                # Track stats per exchange
                if symbol_exchange_code not in exchange_stats:
                    exchange_stats[symbol_exchange_code] = {'created': 0, 'updated': 0}
                if symbol.created_at == symbol.updated_at:
                    exchange_stats[symbol_exchange_code]['created'] += 1
                else:
                    exchange_stats[symbol_exchange_code]['updated'] += 1
            else:
                error_count += 1
        
        return {
            'exchange': exchange_code,
            'total': len(symbols_data),
            'created': created_count,
            'updated': updated_count,
            'errors': error_count,
            'exchange_stats': exchange_stats
        }
    
    @classmethod
    def import_symbols_from_multiple_exchanges(cls, exchange_codes: List[str]) -> Dict:
        """
        Import symbols from multiple exchanges
        Returns:
            Dictionary with import statistics for each exchange
        """
        results = {}
        for exchange_code in exchange_codes:
            results[exchange_code] = cls.import_symbols_from_exchange(exchange_code)
        return results
    
    @classmethod
    def import_symbols_from_all_exchanges(cls) -> Dict:
        """
        Import symbols from all available exchanges
        Returns:
            Dictionary with import statistics
        """
        exchanges = EODAPIProvider.get_exchanges_list()
        exchange_codes = [ex.get('Code', '') for ex in exchanges if ex.get('Code')]
        return cls.import_symbols_from_multiple_exchanges(exchange_codes)

