"""
Symbol Resolution Service
Ensures symbols exist in DB via EOD search/import before OHLCV fetch.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..models import Symbol, Exchange
from ..providers.eod_api import EODAPIProvider
from .symbol_service import SymbolService


@dataclass
class SymbolCandidate:
    ticker: str
    exchange_code: str
    name: str
    symbol_type: str

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'exchange_code': self.exchange_code,
            'name': self.name,
            'type': self.symbol_type,
        }


def _candidate_from_eod(item: Dict) -> SymbolCandidate:
    return SymbolCandidate(
        ticker=item.get('Code', ''),
        exchange_code=item.get('Exchange', ''),
        name=item.get('Name', item.get('Code', '')),
        symbol_type=item.get('Type', 'stock'),
    )


def search_symbol_candidates(ticker: str, limit: int = 20) -> List[SymbolCandidate]:
    """Search EOD for symbol candidates matching ticker."""
    query = (ticker or '').strip()
    if not query:
        return []
    raw = EODAPIProvider.search_symbols(query, limit=limit)
    return [_candidate_from_eod(item) for item in raw if item.get('Code')]


def resolve_symbol(ticker: str) -> Dict:
    """
    Resolve a ticker to an existing or creatable symbol.

    Returns dict with status: resolved | ambiguous | not_found
    """
    query = (ticker or '').strip().upper()
    if not query:
        return {'status': 'not_found', 'message': 'Ticker is required'}

    try:
        symbol = Symbol.objects.select_related('exchange').get(ticker=query)
        return {
            'status': 'resolved',
            'symbol': {
                'ticker': symbol.ticker,
                'exchange_code': symbol.exchange.code if symbol.exchange else None,
                'name': symbol.name,
                'status': symbol.status,
            },
        }
    except Symbol.DoesNotExist:
        pass

    candidates = search_symbol_candidates(query)
    if not candidates:
        return {'status': 'not_found', 'message': f'No matches found for {query}'}

    if len(candidates) == 1:
        symbol = ensure_symbol(query, exchange_code=candidates[0].exchange_code)
        if symbol:
            return {
                'status': 'resolved',
                'symbol': {
                    'ticker': symbol.ticker,
                    'exchange_code': symbol.exchange.code if symbol.exchange else None,
                    'name': symbol.name,
                    'status': symbol.status,
                },
            }
        return {'status': 'not_found', 'message': f'Could not create symbol {query}'}

    return {
        'status': 'ambiguous',
        'candidates': [c.to_dict() for c in candidates],
    }


def ensure_exchange(exchange_code: str) -> Optional[Exchange]:
    """Ensure exchange exists in DB, creating from EOD list if needed."""
    code = (exchange_code or '').strip()
    if not code:
        return None
    try:
        return Exchange.objects.get(code=code)
    except Exchange.DoesNotExist:
        exchanges = EODAPIProvider.get_exchanges_list()
        match = next((ex for ex in exchanges if ex.get('Code') == code), None)
        if match:
            return SymbolService.get_or_create_exchange(match)
        return Exchange.objects.create(
            code=code,
            name=code,
            country='',
            timezone='UTC',
        )


def ensure_symbol(ticker: str, exchange_code: Optional[str] = None) -> Optional[Symbol]:
    """
    Ensure symbol exists in DB, creating from EOD data if needed.

    Args:
        ticker: Symbol ticker
        exchange_code: Optional exchange code for disambiguation

    Returns:
        Symbol instance or None
    """
    query = (ticker or '').strip().upper()
    if not query:
        return None

    try:
        return Symbol.objects.get(ticker=query)
    except Symbol.DoesNotExist:
        pass

    candidates = search_symbol_candidates(query, limit=50)
    if exchange_code:
        candidates = [c for c in candidates if c.exchange_code == exchange_code]

    if not candidates:
        return None

    chosen = candidates[0]
    exchange = ensure_exchange(chosen.exchange_code)
    if not exchange:
        return None

    symbol_data = {
        'Code': chosen.ticker,
        'Name': chosen.name,
        'Exchange': chosen.exchange_code,
        'Type': chosen.symbol_type,
        'Description': '',
    }
    return SymbolService.create_symbol_from_eod_data(symbol_data, exchange)


def ensure_exchange_symbols(exchange_code: str) -> Dict:
    """
    Import symbols from EOD if exchange has none in DB.

    Returns import statistics from SymbolService.
    """
    exchange = ensure_exchange(exchange_code)
    if not exchange:
        return {'created': 0, 'updated': 0, 'errors': 1, 'message': f'Exchange {exchange_code} not found'}

    count = Symbol.objects.filter(exchange=exchange).count()
    if count > 0:
        return {'created': 0, 'updated': 0, 'errors': 0, 'skipped': True, 'existing_count': count}

    return SymbolService.import_symbols_from_exchange(exchange_code)
