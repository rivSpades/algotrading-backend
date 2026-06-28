"""
Hardcoded OHLCV provider catalog for UI and orchestration.
Credentials are read from .env (see credentials.py); DB rows are optional fallback.
"""

from typing import List, Dict, Optional

from ..models import Provider
from . import credentials as provider_credentials

OHLCV_PROVIDERS: List[Dict] = [
    {
        'code': 'YAHOO',
        'name': 'Yahoo Finance',
        'requires_credentials': False,
    },
    {
        'code': 'ALPACA',
        'name': 'Alpaca',
        'requires_credentials': True,
    },
    {
        'code': 'ALPHA_VANTAGE',
        'name': 'Alpha Vantage',
        'requires_credentials': True,
    },
]


def ensure_ohlcv_providers_in_db() -> None:
    """Ensure catalog providers exist in DB (metadata only when using .env creds)."""
    for entry in OHLCV_PROVIDERS:
        Provider.objects.get_or_create(
            code=entry['code'],
            defaults={
                'name': entry['name'],
                'is_active': True,
            },
        )


def _is_configured(entry: Dict) -> bool:
    if not entry['requires_credentials']:
        return True
    if entry['code'] == 'ALPACA':
        return provider_credentials.is_alpaca_configured()
    if entry['code'] == 'ALPHA_VANTAGE':
        return provider_credentials.is_alpha_vantage_configured()
    return False


def get_ohlcv_provider_catalog() -> List[Dict]:
    """Return catalog merged with configuration state (.env + optional DB)."""
    ensure_ohlcv_providers_in_db()
    db_by_code = {
        p.code: p
        for p in Provider.objects.filter(code__in=[e['code'] for e in OHLCV_PROVIDERS])
    }

    catalog = []
    for entry in OHLCV_PROVIDERS:
        db_provider: Optional[Provider] = db_by_code.get(entry['code'])
        catalog.append({
            'code': entry['code'],
            'name': entry['name'],
            'requires_credentials': entry['requires_credentials'],
            'configured': _is_configured(entry),
            'is_active': db_provider.is_active if db_provider else True,
        })
    return catalog
