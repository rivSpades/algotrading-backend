"""
OHLCV data provider credentials — broker paper keys, then .env, then Provider DB.
Falls back when earlier sources are missing or invalid (401).
"""

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import requests

from ..models import Provider


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    api_secret: str
    base_url: str = 'https://data.alpaca.markets'
    source: str = 'unknown'


_PROBE_CACHE: Optional[Tuple[str, AlpacaCredentials]] = None


def _env(name: str) -> str:
    return (os.environ.get(name) or '').strip()


def _broker_alpaca_credentials() -> Optional[AlpacaCredentials]:
    try:
        from live_trading.models import Broker
        broker = Broker.objects.filter(code='ALPACA').first()
    except Exception:
        return None
    if not broker or not broker.paper_trading_api_key or not broker.paper_trading_secret_key:
        return None
    return AlpacaCredentials(
        api_key=broker.paper_trading_api_key,
        api_secret=broker.paper_trading_secret_key,
        base_url='https://data.alpaca.markets',
        source='broker',
    )


def _env_alpaca_credentials() -> Optional[AlpacaCredentials]:
    base_url = _env('ALPACA_DATA_BASE_URL') or 'https://data.alpaca.markets'
    api_key = _env('ALPACA_API_KEY')
    api_secret = _env('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        return None
    return AlpacaCredentials(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        source='env',
    )


def _provider_alpaca_credentials() -> Optional[AlpacaCredentials]:
    base_url = _env('ALPACA_DATA_BASE_URL') or 'https://data.alpaca.markets'
    try:
        provider = Provider.objects.get(code='ALPACA', is_active=True)
        if provider.api_key and provider.secret_access_key:
            return AlpacaCredentials(
                api_key=provider.api_key,
                api_secret=provider.secret_access_key,
                base_url=provider.base_url or base_url,
                source='provider',
            )
    except Provider.DoesNotExist:
        pass
    return None


def _iter_alpaca_credential_candidates() -> Iterator[AlpacaCredentials]:
    """Yield unique credential sets in priority order."""
    seen: set = set()
    for creds in (
        _broker_alpaca_credentials(),
        _env_alpaca_credentials(),
        _provider_alpaca_credentials(),
    ):
        if not creds:
            continue
        key = (creds.api_key, creds.api_secret, creds.base_url)
        if key in seen:
            continue
        seen.add(key)
        yield creds


def _probe_alpaca_credentials(creds: AlpacaCredentials) -> bool:
    """Quick check against the Alpaca data API."""
    base = creds.base_url.rstrip('/').rstrip('/v2')
    try:
        response = requests.get(
            f'{base}/v2/stocks/bars',
            headers={
                'APCA-API-KEY-ID': creds.api_key,
                'APCA-API-SECRET-KEY': creds.api_secret,
            },
            params={
                'symbols': 'AAPL',
                'timeframe': '1Day',
                'limit': 1,
                'feed': 'iex',
            },
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def get_alpaca_credentials() -> Optional[AlpacaCredentials]:
    """
    Resolve working Alpaca market-data credentials.

    Priority: live_trading Broker (paper) → .env → market_data Provider row.
    When multiple sources exist, the first that passes a lightweight API probe wins.
    """
    global _PROBE_CACHE

    if _PROBE_CACHE:
        _source, cached = _PROBE_CACHE
        if _probe_alpaca_credentials(cached):
            return cached
        _PROBE_CACHE = None

    candidates: List[AlpacaCredentials] = list(_iter_alpaca_credential_candidates())
    if not candidates:
        return None

    for creds in candidates:
        if _probe_alpaca_credentials(creds):
            _PROBE_CACHE = (creds.source, creds)
            return creds

    # Offline or probe failed — use highest-priority source without validation
    return candidates[0]


def is_alpaca_configured() -> bool:
    return get_alpaca_credentials() is not None


def get_alpha_vantage_api_key() -> str:
    """Resolve Alpha Vantage API key from env or Provider DB."""
    key = _env('ALPHA_VANTAGE_API_KEY')
    if key:
        return key
    try:
        provider = Provider.objects.get(code='ALPHA_VANTAGE', is_active=True)
        return (provider.api_key or '').strip()
    except Provider.DoesNotExist:
        return ''


def is_alpha_vantage_configured() -> bool:
    return bool(get_alpha_vantage_api_key())


def get_polygon_credentials() -> Optional[Tuple[str, str, str, str]]:
    """S3-style Polygon credentials from env or DB."""
    access_key = _env('POLYGON_ACCESS_KEY_ID')
    secret_key = _env('POLYGON_SECRET_ACCESS_KEY')
    endpoint = _env('POLYGON_ENDPOINT_URL')
    bucket = _env('POLYGON_BUCKET_NAME')

    if all([access_key, secret_key, endpoint, bucket]):
        return access_key, secret_key, endpoint, bucket

    try:
        provider = Provider.objects.get(code='POLYGON', is_active=True)
    except Provider.DoesNotExist:
        return None

    if all([
        provider.access_key_id,
        provider.secret_access_key,
        provider.endpoint_url,
        provider.bucket_name,
    ]):
        return (
            provider.access_key_id,
            provider.secret_access_key,
            provider.endpoint_url,
            provider.bucket_name,
        )
    return None
