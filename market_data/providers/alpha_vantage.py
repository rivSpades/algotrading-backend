"""
Alpha Vantage Provider
Handles fetching OHLCV data from Alpha Vantage API.
Free tier: 25 requests/day — rate limiting handled with basic retry.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from django.utils import timezone


class AlphaVantageProvider:
    """Provider for Alpha Vantage daily OHLCV data."""

    BASE_URL = 'https://www.alphavantage.co/query'
    _api_key: Optional[str] = None
    _initialized = False

    @classmethod
    def initialize(cls, api_key: str) -> None:
        cls._api_key = api_key
        cls._initialized = True

    @classmethod
    def _check_initialized(cls) -> None:
        if not cls._initialized or not cls._api_key:
            raise ValueError(
                'Alpha Vantage provider not initialized. Call initialize() first.'
            )

    @classmethod
    def _resolve_api_key(cls) -> str:
        if cls._api_key:
            return cls._api_key
        env_key = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        if env_key:
            return env_key
        raise ValueError('Alpha Vantage API key not configured')

    @staticmethod
    def _parse_timestamp(date_str: str) -> datetime:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt)
        return dt

    @classmethod
    def get_historical_data(
        cls,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d',
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a symbol.

        Uses TIME_SERIES_DAILY_ADJUSTED for daily interval.
        """
        if interval != '1d':
            raise ValueError('Alpha Vantage provider only supports daily (1d) interval')

        api_key = cls._resolve_api_key()
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full' if (start_date or period) else 'compact',
            'apikey': api_key,
        }

        for attempt in range(3):
            response = requests.get(cls.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()

            if 'Note' in payload or 'Information' in payload:
                if attempt < 2:
                    time.sleep(12 * (attempt + 1))
                    continue
                raise ValueError(
                    payload.get('Note') or payload.get('Information') or 'Rate limit exceeded'
                )

            if 'Error Message' in payload:
                raise ValueError(payload['Error Message'])

            series = payload.get('Time Series (Daily)')
            if not series:
                return []

            data: List[Dict] = []
            for date_str, row in series.items():
                ts = cls._parse_timestamp(date_str)
                if start_date and ts.date() < start_date.date():
                    continue
                if end_date and ts.date() > end_date.date():
                    continue
                data.append({
                    'timestamp': ts,
                    'open': float(row['1. open']),
                    'high': float(row['2. high']),
                    'low': float(row['3. low']),
                    'close': float(row['4. close']),
                    'volume': int(float(row['6. volume'])),
                })

            data.sort(key=lambda x: x['timestamp'])
            return data

        return []
