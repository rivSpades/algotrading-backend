"""
Market Data Service — provider-agnostic facade over provider blackboxes.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union

from django.utils import timezone

from ..providers.factory import ProviderFactory


DateInput = Optional[Union[str, datetime, date]]


def _parse_date(value: DateInput) -> Optional[datetime]:
    """Parse ISO string, date, or datetime into timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime.combine(value, datetime.min.time())
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        return None
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    return dt


def normalize_date_range(
    start_date: DateInput = None,
    end_date: DateInput = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Normalize date inputs for OHLCV fetch.

    If end_date is empty, defaults to today (end of day in current timezone).
    """
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    if end_dt is None:
        today = timezone.localdate()
        end_dt = timezone.make_aware(
            datetime.combine(today, datetime.max.time().replace(microsecond=0))
        )

    return start_dt, end_dt


def parse_task_dates(
    start_date: DateInput = None,
    end_date: DateInput = None,
    period: Optional[str] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse dates for Celery OHLCV tasks.

    When period is set, dates are optional (provider derives range from period).
    Otherwise end_date defaults to today via normalize_date_range.
    """
    if period:
        start_dt = _parse_date(start_date)
        if end_date:
            _, end_dt = normalize_date_range(None, end_date)
        elif start_dt:
            _, end_dt = normalize_date_range(start_dt, None)
        else:
            end_dt = None
        return start_dt, end_dt
    return normalize_date_range(start_date, end_date)


def get_daily_data(
    provider_code: str,
    ticker: str,
    start_date: DateInput = None,
    end_date: DateInput = None,
    period: Optional[str] = None,
    interval: str = '1d',
) -> List[Dict]:
    """
    Fetch daily OHLCV data from the given provider blackbox.

    Unified entry point for all OHLCV providers.
    """
    start_dt, end_dt = normalize_date_range(start_date, end_date)
    data_provider = ProviderFactory.get_provider(provider_code)

    if hasattr(data_provider, 'get_historical_data'):
        return data_provider.get_historical_data(
            ticker=ticker,
            start_date=start_dt,
            end_date=end_dt,
            period=period,
            interval=interval,
        )

    raise ValueError(f'Provider {provider_code} does not support get_historical_data')


def get_daily_data_bulk(
    provider_code: str,
    tickers: List[str],
    start_date: DateInput = None,
    end_date: DateInput = None,
    period: Optional[str] = None,
    interval: str = '1d',
) -> Dict[str, List[Dict]]:
    """
    Fetch daily OHLCV for multiple tickers when the provider supports bulk fetch.
    Falls back to per-ticker get_daily_data otherwise.
    """
    start_dt, end_dt = normalize_date_range(start_date, end_date)
    data_provider = ProviderFactory.get_provider(provider_code)

    if hasattr(data_provider, 'get_historical_data_bulk'):
        return data_provider.get_historical_data_bulk(
            tickers=tickers,
            start_date=start_dt,
            end_date=end_dt,
            period=period,
            interval=interval,
        )

    result: Dict[str, List[Dict]] = {}
    for ticker in tickers:
        result[ticker] = get_daily_data(
            provider_code=provider_code,
            ticker=ticker,
            start_date=start_dt,
            end_date=end_dt,
            period=period,
            interval=interval,
        )
    return result
