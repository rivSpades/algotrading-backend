"""Synchronous OHLCV refresh via existing Celery tasks (in-process ``.apply()``).

Live trading and other callers should use these helpers so fetch parameters
stay consistent with ``market_data.tasks`` while avoiding duplicated ``apply``
boilerplate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def fetch_ohlcv_single_sync(
    *,
    ticker: str,
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    replace_existing: bool = False,
    provider_code: str = 'YAHOO',
    broker_id: Optional[int] = None,
) -> Any:
    """Run :func:`market_data.tasks.fetch_ohlcv_data_task` synchronously."""
    from market_data.tasks import fetch_ohlcv_data_task

    res = fetch_ohlcv_data_task.apply(
        kwargs={
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'period': period,
            'replace_existing': replace_existing,
            'broker_id': broker_id,
            'provider_code': provider_code,
        },
    )
    return res.result if hasattr(res, 'result') else res


def fetch_ohlcv_bulk_symbols_sync(
    *,
    tickers: List[str],
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    replace_existing: bool = False,
    provider_code: str = 'YAHOO',
    broker_id: Optional[int] = None,
) -> Any:
    """Run :func:`market_data.tasks.fetch_ohlcv_data_multiple_symbols_task` synchronously."""
    from market_data.tasks import fetch_ohlcv_data_multiple_symbols_task

    res = fetch_ohlcv_data_multiple_symbols_task.apply(
        kwargs={
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'period': period,
            'replace_existing': replace_existing,
            'broker_id': broker_id,
            'provider_code': provider_code,
        },
    )
    return res.result if hasattr(res, 'result') else res


def bulk_per_ticker_results(fetch_result: Any) -> Optional[Dict[str, Any]]:
    """Normalize apply() return value to per-ticker dict when present."""
    if not isinstance(fetch_result, dict):
        return None
    inner = fetch_result.get('result')
    return inner if isinstance(inner, dict) else None
