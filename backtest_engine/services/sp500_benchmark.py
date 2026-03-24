"""
S&P 500 buy-and-hold benchmark curve for backtest equity comparison (^GSPC via Yahoo).
Uses DB OHLCV when the range is covered; otherwise fetches Yahoo and persists new bars.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd
from django.utils import timezone

from market_data.models import Exchange, OHLCV, Symbol
from market_data.providers.yahoo_finance import YahooFinanceProvider
from market_data.services.ohlcv_service import OHLCVService

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "^GSPC"
BENCHMARK_EXCHANGE_CODE = "BENCHMARK"


def _normalize_ts(ts) -> datetime:
    if ts is None:
        raise ValueError("timestamp is None")
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if not isinstance(ts, datetime):
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")
    if timezone.is_naive(ts):
        ts = timezone.make_aware(ts)
    return ts


def get_or_create_benchmark_symbol() -> Symbol:
    """Ensure ^GSPC exists for OHLCV storage."""
    exchange, _ = Exchange.objects.get_or_create(
        code=BENCHMARK_EXCHANGE_CODE,
        defaults={
            "name": "Benchmark indices",
            "country": "US",
            "timezone": "America/New_York",
        },
    )
    provider = OHLCVService.get_or_create_yahoo_provider()
    sym, _ = Symbol.objects.get_or_create(
        ticker=BENCHMARK_TICKER,
        defaults={
            "exchange": exchange,
            "provider": provider,
            "type": "etf",
            "name": "S&P 500",
            "status": "active",
        },
    )
    return sym


def _load_db_bars(symbol: Symbol, start: datetime, end: datetime) -> List[Tuple[datetime, float]]:
    qs = (
        OHLCV.objects.filter(
            symbol=symbol,
            timeframe="daily",
            timestamp__gte=start,
            timestamp__lte=end,
        )
        .order_by("timestamp")
        .values_list("timestamp", "close")
    )
    return [(row[0], float(row[1])) for row in qs]


def compute_sp500_buy_hold_curve(
    first_equity_timestamp,
    last_equity_timestamp,
    initial_capital: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build daily buy-and-hold equity vs first bar in range, aligned to the strategy curve window.

    Returns:
        (equity_curve_points, meta) where each point is {'timestamp': iso str, 'equity': float}
        meta includes: ticker, source ('db'|'yahoo'|'db+yahoo'), optional 'error'
    """
    meta: Dict[str, Any] = {"ticker": BENCHMARK_TICKER, "source": "db"}
    try:
        t0 = _normalize_ts(first_equity_timestamp)
        t1 = _normalize_ts(last_equity_timestamp)
    except (ValueError, TypeError) as e:
        return [], {"ticker": BENCHMARK_TICKER, "error": str(e), "source": "none"}

    if t1 < t0:
        t0, t1 = t1, t0

    start_query = t0 - timedelta(days=5)
    end_query = t1 + timedelta(days=5)
    initial_capital = float(initial_capital)

    symbol = get_or_create_benchmark_symbol()
    bars = _load_db_bars(symbol, start_query, end_query)

    covered, _ = OHLCVService.check_date_range_coverage(
        symbol, start_query, end_query, "daily"
    )

    if not covered or len(bars) < 2:
        try:
            yahoo_rows = YahooFinanceProvider.get_historical_data(
                BENCHMARK_TICKER,
                start_date=start_query,
                end_date=end_query,
                interval="1d",
            )
            if yahoo_rows:
                OHLCVService.save_ohlcv_data(
                    symbol,
                    yahoo_rows,
                    timeframe="daily",
                    provider=OHLCVService.get_or_create_yahoo_provider(),
                    replace_existing=False,
                )
                meta["source"] = "db+yahoo" if bars else "yahoo"
                bars = _load_db_bars(symbol, start_query, end_query)
            elif not bars:
                meta["source"] = "none"
                meta["error"] = "No ^GSPC data from DB or Yahoo"
                return [], meta
        except Exception as ex:
            logger.warning("Yahoo fetch failed for ^GSPC benchmark: %s", ex, exc_info=True)
            meta["error"] = str(ex)
            if len(bars) < 2:
                return [], meta

    if len(bars) < 2:
        meta["error"] = meta.get("error") or "Insufficient ^GSPC bars in range"
        return [], meta

    # Restrict to window [t0, t1] (inclusive by calendar date)
    t0d = t0.date()
    t1d = t1.date()

    filtered = []
    for bt, close in bars:
        bd = bt.date() if isinstance(bt, datetime) else bt
        if t0d <= bd <= t1d:
            filtered.append((bt, close))
    if len(filtered) < 2:
        # Wider window: include any daily bar whose calendar date overlaps [t0d, t1d] using full query range
        filtered = list(bars)
    if len(filtered) < 2:
        meta["error"] = meta.get("error") or "No ^GSPC bars inside equity window"
        return [], meta

    base_close = float(filtered[0][1])
    if base_close <= 0:
        return [], {**meta, "error": "Invalid benchmark base close"}

    curve: List[Dict[str, Any]] = []
    for bt, close in filtered:
        eq = initial_capital * (float(close) / base_close)
        ts_out = bt
        if timezone.is_naive(ts_out):
            ts_out = timezone.make_aware(ts_out)
        curve.append(
            {
                "timestamp": ts_out.isoformat(),
                "equity": round(eq, 2),
            }
        )

    return curve, meta
