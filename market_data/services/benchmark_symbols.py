"""Idempotent benchmark exchange and symbol helpers for backtests and hedge overlays."""

from __future__ import annotations

from django.db import IntegrityError
from django.db.models import Q

from market_data.models import Exchange, Symbol
from market_data.services.ohlcv_service import OHLCVService

BENCHMARK_EXCHANGE_CODE = "BENCHMARK"
BENCHMARK_EXCHANGE_NAME = "Benchmark indices"


def get_or_create_benchmark_exchange() -> Exchange:
    """Return the shared benchmark exchange, safe under concurrent workers."""
    existing = Exchange.objects.filter(
        Q(code=BENCHMARK_EXCHANGE_CODE) | Q(name=BENCHMARK_EXCHANGE_NAME)
    ).first()
    if existing is not None:
        return existing
    try:
        exchange, _ = Exchange.objects.get_or_create(
            code=BENCHMARK_EXCHANGE_CODE,
            defaults={
                "name": BENCHMARK_EXCHANGE_NAME,
                "country": "US",
                "timezone": "America/New_York",
            },
        )
        return exchange
    except IntegrityError:
        return Exchange.objects.get(
            Q(code=BENCHMARK_EXCHANGE_CODE) | Q(name=BENCHMARK_EXCHANGE_NAME)
        )


def get_or_create_benchmark_symbol(
    ticker: str,
    *,
    name: str | None = None,
    symbol_type: str = "etf",
) -> Symbol:
    """Ensure a benchmark/hedge symbol exists; safe under concurrent workers."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    exchange = get_or_create_benchmark_exchange()
    provider = OHLCVService.get_or_create_yahoo_provider()
    defaults = {
        "exchange": exchange,
        "provider": provider,
        "type": symbol_type,
        "name": name or ticker,
        "status": "active",
    }
    try:
        sym, _ = Symbol.objects.get_or_create(ticker=ticker, defaults=defaults)
        return sym
    except IntegrityError:
        return Symbol.objects.get(ticker=ticker)
