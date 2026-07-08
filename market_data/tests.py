"""Tests for idempotent benchmark symbol helpers."""

from django.db import IntegrityError
from django.test import TestCase

from market_data.models import Exchange, Symbol
from market_data.services.benchmark_symbols import (
    BENCHMARK_EXCHANGE_CODE,
    BENCHMARK_EXCHANGE_NAME,
    get_or_create_benchmark_exchange,
    get_or_create_benchmark_symbol,
)


class BenchmarkSymbolsTestCase(TestCase):
    def test_get_or_create_exchange_by_existing_name(self):
        Exchange.objects.create(
            code="OTHER",
            name=BENCHMARK_EXCHANGE_NAME,
            country="US",
            timezone="America/New_York",
        )
        exchange = get_or_create_benchmark_exchange()
        self.assertEqual(exchange.name, BENCHMARK_EXCHANGE_NAME)
        self.assertEqual(Exchange.objects.count(), 1)

    def test_get_or_create_exchange_race_recovery(self):
        Exchange.objects.create(
            code=BENCHMARK_EXCHANGE_CODE,
            name=BENCHMARK_EXCHANGE_NAME,
            country="US",
            timezone="America/New_York",
        )
        exchange = get_or_create_benchmark_exchange()
        self.assertEqual(exchange.code, BENCHMARK_EXCHANGE_CODE)

    def test_get_or_create_symbol_existing_ticker(self):
        exchange = Exchange.objects.create(
            code="NYSE",
            name="NYSE",
            country="US",
            timezone="America/New_York",
        )
        Symbol.objects.create(
            ticker="SPY",
            exchange=exchange,
            type="etf",
            name="SPY",
            status="active",
        )
        sym = get_or_create_benchmark_symbol("SPY")
        self.assertEqual(sym.ticker, "SPY")
        self.assertEqual(Symbol.objects.filter(ticker="SPY").count(), 1)

    def test_get_or_create_symbol_integrity_error_recovery(self):
        exchange = Exchange.objects.create(
            code=BENCHMARK_EXCHANGE_CODE,
            name=BENCHMARK_EXCHANGE_NAME,
            country="US",
            timezone="America/New_York",
        )
        Symbol.objects.create(
            ticker="VIXY",
            exchange=exchange,
            type="etf",
            name="VIXY",
            status="active",
        )

        original = Symbol.objects.get_or_create

        def race_get_or_create(*args, **kwargs):
            if kwargs.get("ticker") == "VIXY":
                raise IntegrityError("duplicate key")
            return original(*args, **kwargs)

        Symbol.objects.get_or_create = race_get_or_create
        try:
            sym = get_or_create_benchmark_symbol("VIXY")
        finally:
            Symbol.objects.get_or_create = original

        self.assertEqual(sym.ticker, "VIXY")
