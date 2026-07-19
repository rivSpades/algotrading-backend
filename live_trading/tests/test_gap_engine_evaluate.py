"""Tests for the `GapUpGapDownLiveEngine.evaluate()` path.

Covers the skipped_reason branches and one happy-path long entry. Indicator
computation is mocked so we don't depend on the full indicator pipeline; the
broker session-open price is provided by a stub adapter.
"""

from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from unittest import mock

import pandas as pd
from django.test import TestCase
from django.utils import timezone

from backtest_engine.models import SymbolBacktestParameterSet
from live_trading.engines.base import SIGNAL_HOLD, SIGNAL_LONG
from live_trading.engines.gap_up_gap_down import GapUpGapDownLiveEngine
from live_trading.models import (
    Broker,
    DeploymentSymbol,
    StrategyDeployment,
    SymbolBrokerAssociation,
)
from live_trading.tests.mock_broker_adapter import PaperTestBrokerAdapter
from market_data.models import Exchange, ExchangeSchedule, OHLCV, Provider, Symbol
from strategies.models import StrategyDefinition


def _parm_sig(salt: str) -> str:
    import hashlib
    return hashlib.sha256(salt.encode()).hexdigest()


class GapEngineEvaluateTestCase(TestCase):
    def setUp(self):
        self.exchange = Exchange.objects.create(name='Test Ex', code='TEX')
        self.provider = Provider.objects.create(name='Yahoo', code='YAHOO')
        self.sym = Symbol.objects.create(
            ticker='GAPX',
            exchange=self.exchange,
            provider=self.provider,
            status='active',
        )
        self.broker = Broker.objects.create(
            name='Paper',
            code='PBGAP',
            paper_trading_endpoint_url='https://paper.test',
            paper_trading_api_key='k',
            paper_trading_secret_key='s',
            paper_trading_active=True,
        )
        SymbolBrokerAssociation.objects.create(
            symbol=self.sym, broker=self.broker, long_active=True, short_active=True,
        )
        ExchangeSchedule.objects.create(
            exchange=self.exchange,
            open_utc=_dt.time(13, 30),
            close_utc=_dt.time(20, 0),
            weekdays='1,2,3,4,5',
            active=True,
        )
        self.strategy, _ = StrategyDefinition.objects.get_or_create(
            name='Gap-Up and Gap-Down',
            defaults={'description_short': '', 'globally_enabled': True},
        )
        self.param_set = SymbolBacktestParameterSet.objects.create(
            signature=_parm_sig('gapx-unit'),
            label='test',
            strategy=self.strategy,
            broker=self.broker,
            parameters={'std_period': 90, 'threshold': 0.25},
        )
        self.deployment = StrategyDeployment.objects.create(
            name='Gap Deploy',
            strategy=self.strategy,
            parameter_set=self.param_set,
            broker=self.broker,
            position_mode='long',
            deployment_type='paper',
            status='active',
            initial_capital=Decimal('100000'),
            bet_size_percentage=100.0,
            strategy_parameters={'std_period': 90, 'threshold': 0.25},
            hedge_enabled=False,
        )
        self.ds = DeploymentSymbol.objects.create(
            deployment=self.deployment,
            symbol=self.sym,
            position_mode='long',
            status='active',
            priority=1,
        )
        self.adapter = PaperTestBrokerAdapter(self.broker, quote=Decimal('100'))

    def _make_engine(self):
        return GapUpGapDownLiveEngine(self.deployment, broker_adapter=self.adapter)

    def _seed_ohlcv(self, n_rows: int, close: float = 100.0):
        now = timezone.now()
        for i in range(n_rows):
            OHLCV.objects.create(
                symbol=self.sym,
                timestamp=now - _dt.timedelta(days=n_rows - i),
                timeframe='daily',
                open=Decimal(str(close - 1)),
                high=Decimal(str(close + 1)),
                low=Decimal(str(close - 2)),
                close=Decimal(str(close)),
                volume=1000,
            )

    def test_insufficient_history_skipped(self):
        self._seed_ohlcv(2)  # < min_bars_required (5)
        engine = self._make_engine()
        ev = engine.evaluate(self.ds, timezone.now())
        self.assertEqual(ev.skipped_reason, 'insufficient_history')
        self.assertEqual(ev.signal.action, SIGNAL_HOLD)

    def test_missing_open_skipped_when_no_broker_open(self):
        self._seed_ohlcv(10)
        # Adapter returns None for session open -> missing_open_or_prev_close.
        self.adapter.get_session_open_price = lambda *a, **k: None
        engine = self._make_engine()
        ev = engine.evaluate(self.ds, timezone.now())
        self.assertEqual(ev.skipped_reason, 'missing_open_or_prev_close')

    def test_invalid_std_skipped(self):
        self._seed_ohlcv(10)
        self.adapter.get_session_open_price = lambda *a, **k: Decimal('101')
        engine = self._make_engine()
        with mock.patch(
            'market_data.services.indicator_service.compute_strategy_indicators_for_ohlcv',
            return_value={},
        ):
            ev = engine.evaluate(self.ds, timezone.now())
        self.assertEqual(ev.skipped_reason, 'invalid_std')

    def test_happy_path_long_entry(self):
        self._seed_ohlcv(10, close=100.0)
        # today_open=101 -> returns = 0.01 > 0.0025 (threshold*std with std=0.01) -> long entry.
        self.adapter.get_session_open_price = lambda *a, **k: Decimal('101')
        std_period = 90
        engine = self._make_engine()
        fake_indicators = {
            f'RollingSTD_{std_period}': {'values': [0.01]},
            'Returns': {'values': [0.01]},
        }
        with mock.patch(
            'market_data.services.indicator_service.compute_strategy_indicators_for_ohlcv',
            return_value=fake_indicators,
        ):
            ev = engine.evaluate(self.ds, timezone.now())
        self.assertIsNone(ev.skipped_reason)
        self.assertEqual(ev.signal.action, SIGNAL_LONG)
        self.assertEqual(ev.signal.price, Decimal('101'))

    def test_indicator_failure_skipped(self):
        self._seed_ohlcv(10)
        self.adapter.get_session_open_price = lambda *a, **k: Decimal('101')
        engine = self._make_engine()
        with mock.patch(
            'market_data.services.indicator_service.compute_strategy_indicators_for_ohlcv',
            side_effect=RuntimeError('boom'),
        ):
            ev = engine.evaluate(self.ds, timezone.now())
        self.assertEqual(ev.skipped_reason, 'indicator_failure')
