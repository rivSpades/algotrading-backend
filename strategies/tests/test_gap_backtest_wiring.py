"""Backtest executor wiring for unified gap signals."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from backtest_engine.services.backtest_executor import BacktestExecutor


class GapBacktestWiringTests(SimpleTestCase):
    """Ensure BacktestExecutor._gap_up_gap_down_signal uses the blackbox."""

    def _executor(self, position_mode='long'):
        backtest = SimpleNamespace(
            strategy=SimpleNamespace(name='Gap-Up and Gap-Down'),
            strategy_parameters={'threshold': 0.25, 'std_period': 90},
            symbols=MagicMock(),
            broker=None,
        )
        ex = BacktestExecutor.__new__(BacktestExecutor)
        ex.backtest = backtest
        ex.strategy = backtest.strategy
        ex.parameters = backtest.strategy_parameters
        ex.position_mode = position_mode
        ex.broker = None
        return ex

    def test_long_entry_buy(self):
        ex = self._executor('long')
        order = ex._gap_up_gap_down_signal(
            row=MagicMock(),
            indicators={'Returns': 0.02, 'RollingSTD_90': 0.02},
            position=None,
            symbol=None,
        )
        self.assertEqual(order, 'buy')

    def test_short_entry_sell(self):
        ex = self._executor('short')
        order = ex._gap_up_gap_down_signal(
            row=MagicMock(),
            indicators={'Returns': -0.02, 'RollingSTD_90': 0.02},
            position=None,
            symbol=None,
        )
        self.assertEqual(order, 'sell')

    def test_exit_long_on_opposite(self):
        ex = self._executor('long')
        order = ex._gap_up_gap_down_signal(
            row=MagicMock(),
            indicators={'Returns': -0.02, 'RollingSTD_90': 0.02},
            position={'type': 'buy'},
            symbol=None,
        )
        self.assertEqual(order, 'sell')

    def test_no_signal_in_noise(self):
        ex = self._executor('long')
        order = ex._gap_up_gap_down_signal(
            row=MagicMock(),
            indicators={'Returns': 0.001, 'RollingSTD_90': 0.02},
            position=None,
            symbol=None,
        )
        self.assertIsNone(order)
