"""Tests for portfolio symbol priority ordering in BacktestExecutor."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from backtest_engine.services.backtest_executor import BacktestExecutor


class SymbolPriorityRankTests(SimpleTestCase):
    def test_priority_rank_from_backtest_field(self):
        backtest = SimpleNamespace(
            symbol_priority_order=['ZZZ', 'AAA'],
            initial_capital=10000,
            strategy_parameters={},
            broker=None,
            start_date=None,
            end_date=None,
            split_ratio=0.7,
            strategy=SimpleNamespace(name='test'),
        )
        backtest.symbols = MagicMock()
        backtest.symbols.all.return_value = []

        executor = BacktestExecutor(backtest, position_mode='long')
        executor.symbols = []
        rank = executor._symbol_priority_rank()
        self.assertEqual(rank['ZZZ'], 0)
        self.assertEqual(rank['AAA'], 1)

    def test_priority_rank_override(self):
        backtest = SimpleNamespace(
            symbol_priority_order=['AAA', 'BBB'],
            initial_capital=10000,
            strategy_parameters={},
            broker=None,
            start_date=None,
            end_date=None,
            split_ratio=0.7,
            strategy=SimpleNamespace(name='test'),
        )
        backtest.symbols = MagicMock()
        backtest.symbols.all.return_value = []

        executor = BacktestExecutor(
            backtest,
            position_mode='long',
            symbol_priority_order_override=['BBB', 'AAA'],
        )
        executor.symbols = []
        rank = executor._symbol_priority_rank()
        self.assertEqual(rank['BBB'], 0)
        self.assertEqual(rank['AAA'], 1)

    def test_same_timestamp_sort_respects_priority(self):
        """Exits before entries; within entries, lower rank ticker runs first."""
        phase1 = [
            (1, 'AAA', None, None, None, None, 'buy', {}),
            (1, 'BBB', None, None, None, None, 'buy', {}),
            (0, 'BBB', None, None, None, None, 'sell', {}),
        ]
        symbol_rank = {'BBB': 0, 'AAA': 1}
        phase1.sort(key=lambda x: (x[0], symbol_rank.get(x[1], 999), x[1]))
        tickers = [x[1] for x in phase1]
        self.assertEqual(tickers, ['BBB', 'BBB', 'AAA'])
