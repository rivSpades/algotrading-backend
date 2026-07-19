"""Pure statistics-helper regression tests (no DB, no strategy signals)."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import SimpleTestCase
from django.utils import timezone

from backtest_engine.services.backtest_executor import BacktestExecutor


def _aware(year, month, day):
    return timezone.make_aware(datetime(year, month, day), timezone=timezone.UTC)


class StatisticsHelpersTests(SimpleTestCase):
    def _executor(self, initial_capital=10000.0):
        backtest = SimpleNamespace(
            strategy=SimpleNamespace(name='Gap-Up and Gap-Down'),
            strategy_parameters={'threshold': 0.25, 'std_period': 90},
            symbols=MagicMock(),
            broker=None,
            initial_capital=initial_capital,
            bet_size_percentage=100.0,
            split_ratio=0.7,
            start_date=None,
            end_date=None,
            hedge_enabled=False,
        )
        backtest.symbols.all.return_value = []
        ex = BacktestExecutor.__new__(BacktestExecutor)
        ex.backtest = backtest
        ex.strategy = backtest.strategy
        ex.parameters = backtest.strategy_parameters
        ex.position_mode = 'long'
        ex.broker = None
        ex.trades = []
        ex.symbols = []
        ex.skipped_trades_count = 0
        return ex

    def test_cumulative_equity_curve_from_closed_trades(self):
        ex = self._executor(10000.0)
        sym = SimpleNamespace(ticker='TEST')
        t1 = _aware(2024, 1, 10)
        t2 = _aware(2024, 1, 20)
        t3 = _aware(2024, 2, 1)
        t4 = _aware(2024, 2, 15)
        trades = [
            {
                'symbol': sym,
                'trade_type': 'buy',
                'entry_timestamp': t1,
                'exit_timestamp': t2,
                'entry_price': 100.0,
                'exit_price': 110.0,
                'quantity': 10.0,
                'pnl': 100.0,
                'pnl_percentage': 10.0,
                'is_winner': True,
            },
            {
                'symbol': sym,
                'trade_type': 'buy',
                'entry_timestamp': t3,
                'exit_timestamp': t4,
                'entry_price': 100.0,
                'exit_price': 90.0,
                'quantity': 10.0,
                'pnl': -100.0,
                'pnl_percentage': -10.0,
                'is_winner': False,
            },
        ]
        total_pnl = 0.0
        curve = ex._build_cumulative_equity_curve_from_trades(trades, 10000.0, total_pnl)
        self.assertGreaterEqual(len(curve), 2)
        self.assertAlmostEqual(curve[0]['equity'], 10000.0, places=2)
        self.assertAlmostEqual(curve[-1]['equity'], 10000.0 + total_pnl, places=2)
        # After first exit equity should be 10100 before second exit
        mid = next(p for p in curve if abs(p['equity'] - 10100.0) < 0.01)
        self.assertIsNotNone(mid)

    def test_drawdown_peak_to_trough(self):
        ex = self._executor()
        curve = [
            (_aware(2024, 1, 1), 10000.0),
            (_aware(2024, 1, 2), 11000.0),
            (_aware(2024, 1, 3), 9900.0),
            (_aware(2024, 1, 4), 10500.0),
        ]
        max_dd, duration = ex._calculate_drawdown(curve)
        # Peak 11000 → trough 9900 = 10% drawdown
        self.assertAlmostEqual(max_dd, (11000.0 - 9900.0) / 11000.0 * 100, places=4)
        self.assertGreaterEqual(duration, 1)

    def test_symbol_statistics_win_rate_and_profit_factor(self):
        ex = self._executor(10000.0)
        sym = SimpleNamespace(ticker='TEST')
        trades = [
            {
                'symbol': sym,
                'trade_type': 'buy',
                'entry_timestamp': _aware(2024, 1, 1),
                'exit_timestamp': _aware(2024, 1, 5),
                'entry_price': 100.0,
                'exit_price': 110.0,
                'quantity': 10.0,
                'pnl': 100.0,
                'pnl_percentage': 10.0,
                'is_winner': True,
                'max_drawdown': 1.0,
            },
            {
                'symbol': sym,
                'trade_type': 'buy',
                'entry_timestamp': _aware(2024, 2, 1),
                'exit_timestamp': _aware(2024, 2, 5),
                'entry_price': 100.0,
                'exit_price': 95.0,
                'quantity': 10.0,
                'pnl': -50.0,
                'pnl_percentage': -5.0,
                'is_winner': False,
                'max_drawdown': 5.0,
            },
        ]
        stats = ex._calculate_symbol_statistics(sym, trades)
        self.assertEqual(stats['total_trades'], 2)
        self.assertEqual(stats['winning_trades'], 1)
        self.assertEqual(stats['losing_trades'], 1)
        self.assertAlmostEqual(stats['win_rate'], 50.0, places=2)
        self.assertAlmostEqual(stats['total_pnl'], 50.0, places=4)
        self.assertAlmostEqual(stats['profit_factor'], 100.0 / 50.0, places=4)
        self.assertAlmostEqual(stats['total_return'], 0.5, places=2)
        self.assertTrue(stats['equity_curve'])
        self.assertAlmostEqual(stats['equity_curve'][0]['equity'], 10000.0, places=2)
        self.assertAlmostEqual(stats['equity_curve'][-1]['equity'], 10050.0, places=2)
