"""Tests for portfolio Monte Carlo aggregate helpers."""

from django.test import SimpleTestCase

from backtest_engine.services.portfolio_monte_carlo import _mean_performance_metrics


class MeanPerformanceMetricsTests(SimpleTestCase):
    def test_mean_includes_reference_run(self):
        paths = [
            {
                'is_reference': True,
                'performance_metrics': {
                    'total_trades': 100,
                    'total_pnl': 2000.0,
                    'win_rate': 50.0,
                },
            },
            {
                'is_reference': False,
                'performance_metrics': {
                    'total_trades': 100,
                    'total_pnl': 1000.0,
                    'win_rate': 40.0,
                },
            },
        ]
        mean = _mean_performance_metrics(paths)
        self.assertAlmostEqual(mean['total_pnl'], 1500.0, places=2)
        self.assertAlmostEqual(mean['win_rate'], 45.0, places=4)
        self.assertAlmostEqual(mean['total_trades'], 100.0, places=2)

    def test_mean_empty_when_no_metrics(self):
        self.assertEqual(_mean_performance_metrics([{'is_reference': True}]), {})
