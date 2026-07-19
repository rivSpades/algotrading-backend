"""
Golden regression: Gap-Up/Gap-Down trades, statistics, and equity curve.

Goldens live under fixtures/. Regenerate only after intentional engine changes:
  python manage.py dump_gap_golden
"""

import json

from django.test import TestCase

from backtest_engine.tests.golden_harness import (
    GOLDEN_LONG_PATH,
    GOLDEN_SHORT_PATH,
    SYNTHETIC_OHLCV_PATH,
    assert_golden_match,
    dump_golden_files,
    run_gap_executor,
)


class GapGoldenResultsTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Ensure fixtures exist (first checkout / missing files).
        if not SYNTHETIC_OHLCV_PATH.exists() or not GOLDEN_LONG_PATH.exists() or not GOLDEN_SHORT_PATH.exists():
            dump_golden_files()

    def test_long_mode_matches_golden(self):
        expected = json.loads(GOLDEN_LONG_PATH.read_text())
        actual = run_gap_executor('long')
        assert_golden_match(actual, expected)
        # Sanity: engineered gaps should produce at least one closed long
        self.assertGreaterEqual(len(actual['trades']), 1)
        self.assertTrue(any(t['trade_type'] == 'buy' for t in actual['trades']))

    def test_short_mode_matches_golden(self):
        expected = json.loads(GOLDEN_SHORT_PATH.read_text())
        actual = run_gap_executor('short')
        assert_golden_match(actual, expected)
        self.assertGreaterEqual(len(actual['trades']), 1)
        self.assertTrue(any(t['trade_type'] == 'sell' for t in actual['trades']))

    def test_equity_curve_bookends(self):
        actual = run_gap_executor('long')
        curve = actual['stats']['equity_curve']
        self.assertTrue(curve)
        self.assertAlmostEqual(curve[0]['equity'], 10000.0, places=2)
        total_pnl = actual['stats']['scalars']['total_pnl'] or 0.0
        self.assertAlmostEqual(curve[-1]['equity'], 10000.0 + float(total_pnl), places=2)
