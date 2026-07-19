"""Golden-value unit tests for `analytical_tools.indicators`.

Each indicator is exercised against a small, hand-computable series so that
any drift in the formula (e.g. SMA vs EMA confusion, wrong shift direction)
surfaces immediately.
"""

from __future__ import annotations

import math

import pandas as pd
from django.test import TestCase

from analytical_tools.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_returns,
    calculate_rsi,
    calculate_rolling_std,
    calculate_sma,
)


def _series(values):
    return pd.Series(values, dtype=float)


class SimpleMovingAverageTestCase(TestCase):
    def test_sma_period_3(self):
        s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = calculate_sma(s, period=3)
        # First two are NaN; the rest are means of the trailing 3.
        self.assertTrue(pd.isna(sma.iloc[0]))
        self.assertTrue(pd.isna(sma.iloc[1]))
        self.assertAlmostEqual(sma.iloc[2], 2.0)
        self.assertAlmostEqual(sma.iloc[3], 3.0)
        self.assertAlmostEqual(sma.iloc[4], 4.0)


class ExponentialMovingAverageTestCase(TestCase):
    def test_ema_first_value_equals_price(self):
        # adjust=False => EMA seeded with the first price.
        s = _series([10.0, 20.0, 30.0])
        ema = calculate_ema(s, period=3)
        self.assertAlmostEqual(ema.iloc[0], 10.0)
        # EMA moves toward later values but stays between min and max.
        self.assertGreater(ema.iloc[2], ema.iloc[0])
        self.assertLess(ema.iloc[2], 30.0)


class RsiTestCase(TestCase):
    def test_all_up_moves_yields_high_rsi(self):
        # Strictly increasing prices -> gain>0, loss=0 -> RSI -> 100 (after warmup).
        prices = _series([1.0 + i for i in range(20)])
        rsi = calculate_rsi(prices, period=5)
        # After the rolling window fills, loss==0 makes RSI 100.
        self.assertEqual(rsi.iloc[-1], 100.0)

    def test_all_down_moves_yields_low_rsi(self):
        prices = _series([20.0 - i for i in range(20)])
        rsi = calculate_rsi(prices, period=5)
        self.assertEqual(rsi.iloc[-1], 0.0)


class MacdTestCase(TestCase):
    def test_macd_components_keys(self):
        prices = _series([float(i) for i in range(50)])
        out = calculate_macd(prices)
        self.assertEqual(set(out.keys()), {'macd', 'signal', 'histogram'})
        # histogram == macd - signal
        pd.testing.assert_series_equal(
            out['histogram'], out['macd'] - out['signal'], check_names=False,
        )


class BollingerBandsTestCase(TestCase):
    def test_bands_symmetric_around_middle(self):
        prices = _series([10.0, 11.0, 12.0, 11.0, 10.0, 11.0, 12.0, 11.0, 10.0, 11.0])
        bb = calculate_bollinger_bands(prices, period=5, num_std=2.0)
        # upper - middle == middle - lower (symmetric)
        diff_up = (bb['upper'] - bb['middle']).dropna()
        diff_lo = (bb['middle'] - bb['lower']).dropna()
        pd.testing.assert_series_equal(diff_up, diff_lo, check_names=False)


class ReturnsTestCase(TestCase):
    def test_gap_return_formula(self):
        opens = _series([100.0, 110.0, 121.0])
        closes = _series([110.0, 121.0, 133.1])
        r = calculate_returns(opens, closes)
        # First return is NaN (no prev close). Then (110-110)/110=0, (121-121)/121=0.
        self.assertTrue(pd.isna(r.iloc[0]))
        self.assertAlmostEqual(r.iloc[1], 0.0)
        self.assertAlmostEqual(r.iloc[2], 0.0)

    def test_gap_return_nonzero(self):
        opens = _series([100.0, 105.0])
        closes = _series([100.0, 100.0])
        r = calculate_returns(opens, closes)
        # (105 - 100) / 100 = 0.05
        self.assertAlmostEqual(r.iloc[1], 0.05)


class RollingStdTestCase(TestCase):
    def test_rolling_std_constant_series_is_zero(self):
        returns = _series([0.01] * 20)
        std = calculate_rolling_std(returns, period=5)
        self.assertAlmostEqual(std.iloc[-1], 0.0)


class AtrTestCase(TestCase):
    def test_atr_constant_range(self):
        # Each bar has high-low = 1, no gaps -> ATR = 1 after warmup.
        high = _series([11.0, 11.0, 11.0, 11.0, 11.0])
        low = _series([10.0, 10.0, 10.0, 10.0, 10.0])
        close = _series([10.5, 10.5, 10.5, 10.5, 10.5])
        atr = calculate_atr(high, low, close, period=3)
        self.assertAlmostEqual(atr.iloc[-1], 1.0)
