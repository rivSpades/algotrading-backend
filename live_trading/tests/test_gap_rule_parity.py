from django.test import SimpleTestCase

from live_trading.engines._rules import gap_up_gap_down_decision


class GapRuleParityTests(SimpleTestCase):
    def test_decision_matches_raw_threshold_algebra(self):
        threshold = 0.25
        std_f = 0.02
        returns_f = 0.01
        long_threshold_manual = threshold * std_f
        short_threshold_manual = -threshold * std_f
        dec = gap_up_gap_down_decision(
            returns=returns_f,
            std=std_f,
            threshold=threshold,
        )
        self.assertAlmostEqual(dec.long_threshold, long_threshold_manual)
        self.assertAlmostEqual(dec.short_threshold, short_threshold_manual)
        self.assertEqual(dec.long_signal, returns_f > long_threshold_manual)
        self.assertEqual(dec.short_signal, returns_f < short_threshold_manual)

    def test_exclusive_long_when_both_signals_false(self):
        dec = gap_up_gap_down_decision(
            returns=0.001,
            std=0.02,
            threshold=0.25,
        )
        self.assertFalse(dec.long_signal)
        self.assertFalse(dec.short_signal)
        self.assertIsNone(dec.direction)

    def test_long_direction_when_above_band(self):
        dec = gap_up_gap_down_decision(
            returns=0.02,
            std=0.02,
            threshold=0.25,
        )
        self.assertTrue(dec.long_signal)
        self.assertEqual(dec.direction, 'long')

    def test_short_direction_when_below_band(self):
        dec = gap_up_gap_down_decision(
            returns=-0.02,
            std=0.02,
            threshold=0.25,
        )
        self.assertTrue(dec.short_signal)
        self.assertEqual(dec.direction, 'short')
