"""Tests for unified gap strategy signal blackbox."""

from django.test import SimpleTestCase

from strategies.signals import (
    GAP_STRATEGY_NAME,
    StrategySignalContext,
    SignalAction,
    PositionState,
    check_strategy_signal,
    to_backtest_order,
    to_live_action,
)
from strategies.signals.rules.gap import (
    classify_gap_position_action,
    gap_up_gap_down_decision,
)


def _gap_ctx(
    *,
    returns=0.02,
    std=0.02,
    threshold=0.25,
    position_mode='long',
    position_state=PositionState.FLAT,
    long_allowed=True,
    short_allowed=True,
):
    return StrategySignalContext(
        returns=returns,
        std=std,
        threshold=threshold,
        position_mode=position_mode,
        position_state=position_state,
        long_allowed=long_allowed,
        short_allowed=short_allowed,
    )


class GapRuleParityTests(SimpleTestCase):
    """Pure rule algebra (migrated from live_trading.tests.test_gap_rule_parity)."""

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
        dec = gap_up_gap_down_decision(returns=0.001, std=0.02, threshold=0.25)
        self.assertFalse(dec.long_signal)
        self.assertFalse(dec.short_signal)
        self.assertIsNone(dec.direction)

    def test_long_direction_when_above_band(self):
        dec = gap_up_gap_down_decision(returns=0.02, std=0.02, threshold=0.25)
        self.assertTrue(dec.long_signal)
        self.assertEqual(dec.direction, 'long')

    def test_short_direction_when_below_band(self):
        dec = gap_up_gap_down_decision(returns=-0.02, std=0.02, threshold=0.25)
        self.assertTrue(dec.short_signal)
        self.assertEqual(dec.direction, 'short')


class GapClassificationMatrixTests(SimpleTestCase):
    """Position/mode/broker classification shared by backtest and live."""

    def _decision_long(self):
        return gap_up_gap_down_decision(returns=0.02, std=0.02, threshold=0.25)

    def _decision_short(self):
        return gap_up_gap_down_decision(returns=-0.02, std=0.02, threshold=0.25)

    def test_gap_up_flat_long_mode(self):
        action, _ = classify_gap_position_action(
            decision=self._decision_long(),
            position_state=PositionState.FLAT,
            position_mode='long',
            long_allowed=True,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.LONG)

    def test_gap_down_flat_short_mode(self):
        action, _ = classify_gap_position_action(
            decision=self._decision_short(),
            position_state=PositionState.FLAT,
            position_mode='short',
            long_allowed=True,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.SHORT)

    def test_long_open_opposite_signal_exits(self):
        action, reason = classify_gap_position_action(
            decision=self._decision_short(),
            position_state=PositionState.LONG,
            position_mode='long',
            long_allowed=True,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.EXIT_LONG)
        self.assertEqual(reason, 'opposite_signal_long')

    def test_long_open_mode_short_exits(self):
        action, _ = classify_gap_position_action(
            decision=self._decision_long(),
            position_state=PositionState.LONG,
            position_mode='short',
            long_allowed=True,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.EXIT_LONG)

    def test_broker_disabled_holds_entry(self):
        action, reason = classify_gap_position_action(
            decision=self._decision_long(),
            position_state=PositionState.FLAT,
            position_mode='long',
            long_allowed=False,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.HOLD)
        self.assertEqual(reason, 'long_disabled_by_broker')

    def test_live_all_mode_allows_long(self):
        action, _ = classify_gap_position_action(
            decision=self._decision_long(),
            position_state=PositionState.FLAT,
            position_mode='all',
            long_allowed=True,
            short_allowed=True,
        )
        self.assertEqual(action, SignalAction.LONG)


class AdapterParityTests(SimpleTestCase):
    """Backtest buy/sell and live action strings stay consistent."""

    def test_check_signal_registered(self):
        self.assertIn(GAP_STRATEGY_NAME, __import__(
            'strategies.signals.registry',
            fromlist=['StrategySignalRegistry'],
        ).StrategySignalRegistry.names())

    def test_long_entry_adapters(self):
        result = check_strategy_signal(GAP_STRATEGY_NAME, _gap_ctx())
        self.assertEqual(result.action, SignalAction.LONG)
        self.assertEqual(to_backtest_order(result), 'buy')
        self.assertEqual(to_live_action(result), 'long')

    def test_short_entry_adapters(self):
        result = check_strategy_signal(
            GAP_STRATEGY_NAME,
            _gap_ctx(returns=-0.02, position_mode='short'),
        )
        self.assertEqual(result.action, SignalAction.SHORT)
        self.assertEqual(to_backtest_order(result), 'sell')
        self.assertEqual(to_live_action(result), 'short')

    def test_exit_long_adapters(self):
        result = check_strategy_signal(
            GAP_STRATEGY_NAME,
            _gap_ctx(
                returns=-0.02,
                position_mode='long',
                position_state=PositionState.LONG,
            ),
        )
        self.assertEqual(result.action, SignalAction.EXIT_LONG)
        self.assertEqual(to_backtest_order(result), 'sell')
        self.assertEqual(to_live_action(result), 'exit_long')

    def test_hold_adapters(self):
        result = check_strategy_signal(
            GAP_STRATEGY_NAME,
            _gap_ctx(returns=0.001, long_allowed=False),
        )
        self.assertEqual(result.action, SignalAction.HOLD)
        self.assertIsNone(to_backtest_order(result))
        self.assertEqual(to_live_action(result), 'hold')
