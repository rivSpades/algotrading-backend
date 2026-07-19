"""Unit tests for pure money-handling helpers in `order_service`.

These functions are the core of position sizing and PnL math; they must be
deterministic and side-effect-free, so we test them directly without a broker
or DB where possible. Where a `StrategyDeployment` is needed we build a
minimal one with a stub broker adapter.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

from django.test import TestCase

from live_trading.engines.base import SIGNAL_LONG, SIGNAL_SHORT
from live_trading.services.order_service import (
    _compute_pnl,
    _compute_quantity,
    _intended_closing_side,
    _safe_pct,
)


class _StubAdapter:
    """Just enough of BaseBrokerAdapter for sizing helpers."""

    def __init__(self, equity: Decimal = Decimal('100000'), caps: dict | None = None):
        self._equity = equity
        self._caps = caps or {}

    def get_account_equity(self) -> Decimal:
        return self._equity

    def get_symbol_capabilities(self, ticker: str) -> dict:
        return self._caps


class _DeploymentStub:
    """Lightweight stand-in for `StrategyDeployment` (no DB needed)."""

    def __init__(self, bet_size_percentage, initial_capital):
        self.bet_size_percentage = bet_size_percentage
        self.initial_capital = initial_capital


class ComputePnlTestCase(TestCase):
    def test_long_profit(self):
        pnl = _compute_pnl(Decimal('100'), Decimal('110'), Decimal('10'), SIGNAL_LONG)
        self.assertEqual(pnl, Decimal('100'))

    def test_long_loss(self):
        pnl = _compute_pnl(Decimal('100'), Decimal('90'), Decimal('10'), SIGNAL_LONG)
        self.assertEqual(pnl, Decimal('-100'))

    def test_short_profit(self):
        # Short: direction -1, exit below entry -> profit
        pnl = _compute_pnl(Decimal('100'), Decimal('90'), Decimal('10'), SIGNAL_SHORT)
        self.assertEqual(pnl, Decimal('100'))

    def test_short_loss(self):
        pnl = _compute_pnl(Decimal('100'), Decimal('110'), Decimal('10'), SIGNAL_SHORT)
        self.assertEqual(pnl, Decimal('-100'))

    def test_none_inputs(self):
        self.assertIsNone(_compute_pnl(None, Decimal('110'), Decimal('10'), SIGNAL_LONG))
        self.assertIsNone(_compute_pnl(Decimal('100'), None, Decimal('10'), SIGNAL_LONG))
        self.assertIsNone(_compute_pnl(Decimal('100'), Decimal('110'), None, SIGNAL_LONG))


class SafePctTestCase(TestCase):
    def test_long_pct(self):
        pct = _safe_pct(Decimal('100'), Decimal('110'), SIGNAL_LONG)
        self.assertEqual(pct, Decimal('10.0000'))

    def test_short_pct(self):
        pct = _safe_pct(Decimal('100'), Decimal('90'), SIGNAL_SHORT)
        self.assertEqual(pct, Decimal('10.0000'))

    def test_zero_entry_returns_none(self):
        self.assertIsNone(_safe_pct(Decimal('0'), Decimal('10'), SIGNAL_LONG))


class ComputeQuantityTestCase(TestCase):
    def test_fractional_sizing(self):
        dep = _DeploymentStub(bet_size_percentage=5, initial_capital=Decimal('100000'))
        adapter = _StubAdapter(equity=Decimal('100000'), caps={'fractionable': True})
        qty = _compute_quantity(dep, adapter, Decimal('50'), ticker='AAPL')
        # 5% of 100k = 5000; / 50 = 100 shares
        self.assertEqual(qty, Decimal('100.000000'))

    def test_whole_share_sizing_rounds_down(self):
        dep = _DeploymentStub(bet_size_percentage=5, initial_capital=Decimal('100000'))
        adapter = _StubAdapter(equity=Decimal('100000'), caps={'fractionable': False})
        qty = _compute_quantity(dep, adapter, Decimal('33'), ticker='AAPL')
        # 5000 / 33 = 151.51... -> 151 whole shares
        self.assertEqual(qty, Decimal('151'))

    def test_too_small_returns_none(self):
        dep = _DeploymentStub(bet_size_percentage=0, initial_capital=Decimal('100000'))
        adapter = _StubAdapter(equity=Decimal('100000'))
        self.assertIsNone(_compute_quantity(dep, adapter, Decimal('50'), ticker='AAPL'))

    def test_zero_equity_returns_none(self):
        dep = _DeploymentStub(bet_size_percentage=5, initial_capital=Decimal('0'))
        adapter = _StubAdapter(equity=Decimal('0'))
        self.assertIsNone(_compute_quantity(dep, adapter, Decimal('50'), ticker='AAPL'))

    def test_caps_failure_falls_back_to_non_fractional(self):
        dep = _DeploymentStub(bet_size_percentage=5, initial_capital=Decimal('100000'))
        adapter = _StubAdapter(equity=Decimal('100000'))
        with mock.patch.object(adapter, 'get_symbol_capabilities', side_effect=Exception('boom')):
            qty = _compute_quantity(dep, adapter, Decimal('50'), ticker='AAPL')
        # 5000 / 50 = 100 whole shares (fallback non-fractional)
        self.assertEqual(qty, Decimal('100'))


class IntendedClosingSideTestCase(TestCase):
    def _trade(self, position_mode, trade_type):
        t = mock.Mock()
        t.position_mode = position_mode
        t.trade_type = trade_type
        return t

    def test_long_position_closes_with_sell(self):
        self.assertEqual(_intended_closing_side(self._trade(SIGNAL_LONG, 'buy')), 'sell')

    def test_short_position_closes_with_buy(self):
        self.assertEqual(_intended_closing_side(self._trade(SIGNAL_SHORT, 'sell')), 'buy')
