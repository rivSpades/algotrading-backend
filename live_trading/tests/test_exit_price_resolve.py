"""Tests for exit execution price resolution (avoid treating Decimal('0') as missing)."""

from decimal import Decimal
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from live_trading.services.order_service import (
    _positive_decimal,
    _resolve_exit_execution_price,
)


class PositiveDecimalTests(SimpleTestCase):
    def test_zero_is_not_positive(self):
        self.assertIsNone(_positive_decimal(0))
        self.assertIsNone(_positive_decimal(Decimal('0')))
        self.assertIsNone(_positive_decimal('0'))

    def test_positive_values(self):
        self.assertEqual(_positive_decimal('1.5'), Decimal('1.5'))
        self.assertEqual(_positive_decimal(Decimal('2')), Decimal('2'))


class ResolveExitExecutionPriceTests(SimpleTestCase):
    def test_zero_order_price_uses_metadata_in_key_order(self):
        px = _resolve_exit_execution_price(
            order_price=Decimal('0'),
            entry_price_fallback=Decimal('10'),
            metadata_exit_subset={'synced_avg_price': '6', 'filled_price': '5'},
        )
        self.assertEqual(px, Decimal('6'))

    def test_positive_order_price_wins_over_metadata(self):
        px = _resolve_exit_execution_price(
            order_price=Decimal('3'),
            entry_price_fallback=Decimal('10'),
            metadata_exit_subset={'filled_price': '5'},
        )
        self.assertEqual(px, Decimal('3'))

    def test_broker_quote_before_entry_fallback(self):
        broker = MagicMock()
        broker.get_current_price.return_value = Decimal('7.77')
        px = _resolve_exit_execution_price(
            order_price=Decimal('0'),
            entry_price_fallback=Decimal('10'),
            metadata_exit_subset={'filled_price': '0'},
            broker_adapter=broker,
            quote_symbol='VIXM',
        )
        self.assertEqual(px, Decimal('7.77'))
        broker.get_current_price.assert_called_once_with('VIXM')

    def test_falls_back_to_entry_when_no_positive_source(self):
        px = _resolve_exit_execution_price(
            order_price=None,
            entry_price_fallback=Decimal('42'),
            metadata_exit_subset={},
        )
        self.assertEqual(px, Decimal('42'))
