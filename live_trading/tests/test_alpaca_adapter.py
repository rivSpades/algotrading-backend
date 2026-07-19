"""Unit tests for the Alpaca broker adapter (no real network).

HTTP calls are intercepted with `unittest.mock.patch` on `requests.get/post`
so the adapter's request parsing, Decimal handling, and enum normalization
can be exercised without hitting Alpaca.
"""

from __future__ import annotations

from datetime import datetime, timezone as dt_tz
from decimal import Decimal
from unittest import mock

from django.test import TestCase

from live_trading.adapters.alpaca import AlpacaBrokerAdapter
from live_trading.adapters.base import OrderSide, OrderStatus
from live_trading.models import Broker


def _make_broker() -> Broker:
    return Broker.objects.create(
        name='Alpaca Paper',
        code='ALPACA',
        paper_trading_active=True,
        paper_trading_endpoint_url='https://paper-api.alpaca.markets',
        paper_trading_api_key='test-key',
        paper_trading_secret_key='test-secret',
    )


def _mock_response(status_code: int, payload=None, text: str = ''):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    resp.text = text
    return resp


class AlpacaAdapterTestCase(TestCase):
    def setUp(self):
        self.broker = _make_broker()
        self.adapter = AlpacaBrokerAdapter(self.broker, paper_trading=True)

    def test_place_order_filled_returns_enum_side_and_raw_status(self):
        payload = {
            'id': 'order-1',
            'symbol': 'AAPL',
            'status': 'filled',
            'filled_qty': '10',
            'filled_avg_price': '150.25',
            'qty': '10',
        }
        with mock.patch('live_trading.adapters.alpaca.requests.post', return_value=_mock_response(200, payload)):
            result = self.adapter.place_order('AAPL', 'buy', Decimal('10'))

        self.assertEqual(result.order_id, 'order-1')
        self.assertEqual(result.side, OrderSide.BUY)
        self.assertEqual(result.filled_quantity, Decimal('10'))
        self.assertEqual(result.price, Decimal('150.25'))
        # Status kept as the raw vendor string (str-enum compares equal).
        self.assertEqual(result.status, 'filled')

    def test_place_order_rejected_on_non_200(self):
        with mock.patch('live_trading.adapters.alpaca.requests.post', return_value=_mock_response(400, {'message': 'bad'})):
            result = self.adapter.place_order('AAPL', 'sell', Decimal('5'))

        self.assertEqual(result.status, 'rejected')
        self.assertEqual(result.side, OrderSide.SELL)
        self.assertEqual(result.filled_quantity, Decimal('0'))
        self.assertIn('bad', (result.error_message or ''))

    def test_place_order_accepts_enum_side_and_order_type(self):
        payload = {'id': 'o2', 'symbol': 'MSFT', 'status': 'accepted', 'filled_qty': '0', 'filled_avg_price': '', 'qty': '3'}
        with mock.patch('live_trading.adapters.alpaca.requests.post', return_value=_mock_response(200, payload)) as posted:
            result = self.adapter.place_order('MSFT', OrderSide.BUY, Decimal('3'), order_type='limit', limit_price=Decimal('330'))

        sent = posted.call_args.kwargs['json']
        self.assertEqual(sent['side'], 'buy')
        self.assertEqual(sent['type'], 'limit')
        self.assertEqual(sent['limit_price'], '330')
        self.assertEqual(result.side, OrderSide.BUY)

    def test_get_order_status_filled(self):
        payload = {
            'id': 'order-1',
            'symbol': 'AAPL',
            'side': 'buy',
            'status': 'filled',
            'filled_qty': '10',
            'filled_avg_price': '150',
            'qty': '10',
        }
        with mock.patch('live_trading.adapters.alpaca.requests.get', return_value=_mock_response(200, payload)):
            result = self.adapter.get_order_status('order-1')

        self.assertEqual(result.status, 'filled')
        self.assertEqual(result.filled_quantity, Decimal('10'))
        self.assertEqual(result.price, Decimal('150'))

    def test_get_order_status_rejected_on_404(self):
        with mock.patch('live_trading.adapters.alpaca.requests.get', return_value=_mock_response(404, text='not found')):
            result = self.adapter.get_order_status('missing')

        self.assertEqual(result.status, 'rejected')
        self.assertEqual(result.filled_quantity, Decimal('0'))

    def test_get_account_equity_parses_decimal(self):
        with mock.patch('live_trading.adapters.alpaca.requests.get', return_value=_mock_response(200, {'equity': '12345.67'})):
            equity = self.adapter.get_account_equity()
        self.assertEqual(equity, Decimal('12345.67'))

    def test_get_account_equity_error_returns_zero(self):
        with mock.patch('live_trading.adapters.alpaca.requests.get', side_effect=Exception('boom')):
            equity = self.adapter.get_account_equity()
        self.assertEqual(equity, Decimal('0'))

    def test_get_session_open_price_uses_snapshot_daily_bar(self):
        payload = {'dailyBar': {'o': '201.5'}}
        with mock.patch('live_trading.adapters.alpaca.requests.get', return_value=_mock_response(200, payload)) as got:
            price = self.adapter.get_session_open_price(
                'AAPL',
                session_open=datetime(2024, 1, 2, 14, 30, tzinfo=dt_tz.utc),
                window_minutes=6,
            )
        self.assertEqual(price, Decimal('201.5'))
        # Snapshot endpoint path used.
        self.assertIn('/v2/stocks/AAPL/snapshot', got.call_args.args[0])

    def test_get_session_open_price_none_when_no_data(self):
        # Snapshot 404 and bars 200 but empty -> None
        def side(*args, **kwargs):
            url = args[0] if args else kwargs.get('url', '')
            if 'snapshot' in url:
                return _mock_response(404, text='nope')
            return _mock_response(200, {'bars': []})

        with mock.patch('live_trading.adapters.alpaca.requests.get', side_effect=side):
            price = self.adapter.get_session_open_price(
                'AAPL',
                session_open=datetime(2024, 1, 2, 14, 30, tzinfo=dt_tz.utc),
                window_minutes=6,
            )
        self.assertIsNone(price)

    def test_order_status_enum_membership(self):
        # Sanity: the canonical enum members compare equal to their string values.
        self.assertEqual(OrderStatus.FILLED, 'filled')
        self.assertEqual(OrderStatus.REJECTED, 'rejected')
