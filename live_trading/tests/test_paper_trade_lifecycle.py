from decimal import Decimal
import hashlib

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from backtest_engine.models import SymbolBacktestParameterSet
from live_trading.engines.base import SIGNAL_LONG, LiveSignal
from live_trading.models import (
    Broker,
    DeploymentSymbol,
    LiveTrade,
    StrategyDeployment,
    SymbolBrokerAssociation,
)
from live_trading.adapters.base import OrderResult
from live_trading.services.order_service import manual_close_live_trade, place_signal_order, update_open_trades
from live_trading.tests.mock_broker_adapter import PaperTestBrokerAdapter
from market_data.models import Exchange, Provider, Symbol
from strategies.models import StrategyDefinition


def _parm_sig(salt: str) -> str:
    return hashlib.sha256(salt.encode()).hexdigest()


class PendingMarketCloseBrokerAdapter(PaperTestBrokerAdapter):
    """Like Alpaca: close order returns ``accepted`` with 0 fill; poll fills later."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_exit: dict = {}
        self._exit_i = 0

    def place_order(
        self,
        symbol,
        side,
        quantity,
        order_type='market',
        limit_price=None,
        stop_price=None,
    ):
        if side and str(side).lower() == 'sell':
            self._exit_i += 1
            oid = f'exit-{self._exit_i}'
            ticker = (symbol or '').strip().upper()
            qty = Decimal(quantity)
            px = self._next_fill_price()
            self._pending_exit[oid] = {
                'symbol': ticker,
                'qty': qty,
                'price': px,
                'released': False,
                'poll_ct': 0,
            }
            return OrderResult(
                order_id=oid,
                symbol=ticker,
                side='sell',
                quantity=qty,
                filled_quantity=Decimal('0'),
                price=Decimal('0'),
                status='accepted',
                timestamp=timezone.now(),
                broker_order_id=oid,
            )
        return super().place_order(
            symbol, side, quantity, order_type,
            limit_price=limit_price, stop_price=stop_price,
        )

    def get_order_status(self, order_id: str):
        oid = str(order_id)
        p = self._pending_exit.get(oid)
        if not p:
            return super().get_order_status(order_id)
        ticker = p['symbol']
        qty = p['qty']
        px = p['price']
        # First polls: Alpaca-style pending acceptance (no ledger change yet).
        p['poll_ct'] = p.get('poll_ct', 0) + 1
        if not p['released'] and p['poll_ct'] == 1:
            return OrderResult(
                order_id=oid,
                symbol=ticker,
                side='sell',
                quantity=qty,
                filled_quantity=Decimal('0'),
                price=Decimal('0'),
                status='accepted',
                timestamp=timezone.now(),
                broker_order_id=oid,
            )
        if not p['released']:
            p['released'] = True
            prev = self._long_qty.get(ticker, Decimal('0'))
            self._long_qty[ticker] = prev - qty
            if self._long_qty[ticker] <= 0:
                self._long_qty.pop(ticker, None)
        return OrderResult(
            order_id=oid,
            symbol=ticker,
            side='sell',
            quantity=qty,
            filled_quantity=qty,
            price=px,
            status='filled',
            timestamp=timezone.now(),
            broker_order_id=oid,
        )


class PaperTradeLifecycleTests(TestCase):
    """Open via `place_signal_order`, verify API list, manual close fills exit fields."""

    def setUp(self):
        self.exchange = Exchange.objects.create(name='Test Ex', code='TEX')
        self.provider = Provider.objects.create(name='Yahoo Finance', code='YAHOO')
        self.sym = Symbol.objects.create(
            ticker='LTGAP1',
            exchange=self.exchange,
            provider=self.provider,
            status='active',
        )

        self.broker = Broker.objects.create(
            name='Paper Test Broker',
            code=f'PB{self.sym.ticker}',
            paper_trading_endpoint_url='https://paper.test',
            paper_trading_api_key='k',
            paper_trading_secret_key='s',
            paper_trading_active=True,
        )

        SymbolBrokerAssociation.objects.create(
            symbol=self.sym,
            broker=self.broker,
            long_active=True,
            short_active=False,
        )

        self.strategy, _ = StrategyDefinition.objects.get_or_create(
            name='Gap-Up and Gap-Down',
            defaults={'description_short': '', 'globally_enabled': True},
        )

        self.param_set = SymbolBacktestParameterSet.objects.create(
            signature=_parm_sig(f'{self.sym.ticker}-unit'),
            label='test',
            strategy=self.strategy,
            broker=self.broker,
            parameters={'std_period': 90, 'threshold': 0.25},
        )

        self.deployment = StrategyDeployment.objects.create(
            name='Unit Test Deployment',
            strategy=self.strategy,
            parameter_set=self.param_set,
            broker=self.broker,
            position_mode='long',
            deployment_type='paper',
            status='active',
            initial_capital=Decimal('100000'),
            bet_size_percentage=100.0,
            strategy_parameters={'std_period': 90, 'threshold': 0.25},
            hedge_enabled=False,
        )

        self.ds = DeploymentSymbol.objects.create(
            deployment=self.deployment,
            symbol=self.sym,
            position_mode='long',
            status='active',
            priority=1,
        )

        self.adapter = PaperTestBrokerAdapter(
            self.broker,
            quote=Decimal('100'),
            fill_price_sequence=[Decimal('100'), Decimal('110')],
        )

    def test_entry_manual_close_and_api_list(self):
        fire_at = timezone.now()
        sig = LiveSignal(
            action=SIGNAL_LONG,
            confidence=1.0,
            price=Decimal('100'),
            bar_timestamp=fire_at,
        )
        out = place_signal_order(
            self.ds,
            sig,
            broker_adapter=self.adapter,
            actor_type='test',
            actor_id='paper_lifecycle',
            fire_at=fire_at,
        )
        self.assertEqual(out.status, 'filled')
        self.assertTrue(out.live_trade_id)

        lt = LiveTrade.objects.get(id=out.live_trade_id)
        self.assertEqual(lt.status, 'open')
        self.assertIsNotNone(lt.metadata.get('bet_amount'))

        m_close = manual_close_live_trade(
            lt,
            actor_type='test',
            actor_id='paper_lifecycle',
            fire_at=fire_at,
            broker_adapter=self.adapter,
        )
        self.assertEqual(m_close.status, 'filled')

        lt.refresh_from_db()
        self.assertEqual(lt.status, 'closed')
        self.assertIsNotNone(lt.exit_price)
        self.assertIsNotNone(lt.exit_timestamp)
        self.assertIsNotNone(lt.pnl)

        expected = (Decimal('110') - Decimal('100')) * lt.quantity
        self.assertEqual(lt.pnl, expected)

        client = APIClient()
        url = reverse('live-trade-list')
        rsp = client.get(url, {'deployment': self.deployment.id})
        self.assertEqual(rsp.status_code, 200)
        payload = rsp.data
        self.assertGreaterEqual(payload.get('count', 0), 1)
        ids = [row['id'] for row in payload.get('results', [])]
        self.assertIn(lt.id, ids)

    def test_manual_close_zero_fill_then_sync_closes_live_trade(self):
        """Broker accepts close without immediate fill JSON; reconciliation must persist close."""
        fire_at = timezone.now()
        adapter = PendingMarketCloseBrokerAdapter(
            self.broker,
            quote=Decimal('100'),
            fill_price_sequence=[Decimal('100'), Decimal('110')],
        )
        sig = LiveSignal(
            action=SIGNAL_LONG,
            confidence=1.0,
            price=Decimal('100'),
            bar_timestamp=fire_at,
        )
        out = place_signal_order(
            self.ds,
            sig,
            broker_adapter=adapter,
            actor_type='test',
            actor_id='deferred_close',
            fire_at=fire_at,
        )
        self.assertEqual(out.status, 'filled')
        lt = LiveTrade.objects.get(id=out.live_trade_id)
        self.assertEqual(lt.status, 'open')

        m_close = manual_close_live_trade(
            lt,
            actor_type='test',
            actor_id='deferred_close',
            fire_at=fire_at,
            broker_adapter=adapter,
        )
        self.assertEqual(m_close.status, 'placed')

        lt.refresh_from_db()
        self.assertEqual(lt.status, 'open')
        self.assertTrue((lt.metadata or {}).get('exit'))

        summary = update_open_trades(
            self.deployment,
            broker_adapter=adapter,
            actor_type='test',
            actor_id='deferred_close_sync',
        )
        self.assertGreaterEqual(summary.get('closed', 0), 1)

        lt.refresh_from_db()
        self.assertEqual(lt.status, 'closed')
        self.assertIsNotNone(lt.exit_timestamp)

    def test_position_sync_shrinks_main_quantity_when_broker_holds_fewer_shares(self):
        """DB drift (claiming more than the account) is aligned to the broker on sync."""
        fire_at = timezone.now()
        sig = LiveSignal(
            action=SIGNAL_LONG,
            confidence=1.0,
            price=Decimal('100'),
            bar_timestamp=fire_at,
        )
        out = place_signal_order(
            self.ds,
            sig,
            broker_adapter=self.adapter,
            actor_type='test',
            actor_id='qty_reconcile',
            fire_at=fire_at,
        )
        self.assertEqual(out.status, 'filled')
        lt = LiveTrade.objects.get(id=out.live_trade_id)
        at_broker = lt.quantity
        LiveTrade.objects.filter(pk=lt.pk).update(quantity=at_broker + Decimal('100'))
        lt.refresh_from_db()
        self.assertEqual(lt.quantity, at_broker + Decimal('100'))

        summary = update_open_trades(
            self.deployment,
            broker_adapter=self.adapter,
            actor_type='test',
            actor_id='qty_reconcile',
        )
        self.assertGreaterEqual(summary.get('quantity_reconciled', 0), 1)

        lt.refresh_from_db()
        self.assertEqual(lt.status, 'open')
        self.assertEqual(lt.quantity, at_broker)
