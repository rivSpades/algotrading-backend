"""Tests for hybrid exit metadata rollup (strategy / hedge PnL on main row)."""

from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase

from live_trading.services.trade_metadata import (
    main_qty_portion_factor,
    merge_exit_rollups_for_main_trade,
    patch_main_trade_hedge_leg_link,
)


class MainQtyPortionFactorTests(SimpleTestCase):
    def test_fraction_matches_closed_over_requested(self):
        meta = {'requested_quantity': '13.136881'}
        f = main_qty_portion_factor(meta, Decimal('1.136881'))
        self.assertAlmostEqual(float(f), 1.136881 / 13.136881, places=9)

    def test_fraction_uses_main_opening_key_first(self):
        meta = {
            'main_opening_strategyleg_qty': '10',
            'requested_quantity': '99',
        }
        f = main_qty_portion_factor(meta, Decimal('5'))
        self.assertEqual(f, Decimal('0.5'))

    def test_fraction_falls_back_to_broker_reconcile_from_db_max(self):
        meta = {
            'broker_quantity_reconcile': [
                {'from_db': '13.136881', 'to_broker_qty': '1.136881'},
            ],
        }
        f = main_qty_portion_factor(meta, Decimal('1.136881'))
        self.assertAlmostEqual(float(f), 1.136881 / 13.136881, places=9)

    def test_fraction_capped_one_when_quantity_exceeds_snapshot(self):
        meta = {'requested_quantity': '2'}
        f = main_qty_portion_factor(meta, Decimal('5'))
        self.assertEqual(f, Decimal('1'))

    def test_fraction_one_when_no_snapshot_quantity(self):
        meta = {'hedge_enabled': True}
        f = main_qty_portion_factor(meta, Decimal('1.5'))
        self.assertEqual(f, Decimal('1'))


class PatchMainHedgeLinkTests(SimpleTestCase):
    def test_patch_sets_ticker_quantity_and_live_trade_id(self):
        md: dict = {}
        patch_main_trade_hedge_leg_link(
            md,
            hedge_live_trade_id=42,
            hedge_ticker='VIXY',
            hedge_quantity=Decimal('17.12500008'),
        )
        self.assertEqual(md['hedge_leg_live_trade_id'], 42)
        self.assertEqual(md['hedge_leg_ticker'], 'VIXY')
        self.assertEqual(md['hedge_leg_quantity'], '17.12500008')


class MergeExitRollupsTests(SimpleTestCase):
    @patch('live_trading.services.trade_metadata.closed_hedge_pnl_aggregate')
    def test_hedge_pnl_scaled_when_main_portion_less_than_requested(self, mock_agg):
        mock_agg.return_value = Decimal('522')
        meta = {'requested_quantity': '13.136881', 'hedge_enabled': True}
        out = merge_exit_rollups_for_main_trade(
            meta,
            main_pnl=Decimal('-3.30'),
            parent_trade_id=99,
            closed_main_quantity=Decimal('1.136881'),
        )
        frac = Decimal('1.136881') / Decimal('13.136881')
        expected = (Decimal('522') * frac).quantize(Decimal('0.0001'))
        self.assertEqual(out['strategy_pnl'], '-3.3000')
        self.assertEqual(out['hedge_pnl_pre_portion'], '522.0000')
        self.assertEqual(out['hedge_pnl'], str(expected))
        self.assertEqual(out.get('hedge_pnl_scale_version'), 2)

    @patch('live_trading.services.trade_metadata.closed_hedge_pnl_aggregate')
    def test_closed_main_quantity_omitted_keeps_full_hedge_sum(self, mock_agg):
        mock_agg.return_value = Decimal('100')
        out = merge_exit_rollups_for_main_trade(
            {'requested_quantity': '10'},
            main_pnl=Decimal('5'),
            parent_trade_id=1,
        )
        self.assertEqual(out['hedge_pnl_pre_portion'], '100.0000')
        self.assertEqual(out['hedge_pnl'], '100.0000')
        self.assertEqual(out.get('hedge_pnl_scale_version'), 2)
