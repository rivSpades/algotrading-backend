from django.test import SimpleTestCase

from backtest_engine.services.order_permutations import (
    cap_variant_request,
    max_variant_runs,
    permutation_count,
    variant_symbol_orders,
)


class OrderPermutationsTests(SimpleTestCase):
    def test_three_symbols_six_orders_five_variants(self):
        self.assertEqual(permutation_count(3), 6)
        self.assertEqual(max_variant_runs(3), 5)

    def test_cap_request(self):
        capped, max_unique = cap_variant_request(3, 500)
        self.assertEqual(max_unique, 5)
        self.assertEqual(capped, 5)

    def test_variant_orders_unique(self):
        tickers = ['AAPL', 'SPY', 'TSLA']
        ref = ['AAPL', 'SPY', 'TSLA']
        orders, max_unique = variant_symbol_orders(tickers, ref, 5)
        self.assertEqual(max_unique, 5)
        self.assertEqual(len(orders), 5)
        keys = {tuple(o) for o in orders}
        self.assertEqual(len(keys), 5)
        self.assertNotIn(tuple(ref), keys)
