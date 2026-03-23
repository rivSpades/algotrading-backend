"""
Forensics for a completed backtest: worst intra-trade drawdown trades, optional symbol drill-down.
"""
from django.core.management.base import BaseCommand
from collections import Counter

from backtest_engine.models import Backtest, Trade


class Command(BaseCommand):
    help = 'Print worst max_drawdown trades and optional symbol detail for a backtest'

    def add_arguments(self, parser):
        parser.add_argument('backtest_id', type=int)
        parser.add_argument('--symbol', type=str, default=None, help='Ticker to print all trades for')

    def handle(self, *args, **options):
        bid = options['backtest_id']
        ticker_filter = (options['symbol'] or '').strip().upper() or None

        try:
            bt = Backtest.objects.get(id=bid)
        except Backtest.DoesNotExist:
            self.stderr.write(self.style.ERROR(f'Backtest {bid} not found'))
            return

        self.stdout.write(f'Backtest {bt.id}: {bt.name!r} status={bt.status}')

        qs = Trade.objects.filter(backtest_id=bid).select_related('symbol')
        total = qs.count()
        if total == 0:
            self.stdout.write('No trades.')
            return

        modes = Counter()
        for t in qs.iterator(chunk_size=500):
            md = t.metadata if isinstance(t.metadata, dict) else {}
            modes[md.get('position_mode', 'long')] += 1
        self.stdout.write(f'Total trades: {total}  by mode: {dict(modes)}')

        long_trades = []
        short_trades = []
        for t in qs.iterator(chunk_size=500):
            md = t.metadata if isinstance(t.metadata, dict) else {}
            pm = md.get('position_mode', 'long')
            if pm == 'short':
                short_trades.append(t)
            else:
                long_trades.append(t)

        def key_dd(t):
            return float(t.max_drawdown) if t.max_drawdown is not None else -1.0

        long_sorted = sorted(long_trades, key=key_dd, reverse=True)
        short_sorted = sorted(short_trades, key=key_dd, reverse=True)

        self.stdout.write('\n--- Top 5 worst max_drawdown (LONG) ---')
        for t in long_sorted[:5]:
            self._print_trade(t)

        self.stdout.write('\n--- Top 5 worst max_drawdown (SHORT) ---')
        for t in short_sorted[:5]:
            self._print_trade(t)

        if ticker_filter:
            self.stdout.write(f'\n--- All trades for {ticker_filter} ---')
            rows = [t for t in qs if t.symbol.ticker.upper() == ticker_filter]
            if not rows:
                self.stdout.write(f'No trades for ticker {ticker_filter}')
            for t in sorted(rows, key=lambda x: x.entry_timestamp):
                self._print_trade(t, implied=True)

    def _print_trade(self, t, implied=False):
        md = t.metadata if isinstance(t.metadata, dict) else {}
        bet = md.get('bet_amount')
        pm = md.get('position_mode', 'long')
        line = (
            f'{t.symbol.ticker} id={t.id} mode={pm} type={t.trade_type} '
            f'entry={t.entry_price} exit={t.exit_price} qty={t.quantity} '
            f'pnl={t.pnl} max_dd={t.max_drawdown} bet={bet}'
        )
        self.stdout.write(line)
        if implied and t.entry_price and t.exit_price and t.quantity is not None:
            ep, xp, q = float(t.entry_price), float(t.exit_price), float(t.quantity)
            if t.trade_type == 'buy':
                implied_pnl = (xp - ep) * q
            else:
                implied_pnl = (ep - xp) * q
            self.stdout.write(f'  implied_pnl (exit-entry)*qty: {implied_pnl:.6f}')
            if bet is not None:
                try:
                    notional = float(ep) * q
                    self.stdout.write(f'  notional_entry (price*qty): {notional:.6f}')
                except (TypeError, ValueError):
                    pass
