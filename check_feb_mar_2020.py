import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade
from datetime import datetime
import pytz

backtest = Backtest.objects.get(id=254)

# Find trades around Feb-Mar 2020
start_date = pytz.UTC.localize(datetime(2020, 2, 1))
end_date = pytz.UTC.localize(datetime(2020, 3, 31))

print('=' * 100)
print(f'Analysis of trades between {start_date.date()} and {end_date.date()}')
print('=' * 100)
print()

# Get ALL mode long trades in this period
all_mode_trades = []
long_mode_trades = []

for trade in Trade.objects.filter(
    backtest=backtest,
    entry_timestamp__gte=start_date,
    entry_timestamp__lte=end_date
).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all' and trade.trade_type == 'buy':
        all_mode_trades.append({
            'symbol': trade.symbol.ticker,
            'entry': trade.entry_timestamp,
            'exit': trade.exit_timestamp,
            'bet_amount': float(metadata.get('bet_amount', 0)),
            'pnl': float(trade.pnl) if trade.pnl else 0
        })
    elif mode == 'long' and trade.trade_type == 'buy':
        long_mode_trades.append({
            'symbol': trade.symbol.ticker,
            'entry': trade.entry_timestamp,
            'exit': trade.exit_timestamp,
            'bet_amount': float(metadata.get('bet_amount', 0)),
            'pnl': float(trade.pnl) if trade.pnl else 0
        })

print(f'ALL mode long trades in period: {len(all_mode_trades)}')
print(f'LONG mode trades in period: {len(long_mode_trades)}')
print()

# Compare side by side
header = f"{'#':<4} {'Date':<12} {'Symbol':<8} {'ALL Bet':<12} {'ALL PnL':<12} {'LONG Bet':<12} {'LONG PnL':<12} {'Bet Diff':<12} {'PnL Diff':<12}"
print(header)
print('-' * 100)

for i in range(min(50, len(all_mode_trades), len(long_mode_trades))):
    a = all_mode_trades[i]
    l = long_mode_trades[i]
    
    date_str = str(a['entry'])[:10] if a['entry'] else 'N/A'
    bet_diff = a['bet_amount'] - l['bet_amount']
    pnl_diff = a['pnl'] - l['pnl']
    
    print(f"{i+1:<4} {date_str:<12} {a['symbol']:<8} "
          f"${a['bet_amount']:>10,.2f} ${a['pnl']:>10,.2f}  "
          f"${l['bet_amount']:>10,.2f} ${l['pnl']:>10,.2f}  "
          f"${bet_diff:>10,.2f} ${pnl_diff:>10,.2f}")
    
    if abs(bet_diff) > 20:
        print(f"      *** LARGE BET DIFFERENCE: ${bet_diff:,.2f} ***")

print()
print('Summary for Feb-Mar 2020:')
all_pnl = sum(t['pnl'] for t in all_mode_trades)
long_pnl = sum(t['pnl'] for t in long_mode_trades)
print(f'ALL mode PnL: ${all_pnl:,.2f}')
print(f'LONG mode PnL: ${long_pnl:,.2f}')
print(f'Difference: ${all_pnl - long_pnl:,.2f}')

print()
print('=' * 100)
print('ALL mode SHORT exits in this period:')
print('=' * 100)

shorts = Trade.objects.filter(
    backtest=backtest,
    trade_type='sell',
    exit_timestamp__gte=start_date,
    exit_timestamp__lte=end_date
).order_by('exit_timestamp')

short_pnl = 0
profitable_shorts = 0
losing_shorts = 0

header2 = f"{'Date':<12} {'Symbol':<8} {'Exit Price':<12} {'PnL':<12} {'Cumulative':<12}"
print(header2)
print('-' * 100)

for trade in shorts:
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        pnl = float(trade.pnl) if trade.pnl else 0
        short_pnl += pnl
        if pnl > 0:
            profitable_shorts += 1
        else:
            losing_shorts += 1
        
        date_str = str(trade.exit_timestamp)[:10] if trade.exit_timestamp else 'N/A'
        print(f"{date_str:<12} {trade.symbol.ticker:<8} ${float(trade.exit_price):>10,.2f} ${pnl:>10,.2f} ${short_pnl:>10,.2f}")

print()
print(f'Total short PnL in period: ${short_pnl:,.2f}')
print(f'Profitable shorts: {profitable_shorts}, Losing shorts: {losing_shorts}')


