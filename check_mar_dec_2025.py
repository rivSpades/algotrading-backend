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

# Check March 2025 to December 2025
start_date = pytz.UTC.localize(datetime(2025, 3, 1))
end_date = pytz.UTC.localize(datetime(2025, 12, 31))

print('=' * 120)
print(f'Analysis: March 2025 to December 2025 - When does the anomaly start?')
print('=' * 120)
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

# Compare side by side - show first 100 trades
header = f"{'#':<4} {'Date':<12} {'Symbol':<8} {'ALL Bet':<14} {'ALL PnL':<12} {'LONG Bet':<14} {'LONG PnL':<12} {'Bet Diff':<14} {'PnL Diff':<12}"
print(header)
print('-' * 120)

all_cumulative = 0
long_cumulative = 0
first_anomaly = None

for i in range(min(100, len(all_mode_trades), len(long_mode_trades))):
    a = all_mode_trades[i]
    l = long_mode_trades[i]
    
    date_str = str(a['entry'])[:10] if a['entry'] else 'N/A'
    bet_diff = a['bet_amount'] - l['bet_amount']
    pnl_diff = a['pnl'] - l['pnl']
    
    if a['exit']:
        all_cumulative += a['pnl']
    if l['exit']:
        long_cumulative += l['pnl']
    
    print(f"{i+1:<4} {date_str:<12} {a['symbol']:<8} "
          f"${a['bet_amount']:>12,.2f} ${a['pnl']:>10,.2f}  "
          f"${l['bet_amount']:>12,.2f} ${l['pnl']:>10,.2f}  "
          f"${bet_diff:>12,.2f} ${pnl_diff:>10,.2f}")
    
    # Flag large differences
    if abs(bet_diff) > 50:
        print(f"      *** LARGE BET DIFFERENCE: ${bet_diff:,.2f} (ALL=${a['bet_amount']:,.2f} vs LONG=${l['bet_amount']:,.2f}) ***")
        if first_anomaly is None:
            first_anomaly = (i+1, date_str, bet_diff, a['bet_amount'], l['bet_amount'])

print()
print('=' * 120)
print('Summary for March-December 2025:')
print('=' * 120)
all_pnl = sum(t['pnl'] for t in all_mode_trades if t['exit'])
long_pnl = sum(t['pnl'] for t in long_mode_trades if t['exit'])
print(f'ALL mode PnL: ${all_pnl:,.2f}')
print(f'LONG mode PnL: ${long_pnl:,.2f}')
print(f'Difference: ${all_pnl - long_pnl:,.2f}')
print()

if first_anomaly:
    print(f'First large anomaly detected at trade #{first_anomaly[0]} on {first_anomaly[1]}:')
    print(f'  Bet difference: ${first_anomaly[2]:,.2f}')
    print(f'  ALL bet: ${first_anomaly[3]:,.2f}, LONG bet: ${first_anomaly[4]:,.2f}')

print()
print('=' * 120)
print('ALL mode SHORT exits in March-December 2025:')
print('=' * 120)

shorts = Trade.objects.filter(
    backtest=backtest,
    trade_type='sell',
    exit_timestamp__gte=start_date,
    exit_timestamp__lte=end_date
).order_by('exit_timestamp')

short_pnl = 0
profitable_shorts = 0
losing_shorts = 0
large_profits = []

header2 = f"{'Date':<12} {'Symbol':<8} {'Exit Price':<12} {'PnL':<12} {'Cumulative':<12}"
print(header2)
print('-' * 120)

for trade in shorts:
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        pnl = float(trade.pnl) if trade.pnl else 0
        short_pnl += pnl
        if pnl > 0:
            profitable_shorts += 1
            if pnl > 50:
                large_profits.append((trade.exit_timestamp, trade.symbol.ticker, pnl))
        else:
            losing_shorts += 1
        
        date_str = str(trade.exit_timestamp)[:10] if trade.exit_timestamp else 'N/A'
        print(f"{date_str:<12} {trade.symbol.ticker:<8} ${float(trade.exit_price):>10,.2f} ${pnl:>10,.2f} ${short_pnl:>10,.2f}")

print()
print(f'Total short PnL in period: ${short_pnl:,.2f}')
print(f'Profitable shorts: {profitable_shorts}, Losing shorts: {losing_shorts}')

if large_profits:
    print()
    print('Large profitable shorts (>$50):')
    for ts, symbol, pnl in large_profits[:20]:
        print(f'  {ts.date()} | {symbol} | ${pnl:,.2f}')


