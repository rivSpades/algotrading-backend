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

# Check late March to early April 2025
start_date = pytz.UTC.localize(datetime(2025, 3, 20))
end_date = pytz.UTC.localize(datetime(2025, 4, 15))

print('=' * 120)
print(f'ALL mode SHORT exits from {start_date.date()} to {end_date.date()}')
print('=' * 120)
print()

shorts = Trade.objects.filter(
    backtest=backtest,
    trade_type='sell',
    exit_timestamp__gte=start_date,
    exit_timestamp__lte=end_date
).order_by('exit_timestamp')

short_pnl = 0
profitable_count = 0
losing_count = 0
large_profits = []

header = "Date         Symbol   Exit Price   PnL          Cumulative  "
print(header)
print('-' * 120)

for trade in shorts:
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        pnl = float(trade.pnl) if trade.pnl else 0
        short_pnl += pnl
        if pnl > 0:
            profitable_count += 1
            if pnl > 50:
                large_profits.append((trade.exit_timestamp, trade.symbol.ticker, pnl))
        else:
            losing_count += 1
        
        date_str = str(trade.exit_timestamp)[:10] if trade.exit_timestamp else 'N/A'
        marker = "***" if pnl > 50 else ""
        print(f"{date_str:<12} {trade.symbol.ticker:<8} ${float(trade.exit_price):>10,.2f} ${pnl:>10,.2f} ${short_pnl:>10,.2f} {marker}")

print()
print(f'Total short PnL in period: ${short_pnl:,.2f}')
print(f'Profitable shorts: {profitable_count}, Losing shorts: {losing_count}')

if large_profits:
    print()
    print('Large profitable shorts (>$50):')
    for ts, symbol, pnl in large_profits:
        print(f'  {ts.date()} | {symbol} | ${pnl:,.2f}')

print()
print('=' * 120)
print('Reconstructing equity timeline to see when ALL mode equity jumps:')
print('=' * 120)

# Get all ALL mode trades (longs + shorts) up to April 15
all_events = []
for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        if trade.entry_timestamp and trade.entry_timestamp <= end_date:
            all_events.append({
                'timestamp': trade.entry_timestamp,
                'type': 'entry',
                'trade_type': trade.trade_type,
                'symbol': trade.symbol.ticker
            })
        if trade.exit_timestamp and trade.exit_timestamp <= end_date:
            all_events.append({
                'timestamp': trade.exit_timestamp,
                'type': 'exit',
                'trade_type': trade.trade_type,
                'symbol': trade.symbol.ticker,
                'pnl': float(trade.pnl) if trade.pnl else 0
            })

all_events.sort(key=lambda x: x['timestamp'])

# Track equity at key points
equity = 1000.0
key_dates = [
    pytz.UTC.localize(datetime(2025, 3, 1)),
    pytz.UTC.localize(datetime(2025, 3, 24)),
    pytz.UTC.localize(datetime(2025, 4, 1)),
    pytz.UTC.localize(datetime(2025, 4, 14)),
    pytz.UTC.localize(datetime(2025, 4, 15))
]

equity_at_dates = {}

for event in all_events:
    if event['type'] == 'exit':
        equity += event['pnl']
    
    # Check if we passed a key date
    for key_date in key_dates:
        if event['timestamp'] <= key_date and key_date not in equity_at_dates:
            equity_at_dates[key_date] = equity

print("Equity at key dates:")
for date in sorted(equity_at_dates.keys()):
    print(f"  {date.date()}: ${equity_at_dates[date]:,.2f}")

# Also check LONG mode equity at these dates
print()
print('LONG mode equity at key dates:')
long_equity = 1000.0
for trade in Trade.objects.filter(backtest=backtest, trade_type='buy').order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    if mode == 'long' and trade.exit_timestamp:
        for key_date in key_dates:
            if trade.exit_timestamp <= key_date:
                long_equity += float(trade.pnl) if trade.pnl else 0
        
        if trade.exit_timestamp <= key_dates[-1]:
            long_equity = 1000.0  # Reset and recalculate properly
            
# Recalculate properly
for key_date in key_dates:
    long_equity = 1000.0
    for trade in Trade.objects.filter(backtest=backtest, trade_type='buy').order_by('entry_timestamp'):
        metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
        mode = metadata.get('position_mode')
        if mode == 'long' and trade.exit_timestamp and trade.exit_timestamp <= key_date:
            long_equity += float(trade.pnl) if trade.pnl else 0
    print(f"  {key_date.date()}: ${long_equity:,.2f}")


