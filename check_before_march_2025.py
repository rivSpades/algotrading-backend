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

# Check what happened in February 2025 - short exits that might have boosted equity
feb_start = pytz.UTC.localize(datetime(2025, 2, 1))
march_start = pytz.UTC.localize(datetime(2025, 3, 1))

print('=' * 100)
print('SHORT exits in ALL mode during February 2025 (before March anomaly):')
print('=' * 100)
print()

shorts = Trade.objects.filter(
    backtest=backtest,
    trade_type='sell',
    exit_timestamp__gte=feb_start,
    exit_timestamp__lt=march_start
).order_by('exit_timestamp')

short_pnl = 0
large_profits = []

header = "Date         Symbol   Exit Price   PnL          Cumulative  "
print(header)
print('-' * 100)

for trade in shorts:
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        pnl = float(trade.pnl) if trade.pnl else 0
        short_pnl += pnl
        if pnl > 100:
            large_profits.append((trade.exit_timestamp, trade.symbol.ticker, pnl))
        
        date_str = str(trade.exit_timestamp)[:10] if trade.exit_timestamp else 'N/A'
        print(f"{date_str:<12} {trade.symbol.ticker:<8} ${float(trade.exit_price):>10,.2f} ${pnl:>10,.2f} ${short_pnl:>10,.2f}")

print()
print(f'Total short PnL in February 2025: ${short_pnl:,.2f}')

if large_profits:
    print()
    print('Large profitable shorts (>$100) in February:')
    for ts, symbol, pnl in large_profits:
        print(f'  {ts.date()} | {symbol} | ${pnl:,.2f}')

# Also check the last few long trades before March to see equity
print()
print('=' * 100)
print('Last 10 long trades BEFORE March 2025 (to see equity levels):')
print('=' * 100)

last_trades = Trade.objects.filter(
    backtest=backtest,
    trade_type='buy',
    entry_timestamp__lt=march_start
).order_by('-entry_timestamp')[:10]

print("Date         Mode  Symbol  Bet Amount")
print('-' * 100)

for trade in reversed(last_trades):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    if mode in ['all', 'long']:
        date_str = str(trade.entry_timestamp)[:10] if trade.entry_timestamp else 'N/A'
        bet = float(metadata.get('bet_amount', 0))
        print(f"{date_str} | {mode.upper():<4} | {trade.symbol.ticker} | ${bet:,.2f}")

# Check cumulative equity by reconstructing from all trades up to March 1
print()
print('=' * 100)
print('Reconstructing equity up to March 1, 2025:')
print('=' * 100)

# Get all ALL mode trades (longs + shorts) up to March 1
all_events = []
for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        if trade.entry_timestamp and trade.entry_timestamp < march_start:
            all_events.append({
                'timestamp': trade.entry_timestamp,
                'type': 'entry',
                'trade_type': trade.trade_type,
                'symbol': trade.symbol.ticker
            })
        if trade.exit_timestamp and trade.exit_timestamp < march_start:
            all_events.append({
                'timestamp': trade.exit_timestamp,
                'type': 'exit',
                'trade_type': trade.trade_type,
                'symbol': trade.symbol.ticker,
                'pnl': float(trade.pnl) if trade.pnl else 0
            })

all_events.sort(key=lambda x: x['timestamp'])

equity = 1000.0
for event in all_events:
    if event['type'] == 'exit':
        equity += event['pnl']

print(f'ALL mode equity at end of February 2025: ${equity:,.2f}')

# Check LONG mode equity
long_equity = 1000.0
for trade in Trade.objects.filter(backtest=backtest, trade_type='buy').order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    if mode == 'long' and trade.exit_timestamp and trade.exit_timestamp < march_start:
        long_equity += float(trade.pnl) if trade.pnl else 0

print(f'LONG mode equity at end of February 2025: ${long_equity:,.2f}')
print(f'Difference: ${equity - long_equity:,.2f}')


