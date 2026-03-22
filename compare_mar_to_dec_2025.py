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

# March 2025 to December 2025
start_date = pytz.UTC.localize(datetime(2025, 3, 1))
end_date = pytz.UTC.localize(datetime(2025, 12, 31))

print('=' * 120)
print(f'Comparison: ALL mode vs LONG mode from {start_date.date()} to {end_date.date()}')
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

# Group by month to see progression
print('Monthly breakdown of PnL differences:')
print('=' * 120)
header = "Month      ALL Mode PnL      LONG Mode PnL    Difference      ALL Trades  LONG Trades"
print(header)
print('-' * 120)

months = {}
for i in range(len(all_mode_trades)):
    a = all_mode_trades[i]
    l = long_mode_trades[i]
    
    month_key = str(a['entry'].date())[:7] if a['entry'] else 'unknown'
    
    if month_key not in months:
        months[month_key] = {
            'all_pnl': 0,
            'long_pnl': 0,
            'all_trades': 0,
            'long_trades': 0,
            'all_bet_total': 0,
            'long_bet_total': 0
        }
    
    if a['exit']:
        months[month_key]['all_pnl'] += a['pnl']
        months[month_key]['all_trades'] += 1
        months[month_key]['all_bet_total'] += a['bet_amount']
    
    if l['exit']:
        months[month_key]['long_pnl'] += l['pnl']
        months[month_key]['long_trades'] += 1
        months[month_key]['long_bet_total'] += l['bet_amount']

for month in sorted(months.keys()):
    data = months[month]
    all_avg_bet = data['all_bet_total'] / data['all_trades'] if data['all_trades'] > 0 else 0
    long_avg_bet = data['long_bet_total'] / data['long_trades'] if data['long_trades'] > 0 else 0
    diff = data['all_pnl'] - data['long_pnl']
    print(f"{month:<10} ${data['all_pnl']:>15,.2f} ${data['long_pnl']:>15,.2f} ${diff:>15,.2f} {data['all_trades']:>10} {data['long_trades']:>10}")

print()
print('=' * 120)
print('Cumulative PnL over time (every 100 trades):')
print('=' * 120)
header2 = "Trade Range    ALL Cumulative  LONG Cumulative  Difference      Date Range"
print(header2)
print('-' * 120)

all_cumulative = 0
long_cumulative = 0

for i in range(0, min(len(all_mode_trades), len(long_mode_trades)), 100):
    end_idx = min(i + 100, len(all_mode_trades))
    
    # Calculate PnL for this range
    for j in range(i, end_idx):
        if all_mode_trades[j]['exit']:
            all_cumulative += all_mode_trades[j]['pnl']
        if long_mode_trades[j]['exit']:
            long_cumulative += long_mode_trades[j]['pnl']
    
    start_date_str = str(all_mode_trades[i]['entry'])[:10] if all_mode_trades[i]['entry'] else 'N/A'
    end_date_str = str(all_mode_trades[end_idx-1]['entry'])[:10] if end_idx > 0 and all_mode_trades[end_idx-1]['entry'] else 'N/A'
    date_range = f"{start_date_str} to {end_date_str}"
    
    print(f"{i+1}-{end_idx:<12} ${all_cumulative:>14,.2f} ${long_cumulative:>14,.2f} ${all_cumulative - long_cumulative:>14,.2f} {date_range:<30}")

print()
print('=' * 120)
print('Summary for entire period (March - December 2025):')
print('=' * 120)
all_pnl = sum(t['pnl'] for t in all_mode_trades if t['exit'])
long_pnl = sum(t['pnl'] for t in long_mode_trades if t['exit'])
print(f'ALL mode Total PnL: ${all_pnl:,.2f}')
print(f'LONG mode Total PnL: ${long_pnl:,.2f}')
print(f'Difference: ${all_pnl - long_pnl:,.2f}')
print(f'ALL mode has {(all_pnl / long_pnl - 1) * 100:.1f}% more PnL than LONG mode')


