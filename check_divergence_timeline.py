import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade

backtest = Backtest.objects.get(id=254)

# Check cumulative PnL over time
print('Cumulative PnL comparison over time (every 500 trades):')
print('=' * 100)

all_mode_long_trades = []
long_mode_trades = []

for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all' and trade.trade_type == 'buy':
        all_mode_long_trades.append({
            'entry': trade.entry_timestamp,
            'exit': trade.exit_timestamp,
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'bet_amount': float(metadata.get('bet_amount', 0))
        })
    elif mode == 'long' and trade.trade_type == 'buy':
        long_mode_trades.append({
            'entry': trade.entry_timestamp,
            'exit': trade.exit_timestamp,
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'bet_amount': float(metadata.get('bet_amount', 0))
        })

header = "Trade Range          ALL Mode PnL       LONG Mode PnL     Difference        Date Range"
print(header)
print('-' * 100)

all_cumulative = 0
long_cumulative = 0

for i in range(0, min(len(all_mode_long_trades), len(long_mode_trades)), 500):
    end_idx = min(i + 500, len(all_mode_long_trades))
    
    # Calculate PnL for this range (only closed trades)
    all_pnl = sum(t['pnl'] for t in all_mode_long_trades[i:end_idx] if t['exit'])
    long_pnl = sum(t['pnl'] for t in long_mode_trades[i:end_idx] if t['exit'])
    
    all_cumulative += all_pnl
    long_cumulative += long_pnl
    
    start_date = str(all_mode_long_trades[i]['entry'])[:10] if all_mode_long_trades[i]['entry'] else 'N/A'
    end_date = str(all_mode_long_trades[end_idx-1]['entry'])[:10] if end_idx > 0 and all_mode_long_trades[end_idx-1]['entry'] else 'N/A'
    date_range = f"{start_date} to {end_date}"
    
    print(f"{i+1}-{end_idx:<18} ${all_cumulative:>16,.2f} ${long_cumulative:>16,.2f} ${all_cumulative - long_cumulative:>16,.2f} {date_range:<30}")

print()
print('=' * 100)
print('When does ALL mode start outperforming?')
print('=' * 100)

# Find where the difference becomes positive
all_cum = 0
long_cum = 0

for i in range(min(len(all_mode_long_trades), len(long_mode_trades))):
    if all_mode_long_trades[i]['exit']:
        all_cum += all_mode_long_trades[i]['pnl']
    if long_mode_trades[i]['exit']:
        long_cum += long_mode_trades[i]['pnl']
    
    diff = all_cum - long_cum
    if diff > 100 and i > 100:  # Find where difference exceeds $100
        date_str = str(all_mode_long_trades[i]['entry'])[:10] if all_mode_long_trades[i]['entry'] else 'N/A'
        print(f"Trade #{i+1} ({date_str}): ALL=${all_cum:,.2f}, LONG=${long_cum:,.2f}, Diff=${diff:,.2f}")
        break


