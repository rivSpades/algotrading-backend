# deep_compare_all_vs_long_v2.py
import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade
from collections import defaultdict

backtest_id = 254
backtest = Backtest.objects.get(id=backtest_id)
initial_capital = float(backtest.initial_capital)

print("=" * 120)
print(f"Deep Comparison: ALL mode vs LONG mode - Showing equity at each long trade entry")
print(f"Initial Capital: ${initial_capital:,.2f}")
print("=" * 120)
print()

# Get ALL mode trades (both longs and shorts) chronologically
all_mode_events = []

for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        # Entry event
        all_mode_events.append({
            'timestamp': trade.entry_timestamp,
            'type': 'entry',
            'trade_type': trade.trade_type,
            'symbol': trade.symbol.ticker,
            'bet_amount': float(metadata.get('bet_amount', 0)),
            'entry_price': float(trade.entry_price),
            'pnl': 0
        })
        
        # Exit event (if exists)
        if trade.exit_timestamp:
            all_mode_events.append({
                'timestamp': trade.exit_timestamp,
                'type': 'exit',
                'trade_type': trade.trade_type,
                'symbol': trade.symbol.ticker,
                'bet_amount': float(metadata.get('bet_amount', 0)),
                'pnl': float(trade.pnl) if trade.pnl else 0
            })

# Sort by timestamp
all_mode_events.sort(key=lambda x: x['timestamp'] or datetime.min)

# Get LONG mode trades
long_mode_trades = []
for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'long' and trade.trade_type == 'buy':
        long_mode_trades.append({
            'symbol': trade.symbol.ticker,
            'entry_timestamp': trade.entry_timestamp,
            'exit_timestamp': trade.exit_timestamp,
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'quantity': float(trade.quantity),
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'bet_amount': float(metadata.get('bet_amount', 0))
        })

# Get ALL mode long trades for comparison
all_mode_long_trades = []
for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all' and trade.trade_type == 'buy':
        all_mode_long_trades.append({
            'symbol': trade.symbol.ticker,
            'entry_timestamp': trade.entry_timestamp,
            'exit_timestamp': trade.exit_timestamp,
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'quantity': float(trade.quantity),
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'bet_amount': float(metadata.get('bet_amount', 0))
        })

print(f"Total ALL mode events: {len(all_mode_events)}")
print(f"Total ALL mode long trades: {len(all_mode_long_trades)}")
print(f"Total LONG mode trades: {len(long_mode_trades)}")
print()

# Build equity timeline for ALL mode
all_mode_equity = initial_capital
equity_at_long_entry = {}  # {trade_index: equity}

# Process events chronologically and record equity at each long entry
long_trade_idx = 0
for event in all_mode_events:
    if event['type'] == 'exit':
        all_mode_equity += event['pnl']
    elif event['type'] == 'entry' and event['trade_type'] == 'buy':
        # This is a long entry - record current equity
        if long_trade_idx < len(all_mode_long_trades):
            equity_at_long_entry[long_trade_idx] = all_mode_equity
            long_trade_idx += 1

# Build equity timeline for LONG mode
long_mode_equity = initial_capital

# Now print comparison
print("=" * 120)
print("Side-by-side comparison: First 100 LONG trades (showing equity at entry)")
print("=" * 120)
print(f"{'#':<4} {'Timestamp':<20} {'Symbol':<8} {'ALL Mode Equity':<18} {'ALL Bet Amt':<14} {'ALL PnL':<12} {'LONG Equity':<16} {'LONG Bet Amt':<14} {'LONG PnL':<12} {'Bet Diff':<12} {'PnL Diff':<12}")
print("-" * 120)

all_equity = initial_capital
long_equity = initial_capital

for i in range(min(100, len(all_mode_long_trades), len(long_mode_trades))):
    all_trade = all_mode_long_trades[i]
    long_trade = long_mode_trades[i]
    
    # Get equity at this entry for ALL mode
    all_equity_at_entry = equity_at_long_entry.get(i, all_equity)
    
    # Update LONG mode equity (from previous trade's exit)
    if i > 0 and long_mode_trades[i-1]['exit_timestamp']:
        long_equity += long_mode_trades[i-1]['pnl']
    
    # Update ALL mode equity (need to process all events up to this entry)
    # Actually, we already have equity_at_long_entry, so use that
    all_equity = all_equity_at_entry
    
    timestamp_str = str(all_trade['entry_timestamp'])[:19] if all_trade['entry_timestamp'] else 'N/A'
    
    bet_diff = all_trade['bet_amount'] - long_trade['bet_amount']
    pnl_diff = all_trade['pnl'] - long_trade['pnl']
    
    print(f"{i+1:<4} {timestamp_str:<20} {all_trade['symbol']:<8} "
          f"${all_equity_at_entry:>16,.2f} ${all_trade['bet_amount']:>12,.2f} ${all_trade['pnl']:>10,.2f}  "
          f"${long_equity:>14,.2f} ${long_trade['bet_amount']:>12,.2f} ${long_trade['pnl']:>10,.2f}  "
          f"${bet_diff:>10,.2f} ${pnl_diff:>10,.2f}")
    
    # Highlight significant differences
    if abs(bet_diff) > 10:
        print(f"      *** LARGE BET DIFFERENCE: ${bet_diff:,.2f} (ALL equity=${all_equity_at_entry:,.2f} vs LONG equity=${long_equity:,.2f}) ***")
    
    # Update equity after exit
    if all_trade['exit_timestamp']:
        all_equity += all_trade['pnl']
    if long_trade['exit_timestamp']:
        long_equity += long_trade['pnl']

print()
print("=" * 120)
print("Summary after first 100 trades:")
print("=" * 120)

all_equity_final = initial_capital
long_equity_final = initial_capital
all_pnl = 0
long_pnl = 0

for i in range(min(100, len(all_mode_long_trades), len(long_mode_trades))):
    if all_mode_long_trades[i]['exit_timestamp']:
        all_equity_final += all_mode_long_trades[i]['pnl']
        all_pnl += all_mode_long_trades[i]['pnl']
    if long_mode_trades[i]['exit_timestamp']:
        long_equity_final += long_mode_trades[i]['pnl']
        long_pnl += long_mode_trades[i]['pnl']

print(f"ALL mode:  Equity = ${all_equity_final:,.2f}, Long trades PnL = ${all_pnl:,.2f}")
print(f"LONG mode: Equity = ${long_equity_final:,.2f}, Long trades PnL = ${long_pnl:,.2f}")
print(f"Difference: Equity = ${all_equity_final - long_equity_final:,.2f}, PnL = ${all_pnl - long_pnl:,.2f}")
print("=" * 120)


