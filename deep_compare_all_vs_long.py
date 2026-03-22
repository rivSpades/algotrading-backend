# deep_compare_all_vs_long.py
import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade
from collections import defaultdict
from datetime import datetime

backtest_id = 254
backtest = Backtest.objects.get(id=backtest_id)
initial_capital = float(backtest.initial_capital)

print("=" * 100)
print(f"Deep Comparison: ALL mode vs LONG mode - Timestamp, Equity, Bet Amount, PnL")
print(f"Initial Capital: ${initial_capital:,.2f}")
print("=" * 100)
print()

# Get all trades from ALL mode (longs only) and LONG mode
all_mode_long_trades = []
long_mode_trades = []

for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    trade_info = {
        'symbol': trade.symbol.ticker,
        'entry_timestamp': trade.entry_timestamp,
        'exit_timestamp': trade.exit_timestamp,
        'entry_price': float(trade.entry_price),
        'exit_price': float(trade.exit_price) if trade.exit_price else None,
        'quantity': float(trade.quantity),
        'pnl': float(trade.pnl) if trade.pnl else 0,
        'bet_amount': float(metadata.get('bet_amount', 0)),
        'trade_id': trade.id
    }
    
    if mode == 'all' and trade.trade_type == 'buy':
        all_mode_long_trades.append(trade_info)
    elif mode == 'long' and trade.trade_type == 'buy':
        long_mode_trades.append(trade_info)

# Reconstruct equity curve for ALL mode (simulating from trades)
print("Reconstructing equity curves...")
print()

# ALL mode: Track equity from all trades (longs + shorts)
all_mode_equity = initial_capital
all_mode_equity_timeline = [(None, initial_capital)]  # (timestamp, equity)

# Get ALL mode shorts too
all_mode_short_trades = []
for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    if mode == 'all' and trade.trade_type == 'sell':
        all_mode_short_trades.append({
            'entry_timestamp': trade.entry_timestamp,
            'exit_timestamp': trade.exit_timestamp,
            'pnl': float(trade.pnl) if trade.pnl else 0
        })

# Combine all ALL mode trades chronologically
all_mode_all_trades = []
for trade in all_mode_long_trades:
    all_mode_all_trades.append({
        'timestamp': trade['entry_timestamp'],
        'type': 'long_entry',
        'pnl': 0,
        'bet_amount': trade['bet_amount'],
        'symbol': trade['symbol']
    })
    if trade['exit_timestamp']:
        all_mode_all_trades.append({
            'timestamp': trade['exit_timestamp'],
            'type': 'long_exit',
            'pnl': trade['pnl'],
            'bet_amount': trade['bet_amount'],
            'symbol': trade['symbol']
        })

for trade in all_mode_short_trades:
    all_mode_all_trades.append({
        'timestamp': trade['entry_timestamp'],
        'type': 'short_entry',
        'pnl': 0,
        'bet_amount': 0,  # We don't have bet_amount for shorts easily
        'symbol': 'SHORT'
    })
    if trade['exit_timestamp']:
        all_mode_all_trades.append({
            'timestamp': trade['exit_timestamp'],
            'type': 'short_exit',
            'pnl': trade['pnl'],
            'bet_amount': 0,
            'symbol': 'SHORT'
        })

# Sort by timestamp
all_mode_all_trades.sort(key=lambda x: x['timestamp'] or datetime.min)

# LONG mode: Track equity from long trades only
long_mode_equity = initial_capital
long_mode_equity_timeline = [(None, initial_capital)]

# Build equity timeline for LONG mode
for trade in long_mode_trades:
    if trade['exit_timestamp']:
        long_mode_equity += trade['pnl']
        long_mode_equity_timeline.append((trade['exit_timestamp'], long_mode_equity))

# Now compare first 50 long trades side by side
print("=" * 100)
print("Side-by-side comparison: First 50 LONG trades")
print("=" * 100)
print(f"{'#':<4} {'Timestamp':<20} {'Symbol':<8} {'ALL Mode':<35} {'LONG Mode':<35} {'Diff'}")
print(f"{'':<4} {'':<20} {'':<8} {'Equity':<12} {'Bet Amt':<12} {'PnL':<12} {'Equity':<12} {'Bet Amt':<12} {'PnL':<12} {'Bet':<10} {'PnL':<10}")
print("-" * 100)

# Build lookup for ALL mode equity at each timestamp
all_mode_equity_lookup = {}
all_equity = initial_capital
for event in all_mode_all_trades[:2000]:  # Process first 2000 events to build lookup
    if event['type'] in ['long_exit', 'short_exit']:
        all_equity += event['pnl']
        all_mode_equity_lookup[event['timestamp']] = all_equity

# Compare trades
all_mode_current_equity = initial_capital
long_mode_current_equity = initial_capital

for i in range(min(50, len(all_mode_long_trades), len(long_mode_trades))):
    all_trade = all_mode_long_trades[i]
    long_trade = long_mode_trades[i]
    
    # Find equity at entry time for ALL mode (use closest previous exit)
    all_equity_at_entry = all_mode_current_equity
    # Find the most recent equity update before this entry
    for prev_timestamp, prev_equity in reversed(all_mode_equity_timeline):
        if prev_timestamp and prev_timestamp <= all_trade['entry_timestamp']:
            all_equity_at_entry = prev_equity
            break
    
    # Update LONG mode equity
    if i > 0 and long_mode_trades[i-1]['exit_timestamp']:
        long_mode_current_equity += long_mode_trades[i-1]['pnl']
    
    # Update ALL mode equity (find all exits before this entry)
    for event in all_mode_all_trades:
        if event['timestamp'] and event['timestamp'] <= all_trade['entry_timestamp']:
            if event['type'] in ['long_exit', 'short_exit']:
                all_mode_current_equity += event['pnl']
        else:
            break
    
    all_equity_before_entry = all_mode_current_equity
    
    # Calculate expected bet amount from equity
    bet_size_pct = 0.10  # 10%
    all_expected_bet = all_equity_before_entry * bet_size_pct
    long_expected_bet = long_mode_current_equity * bet_size_pct
    
    timestamp_str = str(all_trade['entry_timestamp'])[:19] if all_trade['entry_timestamp'] else 'N/A'
    
    print(f"{i+1:<4} {timestamp_str:<20} {all_trade['symbol']:<8} "
          f"${all_equity_before_entry:>10,.2f} ${all_trade['bet_amount']:>10,.2f} ${all_trade['pnl']:>10,.2f}  "
          f"${long_mode_current_equity:>10,.2f} ${long_trade['bet_amount']:>10,.2f} ${long_trade['pnl']:>10,.2f}  "
          f"${all_trade['bet_amount'] - long_trade['bet_amount']:>8,.2f} ${all_trade['pnl'] - long_trade['pnl']:>8,.2f}")
    
    # Update equity after this trade's exit (if it has exited)
    if all_trade['exit_timestamp']:
        all_mode_current_equity += all_trade['pnl']
        all_mode_equity_timeline.append((all_trade['exit_timestamp'], all_mode_current_equity))

print()
print("=" * 100)
print("Summary after first 50 trades:")
print("=" * 100)

# Recalculate properly
all_equity = initial_capital
long_equity = initial_capital
all_cumulative_pnl = 0
long_cumulative_pnl = 0

for i in range(min(50, len(all_mode_long_trades), len(long_mode_trades))):
    if all_mode_long_trades[i]['exit_timestamp']:
        all_equity += all_mode_long_trades[i]['pnl']
        all_cumulative_pnl += all_mode_long_trades[i]['pnl']
    if long_mode_trades[i]['exit_timestamp']:
        long_equity += long_mode_trades[i]['pnl']
        long_cumulative_pnl += long_mode_trades[i]['pnl']

print(f"ALL mode:  Equity = ${all_equity:,.2f}, Cumulative PnL = ${all_cumulative_pnl:,.2f}")
print(f"LONG mode: Equity = ${long_equity:,.2f}, Cumulative PnL = ${long_cumulative_pnl:,.2f}")
print(f"Difference: Equity = ${all_equity - long_equity:,.2f}, PnL = ${all_cumulative_pnl - long_cumulative_pnl:,.2f}")
print("=" * 100)


