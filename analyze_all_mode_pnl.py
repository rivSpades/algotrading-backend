# analyze_all_mode_pnl.py
import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade

backtest_id = 254
backtest = Backtest.objects.get(id=backtest_id)

print("=" * 80)
print(f"Analyzing ALL mode PnL breakdown for Backtest {backtest_id}")
print("=" * 80)
print()

# Get all trades from ALL mode
all_mode_long_trades = []
all_mode_short_trades = []

for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        trade_info = {
            'symbol': trade.symbol.ticker,
            'entry_timestamp': trade.entry_timestamp,
            'exit_timestamp': trade.exit_timestamp,
            'trade_type': trade.trade_type,
            'entry_price': float(trade.entry_price),
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'quantity': float(trade.quantity),
            'pnl': float(trade.pnl) if trade.pnl else 0,
            'bet_amount': float(metadata.get('bet_amount', 0))
        }
        
        if trade.trade_type == 'buy':
            all_mode_long_trades.append(trade_info)
        elif trade.trade_type == 'sell':
            all_mode_short_trades.append(trade_info)

# Calculate totals
all_mode_long_pnl = sum(t['pnl'] for t in all_mode_long_trades)
all_mode_short_pnl = sum(t['pnl'] for t in all_mode_short_trades)
all_mode_total_pnl = all_mode_long_pnl + all_mode_short_pnl

print(f"ALL mode LONG trades: {len(all_mode_long_trades)}")
print(f"ALL mode SHORT trades: {len(all_mode_short_trades)}")
print(f"ALL mode TOTAL trades: {len(all_mode_long_trades) + len(all_mode_short_trades)}")
print()

print("=" * 80)
print("ALL Mode PnL Breakdown:")
print("=" * 80)
print(f"  Long trades Total PnL:  ${all_mode_long_pnl:,.2f}")
print(f"  Short trades Total PnL: ${all_mode_short_pnl:,.2f}")
print(f"  Combined Total PnL:     ${all_mode_total_pnl:,.2f}")
print()

# Compare with separate modes
long_mode_trades = []
short_mode_trades = []

for trade in Trade.objects.filter(backtest=backtest).order_by('entry_timestamp'):
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'long':
        long_mode_trades.append({
            'pnl': float(trade.pnl) if trade.pnl else 0
        })
    elif mode == 'short':
        short_mode_trades.append({
            'pnl': float(trade.pnl) if trade.pnl else 0
        })

long_mode_pnl = sum(t['pnl'] for t in long_mode_trades)
short_mode_pnl = sum(t['pnl'] for t in short_mode_trades)
separate_modes_total = long_mode_pnl + short_mode_pnl

print("=" * 80)
print("Comparison with Separate LONG and SHORT modes:")
print("=" * 80)
print(f"  LONG mode Total PnL:  ${long_mode_pnl:,.2f}")
print(f"  SHORT mode Total PnL: ${short_mode_pnl:,.2f}")
print(f"  Combined Total PnL:   ${separate_modes_total:,.2f}")
print()

print("=" * 80)
print("Differences (ALL mode vs Separate modes):")
print("=" * 80)
print(f"  Long PnL difference:  ${all_mode_long_pnl - long_mode_pnl:,.2f}")
print(f"  Short PnL difference: ${all_mode_short_pnl - short_mode_pnl:,.2f}")
print(f"  Total PnL difference: ${all_mode_total_pnl - separate_modes_total:,.2f}")
print("=" * 80)


