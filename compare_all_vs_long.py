# compare_all_vs_long.py
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

print("=" * 80)
print(f"Comparing ALL mode vs LONG mode trades for Backtest {backtest_id}")
print("=" * 80)
print()

# Get all LONG trades from ALL mode
all_long_trades = []
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
        all_long_trades.append(trade_info)
    elif mode == 'long' and trade.trade_type == 'buy':
        long_mode_trades.append(trade_info)

print(f"ALL mode long trades (buy only): {len(all_long_trades)}")
print(f"LONG mode trades (buy only): {len(long_mode_trades)}")
print()

# Compare totals
all_long_pnl = sum(t['pnl'] for t in all_long_trades)
long_mode_pnl = sum(t['pnl'] for t in long_mode_trades)

print(f"ALL mode long trades Total PnL: ${all_long_pnl:.2f}")
print(f"LONG mode trades Total PnL: ${long_mode_pnl:.2f}")
print(f"Difference: ${all_long_pnl - long_mode_pnl:.2f}")
print()

# Compare by symbol
all_long_by_symbol = defaultdict(list)
long_mode_by_symbol = defaultdict(list)

for trade in all_long_trades:
    all_long_by_symbol[trade['symbol']].append(trade)

for trade in long_mode_trades:
    long_mode_by_symbol[trade['symbol']].append(trade)

# Find symbols with differences
print("=" * 80)
print("Symbols with differences in trade count or PnL:")
print("=" * 80)
differences = []
for symbol in sorted(set(list(all_long_by_symbol.keys()) + list(long_mode_by_symbol.keys()))):
    all_trades_list = all_long_by_symbol.get(symbol, [])
    long_trades_list = long_mode_by_symbol.get(symbol, [])
    
    all_count = len(all_trades_list)
    long_count = len(long_trades_list)
    all_pnl = sum(t['pnl'] for t in all_trades_list)
    long_pnl = sum(t['pnl'] for t in long_trades_list)
    
    if all_count != long_count or abs(all_pnl - long_pnl) > 0.01:
        differences.append((symbol, all_count, long_count, all_pnl, long_pnl, all_trades_list, long_trades_list))

if differences:
    print(f"\nFound {len(differences)} symbols with differences:\n")
    for symbol, all_c, long_c, all_p, long_p, all_trades_list, long_trades_list in differences[:30]:
        print(f"{symbol}:")
        print(f"  ALL mode:  {all_c} trades, Total PnL: ${all_p:.2f}")
        print(f"  LONG mode: {long_c} trades, Total PnL: ${long_p:.2f}")
        print(f"  Difference: {all_c - long_c} trades, ${all_p - long_p:.2f} PnL")
        
        # Compare first few trades in detail
        if all_trades_list and long_trades_list:
            min_trades = min(len(all_trades_list), len(long_trades_list), 3)
            print(f"  First {min_trades} trades comparison:")
            for i in range(min_trades):
                a = all_trades_list[i]
                l = long_trades_list[i] if i < len(long_trades_list) else None
                print(f"    Trade {i+1}:")
                print(f"      ALL:  Entry={a['entry_timestamp']}, BetAmount=${a['bet_amount']:.2f}, PnL=${a['pnl']:.2f}")
                if l:
                    print(f"      LONG: Entry={l['entry_timestamp']}, BetAmount=${l['bet_amount']:.2f}, PnL=${l['pnl']:.2f}")
                    if abs(a['pnl'] - l['pnl']) > 0.01:
                        print(f"      *** PnL DIFFERENCE: ${a['pnl'] - l['pnl']:.2f} ***")
                    if abs(a['bet_amount'] - l['bet_amount']) > 0.01:
                        print(f"      *** Bet Amount DIFFERENCE: ${a['bet_amount'] - l['bet_amount']:.2f} ***")
        print()
else:
    print("No differences found - all symbols match!")

print("=" * 80)
print("Summary:")
print(f"  Total ALL mode long trades: {len(all_long_trades)}")
print(f"  Total LONG mode trades: {len(long_mode_trades)}")
print(f"  ALL mode Total PnL: ${all_long_pnl:.2f}")
print(f"  LONG mode Total PnL: ${long_mode_pnl:.2f}")
print(f"  Difference: ${all_long_pnl - long_mode_pnl:.2f}")
print("=" * 80)


