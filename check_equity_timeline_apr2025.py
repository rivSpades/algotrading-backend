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

# Get equity at key dates by reconstructing from all trades
def get_equity_at_date(target_date, mode='all'):
    equity = 1000.0  # Start with initial capital
    
    # Get all trades that exited before or at target_date
    for trade in Trade.objects.filter(backtest=backtest).order_by('exit_timestamp'):
        metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
        trade_mode = metadata.get('position_mode')
        
        if trade.exit_timestamp and trade.exit_timestamp <= target_date:
            if mode == 'all' and trade_mode == 'all':
                equity += float(trade.pnl) if trade.pnl else 0
            elif mode == 'long' and trade_mode == 'long' and trade.trade_type == 'buy':
                equity += float(trade.pnl) if trade.pnl else 0
    
    return equity

print('=' * 100)
print('Equity at key dates (reconstructed from actual trade exits):')
print('=' * 100)

key_dates = [
    pytz.UTC.localize(datetime(2025, 3, 1)),
    pytz.UTC.localize(datetime(2025, 3, 24)),
    pytz.UTC.localize(datetime(2025, 4, 1)),
    pytz.UTC.localize(datetime(2025, 4, 8)),
    pytz.UTC.localize(datetime(2025, 4, 14)),
    pytz.UTC.localize(datetime(2025, 4, 15))
]

print("Date         ALL Mode Equity    LONG Mode Equity   Difference")
print('-' * 100)

for date in key_dates:
    all_equity = get_equity_at_date(date, 'all')
    long_equity = get_equity_at_date(date, 'long')
    diff = all_equity - long_equity
    print(f"{date.date()}  ${all_equity:>16,.2f}  ${long_equity:>16,.2f}  ${diff:>16,.2f}")

print()
print('=' * 100)
print('April 8, 2025 - Short exits detail (the big day):')
print('=' * 100)

april8 = pytz.UTC.localize(datetime(2025, 4, 8, 23, 59, 59))
shorts_april8 = Trade.objects.filter(
    backtest=backtest,
    trade_type='sell',
    exit_timestamp__date=april8.date()
).order_by('exit_timestamp')

total_april8_pnl = 0
print("Symbol   Exit Price   PnL")
print('-' * 100)

for trade in shorts_april8:
    metadata = trade.metadata if isinstance(trade.metadata, dict) else {}
    mode = metadata.get('position_mode')
    
    if mode == 'all':
        pnl = float(trade.pnl) if trade.pnl else 0
        total_april8_pnl += pnl
        print(f"{trade.symbol.ticker:<8} ${float(trade.exit_price):>10,.2f} ${pnl:>10,.2f}")

print('-' * 100)
print(f"Total short PnL on April 8, 2025: ${total_april8_pnl:,.2f}")

print()
print('=' * 100)
print('Equity BEFORE and AFTER April 8:')
print('=' * 100)

april7 = pytz.UTC.localize(datetime(2025, 4, 7, 23, 59, 59))
april9 = pytz.UTC.localize(datetime(2025, 4, 9, 0, 0, 0))

all_equity_before = get_equity_at_date(april7, 'all')
all_equity_after = get_equity_at_date(april9, 'all')
long_equity_before = get_equity_at_date(april7, 'long')
long_equity_after = get_equity_at_date(april9, 'long')

print(f"BEFORE April 8:")
print(f"  ALL mode:  ${all_equity_before:,.2f}")
print(f"  LONG mode: ${long_equity_before:,.2f}")
print(f"  Difference: ${all_equity_before - long_equity_before:,.2f}")
print()
print(f"AFTER April 8:")
print(f"  ALL mode:  ${all_equity_after:,.2f}")
print(f"  LONG mode: ${long_equity_after:,.2f}")
print(f"  Difference: ${all_equity_after - long_equity_after:,.2f}")
print()
print(f"ALL mode equity change on April 8: ${all_equity_after - all_equity_before:,.2f}")
print(f"LONG mode equity change on April 8: ${long_equity_after - long_equity_before:,.2f}")


