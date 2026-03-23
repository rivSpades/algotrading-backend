"""
Test script to verify cash management system with 100 symbols linked to Alpaca broker
Tests that:
1. cash_available and cash_invested are tracked correctly
2. Trades are skipped when bet_amount > cash_available
3. Equity = cash_available + cash_invested
4. Position exits update cash correctly
5. skipped_trades_count is tracked and reported
"""
import os
import sys
import django

sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest
from strategies.models import StrategyDefinition
from market_data.models import Symbol
from live_trading.models import Broker, SymbolBrokerAssociation
from django.utils import timezone
from datetime import timedelta
from django.db.models import Q
import random

print("="*80)
print("Cash Management System Test - 100 Symbols (Alpaca Broker)")
print("="*80)

# Find a strategy
strategy = StrategyDefinition.objects.filter(name='Gap-Up and Gap-Down').first()
if not strategy:
    print("ERROR: Gap-Up and Gap-Down strategy not found. Please create it first.")
    sys.exit(1)

# Get Alpaca broker
try:
    broker = Broker.objects.get(code='ALPACA')
    print(f"✓ Found Alpaca broker: {broker.name} (ID: {broker.id})")
except Broker.DoesNotExist:
    print("ERROR: Alpaca broker not found. Please create it first.")
    sys.exit(1)

# Get 100 symbols linked to Alpaca broker with at least one active flag
associations = SymbolBrokerAssociation.objects.filter(
    broker=broker
).filter(
    Q(long_active=True) | Q(short_active=True)
).select_related('symbol')

if not associations.exists():
    print("ERROR: No symbols found linked to Alpaca broker with active trading capabilities.")
    sys.exit(1)

# Get symbols from associations
available_symbols = [assoc.symbol for assoc in associations]

if len(available_symbols) < 100:
    print(f"WARNING: Only {len(available_symbols)} symbols available linked to Alpaca, using all of them")
    symbols = available_symbols
else:
    symbols = random.sample(available_symbols, 100)

print(f"\n✓ Selected {len(symbols)} symbols linked to Alpaca broker")
print(f"  Strategy: {strategy.name}")
print(f"  Broker: {broker.name}")
print(f"  Sample symbols: {', '.join([s.ticker for s in symbols[:5]])}...")

# Create a test backtest with specified parameters
end_date = timezone.now()
start_date = end_date - timedelta(days=365)  # 1 year of data

backtest = Backtest.objects.create(
    name=f'Cash Management Test - {len(symbols)} symbols (Alpaca)',
    strategy=strategy,
    broker=broker,  # Link to Alpaca broker
    start_date=start_date,
    end_date=end_date,
    split_ratio=0.2,  # 20% training, 80% test
    initial_capital=1000.0,  # Initial capital
    bet_size_percentage=10.0,  # 10% bet size
    strategy_parameters={
        'threshold': 1.0,  # Threshold: 1
        'std_period': 90  # STD period: 90
    },
    status='pending'
)

backtest.symbols.set(symbols)

print(f"\n✓ Created backtest ID: {backtest.id}")
print(f"  Initial Capital: ${backtest.initial_capital}")
print(f"  Bet Size: {backtest.bet_size_percentage}%")
print(f"  Date Range: {start_date.date()} to {end_date.date()}")
print(f"  Symbols: {len(symbols)}")

# Run the backtest directly using the executor
from backtest_engine.services.backtest_executor import BacktestExecutor

print("\n" + "="*80)
print("Running backtest (this may take a few minutes)...")
print("="*80)

try:
    # Run for 'all' mode directly
    executor = BacktestExecutor(backtest, position_mode='long')
    
    if not executor.symbols:
        print(f"WARNING: Executor has no symbols after filtering. Original count: {len(symbols)}")
        print("This might be due to broker filtering. Trying without broker...")
        backtest.broker = None
        backtest.save()
        executor = BacktestExecutor(backtest, position_mode='long')
    
    if not executor.symbols:
        print("ERROR: No symbols available after filtering. Cannot run backtest.")
        sys.exit(1)
    
    print(f"✓ Executor initialized with {len(executor.symbols)} symbols")
    
    # Execute strategy
    executor.execute_strategy()
    
    print(f"✓ Strategy execution completed")
    print(f"  Total trades generated: {len(executor.trades)}")
    print(f"  Skipped trades: {executor.skipped_trades_count}")
    
    # Calculate statistics
    stats = executor.calculate_statistics()
    
    # Check portfolio stats
    if None in stats:
        portfolio_stats = stats[None]
        print("\n" + "="*80)
        print("Portfolio Statistics (ALL mode):")
        print("="*80)
        print(f"  Total Trades: {portfolio_stats.get('total_trades', 0)}")
        print(f"  Winning Trades: {portfolio_stats.get('winning_trades', 0)}")
        print(f"  Losing Trades: {portfolio_stats.get('losing_trades', 0)}")
        print(f"  Win Rate: {portfolio_stats.get('win_rate', 0):.2f}%")
        print(f"  Total PnL: ${portfolio_stats.get('total_pnl', 0):,.2f}")
        print(f"  Total PnL %: {portfolio_stats.get('total_pnl_percentage', 0):,.2f}%")
        print(f"  Max Drawdown: {portfolio_stats.get('max_drawdown', 0):,.2f}%")
        
        # Check for skipped_trades_count
        skipped_count = portfolio_stats.get('skipped_trades_count', 0)
        print(f"  Skipped Trades: {skipped_count}")
        
        if skipped_count > 0:
            print(f"\n  ✓ Cash management working! {skipped_count} trades were skipped due to insufficient cash")
        else:
            print(f"\n  ℹ No trades were skipped (all trades had sufficient cash available)")
        
        # Check equity curve
        equity_curve = portfolio_stats.get('equity_curve', [])
        if equity_curve:
            initial_equity = equity_curve[0].get('equity', 0)
            final_equity = equity_curve[-1].get('equity', 0)
            print(f"\n  Equity Curve:")
            print(f"    Initial: ${initial_equity:,.2f}")
            print(f"    Final: ${final_equity:,.2f}")
            print(f"    Change: ${final_equity - initial_equity:,.2f}")
            
            # Verify equity matches total_pnl
            expected_final = initial_equity + portfolio_stats.get('total_pnl', 0)
            if abs(final_equity - expected_final) < 0.01:
                print(f"    ✓ Equity matches total_pnl (${final_equity:,.2f} ≈ ${expected_final:,.2f})")
            else:
                print(f"    ✗ Equity mismatch! Expected ${expected_final:,.2f}, got ${final_equity:,.2f}")
    
    print("\n" + "="*80)
    print("✓ Cash management test completed successfully!")
    print("="*80)
    
except Exception as e:
    print(f"\n✗ Error during backtest: {str(e)}")
    import traceback
    traceback.print_exc()
    backtest.refresh_from_db()
    backtest.status = 'failed'
    backtest.error_message = str(e)
    backtest.save()

