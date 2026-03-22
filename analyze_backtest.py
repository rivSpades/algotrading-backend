"""
Script to analyze backtest results and verify cash management logic
"""
import os
import django
import sys

# Setup Django
sys.path.insert(0, '/home/ric/Projects/Trading/algo_trading_backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'algo_trading_backend.settings')
django.setup()

from backtest_engine.models import Backtest, Trade, BacktestStatistics
from django.db.models import Sum, Count, Q
import time

def wait_for_completion(backtest_id, timeout=3600):
    """Wait for backtest to complete"""
    print(f"\n⏳ Waiting for backtest {backtest_id} to complete...")
    start_time = time.time()
    check_interval = 10  # Check every 10 seconds
    
    while time.time() - start_time < timeout:
        try:
            backtest = Backtest.objects.get(id=backtest_id)
            backtest.refresh_from_db()
            status = backtest.status
            
            if status == 'completed':
                elapsed = int(time.time() - start_time)
                print(f"\n✅ Backtest completed! (took {elapsed}s)")
                return True
            elif status == 'failed':
                print(f"\n❌ Backtest failed!")
                print(f"   Error: {backtest.error_message}")
                return False
            else:
                elapsed = int(time.time() - start_time)
                print(f"   Status: {status} (elapsed: {elapsed}s)", end='\r', flush=True)
                time.sleep(check_interval)
        except Backtest.DoesNotExist:
            print(f"\n❌ Backtest {backtest_id} not found!")
            return False
        except Exception as e:
            print(f"\n❌ Error checking status: {e}")
            return False
    
    print(f"\n⏰ Timeout waiting for backtest to complete")
    return False

def analyze_results(backtest_id):
    """Analyze backtest results"""
    print(f"\n📊 Analyzing backtest {backtest_id} results...")
    
    backtest = Backtest.objects.get(id=backtest_id)
    
    # Get portfolio-level statistics
    portfolio_stats = BacktestStatistics.objects.filter(
        backtest=backtest,
        symbol__isnull=True
    ).first()
    
    if not portfolio_stats:
        print("❌ No portfolio statistics found")
        return
    
    print(f"\n=== BACKTEST SUMMARY ===")
    print(f"Strategy: {backtest.strategy.name}")
    print(f"Broker: {backtest.broker.name if backtest.broker else 'None'}")
    print(f"Initial Capital: ${backtest.initial_capital}")
    print(f"Bet Size: {backtest.bet_size_percentage}%")
    print(f"Split Ratio: {backtest.split_ratio*100:.0f}% training / {(1-backtest.split_ratio)*100:.0f}% test")
    print(f"Strategy Parameters: {backtest.strategy_parameters}")
    
    # Count trades by mode
    trades_all = Trade.objects.filter(
        backtest=backtest,
        metadata__position_mode='all'
    )
    trades_long = Trade.objects.filter(
        backtest=backtest,
        metadata__position_mode='long'
    )
    trades_short = Trade.objects.filter(
        backtest=backtest,
        metadata__position_mode='short'
    )
    
    print(f"\n=== TRADE COUNTS ===")
    print(f"ALL mode: {trades_all.count()} trades")
    print(f"LONG mode: {trades_long.count()} trades")
    print(f"SHORT mode: {trades_short.count()} trades")
    
    # Portfolio statistics for ALL mode
    print(f"\n=== PORTFOLIO STATISTICS (ALL MODE) ===")
    all_stats = portfolio_stats.additional_stats.get('all', {})
    print(f"Total Trades: {portfolio_stats.total_trades}")
    print(f"Winning Trades: {portfolio_stats.winning_trades}")
    print(f"Losing Trades: {portfolio_stats.losing_trades}")
    print(f"Win Rate: {portfolio_stats.win_rate:.2f}%")
    print(f"Total PnL: ${portfolio_stats.total_pnl:.2f}")
    print(f"Total PnL %: {portfolio_stats.total_pnl_percentage:.2f}%")
    print(f"Average PnL: ${portfolio_stats.average_pnl:.2f}")
    print(f"Profit Factor: {portfolio_stats.profit_factor:.2f}")
    print(f"Max Drawdown: {portfolio_stats.max_drawdown:.2f}%")
    
    # Check for skipped trades (this would be in executor, but we can check trade patterns)
    print(f"\n=== CASH MANAGEMENT VERIFICATION ===")
    
    # Get sample trades to check bet_amounts
    sample_trades = Trade.objects.filter(
        backtest=backtest,
        metadata__position_mode='all'
    ).exclude(metadata__bet_amount__isnull=True)[:10]
    
    if sample_trades:
        print(f"Sample trades (first 10):")
        for trade in sample_trades:
            bet_amount = trade.metadata.get('bet_amount', 'N/A')
            print(f"  {trade.symbol.ticker}: {trade.trade_type} @ ${trade.entry_price:.2f}, bet_amount=${bet_amount}, PnL=${trade.pnl:.2f}")
    
    # Check equity curve
    equity_curve = portfolio_stats.equity_curve
    if equity_curve:
        print(f"\n=== EQUITY CURVE ===")
        print(f"Equity curve points: {len(equity_curve)}")
        if len(equity_curve) > 0:
            initial_equity = equity_curve[0].get('equity', backtest.initial_capital)
            final_equity = equity_curve[-1].get('equity', initial_equity)
            print(f"Initial Equity: ${initial_equity:.2f}")
            print(f"Final Equity: ${final_equity:.2f}")
            print(f"Total Return: ${final_equity - float(initial_equity):.2f} ({((final_equity / float(initial_equity) - 1) * 100):.2f}%)")
            
            # Check for negative equity (account blow-up)
            min_equity = min(point.get('equity', initial_equity) for point in equity_curve)
            if min_equity <= 0:
                print(f"⚠️  WARNING: Minimum equity was ${min_equity:.2f} (account blow-up detected)")
            else:
                print(f"✓ Minimum equity: ${min_equity:.2f} (no blow-up)")
    
    # Check for symbol-level statistics
    symbol_stats_count = BacktestStatistics.objects.filter(
        backtest=backtest,
        symbol__isnull=False
    ).count()
    print(f"\n=== SYMBOL-LEVEL STATISTICS ===")
    print(f"Symbols with statistics: {symbol_stats_count}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    # Get the latest backtest ID
    from backtest_engine.models import Backtest
    latest = Backtest.objects.order_by('-id').first()
    backtest_id = latest.id if latest else 232
    print(f"Analyzing backtest ID: {backtest_id}")
    
    # Wait for completion
    if wait_for_completion(backtest_id, timeout=3600):
        # Analyze results
        analyze_results(backtest_id)
    else:
        print("❌ Backtest did not complete successfully")

