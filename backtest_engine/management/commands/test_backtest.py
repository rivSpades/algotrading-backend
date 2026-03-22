"""
Management command to create and run a test backtest with specific parameters
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from strategies.models import StrategyDefinition
from live_trading.models import Broker, SymbolBrokerAssociation
from market_data.models import Symbol
from backtest_engine.models import Backtest
from backtest_engine.tasks import run_backtest_task
from django.db.models import Q
import random


class Command(BaseCommand):
    help = 'Create and run a test backtest with Gap-Up and Gap-Down strategy'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbol-count',
            type=int,
            default=30,
            help='Number of random symbols to use (default: 30)'
        )

    def handle(self, *args, **options):
        symbol_count = options['symbol_count']
        
        self.stdout.write("Creating test backtest...")
        
        # Get Gap-Up and Gap-Down strategy
        try:
            strategy = StrategyDefinition.objects.get(name='Gap-Up and Gap-Down')
            self.stdout.write(self.style.SUCCESS(f"✓ Found strategy: {strategy.name}"))
        except StrategyDefinition.DoesNotExist:
            self.stdout.write(self.style.ERROR("✗ Strategy 'Gap-Up and Gap-Down' not found"))
            return
        
        # Get Alpaca broker
        try:
            broker = Broker.objects.get(name='Alpaca')
            self.stdout.write(self.style.SUCCESS(f"✓ Found broker: {broker.name}"))
        except Broker.DoesNotExist:
            self.stdout.write(self.style.ERROR("✗ Broker 'Alpaca' not found"))
            return
        
        # Get 30 random symbols linked to Alpaca broker
        associations = SymbolBrokerAssociation.objects.filter(
            broker=broker,
            symbol__status='active'
        ).filter(
            Q(long_active=True) | Q(short_active=True)
        ).select_related('symbol')
        
        available_symbols = [assoc.symbol for assoc in associations]
        
        if len(available_symbols) < symbol_count:
            self.stdout.write(
                self.style.WARNING(
                    f"⚠ Only {len(available_symbols)} symbols available, using all of them instead of {symbol_count}"
                )
            )
            symbols = available_symbols
        else:
            symbols = random.sample(available_symbols, symbol_count)
        
        self.stdout.write(self.style.SUCCESS(f"✓ Selected {len(symbols)} symbols"))
        self.stdout.write(f"  Symbols: {', '.join([s.ticker for s in symbols[:10]])}{'...' if len(symbols) > 10 else ''}")
        
        # Set dates (use last year as default)
        end_date = timezone.now()
        start_date = end_date - timedelta(days=365)
        
        # Create backtest
        backtest = Backtest.objects.create(
            name=f'Test Backtest - {len(symbols)} symbols - Gap-Up/Gap-Down',
            strategy=strategy,
            broker=broker,
            start_date=start_date,
            end_date=end_date,
            split_ratio=0.2,  # 20% training, 80% test
            initial_capital=1000.0,
            bet_size_percentage=10.0,  # 10% bet size
            strategy_parameters={
                'threshold': 1.0,
                'std_period': 90
            },
            status='pending'
        )
        
        # Add symbols
        backtest.symbols.set(symbols)
        
        self.stdout.write(self.style.SUCCESS(f"✓ Created backtest (ID: {backtest.id})"))
        self.stdout.write(f"  Strategy: {strategy.name}")
        self.stdout.write(f"  Broker: {broker.name}")
        self.stdout.write(f"  Symbols: {len(symbols)}")
        self.stdout.write(f"  Initial Capital: $1000")
        self.stdout.write(f"  Bet Size: 10%")
        self.stdout.write(f"  Split Ratio: 20% training / 80% test")
        self.stdout.write(f"  Strategy Parameters: threshold=1.0, std_period=90")
        self.stdout.write(f"  Date Range: {start_date.date()} to {end_date.date()}")
        
        # Start backtest task
        try:
            task = run_backtest_task.delay(backtest.id)
            self.stdout.write(self.style.SUCCESS(f"✓ Started backtest task (Task ID: {task.id})"))
            self.stdout.write(f"  Backtest ID: {backtest.id}")
            self.stdout.write(f"  You can monitor progress via the API or admin panel")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"✗ Error starting backtest task: {str(e)}"))
            backtest.status = 'failed'
            backtest.error_message = f"Error starting backtest task: {str(e)}"
            backtest.save()


