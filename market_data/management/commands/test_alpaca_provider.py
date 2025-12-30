"""
Management command to test Alpaca provider with random symbols
Usage: python manage.py test_alpaca_provider
"""

from django.core.management.base import BaseCommand
from django.db.models import Q
from market_data.models import Symbol, Provider, OHLCV
from live_trading.models import Broker, SymbolBrokerAssociation
import random


class Command(BaseCommand):
    help = 'Test Alpaca provider by fetching OHLCV data for 10 random symbols linked to Alpaca broker without OHLCV data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            type=int,
            default=10,
            help='Number of symbols to test (default: 10)'
        )
        parser.add_argument(
            '--broker-id',
            type=int,
            default=None,
            help='Specific broker ID (default: finds Alpaca broker automatically)'
        )
        parser.add_argument(
            '--start-date',
            type=str,
            default=None,
            help='Start date in YYYY-MM-DD format (default: 30 days ago)'
        )
        parser.add_argument(
            '--end-date',
            type=str,
            default=None,
            help='End date in YYYY-MM-DD format (default: today)'
        )

    def handle(self, *args, **options):
        num_symbols = options['symbols']
        broker_id = options['broker_id']
        
        # Get Alpaca broker
        if broker_id:
            try:
                broker = Broker.objects.get(id=broker_id)
                if broker.code != 'ALPACA':
                    self.stdout.write(
                        self.style.WARNING(f'Warning: Broker {broker_id} is not ALPACA (code: {broker.code})')
                    )
            except Broker.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR(f'Broker with ID {broker_id} not found')
                )
                return
        else:
            try:
                broker = Broker.objects.get(code='ALPACA')
            except Broker.DoesNotExist:
                self.stdout.write(
                    self.style.ERROR('Alpaca broker not found. Please create it first.')
                )
                return
        
        self.stdout.write(f'Using broker: {broker.name} (ID: {broker.id}, Code: {broker.code})')
        
        # Get Alpaca provider
        try:
            provider = Provider.objects.get(code='ALPACA', is_active=True)
            self.stdout.write(f'Using provider: {provider.name} (Code: {provider.code})')
        except Provider.DoesNotExist:
            self.stdout.write(
                self.style.ERROR('Alpaca provider not found. Please create it first using: python manage.py create_alpaca_provider --api-key <KEY> --api-secret <SECRET>')
            )
            return
        
        # Get symbols linked to Alpaca broker with at least one active flag
        associations = SymbolBrokerAssociation.objects.filter(
            broker=broker
        ).filter(
            Q(long_active=True) | Q(short_active=True)
        ).select_related('symbol')
        
        if not associations.exists():
            self.stdout.write(
                self.style.ERROR('No symbols found linked to Alpaca broker with active trading flags')
            )
            return
        
        # Filter symbols that don't have OHLCV data yet
        symbols_without_data = []
        for assoc in associations:
            symbol = assoc.symbol
            # Check if symbol has any OHLCV data
            has_data = OHLCV.objects.filter(symbol=symbol).exists()
            if not has_data:
                symbols_without_data.append(symbol)
        
        if not symbols_without_data:
            self.stdout.write(
                self.style.WARNING('All linked symbols already have OHLCV data. No symbols to test.')
            )
            return
        
        self.stdout.write(f'Found {len(symbols_without_data)} symbols without OHLCV data')
        
        # Select random symbols
        num_to_select = min(num_symbols, len(symbols_without_data))
        selected_symbols = random.sample(symbols_without_data, num_to_select)
        
        self.stdout.write(self.style.SUCCESS(f'\nSelected {num_to_select} symbols for testing:'))
        for symbol in selected_symbols:
            self.stdout.write(f'  - {symbol.ticker} ({symbol.exchange.code})')
        
        # Test fetching data using the task
        self.stdout.write(self.style.SUCCESS(f'\nFetching OHLCV data using Alpaca provider...'))
        
        from market_data.tasks import fetch_ohlcv_data_multiple_symbols_task
        from datetime import datetime, timedelta
        from django.utils import timezone
        
        # Use provided dates or default to last 30 days
        if options['end_date']:
            end_date = datetime.strptime(options['end_date'], '%Y-%m-%d')
            if timezone.is_naive(end_date):
                end_date = timezone.make_aware(end_date)
        else:
            end_date = timezone.now()
        
        if options['start_date']:
            start_date = datetime.strptime(options['start_date'], '%Y-%m-%d')
            if timezone.is_naive(start_date):
                start_date = timezone.make_aware(start_date)
        else:
            start_date = end_date - timedelta(days=30)
        
        tickers = [symbol.ticker for symbol in selected_symbols]
        
        self.stdout.write(f'Date range: {start_date.date()} to {end_date.date()}')
        self.stdout.write(f'Starting task...')
        
        # Use Celery's apply() for synchronous execution
        # apply() runs the task synchronously and returns the result
        result = fetch_ohlcv_data_multiple_symbols_task.apply(
            args=(tickers,),
            kwargs={
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'period': None,
                'replace_existing': False,
                'broker_id': None,
                'provider_code': 'ALPACA'
            }
        )
        
        # Get the result value (apply() returns an AsyncResult)
        result = result.result
        
        self.stdout.write(self.style.SUCCESS(f'\nTask completed!'))
        self.stdout.write(f'Result status: {result.get("status", "unknown")}')
        
        if isinstance(result, dict) and 'result' in result:
            results = result['result']
            total_created = sum(r.get('created', 0) for r in results.values())
            total_updated = sum(r.get('updated', 0) for r in results.values())
            total_errors = sum(r.get('errors', 0) for r in results.values())
            
            self.stdout.write(f'\nSummary:')
            self.stdout.write(f'  Created: {total_created} records')
            self.stdout.write(f'  Updated: {total_updated} records')
            self.stdout.write(f'  Errors: {total_errors} symbols')
            
            self.stdout.write(f'\nPer-symbol results:')
            for ticker, symbol_result in results.items():
                created = symbol_result.get('created', 0)
                updated = symbol_result.get('updated', 0)
                errors = symbol_result.get('errors', 0)
                message = symbol_result.get('message', '')
                status = '✓' if errors == 0 else '✗'
                
                self.stdout.write(f'  {status} {ticker}: Created={created}, Updated={updated}, Errors={errors}')
                if message and 'error' in message.lower():
                    self.stdout.write(f'      Message: {message[:100]}')
        
        # Verify data was saved
        self.stdout.write(f'\nVerifying saved data...')
        for symbol in selected_symbols:
            count = OHLCV.objects.filter(symbol=symbol).count()
            if count > 0:
                self.stdout.write(self.style.SUCCESS(f'  ✓ {symbol.ticker}: {count} OHLCV records'))
            else:
                self.stdout.write(self.style.WARNING(f'  ✗ {symbol.ticker}: No OHLCV records found'))

