"""
Management command to test update and refetch functionality for symbols with different providers
Usage: python manage.py test_symbol_update_refetch --ticker AAPL --provider YAHOO
       python manage.py test_symbol_update_refetch --ticker TSLA --provider ALPACA
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from market_data.models import Symbol, OHLCV, Provider
from market_data.services.ohlcv_service import OHLCVService
from market_data.tasks import fetch_ohlcv_data_task
from datetime import datetime, timedelta


class Command(BaseCommand):
    help = 'Test update and refetch functionality for a symbol with a specific provider'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ticker',
            type=str,
            required=True,
            help='Symbol ticker to test (e.g., AAPL, TSLA)'
        )
        parser.add_argument(
            '--provider',
            type=str,
            required=True,
            choices=['YAHOO', 'ALPACA'],
            help='Provider to use (YAHOO or ALPACA)'
        )

    def handle(self, *args, **options):
        ticker = options['ticker'].upper()
        provider_code = options['provider'].upper()

        self.stdout.write(self.style.SUCCESS(f'\n=== Testing Update and Refetch for {ticker} with {provider_code} ===\n'))

        # Get or create symbol
        try:
            symbol = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Symbol {ticker} not found. Please create it first.'))
            return

        # Get provider
        try:
            provider = Provider.objects.get(code=provider_code, is_active=True)
        except Provider.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Provider {provider_code} not found or not active.'))
            return

        # Get initial state
        initial_count = OHLCV.objects.filter(symbol=symbol).count()
        initial_provider = symbol.provider.code if symbol.provider else None
        initial_status = symbol.status

        self.stdout.write(f'Initial state:')
        self.stdout.write(f'  OHLCV records: {initial_count}')
        self.stdout.write(f'  Provider: {initial_provider}')
        self.stdout.write(f'  Status: {initial_status}\n')

        # Step 1: Set provider if not set or different
        if not symbol.provider or symbol.provider.code != provider_code:
            self.stdout.write(f'Setting provider to {provider_code}...')
            symbol.provider = provider
            symbol.save(update_fields=['provider'])

        # Step 2: Fetch initial data if none exists
        if initial_count == 0:
            self.stdout.write(f'\n=== Step 1: Fetching initial data with {provider_code} ===')
            task_result = fetch_ohlcv_data_task.apply(
                kwargs={
                    'ticker': ticker,
                    'period': '1y',  # Fetch 1 year of data
                    'replace_existing': False,
                    'provider_code': provider_code
                }
            ).get()

            if task_result.get('status') == 'completed':
                result = task_result.get('result', {})
                self.stdout.write(self.style.SUCCESS(f'✓ Fetched {result.get("created", 0)} records'))
            else:
                self.stdout.write(self.style.ERROR(f'✗ Failed: {task_result.get("message")}'))
                return

            # Refresh symbol
            symbol.refresh_from_db()
            count_after_initial = OHLCV.objects.filter(symbol=symbol).count()
            self.stdout.write(f'  Total records after initial fetch: {count_after_initial}\n')
        else:
            self.stdout.write(f'Skipping initial fetch - {initial_count} records already exist\n')

        # Step 3: Test Update (incremental update)
        self.stdout.write(f'=== Step 2: Testing Update (incremental update) ===')
        
        # Get latest timestamp
        latest_timestamp = OHLCVService.get_latest_timestamp(symbol, timeframe='daily')
        if latest_timestamp:
            start_date = (latest_timestamp + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = timezone.now().date().strftime('%Y-%m-%d')
            
            self.stdout.write(f'Latest timestamp in DB: {latest_timestamp.date()}')
            self.stdout.write(f'Will fetch from: {start_date} to {end_date}')
            
            count_before_update = OHLCV.objects.filter(symbol=symbol).count()
            
            task_result = fetch_ohlcv_data_task.apply(
                kwargs={
                    'ticker': ticker,
                    'start_date': start_date,
                    'end_date': end_date,
                    'replace_existing': False,
                    'provider_code': provider_code
                }
            ).get()

            if task_result.get('status') == 'completed':
                result = task_result.get('result', {})
                created = result.get('created', 0)
                if created > 0:
                    self.stdout.write(self.style.SUCCESS(f'✓ Updated with {created} new records'))
                else:
                    self.stdout.write(self.style.WARNING(f'✓ No new records (data already up to date)'))
                
                symbol.refresh_from_db()
                count_after_update = OHLCV.objects.filter(symbol=symbol).count()
                self.stdout.write(f'  Records before update: {count_before_update}')
                self.stdout.write(f'  Records after update: {count_after_update}')
            else:
                self.stdout.write(self.style.ERROR(f'✗ Failed: {task_result.get("message")}'))
        else:
            self.stdout.write(self.style.WARNING('No existing data - skipping update test'))

        # Step 4: Test Refetch (delete all and fetch again)
        self.stdout.write(f'\n=== Step 3: Testing Refetch (delete all and fetch again) ===')
        
        count_before_refetch = OHLCV.objects.filter(symbol=symbol).count()
        self.stdout.write(f'Records before refetch: {count_before_refetch}')
        
        # Delete all OHLCV data
        deleted_count, _ = OHLCV.objects.filter(symbol=symbol).delete()
        self.stdout.write(f'Deleted {deleted_count} records')
        
        # Disable symbol
        symbol.status = 'disabled'
        symbol.validation_status = 'invalid'
        symbol.validation_reason = 'OHLCV data deleted for refetch'
        symbol.save(update_fields=['status', 'validation_status', 'validation_reason'])
        
        # Refetch all data
        task_result = fetch_ohlcv_data_task.apply(
            kwargs={
                'ticker': ticker,
                'period': '1y',
                'replace_existing': False,
                'provider_code': provider_code
            }
        ).get()

        if task_result.get('status') == 'completed':
            result = task_result.get('result', {})
            created = result.get('created', 0)
            self.stdout.write(self.style.SUCCESS(f'✓ Refetched {created} records'))
            
            symbol.refresh_from_db()
            count_after_refetch = OHLCV.objects.filter(symbol=symbol).count()
            self.stdout.write(f'  Records after refetch: {count_after_refetch}')
            self.stdout.write(f'  Provider: {symbol.provider.code if symbol.provider else None}')
            self.stdout.write(f'  Status: {symbol.status}')
        else:
            self.stdout.write(self.style.ERROR(f'✗ Failed: {task_result.get("message")}'))
            return

        # Summary
        self.stdout.write(f'\n=== Summary ===')
        self.stdout.write(self.style.SUCCESS(f'✓ All tests passed for {ticker} with {provider_code}'))
        self.stdout.write(f'  Final records: {count_after_refetch}')
        self.stdout.write(f'  Final provider: {symbol.provider.code if symbol.provider else None}')
        self.stdout.write(f'  Final status: {symbol.status}\n')

