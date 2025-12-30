"""
Management command to create Alpaca provider
Usage: python manage.py create_alpaca_provider --api-key <KEY> --api-secret <SECRET> [--base-url <URL>]
"""

from django.core.management.base import BaseCommand
from market_data.models import Provider


class Command(BaseCommand):
    help = 'Create or update Alpaca provider with API credentials'

    def add_arguments(self, parser):
        parser.add_argument(
            '--api-key',
            type=str,
            required=True,
            help='Alpaca API Key ID'
        )
        parser.add_argument(
            '--api-secret',
            type=str,
            required=True,
            help='Alpaca API Secret Key'
        )
        parser.add_argument(
            '--base-url',
            type=str,
            default=None,
            help='Data API Base URL (defaults to https://data.alpaca.markets if not provided. Note: This is the DATA API, not the trading API)'
        )

    def handle(self, *args, **options):
        api_key = options['api_key']
        api_secret = options['api_secret']
        # Use data API URL for market data (not trading API)
        base_url = options['base_url'] or 'https://data.alpaca.markets'

        provider, created = Provider.objects.get_or_create(
            code='ALPACA',
            defaults={
                'name': 'Alpaca Markets',
                'api_key': api_key,
                'secret_access_key': api_secret,
                'base_url': base_url,
                'is_active': True
            }
        )

        if not created:
            # Update existing provider
            provider.api_key = api_key
            provider.secret_access_key = api_secret
            provider.base_url = base_url
            provider.is_active = True
            provider.save()
            self.stdout.write(
                self.style.SUCCESS(f'Successfully updated Alpaca provider')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created Alpaca provider')
            )

        self.stdout.write(f'Provider: {provider.name}')
        self.stdout.write(f'Code: {provider.code}')
        self.stdout.write(f'Base URL: {provider.base_url}')
        self.stdout.write(f'Active: {provider.is_active}')

