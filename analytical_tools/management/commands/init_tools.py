"""
Management command to initialize default analytical tools
"""

from django.core.management.base import BaseCommand
from analytical_tools.models import ToolDefinition


class Command(BaseCommand):
    help = 'Initialize default analytical tools/indicators'

    def handle(self, *args, **options):
        tools = [
            {
                'name': 'RSI',
                'description': 'Relative Strength Index - Momentum oscillator that measures speed and magnitude of price changes',
                'category': 'indicator',
                'default_parameters': {'period': 14}
            },
            {
                'name': 'ATR',
                'description': 'Average True Range - Measures market volatility',
                'category': 'indicator',
                'default_parameters': {'period': 14}
            },
            {
                'name': 'MACD',
                'description': 'Moving Average Convergence Divergence - Trend-following momentum indicator',
                'category': 'indicator',
                'default_parameters': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                }
            },
            {
                'name': 'Variance',
                'description': 'Statistical variance - Measures the spread of price data',
                'category': 'statistical',
                'default_parameters': {'period': 20}
            },
            {
                'name': 'SMA',
                'description': 'Simple Moving Average - Average of prices over a period',
                'category': 'indicator',
                'default_parameters': {'period': 20}
            },
            {
                'name': 'EMA',
                'description': 'Exponential Moving Average - Weighted average giving more importance to recent prices',
                'category': 'indicator',
                'default_parameters': {'period': 20}
            },
        ]

        created_count = 0
        updated_count = 0

        for tool_data in tools:
            tool, created = ToolDefinition.objects.update_or_create(
                name=tool_data['name'],
                defaults={
                    'description': tool_data['description'],
                    'category': tool_data['category'],
                    'default_parameters': tool_data['default_parameters']
                }
            )
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created tool: {tool.name}')
                )
            else:
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'Updated tool: {tool.name}')
                )

        self.stdout.write(
            self.style.SUCCESS(
                f'\nInitialization complete: {created_count} created, {updated_count} updated'
            )
        )

