"""
Signals for automatically creating/updating strategy definitions
"""

from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import StrategyDefinition
from .config import STRATEGY_DEFINITIONS


@receiver(post_migrate)
def create_strategy_definitions(sender, **kwargs):
    """Create or update strategy definitions after migrations"""
    if sender.name != 'strategies':
        return
    
    for strategy_data in STRATEGY_DEFINITIONS:
        strategy, created = StrategyDefinition.objects.update_or_create(
            name=strategy_data['name'],
            defaults={
                'description_short': strategy_data.get('description_short', ''),
                'description_long': strategy_data.get('description_long', ''),
                'default_parameters': strategy_data.get('default_parameters', {}),
                'analytic_tools_used': strategy_data.get('analytic_tools_used', []),
                'required_tool_configs': strategy_data.get('required_tool_configs', []),
                'example_code': strategy_data.get('example_code', ''),
                'globally_enabled': strategy_data.get('globally_enabled', False),
            }
        )
        if created:
            print(f"Created strategy definition: {strategy.name}")
        else:
            print(f"Updated strategy definition: {strategy.name}")

